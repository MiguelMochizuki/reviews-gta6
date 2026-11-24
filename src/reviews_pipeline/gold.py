"""Gold layer: daily and hourly aggregates of silver, with no review-level data."""

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import duckdb

from reviews_pipeline.config import PLATFORMS

# Score bands of the published aggregates. They match Metacritic's sentiment filters
# only partly: its filters are 0-4 / 5-7 / 8-10, used below to find the collection window.
BANDS = {"low": (0, 1), "mid": (2, 8), "high": (9, 10)}
_SENTIMENT_BANDS = ((0, 4), (5, 7), (8, 10))


@dataclass
class Coverage:
    """Where the collected window of a platform starts.

    Attributes:
        window_start: Oldest `version_ms` that every sentiment query reaches, or None
            when the platform has no reviews.
        complete_from: First day with complete coverage, or None.
    """

    window_start: datetime | None
    complete_from: date | None


def valid_days(now: datetime) -> date:
    """Return the last `date` accepted in aggregates.

    Metacritic dates a few reviews one day ahead (time zones) and a few years ahead (junk).

    Args:
        now: Time of the run.

    Returns:
        date: The day after `now`, in UTC.
    """
    return now.astimezone(UTC).date() + timedelta(days=1)


def coverage(con: duckdb.DuckDBPyConnection) -> dict[str, Coverage]:
    """Find, per platform, the day from which the collected reviews are complete.

    Each sentiment query reaches at most 10,000 newest reviews, so a capped band only
    covers time since its oldest `version_ms`. The window starts at the latest such
    start over the bands that have rows, and the first complete day is the day after.

    Args:
        con: Initialized silver connection.

    Returns:
        dict[str, Coverage]: One entry per platform in `config.PLATFORMS`.
    """
    result = {}
    for platform in PLATFORMS:
        starts = []
        for low, high in _SENTIMENT_BANDS:
            (start,) = con.execute(
                "SELECT min(version_ms) FROM reviews "
                "WHERE platform = ? AND score BETWEEN ? AND ?",
                [platform, low, high],
            ).fetchone()
            if start is not None:
                starts.append(start)
        if not starts:
            result[platform] = Coverage(None, None)
            continue
        window_start = datetime.fromtimestamp(max(starts) / 1000, tz=UTC)
        result[platform] = Coverage(window_start, window_start.date() + timedelta(days=1))
    return result


def _band_counts() -> str:
    return ", ".join(
        f"count(*) FILTER (WHERE score BETWEEN {lo} AND {hi}) AS {name}"
        for name, (lo, hi) in BANDS.items()
    )


def daily(con: duckdb.DuckDBPyConnection, now: datetime) -> dict[str, list[dict]]:
    """Aggregate silver per platform and day.

    Args:
        con: Initialized silver connection.
        now: Time of the run. Days after `valid_days(now)` are left out.

    Returns:
        dict[str, list[dict]]: Per platform, rows sorted by date with the keys `date`, `n`,
            `mean`, `median`, `low`, `mid`, `high`, `repeated` (share of reviews whose
            normalized text appears in two or more reviews; empty text never counts) and
            `complete` (whether the day is at or after the platform's `complete_from`).
    """
    cover = coverage(con)
    last_day = valid_days(now)
    out: dict[str, list[dict]] = {}
    for platform in PLATFORMS:
        rows = con.execute(
            f"""
            WITH norm AS (
                SELECT date, score,
                       lower(trim(regexp_replace(text, '\\s+', ' ', 'g'))) AS key
                FROM reviews WHERE platform = ? AND date <= ?
            ), seen AS (
                SELECT key, count(*) AS c FROM norm WHERE key <> '' GROUP BY key
            )
            SELECT norm.date, count(*), avg(norm.score), median(norm.score), {_band_counts()},
                   avg(CASE WHEN seen.c >= 2 THEN 1.0 ELSE 0.0 END)
            FROM norm LEFT JOIN seen USING (key)
            GROUP BY norm.date ORDER BY norm.date
            """,
            [platform, last_day],
        ).fetchall()
        from_day = cover[platform].complete_from
        out[platform] = [
            {
                "date": day.isoformat(),
                "n": n,
                "mean": round(mean, 2),
                "median": median,
                "low": low,
                "mid": mid,
                "high": high,
                "repeated": round(repeated, 4),
                "complete": from_day is not None and day >= from_day,
            }
            for day, n, mean, median, low, mid, high, repeated in rows
        ]
    return out


def _iso(moment: datetime | None) -> str | None:
    return None if moment is None else moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def hourly(con: duckdb.DuckDBPyConnection, now: datetime) -> dict[str, list[dict]]:
    """Count reviews per UTC hour of their last write (`version_ms`), in complete hours only.

    Args:
        con: Initialized silver connection.
        now: Time of the run. Later versions are left out.

    Returns:
        dict[str, list[dict]]: Per platform, rows `{"hour": "...Z", "n": int}` sorted by hour,
            starting at the first full hour at or after the platform's window start.
    """
    now_ms = int(now.timestamp() * 1000)
    out: dict[str, list[dict]] = {}
    for platform, cover in coverage(con).items():
        if cover.window_start is None:
            out[platform] = []
            continue
        start = cover.window_start
        first_ms = int(start.replace(minute=0, second=0, microsecond=0).timestamp() * 1000)
        if start.minute or start.second or start.microsecond:
            first_ms += 3_600_000
        rows = con.execute(
            "SELECT version_ms // 3600000 * 3600000 AS hour_ms, count(*) FROM reviews "
            "WHERE platform = ? AND version_ms >= ? AND version_ms <= ? "
            "GROUP BY hour_ms ORDER BY hour_ms",
            [platform, first_ms, now_ms],
        ).fetchall()
        out[platform] = [
            {"hour": _iso(datetime.fromtimestamp(hour_ms / 1000, tz=UTC)), "n": n}
            for hour_ms, n in rows
        ]
    return out


DRIFT_THRESHOLD_PP = 15.0
DRIFT_MIN_PRIOR_DAYS = 3


def _drift(days: list[dict]) -> dict | None:
    """Compare the latest complete day's low/high score shares with the earlier days' mean."""
    complete = [day for day in days if day["complete"]]
    if len(complete) <= DRIFT_MIN_PRIOR_DAYS:
        return None
    *prior, latest = complete
    best = None
    for metric in ("low", "high"):
        mean_share = sum(day[metric] / day["n"] for day in prior) / len(prior)
        delta = (latest[metric] / latest["n"] - mean_share) * 100
        if abs(delta) > DRIFT_THRESHOLD_PP and (best is None or abs(delta) > abs(best[1])):
            best = (metric, delta)
    if best is None:
        return None
    return {"date": latest["date"], "metric": best[0], "delta_pp": round(best[1], 1)}


def meta(con: duckdb.DuckDBPyConnection, now: datetime) -> dict:
    """Describe how fresh and how complete the aggregates are.

    Args:
        con: Initialized silver connection.
        now: Time of the run.

    Returns:
        dict: `generated_at`, `last_success_at` (latest `last_seen_at`), `newest_review_at`
            (latest `version_ms` not after `now`), `rows`, `excluded_future_dates`,
            `coverage` and `drift` per platform. Times are ISO UTC strings or None.
    """
    now = now.astimezone(UTC)
    now_ms = int(now.timestamp() * 1000)
    last_seen, newest_ms, rows, excluded = con.execute(
        "SELECT max(last_seen_at), max(version_ms) FILTER (WHERE version_ms <= ?), "
        "count(*), count(*) FILTER (WHERE date > ?) FROM reviews",
        [now_ms, valid_days(now)],
    ).fetchone()
    days = daily(con, now)
    return {
        "generated_at": _iso(now),
        "last_success_at": _iso(last_seen.replace(tzinfo=UTC)) if last_seen else None,
        "newest_review_at": (
            _iso(datetime.fromtimestamp(newest_ms / 1000, tz=UTC)) if newest_ms else None
        ),
        "rows": rows,
        "excluded_future_dates": excluded,
        "coverage": {
            platform: {
                "window_start": _iso(cover.window_start),
                "complete_from": cover.complete_from.isoformat() if cover.complete_from else None,
            }
            for platform, cover in coverage(con).items()
        },
        "drift": {platform: _drift(days[platform]) for platform in PLATFORMS},
    }


def build(con: duckdb.DuckDBPyConnection, now: datetime) -> dict[str, dict]:
    """Build the three published documents.

    Args:
        con: Initialized silver connection.
        now: Time of the run.

    Returns:
        dict[str, dict]: `meta`, `daily` and `hourly`, ready to be written as JSON.
    """
    return {
        "meta": meta(con, now),
        "daily": {"platforms": daily(con, now)},
        "hourly": {"platforms": hourly(con, now)},
    }

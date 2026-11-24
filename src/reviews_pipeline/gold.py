"""Gold layer: daily and hourly aggregates of silver, with no review-level data."""

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb

from reviews_pipeline import silver
from reviews_pipeline.config import PLATFORMS

log = logging.getLogger("reviews_pipeline")

ALL = "all"  # pseudo-platform: both platforms together
VIEWS = (*PLATFORMS, ALL)

# Score bands of the published aggregates. They match Metacritic's sentiment filters
# only partly: its filters are 0-4 / 5-7 / 8-10, used below to find the collection window.
BANDS = {"low": (0, 1), "mid": (2, 8), "high": (9, 10)}
_SENTIMENT_BANDS = ((0, 4), (5, 7), (8, 10))
# A sentiment query reaches at most 10,000 reviews (config.MAX_OFFSET + PAGE_LIMIT); a band
# holding at least this many rows is treated as capped, with some room for duplicates.
CAPPED_ROWS = 9_900


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

    Metacritic dates a handful of reviews tomorrow or years ahead (junk, time zones); they
    would show up as points in the future, so they are excluded and counted instead.

    Args:
        now: Time of the run.

    Returns:
        date: Today, in UTC.
    """
    return now.astimezone(UTC).date()


def coverage(con: duckdb.DuckDBPyConnection) -> dict[str, Coverage]:
    """Find, per platform, the day from which the collected reviews are complete.

    Each sentiment query reaches at most 10,000 newest reviews, so a capped band only
    covers time since its oldest `version_ms`. The window starts at the latest such
    start over the capped bands (those with at least `CAPPED_ROWS` rows), and the first
    complete day is the day after. When no band is capped, every review the platform has
    is collected, so it is complete from the day of its first review.

    Args:
        con: Initialized silver connection.

    Returns:
        dict[str, Coverage]: One entry per platform in `config.PLATFORMS`, plus `all`, which
            is complete only from the day every platform with reviews is complete.
    """
    result = {}
    for platform in PLATFORMS:
        starts, capped = [], []
        for low, high in _SENTIMENT_BANDS:
            start, rows = con.execute(
                "SELECT min(version_ms), count(*) FROM reviews "
                "WHERE platform = ? AND score BETWEEN ? AND ?",
                [platform, low, high],
            ).fetchone()
            if start is not None:
                starts.append(start)
                if rows >= CAPPED_ROWS:
                    capped.append(start)
        if not starts:
            result[platform] = Coverage(None, None)
        elif capped:
            window_start = datetime.fromtimestamp(max(capped) / 1000, tz=UTC)
            result[platform] = Coverage(window_start, window_start.date() + timedelta(days=1))
        else:
            window_start = datetime.fromtimestamp(min(starts) / 1000, tz=UTC)
            result[platform] = Coverage(window_start, window_start.date())
    known = [cover for cover in result.values() if cover.window_start is not None]
    result[ALL] = (
        Coverage(max(c.window_start for c in known), max(c.complete_from for c in known))
        if known else Coverage(None, None)
    )
    return result


def _band_counts() -> str:
    return ", ".join(
        f"count(*) FILTER (WHERE score BETWEEN {lo} AND {hi}) AS {name}"
        for name, (lo, hi) in BANDS.items()
    )


def daily(con: duckdb.DuckDBPyConnection, now: datetime) -> dict[str, list[dict]]:
    """Aggregate silver per platform and day, plus `all` for both platforms together.

    Args:
        con: Initialized silver connection.
        now: Time of the run. Days after `valid_days(now)` (today) are left out.

    Returns:
        dict[str, list[dict]]: Per platform and `all`, rows sorted by date with the keys `date`, `n`,
            `mean`, `median`, `low`, `mid`, `high`, `repeated` (share of reviews whose
            normalized text appears in two or more reviews on any platform; empty text never
            counts) and `complete` (the day is at or after the platform's `complete_from`
            and is over, so today is never complete).
    """
    cover = coverage(con)
    last_day = valid_days(now)
    out: dict[str, list[dict]] = {}
    for platform in VIEWS:
        rows = con.execute(
            f"""
            WITH norm AS (
                SELECT platform, date, score,
                       lower(trim(regexp_replace(text, '\\s+', ' ', 'g'))) AS key
                FROM reviews WHERE date <= ?
            ), seen AS (
                SELECT key, count(*) AS c FROM norm WHERE key <> '' GROUP BY key
            )
            SELECT norm.date, count(*), avg(norm.score), median(norm.score), {_band_counts()},
                   avg(CASE WHEN seen.c >= 2 THEN 1.0 ELSE 0.0 END)
            FROM norm LEFT JOIN seen USING (key)
            WHERE (? = 'all' OR norm.platform = ?)
            GROUP BY norm.date ORDER BY norm.date
            """,
            [last_day, platform, platform],
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
                "complete": from_day is not None and from_day <= day < last_day,
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
        now: Time of the run.

    Returns:
        dict[str, list[dict]]: Per platform and `all`, rows `{"hour": "...Z", "n": int}` sorted by hour,
            from the first full hour at or after the platform's window start up to the hour
            before the latest collection started (an hour still being filled is left out).
    """
    (last_seen,) = con.execute("SELECT max(last_seen_at) FROM reviews").fetchone()
    if last_seen is None:
        return {view: [] for view in VIEWS}
    # The collection that wrote silver started at `last_seen`: the hour it started in was
    # still being filled, so only hours before it are complete.
    cutoff = min(now, last_seen.replace(tzinfo=UTC)).replace(minute=0, second=0, microsecond=0)
    cutoff_ms = int(cutoff.timestamp() * 1000)
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
            "WHERE (? = 'all' OR platform = ?) AND version_ms >= ? AND version_ms < ? "
            "GROUP BY hour_ms ORDER BY hour_ms",
            [platform, platform, first_ms, cutoff_ms],
        ).fetchall()
        out[platform] = [
            {"hour": _iso(datetime.fromtimestamp(hour_ms / 1000, tz=UTC)), "n": n}
            for hour_ms, n in rows
        ]
    return out


DRIFT_THRESHOLD_PP = 15.0
DRIFT_MIN_PRIOR_DAYS = 3
DRIFT_MIN_LATEST_REVIEWS = 30  # below this a share swings by chance, and the page draws none


def _drift(days: list[dict], today: date) -> dict | None:
    """Compare the latest finished complete day's low/high shares with the earlier days, pooled."""
    complete = [day for day in days if day["complete"] and day["date"] < today.isoformat()]
    if len(complete) <= DRIFT_MIN_PRIOR_DAYS:
        return None
    *prior, latest = complete
    if latest["n"] < DRIFT_MIN_LATEST_REVIEWS:
        return None
    best = None
    prior_reviews = sum(day["n"] for day in prior)
    for metric in ("low", "high"):
        prior_share = sum(day[metric] for day in prior) / prior_reviews  # pooled, not a mean
        delta = (latest[metric] / latest["n"] - prior_share) * 100
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
        "drift": {view: _drift(days[view], valid_days(now)) for view in VIEWS},
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


def write(con: duckdb.DuckDBPyConnection, out_dir: Path, now: datetime) -> None:
    """Write `meta.json`, `daily.json` and `hourly.json` for the site.

    Args:
        con: Initialized silver connection.
        now: Time of the run.
        out_dir: Target directory, created if missing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, document in build(con, now).items():
        (out_dir / f"{name}.json").write_text(
            json.dumps(document, sort_keys=True, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> int:
    """Build the site data from saved silver files.

    Args:
        argv: Command-line arguments (`--state` directory with `reviews.parquet`, `--out`
            directory for the JSON files). Defaults to `sys.argv`.

    Returns:
        int: 0 on success, 1 when the state has no silver files (nothing is written).
    """
    parser = argparse.ArgumentParser(description="Build gold aggregates for the site.")
    parser.add_argument("--state", type=Path, default=Path("data/state"))
    parser.add_argument("--out", type=Path, default=Path("site/data"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not (args.state / "reviews.parquet").exists():
        log.error("no reviews.parquet in %s; nothing to aggregate", args.state)
        return 1
    con = duckdb.connect()
    silver.load(con, args.state)
    write(con, args.out, datetime.now(UTC))
    log.info("gold written to %s from %d reviews", args.out, silver.count(con))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

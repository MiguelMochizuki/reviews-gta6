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

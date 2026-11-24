"""Data contracts for raw items and silver, plus the bronze writer."""

import json
import uuid
from datetime import date, datetime
from pathlib import Path

import duckdb

from reviews_pipeline.config import PLATFORMS


class ContractError(Exception):
    """A raw page or the silver table broke the data contract."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors[:5]))
        self.errors = errors


def _is_uuid(value) -> bool:
    try:
        uuid.UUID(str(value))
    except ValueError:
        return False
    return True


def _is_iso_date(value) -> bool:
    try:
        date.fromisoformat(str(value))
    except ValueError:
        return False
    return True


def check_raw_items(items: list[dict]) -> list[str]:
    """Check raw items against the data contract.

    Args:
        items: Raw items as returned by the endpoint.

    Returns:
        list[str]: One message per broken field, never including review text. Empty
            when every item is valid.
    """
    errors = []
    for index, item in enumerate(items):
        if not _is_uuid(item.get("id")):
            errors.append(f"item {index}: invalid id")
        score = item.get("score")
        if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 10:
            errors.append(f"item {index}: score out of range 0-10")
        if not _is_iso_date(item.get("date")):
            errors.append(f"item {index}: invalid date")
        version = item.get("version")
        if not isinstance(version, int) or isinstance(version, bool):
            errors.append(f"item {index}: invalid version")
    return errors


def check_silver(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Check silver for scores outside 0-10 and unknown platforms.

    Args:
        con: Initialized connection.

    Returns:
        list[str]: One message per problem found. Empty when silver is valid.
    """
    errors = []
    bad_scores = con.execute(
        "SELECT count(*) FROM reviews WHERE score IS NULL OR score NOT BETWEEN 0 AND 10"
    ).fetchone()[0]
    if bad_scores:
        errors.append(f"silver: {bad_scores} rows with score outside 0-10")
    bad_platforms = con.execute(
        f"SELECT count(*) FROM reviews WHERE platform NOT IN ({','.join('?' * len(PLATFORMS))})",
        list(PLATFORMS),
    ).fetchone()[0]
    if bad_platforms:
        errors.append(f"silver: {bad_platforms} rows with unknown platform")
    return errors


def write_bronze(
    items: list[dict],
    *,
    fetched_at: datetime,
    platform: str,
    sentiment: str,
    sort: str,
    path: Path,
) -> None:
    """Write raw items, as returned, to one Parquet file. Private storage only.

    Args:
        items: Raw items to store.
        fetched_at: When the items were fetched.
        platform: Platform of the query.
        sentiment: Sentiment filter of the query.
        sort: Sort order of the query.
        path: Target Parquet file. Parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE bronze (fetched_at TIMESTAMP, platform VARCHAR, "
        "sentiment VARCHAR, sort VARCHAR, raw_json VARCHAR)"
    )
    con.executemany(
        "INSERT INTO bronze VALUES (?, ?, ?, ?, ?)",
        [
            (fetched_at, platform, sentiment, sort, json.dumps(item, ensure_ascii=False))
            for item in items
        ],
    )
    con.execute(f"COPY bronze TO '{path}' (FORMAT PARQUET)")

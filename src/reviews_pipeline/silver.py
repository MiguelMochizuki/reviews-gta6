"""Silver layer: latest review per id plus every distinct version, in DuckDB."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb

_COLS = (
    "source, source_review_id, platform, score, date, version_ms, "
    "author_hash, text, spoiler, content_hash"
)
_VERSION_KEY = "source, source_review_id, version_ms, content_hash"
_TABLES = ("reviews", "review_versions")


@dataclass
class UpsertStats:
    """Outcome of one upsert."""

    inserted: int
    updated: int
    unchanged: int


def init(con: duckdb.DuckDBPyConnection) -> None:
    """Create the `reviews` and `review_versions` tables if they do not exist.

    Args:
        con: Connection to initialize.
    """
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            source VARCHAR, source_review_id VARCHAR, platform VARCHAR, score INTEGER,
            date DATE, version_ms BIGINT, author_hash VARCHAR, text VARCHAR,
            spoiler BOOLEAN, content_hash VARCHAR,
            first_seen_at TIMESTAMP, last_seen_at TIMESTAMP,
            PRIMARY KEY (source, source_review_id))
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS review_versions (
            source VARCHAR, source_review_id VARCHAR, version_ms BIGINT,
            content_hash VARCHAR, score INTEGER, text VARCHAR, seen_at TIMESTAMP,
            PRIMARY KEY (source, source_review_id, version_ms, content_hash))
        """
    )


def count(con: duckdb.DuckDBPyConnection) -> int:
    """Return the number of reviews in silver.

    Args:
        con: Initialized connection.

    Returns:
        int: Row count of `reviews`.
    """
    return con.execute("SELECT count(*) FROM reviews").fetchone()[0]


def _naive_utc(moment: datetime) -> datetime:
    return moment.astimezone(UTC).replace(tzinfo=None) if moment.tzinfo else moment


def upsert(con: duckdb.DuckDBPyConnection, rows: list[dict], seen_at: datetime) -> UpsertStats:
    """Insert new reviews, record changed ones as new versions, touch unchanged ones.

    Args:
        con: Initialized connection.
        rows: Normalized rows. Repeated ids in one batch keep the last row.
        seen_at: Observation time. Aware datetimes are converted to naive UTC.

    Returns:
        UpsertStats: How many rows were inserted, updated and left unchanged.
    """
    if not rows:
        return UpsertStats(0, 0, 0)
    rows = list({(r["source"], r["source_review_id"]): r for r in rows}.values())
    seen_at = _naive_utc(seen_at)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE incoming (
            source VARCHAR, source_review_id VARCHAR, platform VARCHAR, score INTEGER,
            date DATE, version_ms BIGINT, author_hash VARCHAR, text VARCHAR,
            spoiler BOOLEAN, content_hash VARCHAR)
        """
    )
    con.executemany(
        "INSERT INTO incoming VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [tuple(r[c.strip()] for c in _COLS.split(",")) for r in rows],
    )
    known = "EXISTS (SELECT 1 FROM review_versions v WHERE {m})".format(
        m=" AND ".join(f"v.{c.strip()} = i.{c.strip()}" for c in _VERSION_KEY.split(","))
    )
    in_reviews = "EXISTS (SELECT 1 FROM reviews r WHERE r.source = i.source "
    in_reviews += "AND r.source_review_id = i.source_review_id)"

    def count_where(where: str) -> int:
        return con.execute(f"SELECT count(*) FROM incoming i WHERE {where}").fetchone()[0]

    stats = UpsertStats(
        inserted=count_where(f"NOT {in_reviews}"),
        updated=count_where(f"{in_reviews} AND NOT {known}"),
        unchanged=count_where(known),
    )
    con.execute(
        f"INSERT INTO reviews SELECT {_COLS}, ?, ? FROM incoming i WHERE NOT {in_reviews}",
        [seen_at, seen_at],
    )
    con.execute(
        """
        UPDATE reviews SET platform = i.platform, score = i.score, date = i.date,
            version_ms = i.version_ms, author_hash = i.author_hash, text = i.text,
            spoiler = i.spoiler, content_hash = i.content_hash
        FROM incoming i
        WHERE reviews.source = i.source AND reviews.source_review_id = i.source_review_id
          AND (reviews.content_hash IS DISTINCT FROM i.content_hash
               OR reviews.version_ms IS DISTINCT FROM i.version_ms)
        """
    )
    con.execute(
        "UPDATE reviews SET last_seen_at = ? FROM incoming i "
        "WHERE reviews.source = i.source AND reviews.source_review_id = i.source_review_id",
        [seen_at],
    )
    con.execute(
        f"INSERT INTO review_versions SELECT {_VERSION_KEY}, score, text, ? "
        f"FROM incoming i WHERE NOT {known}",
        [seen_at],
    )
    return stats


def is_known(con: duckdb.DuckDBPyConnection, rows: list[dict]) -> bool:
    """Check whether every row's id, version and content are already stored.

    Args:
        con: Initialized connection.
        rows: Normalized rows to look up.

    Returns:
        bool: True when all rows are known. False for an empty list.
    """
    if not rows:
        return False
    placeholders = ",".join(["(?, ?, ?, ?)"] * len(rows))
    params = [
        value
        for r in rows
        for value in (r["source"], r["source_review_id"], r["version_ms"], r["content_hash"])
    ]
    found = con.execute(
        f"SELECT count(*) FROM review_versions WHERE ({_VERSION_KEY}) IN (VALUES {placeholders})",
        params,
    ).fetchone()[0]
    return found == len(rows)


def save(con: duckdb.DuckDBPyConnection, directory: Path) -> None:
    """Export both tables to Parquet files in a directory.

    Args:
        con: Initialized connection.
        directory: Target directory, created if missing.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for table in _TABLES:
        con.execute(f"COPY {table} TO '{directory / (table + '.parquet')}' (FORMAT PARQUET)")


def load(con: duckdb.DuckDBPyConnection, directory: Path) -> None:
    """Create the tables and fill them from Parquet files, when present.

    Args:
        con: Connection to load into.
        directory: Directory with the Parquet files. A missing directory gives empty tables.
    """
    init(con)
    for table in _TABLES:
        path = directory / f"{table}.parquet"
        if path.exists():
            con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM read_parquet('{path}')")

"""Head and deep sweeps over one platform and sentiment."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import duckdb

from reviews_pipeline import silver
from reviews_pipeline.http import PoliteClient
from reviews_pipeline.quality import ContractError, check_raw_items, write_bronze
from reviews_pipeline.sources import metacritic


class EmptyRunError(Exception):
    """A run fetched nothing although the silver table already has reviews."""


@dataclass
class RunStats:
    fetched: int
    inserted: int
    updated: int
    unchanged: int
    pages: int

    def __add__(self, other: "RunStats") -> "RunStats":
        return RunStats(
            self.fetched + other.fetched,
            self.inserted + other.inserted,
            self.updated + other.updated,
            self.unchanged + other.unchanged,
            self.pages + other.pages,
        )


def sweep(
    client: PoliteClient,
    con: duckdb.DuckDBPyConnection,
    platform: str,
    sentiment: str,
    *,
    mode: Literal["head", "deep"],
    salt: str,
    seen_at: datetime,
    bronze_dir: Path,
) -> RunStats:
    """Walk pages newest first, validate, upsert into silver and write raw items to bronze.

    A head sweep stops after the first page whose reviews are all already stored; a deep
    sweep walks to the end. Whatever was collected before an error stays in silver and bronze.
    """
    stats = RunStats(0, 0, 0, 0, 0)
    fetched_items: list[dict] = []
    try:
        for page in metacritic.iter_pages(client, platform, sentiment):
            errors = check_raw_items(page)
            if errors:
                raise ContractError(errors)
            rows = [metacritic.normalize(item, platform, salt) for item in page]
            page_known = silver.is_known(con, rows)
            result = silver.upsert(con, rows, seen_at)
            fetched_items.extend(page)
            stats += RunStats(
                len(page), result.inserted, result.updated, result.unchanged, 1
            )
            if mode == "head" and page_known:
                break
    finally:
        if fetched_items:
            name = f"{seen_at:%Y%m%dT%H%M%S}-{platform}-{sentiment}-{mode}.parquet"
            write_bronze(
                fetched_items,
                fetched_at=seen_at,
                platform=platform,
                sentiment=sentiment,
                sort="date",
                path=bronze_dir / name,
            )
    return stats


def check_volume(stats: RunStats, previous_count: int) -> None:
    if stats.fetched == 0 and previous_count > 0:
        raise EmptyRunError("run fetched no reviews although silver already has some")

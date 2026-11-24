"""Head and deep sweeps over one platform and sentiment."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import duckdb

from reviews_pipeline import silver
from reviews_pipeline.http import PoliteClient
from reviews_pipeline.quality import ContractError, check_raw_items
from reviews_pipeline.sources import metacritic


class EmptyRunError(Exception):
    """A run fetched nothing although the silver table already has reviews."""


@dataclass
class RunStats:
    """Counts collected by one or more sweeps."""

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
    bronze: list[tuple[str, str, dict]],
) -> RunStats:
    """Walk pages newest first, validate, upsert into silver and collect raw items for bronze.

    A head sweep stops after the first page whose reviews are all already stored; a deep
    sweep walks to the end. Whatever was collected before an error stays in silver and in
    `bronze`.

    Args:
        client: Client used for every request.
        con: Initialized silver connection.
        platform: One of `config.PLATFORMS`.
        sentiment: `positive`, `neutral` or `negative`.
        mode: When to stop walking pages.
        salt: Secret used to hash author names.
        seen_at: Observation time.
        bronze: Collector the caller writes to bronze; each fetched item is appended as
            `(platform, sentiment, raw_item)`.

    Returns:
        RunStats: Counts of fetched items, inserted, updated and unchanged reviews, and pages.

    Raises:
        ContractError: If a page breaks the raw data contract.
        CircuitOpen: If the server keeps failing; earlier pages stay stored.
        ShortPage: If the server returns a smaller page than requested mid-walk.
    """
    stats = RunStats(0, 0, 0, 0, 0)
    for page in metacritic.iter_pages(client, platform, sentiment):
        errors = check_raw_items(page)
        if errors:
            raise ContractError(errors)
        rows = [metacritic.normalize(item, platform, salt) for item in page]
        page_known = silver.is_known(con, rows)
        result = silver.upsert(con, rows, seen_at)
        bronze.extend((platform, sentiment, item) for item in page)
        stats += RunStats(len(page), result.inserted, result.updated, result.unchanged, 1)
        if mode == "head" and page_known:
            break
    return stats


def check_volume(stats: RunStats, previous_count: int) -> None:
    """Fail loudly when a run fetched nothing although silver already has reviews.

    Args:
        stats: Totals of the run.
        previous_count: Reviews in silver before the run.

    Raises:
        EmptyRunError: If `stats.fetched` is 0 and `previous_count` is above 0.
    """
    if stats.fetched == 0 and previous_count > 0:
        raise EmptyRunError("run fetched no reviews although silver already has some")

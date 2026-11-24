"""Collection entry point: python -m reviews_pipeline.run --mode head|deep."""

import argparse
import logging
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import duckdb

from reviews_pipeline import silver
from reviews_pipeline.config import PLATFORMS, SENTIMENTS
from reviews_pipeline.http import CircuitOpen, PoliteClient
from reviews_pipeline.quality import ContractError, check_silver, write_bronze
from reviews_pipeline.sources.metacritic import ShortPage
from reviews_pipeline.state import GhReleaseStore
from reviews_pipeline.sweeps import EmptyRunError, RunStats, check_volume, sweep

log = logging.getLogger("reviews_pipeline")


def run(
    mode: Literal["head", "deep"],
    *,
    client,
    store,
    workdir: Path,
    salt: str,
    now: datetime,
) -> int:
    """Pull state, sweep every platform and sentiment, check, save and push.

    Only counts and parameters are logged, never review text or author names.

    Args:
        mode: Sweep mode.
        client: Client used for every request.
        store: Where state is pulled from and pushed to.
        workdir: Local directory for state and bronze files.
        salt: Secret used to hash author names. Empty means refuse to run.
        now: Observation time of the run.

    Returns:
        int: 0 on success, 1 on any failure.
    """
    if not salt:
        log.error("AUTHOR_SALT is not set; refusing to run")
        return 1
    store.pull(workdir)
    con = duckdb.connect()
    silver.load(con, workdir)
    previous = silver.count(con)
    bronze_dir = workdir / "bronze"
    shutil.rmtree(bronze_dir, ignore_errors=True)
    bronze: list[tuple[str, str, dict]] = []

    total, interrupted = RunStats(0, 0, 0, 0, 0), False
    for platform in PLATFORMS:
        for sentiment in SENTIMENTS:
            try:
                stats = sweep(client, con, platform, sentiment, mode=mode, salt=salt,
                              seen_at=now, bronze=bronze)
            except (CircuitOpen, ShortPage) as error:
                log.error("%s/%s: %s; stopping", platform, sentiment, error)
                interrupted = True
                break
            except ContractError as error:
                log.error("%s/%s: contract failed: %s", platform, sentiment, error.errors[:5])
                return 1
            log.info("%s/%s %s: %s", platform, sentiment, mode, stats)
            total += stats
        if interrupted:
            break

    errors = check_silver(con)
    if errors:
        log.error("silver contract failed: %s", errors)
        return 1
    if not interrupted:
        try:
            check_volume(total, previous)
        except EmptyRunError as error:
            log.error("%s", error)
            return 1
    if bronze:
        name = f"{now:%Y%m%dT%H%M%S}-{mode}.parquet"
        write_bronze(bronze, fetched_at=now, sort="date", path=bronze_dir / name)
    silver.save(con, workdir)
    store.push(workdir)
    log.info("done: %s, silver rows %d -> %d", total, previous, silver.count(con))
    return 1 if interrupted else 0


def main(argv: list[str] | None = None) -> int:
    """Run the collection from the command line.

    Reads `STATE_REPO` and `AUTHOR_SALT` from the environment.

    Args:
        argv: Command-line arguments. Defaults to `sys.argv`.

    Returns:
        int: Exit code, 0 on success.
    """
    parser = argparse.ArgumentParser(description="Collect Metacritic user reviews.")
    parser.add_argument("--mode", choices=["head", "deep"], required=True)
    parser.add_argument("--workdir", type=Path, default=Path("data/state"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    repo = os.environ.get("STATE_REPO", "")
    if not repo:
        log.error("STATE_REPO is not set; refusing to run")
        return 1
    return run(
        args.mode,
        client=PoliteClient(),
        store=GhReleaseStore(repo),
        workdir=args.workdir,
        salt=os.environ.get("AUTHOR_SALT", ""),
        now=datetime.now(UTC),
    )


if __name__ == "__main__":
    raise SystemExit(main())

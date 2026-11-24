from datetime import datetime

import duckdb
import pytest
from helpers import FakeClient, make_item

from reviews_pipeline import silver
from reviews_pipeline.config import PAGE_LIMIT
from reviews_pipeline.http import CircuitOpen
from reviews_pipeline.quality import ContractError
from reviews_pipeline.sources import metacritic
from reviews_pipeline.sweeps import EmptyRunError, RunStats, check_volume, sweep

L = PAGE_LIMIT
NOW = datetime(2026, 11, 24, 12, 0)
PLATFORM, SENTIMENT, SALT = "playstation-5", "negative", "salt"


def fresh():
    con = duckdb.connect()
    silver.init(con)
    return con


def seed(con, ids):
    rows = [metacritic.normalize(make_item(i), PLATFORM, SALT) for i in ids]
    silver.upsert(con, rows, datetime(2026, 11, 24, 9, 0))


def run(client, con, tmp_path, mode="head", bronze=None):
    return sweep(
        client, con, PLATFORM, SENTIMENT, mode=mode, salt=SALT, seen_at=NOW,
        bronze=[] if bronze is None else bronze,
    )


def test_head_sweep_stops_at_first_fully_known_page(tmp_path):
    con = fresh()
    seed(con, range(2 * L, 4 * L))
    client = FakeClient(served=4 * L)
    stats = run(client, con, tmp_path, "head")
    assert client.offsets == [0, L, 2 * L]
    assert stats == RunStats(fetched=3 * L, inserted=2 * L, updated=0, unchanged=L, pages=3)


def test_deep_sweep_walks_every_page(tmp_path):
    con = fresh()
    seed(con, range(2 * L, 4 * L))
    client = FakeClient(served=4 * L)
    stats = run(client, con, tmp_path, "deep")
    assert client.offsets == [0, L, 2 * L, 3 * L]
    assert (stats.inserted, stats.unchanged, stats.pages) == (2 * L, 2 * L, 4)


def test_duplicate_id_within_run_is_counted_once(tmp_path):
    con = fresh()
    client = FakeClient(served=2 * L, item=lambda i: make_item(i % (L + 5)))
    stats = run(client, con, tmp_path, "deep")
    assert silver.count(con) == L + 5
    assert stats.fetched == 2 * L
    assert stats.inserted == L + 5


def test_sweep_collects_every_fetched_item_for_bronze(tmp_path):
    con = fresh()
    bronze = []
    stats = run(FakeClient(served=L + 10), con, tmp_path, "deep", bronze)
    assert len(bronze) == stats.fetched == L + 10
    assert bronze[0] == (PLATFORM, SENTIMENT, make_item(0))


def test_sweep_raises_contract_error_on_invalid_page(tmp_path):
    con = fresh()
    client = FakeClient(served=L, item=lambda i: make_item(i, score=11))
    with pytest.raises(ContractError):
        run(client, con, tmp_path, "deep")
    assert silver.count(con) == 0


def test_circuit_open_keeps_partial_silver_and_collected_items(tmp_path):
    con = fresh()
    bronze = []
    client = FakeClient(served=3 * L, fail_at_offset=L)
    with pytest.raises(CircuitOpen):
        run(client, con, tmp_path, "deep", bronze)
    assert silver.count(con) == L
    assert len(bronze) == L


def test_check_volume_raises_on_empty_run_with_history():
    with pytest.raises(EmptyRunError):
        check_volume(RunStats(0, 0, 0, 0, 0), previous_count=10)


def test_check_volume_ok_on_first_run_and_on_normal_run():
    check_volume(RunStats(0, 0, 0, 0, 0), previous_count=0)
    check_volume(RunStats(5, 5, 0, 0, 1), previous_count=10)


def test_run_stats_add():
    total = RunStats(1, 2, 3, 4, 5) + RunStats(10, 20, 30, 40, 50)
    assert total == RunStats(11, 22, 33, 44, 55)

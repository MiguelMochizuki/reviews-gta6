import logging
from datetime import datetime

import duckdb
from helpers import FakeClient, make_item

from reviews_pipeline import silver
from reviews_pipeline.config import PAGE_LIMIT
from reviews_pipeline.run import main, run
from reviews_pipeline.sources import metacritic

L = PAGE_LIMIT
NOW = datetime(2026, 11, 24, 12, 0)


class FakeStore:
    def __init__(self, prepare_error=None):
        self.pulls = 0
        self.pushes = 0
        self.pushed_count = None
        self.prepare_error = prepare_error
        self.order = []

    def prepare(self):
        self.order.append("prepare")
        if self.prepare_error:
            raise RuntimeError(self.prepare_error)

    def pull(self, dest):
        self.order.append("pull")
        self.pulls += 1

    def push(self, src):
        self.pushes += 1
        con = duckdb.connect()
        silver.load(con, src)
        self.pushed_count = silver.count(con)


def go(client, store, tmp_path, salt="salt"):
    return run("head", client=client, store=store, workdir=tmp_path, salt=salt, now=NOW)


def test_run_happy_path(tmp_path):
    store = FakeStore()
    assert go(FakeClient(served=5), store, tmp_path) == 0
    assert (store.pulls, store.pushes, store.pushed_count) == (1, 1, 5)
    assert (tmp_path / "reviews.parquet").exists()
    files = list((tmp_path / "bronze").glob("*.parquet"))
    assert [f.name for f in files] == ["20261124T120000-head.parquet"]  # one file per run
    count = duckdb.connect().execute(f"SELECT count(*) FROM read_parquet('{files[0]}')")
    assert count.fetchone()[0] == 5 * 6  # five items for each of 6 platform x sentiment queries


def test_run_circuit_open_keeps_partial_and_fails(tmp_path):
    store = FakeStore()
    assert go(FakeClient(served=3 * L, fail_at_offset=L), store, tmp_path) == 1
    assert store.pushes == 1
    assert store.pushed_count == L


def test_run_short_pages_keep_partial_and_fail(tmp_path):
    store = FakeStore()
    assert go(FakeClient(served=1000, cap=100), store, tmp_path) == 1
    assert store.pushes == 1
    assert store.pushed_count == 100


def test_run_survives_network_error_keeping_partial(tmp_path):
    store = FakeStore()
    assert go(FakeClient(served=3 * L, fail_at_offset=L), store, tmp_path) == 1
    assert (tmp_path / "bronze" / "20261124T120000-head.parquet").exists()


def test_run_contract_failure_does_not_push(tmp_path):
    store = FakeStore()
    client = FakeClient(served=5, item=lambda i: make_item(i, score=11))
    assert go(client, store, tmp_path) == 1
    assert store.pushes == 0


def test_run_empty_result_with_history_fails(tmp_path):
    con = duckdb.connect()
    silver.init(con)
    silver.upsert(con, [metacritic.normalize(make_item(1), "playstation-5", "salt")], NOW)
    silver.save(con, tmp_path)
    store = FakeStore()
    assert go(FakeClient(served=0), store, tmp_path) == 1
    assert store.pushes == 0


def test_missing_salt_fails_before_any_request(tmp_path):
    store, client = FakeStore(), FakeClient(served=5)
    assert go(client, store, tmp_path, salt="") == 1
    assert client.offsets == []
    assert store.pulls == 0


def test_main_without_env_exits_nonzero(monkeypatch):
    monkeypatch.delenv("AUTHOR_SALT", raising=False)
    monkeypatch.delenv("STATE_REPO", raising=False)
    assert main(["--mode", "head"]) == 1


def test_logs_contain_no_review_text_or_author(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    go(FakeClient(served=5), FakeStore(), tmp_path)
    go(FakeClient(served=5, item=lambda i: make_item(i, score=11)), FakeStore(), tmp_path)
    for needle in ("text 1", "user1", "salt"):
        assert needle not in caplog.text


def test_run_prepares_the_store_before_pulling_or_requesting(tmp_path):
    store = FakeStore()
    go(FakeClient(served=5), store, tmp_path)
    assert store.order[:2] == ["prepare", "pull"]


def test_run_fails_before_any_request_when_the_store_is_not_ready(tmp_path):
    store, client = FakeStore(prepare_error="Repository is empty."), FakeClient(served=5)
    assert go(client, store, tmp_path) == 1
    assert client.offsets == []

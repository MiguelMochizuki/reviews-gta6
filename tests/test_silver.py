from datetime import datetime

import duckdb

from reviews_pipeline import silver
from reviews_pipeline.silver import UpsertStats

T0 = datetime(2026, 11, 24, 10, 0)
T1 = datetime(2026, 11, 24, 11, 0)


def row(i=1, score=7, text="great", version=1000, platform="playstation-5"):
    return {
        "source": "metacritic",
        "source_review_id": f"id-{i}",
        "platform": platform,
        "score": score,
        "date": "2026-11-24",
        "version_ms": version,
        "author_hash": "ab12",
        "text": text,
        "spoiler": False,
        "content_hash": f"h-{score}-{text}",
    }


def fresh():
    con = duckdb.connect()
    silver.init(con)
    return con


def n(con, table, where="true"):
    return con.execute(f"SELECT count(*) FROM {table} WHERE {where}").fetchone()[0]


def test_upsert_inserts_new_review():
    con = fresh()
    assert silver.upsert(con, [row()], T0) == UpsertStats(1, 0, 0)
    assert silver.count(con) == 1
    assert n(con, "review_versions") == 1


def test_upsert_same_item_is_unchanged_and_advances_last_seen():
    con = fresh()
    silver.upsert(con, [row()], T0)
    assert silver.upsert(con, [row()], T1) == UpsertStats(0, 0, 1)
    first, last = con.execute("SELECT first_seen_at, last_seen_at FROM reviews").fetchone()
    assert (first, last) == (T0, T1)
    assert n(con, "review_versions") == 1


def test_upsert_changed_score_adds_version_and_keeps_first_seen():
    con = fresh()
    silver.upsert(con, [row(score=7)], T0)
    assert silver.upsert(con, [row(score=2, version=2000)], T1) == UpsertStats(0, 1, 0)
    assert n(con, "review_versions", "source_review_id = 'id-1'") == 2
    score, first = con.execute("SELECT score, first_seen_at FROM reviews").fetchone()
    assert (score, first) == (2, T0)


def test_upsert_dedupes_ids_within_batch_last_wins():
    con = fresh()
    stats = silver.upsert(con, [row(score=7), row(score=3, version=2000)], T0)
    assert stats == UpsertStats(1, 0, 0)
    assert con.execute("SELECT score FROM reviews").fetchone()[0] == 3


def test_is_known():
    con = fresh()
    assert silver.is_known(con, [row()]) is False
    silver.upsert(con, [row()], T0)
    assert silver.is_known(con, [row()]) is True
    assert silver.is_known(con, [row(version=2000)]) is False
    assert silver.is_known(con, []) is False


def test_unicode_and_long_text_round_trip(tmp_path):
    texts = ["héllo 🎮 مرحبا", "x" * 5000]
    con = fresh()
    silver.upsert(con, [row(1, text=texts[0]), row(2, text=texts[1])], T0)
    silver.save(con, tmp_path)
    other = duckdb.connect()
    silver.load(other, tmp_path)
    got = dict(other.execute("SELECT source_review_id, text FROM reviews").fetchall())
    assert got == {"id-1": texts[0], "id-2": texts[1]}
    assert n(other, "review_versions") == 2


def test_load_missing_directory_gives_empty_tables(tmp_path):
    con = duckdb.connect()
    silver.load(con, tmp_path / "does-not-exist")
    assert silver.count(con) == 0
    assert n(con, "review_versions") == 0

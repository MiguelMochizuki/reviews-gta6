import json
import uuid
from datetime import datetime

import duckdb
import pytest

from reviews_pipeline import quality, silver

GOOD = {"id": str(uuid.UUID(int=1)), "score": 7, "date": "2026-11-24", "version": 1795548626092}


def silver_row(**overrides):
    row = {
        "source": "metacritic",
        "source_review_id": "id-1",
        "platform": "playstation-5",
        "score": 7,
        "date": "2026-11-24",
        "version_ms": 1000,
        "author_hash": None,
        "text": "t",
        "spoiler": False,
        "content_hash": "h",
    }
    row.update(overrides)
    return row


def test_raw_items_valid_passes():
    assert quality.check_raw_items([GOOD]) == []


@pytest.mark.parametrize(
    ("field", "bad"), [("id", "x"), ("score", 11), ("date", "nope"), ("version", "soon")]
)
def test_raw_items_flag_bad_field(field, bad):
    errors = quality.check_raw_items([{**GOOD, field: bad}])
    assert len(errors) == 1
    assert field in errors[0]


def test_silver_flags_score_out_of_range_and_unknown_platform():
    con = duckdb.connect()
    silver.init(con)
    silver.upsert(con, [silver_row(score=11), silver_row(source_review_id="id-2", platform="switch")],
                  datetime(2026, 11, 24))
    errors = quality.check_silver(con)
    assert any("score" in e for e in errors)
    assert any("platform" in e for e in errors)


def test_silver_clean_table_has_no_errors():
    con = duckdb.connect()
    silver.init(con)
    silver.upsert(con, [silver_row()], datetime(2026, 11, 24))
    assert quality.check_silver(con) == []


def test_silver_duplicate_ids_are_impossible_by_primary_key():
    con = duckdb.connect()
    silver.init(con)
    silver.upsert(con, [silver_row()], datetime(2026, 11, 24))
    with pytest.raises(duckdb.ConstraintException):
        con.execute("INSERT INTO reviews SELECT * FROM reviews")


def test_bronze_round_trip(tmp_path):
    items = [{**GOOD, "quote": "héllo 🎮"}, {**GOOD, "id": str(uuid.UUID(int=2))}]
    path = tmp_path / "bronze" / "run.parquet"
    quality.write_bronze(
        items,
        fetched_at=datetime(2026, 11, 24, 10, 0),
        platform="playstation-5",
        sentiment="negative",
        sort="date",
        path=path,
    )
    rows = duckdb.connect().execute(
        f"SELECT platform, sentiment, sort, raw_json FROM read_parquet('{path}')"
    ).fetchall()
    assert len(rows) == 2
    assert [json.loads(r[3]) for r in rows] == items
    assert {r[:3] for r in rows} == {("playstation-5", "negative", "date")}

from datetime import datetime

import duckdb
import pytest
from nlp_helpers import synthetic_docs

from reviews_pipeline import enrich, silver
from reviews_pipeline.nlp import model as topics

NOW = datetime(2026, 11, 24, 12, 0)


@pytest.fixture(scope="module")
def model():
    return topics.train(synthetic_docs((60, 60, 60)), k=3, passes=5)


def review(i, text, *, version=1, day="2026-11-24"):
    return {
        "source": "metacritic", "source_review_id": f"r{i}", "platform": "playstation-5",
        "score": 5, "date": day, "version_ms": version, "author_hash": f"a{i}", "text": text,
        "spoiler": False, "content_hash": f"h:{text}",
    }


def words(group, n=20):
    return " ".join(f"topic{group}_w{i % 10}" for i in range(n))


class Detect:
    def __init__(self):
        self.calls = 0

    def __call__(self, text):
        self.calls += 1
        return "portuguese" if text.startswith("pt:") else "english"


def silver_con(*rows):
    con = duckdb.connect()
    silver.init(con)
    enrich.init(con)
    silver.upsert(con, list(rows), NOW)
    return con


def rows(con):
    cols = "source_review_id, language, n_tokens, model_version, topic, topic_prob"
    return {r[0]: r[1:] for r in con.execute(f"SELECT {cols} FROM review_topics").fetchall()}


def run(con, model, detect=None, version=None):
    return enrich.enrich_pending(con, model, tokenize=str.split, detect=detect or Detect())


def test_new_reviews_get_language_tokens_and_topic(model):
    con = silver_con(review(1, words(0)), review(2, "pt: " + words(1)))
    assert run(con, model) == 2
    english, portuguese = rows(con)["r1"], rows(con)["r2"]
    assert english[0] == "english" and english[1] == 20 and english[3] is not None
    assert 0 < english[4] <= 1 and english[2] == 1
    assert portuguese == ("portuguese", 0, 1, None, None)


def test_nothing_pending_afterwards(model):
    con = silver_con(review(1, words(0)))
    run(con, model)
    assert run(con, model) == 0


def test_an_edited_review_is_enriched_again(model):
    con = silver_con(review(1, words(0)))
    run(con, model)
    first = rows(con)["r1"][3]
    silver.upsert(con, [review(1, words(1), version=2)], NOW)
    assert run(con, model) == 1
    assert rows(con)["r1"][3] != first


def test_a_new_model_version_reclassifies_without_detecting_again(model):
    con = silver_con(review(1, words(0)), review(2, words(1)))
    detect = Detect()
    run(con, model, detect)
    newer = topics.TopicModel(model.lda, model.dictionary, model.ids, {**model.info, "version": 2})
    assert run(con, newer, detect) == 2
    assert detect.calls == 2
    assert {v[2] for v in rows(con).values()} == {2}


def test_short_review_has_a_language_but_no_topic(model):
    con = silver_con(review(1, "topic0_w1 topic0_w2 topic0_w3"))
    run(con, model)
    assert rows(con)["r1"] == ("english", 3, 1, None, None)
    assert run(con, model) == 0


def test_without_a_model_rows_have_version_zero_and_no_topic(model):
    con = silver_con(review(1, words(0)))
    run(con, None)
    assert rows(con)["r1"][2:] == (0, None, None)
    assert run(con, model) == 1
    assert rows(con)["r1"][3] is not None


def test_save_and_load_roundtrip(model, tmp_path):
    con = silver_con(review(1, words(0)))
    run(con, model)
    enrich.save(con, tmp_path)
    other = duckdb.connect()
    enrich.init(other)
    enrich.load(other, tmp_path)
    assert rows(other) == rows(con)
    empty = duckdb.connect()
    enrich.init(empty)
    enrich.load(empty, tmp_path / "nothing")
    assert rows(empty) == {}

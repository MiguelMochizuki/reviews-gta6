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


# ---- training corpus, retraining and the entry point ----

import logging
import shutil

from reviews_pipeline.nlp.model import load as load_model

CORPUS = synthetic_docs((60, 60, 60))


def corpus_reviews(start=0, day="2026-11-24", docs=CORPUS, **extra):
    return [review(start + i, " ".join(d), day=day, **extra) for i, d in enumerate(docs)]


def trained_con(*reviews):
    con = silver_con(*reviews)
    enrich.enrich_pending(con, None, tokenize=str.split, detect=Detect())
    return con


def test_training_docs_keep_only_eligible_reviews():
    old = review(900, words(0), day="2026-11-01")
    portuguese = review(901, "pt: " + words(0))
    short = review(902, "topic0_w1 topic0_w2")
    repeat = review(903, words(0, 21))
    repeat_again = {**review(904, words(0, 21)), "content_hash": "other"}
    con = trained_con(old, portuguese, short, repeat, repeat_again, review(905, words(1)))
    docs = enrich.training_docs(con, NOW, tokenize=str.split, min_docs=1)
    assert sorted(len(d) for d in docs) == [20, 21]  # one copy of the repeated text, no others


def test_training_docs_sample_evenly_and_reproducibly():
    reviews = []
    for n, day in enumerate(("2026-11-22", "2026-11-23", "2026-11-24")):
        reviews += [review(n * 100 + i, words(0, 20 + i), day=day) for i in range(30)]
    con = trained_con(*reviews)
    args = {"tokenize": str.split, "max_docs": 30, "min_docs": 1}
    docs = enrich.training_docs(con, NOW, **args)
    assert len(docs) == 30 and docs == enrich.training_docs(con, NOW, **args)
    lengths = sorted(len(d) for d in docs)
    assert lengths.count(20) + lengths.count(21) + lengths.count(22) >= 3  # every day appears


def test_training_docs_fall_back_to_all_dates_when_the_window_is_thin():
    con = trained_con(review(1, words(0), day="2026-10-01"), review(2, words(1)))
    assert len(enrich.training_docs(con, NOW, tokenize=str.split, min_docs=2)) == 2
    assert len(enrich.training_docs(con, NOW, tokenize=str.split, min_docs=1)) == 1


def test_retrain_without_a_previous_model_is_version_one():
    con = trained_con(*corpus_reviews())
    model = enrich.retrain(con, None, NOW, tokenize=str.split, k=3)
    assert model.info["version"] == 1 and model.info["trained_at"] == "2026-11-24T12:00:00Z"


def test_retrain_with_a_previous_model_keeps_ids_and_bumps_the_version():
    con = trained_con(*corpus_reviews())
    first = enrich.retrain(con, None, NOW, tokenize=str.split, k=3)
    second = enrich.retrain(con, first, NOW, tokenize=str.split, k=3)
    assert second.info["version"] == 2
    for doc in CORPUS[::30]:
        assert topics.infer(first, doc)[0] == topics.infer(second, doc)[0]


class FakeStore:
    """Keeps files in a `remote` directory like the state release."""

    def __init__(self, remote):
        self.remote = remote
        remote.mkdir(exist_ok=True)

    def pull(self, dest):
        dest.mkdir(parents=True, exist_ok=True)
        for f in self.remote.iterdir():
            shutil.copy(f, dest / f.name)

    def push(self, src):
        for f in src.iterdir():
            if f.is_file():
                shutil.copy(f, self.remote / f.name)


def seed_remote(remote, *reviews):
    remote.mkdir(exist_ok=True)
    con = silver_con(*reviews)
    silver.save(con, remote)


def go(tmp_path, mode="head", detect=None):
    store = FakeStore(tmp_path / "remote")
    code = enrich.run(mode, store=store, workdir=tmp_path / "work", now=NOW,
                      tokenize=str.split, detect=detect or Detect(), k=3)
    return code, store


def remote_version(tmp_path):
    return load_model(tmp_path / "remote" / "topic_model.zip").info["version"]


def remote_rows(tmp_path):
    con = duckdb.connect()
    enrich.init(con)
    enrich.load(con, tmp_path / "remote")
    return con.execute("SELECT model_version, topic FROM review_topics").fetchall()


def test_first_run_trains_assigns_and_pushes_without_silver(tmp_path):
    seed_remote(tmp_path / "remote", *corpus_reviews())
    code, _ = go(tmp_path)
    assert code == 0 and remote_version(tmp_path) == 1
    assert {v for v, _ in remote_rows(tmp_path)} == {1}
    assert sum(t is not None for _, t in remote_rows(tmp_path)) > 150
    pushed_by_run = {p.name for p in (tmp_path / "work").iterdir()}
    assert "reviews.parquet" not in pushed_by_run  # silver is never re-pushed by this step


def test_head_run_does_not_retrain_and_only_adds_new_reviews(tmp_path):
    seed_remote(tmp_path / "remote", *corpus_reviews())
    go(tmp_path)
    con = duckdb.connect()
    silver.init(con)
    silver.load(con, tmp_path / "remote")
    silver.upsert(con, corpus_reviews(start=5000, docs=synthetic_docs((5, 5, 5), seed=9)), NOW)
    silver.save(con, tmp_path / "remote")
    code, _ = go(tmp_path, "head")
    assert code == 0 and remote_version(tmp_path) == 1
    assert len(remote_rows(tmp_path)) == 195


def test_deep_run_retrains_and_reclassifies_everything(tmp_path):
    seed_remote(tmp_path / "remote", *corpus_reviews())
    go(tmp_path)
    code, _ = go(tmp_path, "deep")
    assert code == 0 and remote_version(tmp_path) == 2
    assert {v for v, _ in remote_rows(tmp_path)} == {2}


def test_a_failed_retrain_keeps_the_old_model_and_still_pushes(tmp_path, monkeypatch):
    seed_remote(tmp_path / "remote", *corpus_reviews())
    go(tmp_path)
    monkeypatch.setattr(enrich, "retrain", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    code, _ = go(tmp_path, "deep")
    assert code == 1 and remote_version(tmp_path) == 1
    assert {v for v, _ in remote_rows(tmp_path)} == {1}


def test_logs_carry_counts_only(tmp_path, caplog):
    seed_remote(tmp_path / "remote", *corpus_reviews())
    with caplog.at_level(logging.INFO):
        go(tmp_path)
    assert caplog.text and "topic0_w" not in caplog.text


def test_pending_counts_reviews_with_a_different_model_version(model):
    con = trained_con(*corpus_reviews()[:3])
    assert len(enrich.pending(con, 0)) == 0 and len(enrich.pending(con, 1)) == 3

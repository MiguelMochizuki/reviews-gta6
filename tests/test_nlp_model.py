from collections import Counter

from nlp_helpers import group_of, synthetic_docs

from reviews_pipeline.nlp import model as topics


def fit(sizes=(60, 60, 60), seed=0):
    docs = synthetic_docs(sizes, seed)
    return docs, topics.train(docs, k=len(sizes), passes=5)


def test_each_group_gets_its_own_topic():
    docs, model = fit()
    dominant = [topics.infer(model, d)[0] for d in docs]
    per_group = [Counter(dominant[i * 60:(i + 1) * 60]) for i in range(3)]
    assert all(c.most_common(1)[0][1] / 60 > 0.95 for c in per_group)
    assert len({c.most_common(1)[0][0] for c in per_group}) == 3


def test_training_is_reproducible():
    _, a = fit()
    _, b = fit()
    for t in range(3):
        assert topics.top_words(a, t) == topics.top_words(b, t)


def test_public_ids_follow_descending_share():
    _, model = fit(sizes=(30, 70, 50))
    group_by_topic = {t: group_of(topics.top_words(model, t, 1)[0][0]) for t in range(3)}
    assert group_by_topic == {0: 1, 1: 2, 2: 0}


def test_infer_is_none_without_enough_known_words():
    _, model = fit()
    assert topics.infer(model, ["unknown_a", "unknown_b"]) is None
    assert topics.infer(model, ["topic0_w0", "topic0_w1", "topic0_w2", "topic0_w3"]) is None
    assert topics.infer(model, [f"topic0_w{i}" for i in range(5)]) is not None


def test_label_format():
    _, model = fit()
    assert topics.label(model, 0).startswith("T00 ") and topics.label(model, 0).count("/") == 2


def test_save_and_load_roundtrip(tmp_path):
    docs, model = fit()
    topics.save(model, tmp_path / "m.zip")
    loaded = topics.load(tmp_path / "m.zip")
    assert loaded.info == model.info
    assert all(topics.infer(loaded, d) == topics.infer(model, d) for d in docs[::20])
    assert topics.load(tmp_path / "missing.zip") is None

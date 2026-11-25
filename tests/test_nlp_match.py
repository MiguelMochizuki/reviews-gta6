import numpy as np
import pytest
from nlp_helpers import group_of, synthetic_docs

from reviews_pipeline.nlp import model as topics
from reviews_pipeline.nlp.match import align, js_distance


def fit(seed, offset=0, sizes=(60, 60, 60)):
    return topics.train(synthetic_docs(sizes, seed, offset), k=len(sizes), passes=5)


def group_by_topic(model):
    return {t: group_of(topics.top_words(model, t, 1)[0][0]) for t in range(model.info["k"])}


def test_js_distance_basics():
    p, q = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert js_distance(p, p) == 0
    assert js_distance(p, q) == pytest.approx(1.0)
    r = np.array([0.5, 0.5])
    assert js_distance(p, r) == pytest.approx(js_distance(r, p))


def test_same_groups_keep_their_ids():
    a, b = fit(0), fit(1)
    out = align(a, b)
    assert group_by_topic(out) == group_by_topic(a)
    assert out.info["version"] == 2 and out.info["changed"] == []


def test_a_new_theme_is_marked_changed():
    a = fit(0)
    docs = synthetic_docs((60, 60), seed=1) + synthetic_docs((60,), seed=2, offset=7)
    out = align(a, topics.train(docs, k=3, passes=5))
    replaced = [t for t, g in group_by_topic(out).items() if g == 7]
    assert out.info["changed"] == replaced and len(replaced) == 1
    kept = {t: g for t, g in group_by_topic(out).items() if g != 7}
    assert all(group_by_topic(a)[t] == g for t, g in kept.items())


def test_different_topic_counts_raise():
    with pytest.raises(ValueError):
        align(fit(0), fit(1, sizes=(60, 60)))

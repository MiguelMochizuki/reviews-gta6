"""Matching the topics of a retrained model to the previous model, so ids stay stable."""

from dataclasses import replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from reviews_pipeline.nlp.model import TopicModel

CHANGED_DISTANCE = 0.6  # Jensen-Shannon distance above which a matched topic counts as changed


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Return the Jensen-Shannon distance (base 2, so between 0 and 1) of two distributions."""
    m = (p + q) / 2

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return float(np.sqrt(max(0.0, (kl(p, m) + kl(q, m)) / 2)))


def _matrix(model: TopicModel, words: list[str]) -> np.ndarray:
    """Topic-word probabilities, rows in public id order, columns over the shared vocabulary."""
    position = {w: i for i, w in enumerate(words)}
    topics = model.lda.get_topics()
    out = np.zeros((len(model.ids), len(words)))
    for public, index in enumerate(model.ids):
        for word_id, prob in enumerate(topics[index]):
            out[public, position[model.dictionary[word_id]]] = prob
    return out


def align(previous: TopicModel, new: TopicModel) -> TopicModel:
    """Give each topic of `new` the public id of its match in `previous`.

    Args:
        previous: The model currently in service.
        new: A model just trained on newer data.

    Returns:
        TopicModel: `new` with permuted `ids`, `info["version"]` one above the previous one,
            `info["changed"]` the ids matched farther than `CHANGED_DISTANCE`, and
            `info["distances"]` the distance per public id.

    Raises:
        ValueError: If the models have different numbers of topics.
    """
    if len(previous.ids) != len(new.ids):
        raise ValueError("models have different numbers of topics")
    words = sorted({*previous.dictionary.token2id, *new.dictionary.token2id})
    old_m, new_m = _matrix(previous, words), _matrix(new, words)
    cost = np.array([[js_distance(o, n) for n in new_m] for o in old_m])
    rows, cols = linear_sum_assignment(cost)
    ids = [0] * len(new.ids)
    distances = {}
    for old_public, new_public in zip(rows, cols):
        ids[int(old_public)] = new.ids[int(new_public)]
        distances[int(old_public)] = round(float(cost[old_public, new_public]), 4)
    info = {
        **new.info,
        "version": previous.info["version"] + 1,
        "changed": sorted(t for t, d in distances.items() if d > CHANGED_DISTANCE),
        "distances": distances,
    }
    return replace(new, ids=ids, info=info)

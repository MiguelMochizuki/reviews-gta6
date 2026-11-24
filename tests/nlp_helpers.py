"""Synthetic corpora with disjoint word groups, so LDA separates them reliably."""

import random


def group_of(word: str) -> int:
    return int(word.split("_")[0].removeprefix("topic"))


def synthetic_docs(sizes=(60, 60, 60), seed=0, offset=0, length=20) -> list[list[str]]:
    """Documents of `length` tokens each drawn from one group of 10 words per topic."""
    rng = random.Random(seed)
    docs = []
    for topic, size in enumerate(sizes):
        words = [f"topic{topic + offset}_w{i}" for i in range(10)]
        docs += [[rng.choice(words) for _ in range(length)] for _ in range(size)]
    return docs

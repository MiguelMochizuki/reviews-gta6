"""LDA topic model with stable public topic ids."""

import json
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from gensim.corpora import Dictionary
from gensim.models import LdaModel

from reviews_pipeline.nlp.text import MIN_BOW


@dataclass
class TopicModel:
    """A fitted LDA model, its dictionary and the mapping from public topic ids.

    Attributes:
        lda: The gensim model.
        dictionary: Vocabulary the model was fitted on.
        ids: `ids[p]` is the gensim topic index behind public topic id `p`.
        info: `version`, `trained_at`, `k`, `seed`, `n_docs`, `changed`, `distances`.
    """

    lda: LdaModel
    dictionary: Dictionary
    ids: list[int]
    info: dict = field(default_factory=dict)


def _bow(dictionary: Dictionary, tokens: list[str]) -> list[tuple[int, int]]:
    return dictionary.doc2bow(tokens)


def _dominant(lda: LdaModel, bow) -> tuple[int, float]:
    return max(lda.get_document_topics(bow, minimum_probability=0.0), key=lambda x: (x[1], -x[0]))


def train(
    docs: list[list[str]], *, k: int = 10, seed: int = 42, passes: int = 10
) -> TopicModel:
    """Fit an LDA model, with public ids ordered by descending share of the training docs.

    Args:
        docs: Tokenized documents.
        k: Number of topics.
        seed: Random seed; the same docs and seed give the same model.
        passes: Passes over the corpus.

    Returns:
        TopicModel: Version 1, with empty `changed` and `distances`.
    """
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [b for b in (_bow(dictionary, d) for d in docs) if len(b) >= MIN_BOW]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, random_state=seed,
                   passes=passes, alpha="auto")
    counts = [0] * k
    for bow in corpus:
        counts[_dominant(lda, bow)[0]] += 1
    ids = sorted(range(k), key=lambda t: (-counts[t], t))
    info = {
        "version": 1, "trained_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "k": k, "seed": seed, "n_docs": len(corpus), "changed": [], "distances": {},
    }
    return TopicModel(lda, dictionary, ids, info)


def infer(model: TopicModel, tokens: list[str]) -> tuple[int, float] | None:
    """Return the dominant public topic of a document and its probability.

    Args:
        model: Fitted model.
        tokens: Tokenized document.

    Returns:
        tuple[int, float] | None: `(topic, probability)`, or `None` when the document has
            fewer than `MIN_BOW` distinct in-vocabulary words.
    """
    bow = _bow(model.dictionary, tokens)
    if len(bow) < MIN_BOW:
        return None
    index, prob = _dominant(model.lda, bow)
    return model.ids.index(index), float(prob)


def top_words(model: TopicModel, topic: int, n: int = 12) -> list[tuple[str, float]]:
    """Return the `n` highest-weight words of a public topic as `(word, weight)`."""
    return [(w, float(p)) for w, p in model.lda.show_topic(model.ids[topic], topn=n)]


def label(model: TopicModel, topic: int) -> str:
    """Return a label generated from the top words, such as `T03 story/character/mission`."""
    words = "/".join(w for w, _ in top_words(model, topic, 3))
    return f"T{topic:02d} {words}"


def save(model: TopicModel, path: Path) -> None:
    """Write the model, its dictionary, ids and info into one zip archive."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        model.lda.save(str(folder / "lda.model"))
        (folder / "meta.json").write_text(json.dumps({"ids": model.ids, "info": model.info}))
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(folder.iterdir()):
                archive.write(file, file.name)


def load(path: Path) -> TopicModel | None:
    """Read a model written by `save`; `None` when the file does not exist."""
    if not path.exists():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(path) as archive:
            archive.extractall(tmp)
        lda = LdaModel.load(str(Path(tmp) / "lda.model"))
        meta = json.loads((Path(tmp) / "meta.json").read_text())
    return TopicModel(lda, lda.id2word, meta["ids"], meta["info"])

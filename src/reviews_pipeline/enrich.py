"""Topic enrichment: language, tokens and topic per review, kept in `review_topics`."""

import logging
from collections.abc import Callable
from pathlib import Path

import duckdb

from reviews_pipeline.nlp import model as topics
from reviews_pipeline.nlp.model import TopicModel
from reviews_pipeline.nlp.text import MIN_TOKENS, detect_language, preprocess

log = logging.getLogger("reviews_pipeline")

_FILE = "review_topics.parquet"


def init(con: duckdb.DuckDBPyConnection) -> None:
    """Create the `review_topics` table if it does not exist.

    Args:
        con: Connection that also holds the silver tables.
    """
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS review_topics (
            source VARCHAR, source_review_id VARCHAR, content_hash VARCHAR, language VARCHAR,
            n_tokens INTEGER, model_version INTEGER, topic INTEGER, topic_prob DOUBLE,
            PRIMARY KEY (source, source_review_id))
        """
    )


def save(con: duckdb.DuckDBPyConnection, directory: Path) -> None:
    """Write `review_topics` to `review_topics.parquet` in `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY review_topics TO '{directory / _FILE}' (FORMAT PARQUET)")


def load(con: duckdb.DuckDBPyConnection, directory: Path) -> None:
    """Load `review_topics.parquet` from `directory`; does nothing when it is absent."""
    path = directory / _FILE
    if path.exists():
        con.execute(f"INSERT INTO review_topics BY NAME SELECT * FROM read_parquet('{path}')")


def pending(con: duckdb.DuckDBPyConnection, model_version: int) -> list[tuple[str, str]]:
    """List the reviews that need enrichment.

    Args:
        con: Connection with silver and `review_topics`.
        model_version: Version of the model that would do the enrichment.

    Returns:
        list[tuple[str, str]]: `(source, source_review_id)` of reviews with no row, a row for
            other content, or a row from a different model version (a restored older state
            can hold rows newer than the model).
    """
    return con.execute(
        """
        SELECT r.source, r.source_review_id FROM reviews r
        LEFT JOIN review_topics t USING (source, source_review_id)
        WHERE t.source IS NULL OR t.content_hash IS DISTINCT FROM r.content_hash
           OR t.model_version <> ?
        """,
        [model_version],
    ).fetchall()


def enrich_pending(
    con: duckdb.DuckDBPyConnection,
    model: TopicModel | None,
    *,
    tokenize: Callable[[str], list[str]] = preprocess,
    detect: Callable[[str], str] = detect_language,
) -> int:
    """Enrich every pending review with language, token count and topic.

    The language is reused when the review's content did not change. Only English reviews
    with at least `MIN_TOKENS` tokens and enough known words get a topic. Without a model
    the topic is always null and the row has version 0.

    Args:
        con: Connection with silver and `review_topics`.
        model: Current model, or `None` before the first training.
        tokenize: Turns text into tokens.
        detect: Turns text into a language name.

    Returns:
        int: Number of reviews enriched.
    """
    version = model.info["version"] if model else 0
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE todo AS
        SELECT r.source, r.source_review_id, r.content_hash, r.text,
               CASE WHEN t.content_hash = r.content_hash THEN t.language END AS known_language
        FROM reviews r LEFT JOIN review_topics t USING (source, source_review_id)
        WHERE t.source IS NULL OR t.content_hash IS DISTINCT FROM r.content_hash
           OR t.model_version <> ?
        """,
        [version],
    )
    out = []
    for source, review_id, content_hash, text, known in con.execute("SELECT * FROM todo").fetchall():
        language = known or detect(text or "")
        n_tokens, topic, prob = 0, None, None
        if language == "english":
            tokens = tokenize(text)
            n_tokens = len(tokens)
            if model and n_tokens >= MIN_TOKENS and (found := topics.infer(model, tokens)):
                topic, prob = found
        out.append((source, review_id, content_hash, language, n_tokens, version, topic, prob))
    if out:
        con.executemany("INSERT OR REPLACE INTO review_topics VALUES (?, ?, ?, ?, ?, ?, ?, ?)", out)
    con.execute("DROP TABLE todo")
    return len(out)

"""Topic enrichment: language, tokens and topic per review, kept in `review_topics`."""

import argparse
import logging
import os
import random
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import duckdb

from reviews_pipeline import silver
from reviews_pipeline.nlp import model as topics
from reviews_pipeline.nlp.match import align
from reviews_pipeline.nlp.model import TopicModel
from reviews_pipeline.nlp.text import MIN_TOKENS, detect_language, preprocess

log = logging.getLogger("reviews_pipeline")

_FILE = "review_topics.parquet"
MODEL_FILE = "topic_model.zip"
WINDOW_DAYS = 14
MAX_TRAIN_DOCS = 50_000
MIN_TRAIN_DOCS = 5_000
SEED = 42


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


def _eligible(con: duckdb.DuckDBPyConnection, since) -> list[tuple]:
    """English reviews with enough tokens since a date, one per distinct normalized text."""
    return con.execute(
        """
        SELECT date, source_review_id, text FROM (
            SELECT r.date, r.source_review_id, r.text,
                   row_number() OVER (
                       PARTITION BY lower(trim(regexp_replace(r.text, '\\s+', ' ', 'g')))
                       ORDER BY r.date, r.source_review_id) AS n
            FROM reviews r JOIN review_topics t USING (source, source_review_id)
            WHERE t.content_hash = r.content_hash AND t.language = 'english'
              AND t.n_tokens >= ? AND (? IS NULL OR r.date >= ?))
        WHERE n = 1 ORDER BY date, source_review_id
        """,
        [MIN_TOKENS, since, since],
    ).fetchall()


def _sample_evenly(rows: list[tuple], limit: int) -> list[tuple]:
    """Pick at most `limit` rows with an equal quota per day, unused quota going to bigger days."""
    if len(rows) <= limit:
        return rows
    by_day = defaultdict(list)
    for row in rows:
        by_day[row[0]].append(row)
    rng = random.Random(SEED)
    chosen, left = [], limit
    for position, day in enumerate(sorted(by_day, key=lambda d: (len(by_day[d]), d))):
        quota = left // (len(by_day) - position)
        take = by_day[day] if len(by_day[day]) <= quota else rng.sample(by_day[day], quota)
        chosen += take
        left -= len(take)
    return sorted(chosen)


def training_docs(
    con: duckdb.DuckDBPyConnection,
    now: datetime,
    *,
    tokenize: Callable[[str], list[str]] = preprocess,
    max_docs: int = MAX_TRAIN_DOCS,
    min_docs: int = MIN_TRAIN_DOCS,
) -> list[list[str]]:
    """Build the documents the daily retraining learns from.

    Args:
        con: Connection with silver and `review_topics` (languages already filled).
        now: Run time; the window is the last `WINDOW_DAYS` days.
        tokenize: Turns text into tokens.
        max_docs: Cap, sampled evenly across days with a fixed seed.
        min_docs: With fewer eligible documents in the window, all dates are used.

    Returns:
        list[list[str]]: Tokens of English reviews with enough tokens, one per distinct text.
    """
    rows = _eligible(con, (now - timedelta(days=WINDOW_DAYS - 1)).date())
    if len(rows) < min_docs:
        rows = _eligible(con, None)
    return [tokenize(text) for _, _, text in _sample_evenly(rows, max_docs)]


def retrain(
    con: duckdb.DuckDBPyConnection,
    previous: TopicModel | None,
    now: datetime,
    *,
    tokenize: Callable[[str], list[str]] = preprocess,
    k: int = 10,
) -> TopicModel:
    """Train a new model and, when there is a previous one, keep its topic ids.

    Args:
        con: Connection with silver and `review_topics`.
        previous: Model in service, or `None` for the first training.
        now: Run time, recorded as `trained_at`.
        tokenize: Turns text into tokens.
        k: Number of topics.

    Returns:
        TopicModel: Version 1 without a previous model, otherwise aligned to it.

    Raises:
        ValueError: If there is nothing to train on or the topic counts differ.
    """
    docs = training_docs(con, now, tokenize=tokenize)
    if not docs:
        raise ValueError("no training documents")
    model = topics.train(docs, k=k, seed=SEED)
    if previous:
        model = align(previous, model)
    model.info["trained_at"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    return model


def run(
    mode: Literal["head", "deep"],
    *,
    store,
    workdir: Path,
    now: datetime,
    tokenize: Callable[[str], list[str]] = preprocess,
    detect: Callable[[str], str] = detect_language,
    k: int = 10,
) -> int:
    """Enrich new and edited reviews; on a deep run, or without a model, retrain first.

    Silver is read, never pushed: only `review_topics.parquet` and `topic_model.zip` go back.
    If retraining fails the previous model keeps serving. Logs carry counts only.

    Args:
        mode: `deep` also retrains and reclassifies everything.
        store: State store with `pull(dir)` and `push(dir)`.
        workdir: Local directory for state files (not shared with the collection step).
        now: Run time.
        tokenize: Turns text into tokens.
        detect: Turns text into a language name.
        k: Number of topics.

    Returns:
        int: 0 on success, 1 if retraining failed.
    """
    logging.getLogger("gensim").setLevel(logging.WARNING)  # its INFO logs print topic words
    store.pull(workdir)
    con = duckdb.connect()
    silver.init(con)
    silver.load(con, workdir)
    init(con)
    load(con, workdir)
    model = topics.load(workdir / MODEL_FILE)
    for name in ("reviews.parquet", "review_versions.parquet"):
        (workdir / name).unlink(missing_ok=True)  # read, not ours to push back
    version = model.info["version"] if model else 0
    log.info("pending reviews: %d (model version %d)", len(pending(con, version)), version)
    enriched = enrich_pending(con, model, tokenize=tokenize, detect=detect)
    failed = False
    if mode == "deep" or model is None:
        try:
            model = retrain(con, model, now, tokenize=tokenize, k=k)
            enriched += enrich_pending(con, model, tokenize=tokenize, detect=detect)
        except Exception as error:  # noqa: BLE001  the old model keeps serving; the job reports the failure
            log.error("retraining failed (%s); keeping model version %d", type(error).__name__, version)
            failed = True
    save(con, workdir)
    if model:
        topics.save(model, workdir / MODEL_FILE)
    store.push(workdir)
    covered = con.execute("SELECT count(topic), count(*) FROM review_topics").fetchone()
    log.info("done: enriched %d, with a topic %d of %d, model version %s", enriched, *covered,
             model.info["version"] if model else None)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    """Run the enrichment from the command line; reads `STATE_REPO` from the environment.

    Args:
        argv: Command-line arguments. Defaults to `sys.argv`.

    Returns:
        int: Exit code, 0 on success.
    """
    from reviews_pipeline.state import GhReleaseStore

    parser = argparse.ArgumentParser(description="Assign topics to collected reviews.")
    parser.add_argument("--mode", choices=["head", "deep"], required=True)
    parser.add_argument("--workdir", type=Path, default=Path("data/topics-state"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    repo = os.environ.get("STATE_REPO", "")
    if not repo:
        log.error("STATE_REPO is not set; refusing to run")
        return 1
    return run(args.mode, store=GhReleaseStore(repo), workdir=args.workdir, now=datetime.now(UTC))


if __name__ == "__main__":
    raise SystemExit(main())

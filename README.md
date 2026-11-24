# reviews-gta6

Collection pipeline and analysis of ~51k Metacritic user reviews of GTA 6, for anyone who wants to see what players praise and criticize.

## Overview

The dataset has 51,385 reviews of GTA 6 on PlayStation 5 (90%) and Xbox Series X (10%), published between 2026-11-19 and 2026-11-24. The repository has two parts:

- A notebook that follows CRISP-DM: it checks data quality, describes scores, platforms, time and languages, then fits a topic model on the English reviews and relates each topic to the score.
- A collection pipeline (`src/reviews_pipeline`) that fetches reviews from the public Metacritic endpoint on a schedule, stores raw responses (bronze) and a deduplicated, versioned review table (silver), and keeps its state in a private GitHub release.

### Status and limitations

- The notebook sample covers about five launch days, so it says nothing about reception after patches.
- The topic model uses English reviews only (78% of the data) and reviews with at least 10 tokens (58% of the data).
- The pipeline covers collection and the silver layer. The gold layer, the published site and topic enrichment are not built yet.
- The endpoint is public but undocumented, and each query reaches at most 10,000 results. The pipeline collects politely: one request every 3 seconds, retries with backoff, and a stop after repeated 429/5xx responses.

## Quick Start

### Prerequisites

- Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).
- Kaggle API credentials, only to download the notebook dataset again.
- The GitHub CLI (`gh`), only to run the collection against a real state release.

### Install

```bash
uv sync
source .venv/bin/activate
```

`uv sync` installs the pipeline, the test tools and the notebook libraries. Use `uv sync --no-default-groups` for the pipeline alone.

### Get the notebook data

`data/raw/reviews.csv` comes from the Kaggle dataset [ektarr/gta-vi-user-reviews](https://www.kaggle.com/datasets/ektarr/gta-vi-user-reviews), which contains a single file, `reviews.csv`. The `data/` directory is git-ignored. Download the file with:

```bash
kaggle datasets download -d ektarr/gta-vi-user-reviews -p data/raw --unzip
```

### Run

```bash
jupyter lab notebooks/reviews_gta6.ipynb
```

## Project Organization

```
reviews-gta6/
├── .github/                # GitHub config
│   └── workflows/          # CI and scheduled collection
│       ├── ci.yml
│       └── collect.yml
├── notebooks/
│   └── reviews_gta6.ipynb  # EDA + LDA
├── scripts/                # one-off live checks
│   └── verify_endpoint.py
├── src/                    # source code
│   └── reviews_pipeline/   # collection pipeline package
│       ├── sources/
│       ├── __init__.py
│       ├── config.py
│       ├── http.py
│       ├── quality.py
│       ├── run.py
│       ├── silver.py
│       ├── state.py
│       └── sweeps.py
├── tests/                  # pytest suite, no network
│   ├── helpers.py
│   ├── test_http.py
│   ├── test_metacritic.py
│   ├── test_quality.py
│   ├── test_run.py
│   ├── test_silver.py
│   ├── test_smoke.py
│   ├── test_state.py
│   ├── test_sweeps.py
│   └── test_workflows.py
├── .gitignore
├── CONTRIBUTING.md
├── pyproject.toml          # dependencies and tool config (uv)
├── README.md
└── uv.lock
```

`data/` is git-ignored. It holds `raw/` (the Kaggle file, never edited), `processed/` (notebook output) and `state/` (local pipeline state).

### Key modules

- `notebooks/reviews_gta6.ipynb` holds the whole analysis. Its sections map to CRISP-DM. Cells share state, so run it top to bottom.
- `src/reviews_pipeline/sources/metacritic.py` is the source adapter: paging and normalization. A new source is a module with the same two functions.
- `src/reviews_pipeline/sweeps.py` runs head and deep sweeps. `silver.py` keeps the review and version tables. `quality.py` holds the data contracts and the bronze writer. `state.py` moves state to and from the private release. `run.py` is the command-line entry point.

## Main Tools & Stack

### Language and runtime

Python 3.12, managed with `uv`. Dependencies are in `pyproject.toml` and locked in `uv.lock`, in three groups: the pipeline itself (`httpx`, `duckdb`), `dev` (`pytest`, `ruff`) and `notebook`.

### Key libraries

- `httpx` for requests and `duckdb` for Parquet and SQL in the pipeline.
- `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn` for the notebook analysis and charts.
- `lingua-language-detector` for language detection, `nltk` for stopwords, POS tagging and lemmatization, `gensim` for LDA and coherence, `pyLDAvis` for the topic map.
- `kaggle` to download the dataset, `jupyterlab` and `ipykernel` to run the notebook.
- GitHub Actions for CI and the schedule, and the `gh` CLI for the state release.

### Declared but unused

`scikit-learn` is listed in the `notebook` group but the notebook does not import it yet.

## Usage

### Tests

```bash
uv run pytest -q
uv run ruff check .
```

The tests use fake clients and a fake `gh`, so they need no network and no secrets.

### Collect reviews

State lives in a release named `state` of a private repository, so the collection needs two repositories: this public one with the code, and a private one that only holds releases. The workflow `.github/workflows/collect.yml` runs a head sweep every hour and a deep sweep every day. It needs:

| Name | Kind | Meaning |
|---|---|---|
| `STATE_REPO` | repository variable | `owner/name` of the private repository |
| `STATE_TOKEN` | repository secret | fine-grained token with write access to releases of the private repository |
| `AUTHOR_SALT` | repository secret | secret key used to hash author names |

Trigger a run by hand from the Actions tab (Collect reviews, Run workflow), choosing `head` or `deep`. Locally:

```bash
export STATE_REPO=owner/name AUTHOR_SALT=some-secret
uv run python -m reviews_pipeline.run --mode head
```

A head sweep walks the newest reviews of each platform and sentiment until it reaches a page it already stores. A deep sweep walks every page up to the 10,000 cap and catches edits to older reviews. Raw responses are written as one bronze file per run, uploaded to a monthly release (`bronze-YYYY-MM`) of the private repository to stay under the 1,000-assets-per-release limit. Author names are only hashed in silver.

### Run the notebook headless

Paths in the notebook are relative to `notebooks/`. A full run takes about six minutes, mostly language detection and the topic-number search.

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace reviews_gta6.ipynb
```

The first run also downloads NLTK data, so it needs network access.

### Input format

`data/raw/reviews.csv` has one row per review:

| Column | Meaning |
|---|---|
| `platform` | `playstation-5` or `xbox-series-x` |
| `review_id` | Metacritic profile hash. It identifies the user, not the review. |
| `published_date` | Publication date, day granularity |
| `score` | Integer from 0 to 10 |
| `review_text` | Free text, multilingual and multi-line |

Texts contain line breaks, so `wc -l` reports about 108k lines for 51,385 rows.

## Key Outputs & Reports

### Modeled reviews

`data/processed/reviews_en_topics.csv` has one row per modeled English review, with the original columns plus derived ones: language, word count, score band, repeated-text flag, cleaned text, dominant topic, topic probability and topic label. Section 6 of the notebook summarizes the findings and proposes a next step, a polarity classifier with a temporal split.

## Documentation Conventions

Python docstrings use the Google style and the notebook is written in English. See [CONTRIBUTING.md](CONTRIBUTING.md#documentation-conventions).

## Contributing

Commit messages follow Conventional Commits in English. Details are in [CONTRIBUTING.md](CONTRIBUTING.md).

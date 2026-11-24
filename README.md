# reviews-gta6

Exploratory analysis and LDA topic modeling of ~51k Metacritic user reviews of GTA 6, for anyone who wants to see what players praise and criticize.

## Overview

The dataset has 51,385 reviews of GTA 6 on PlayStation 5 (90%) and Xbox Series X (10%), published between 2026-11-19 and 2026-11-24. The notebook follows CRISP-DM: it checks data quality, describes scores, platforms, time and languages, then fits a topic model on the English reviews and relates each topic to the score.

### Status and limitations

- The sample covers about five launch days, so it says nothing about reception after patches.
- The topic model uses English reviews only (78% of the data) and reviews with at least 10 tokens (58% of the data).
- The notebook is the only code. There are no tests.

## Quick Start

### Prerequisites

- Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).
- Kaggle API credentials, only if you need to download the dataset again.

### Install

```bash
uv sync
source .venv/bin/activate
```

### Get the data

`data/raw/reviews.csv` comes from the Kaggle dataset [ektarr/gta-vi-user-reviews](https://www.kaggle.com/datasets/ektarr/gta-vi-user-reviews), which contains a single file, `reviews.csv`. If the file is missing, download it with:

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
├── data/
│   ├── processed/          # notebook output: English reviews with topics
│   │   └── reviews_en_topics.csv
│   └── raw/                # Kaggle dataset, never edited
│       └── reviews.csv
├── notebooks/
│   └── reviews_gta6.ipynb  # EDA + LDA, all the code
├── .gitignore
├── CONTRIBUTING.md
├── pyproject.toml          # dependencies only (uv)
└── README.md
```

### Key modules

- `notebooks/reviews_gta6.ipynb` holds the whole pipeline. Its sections map to CRISP-DM: business understanding, data understanding, data preparation, modeling, evaluation and findings. Cells share state, so run it top to bottom.

## Main Tools & Stack

### Language and runtime

Python 3.12, managed with `uv`. Dependencies are listed in `pyproject.toml` and there is no lock file.

### Key libraries

- `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn` for analysis and charts.
- `lingua-language-detector` for language detection.
- `nltk` for stopwords, POS tagging and lemmatization.
- `gensim` for LDA and coherence, `pyLDAvis` for the topic map.
- `kaggle` to download the dataset.
- `jupyterlab` and `ipykernel` to run the notebook.

### Declared but unused

`scikit-learn` is listed in `pyproject.toml` but the notebook does not import it yet.

## Usage

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

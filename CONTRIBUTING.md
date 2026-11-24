# Contributing

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/) in English, for example `feat(notebook): add polarity classifier` or `docs: explain how to download the dataset`.

## Documentation Conventions

### Python

Functions in `src/` and in notebook cells use Google-style docstrings. Notebook functions have no type hints, so their docstrings state each type as `name (type): meaning`. In `src/` the type hints carry the types, so `Args` entries are just `name: meaning`.

```python
def detect_lang(text):
    """Return the language of a review text.

    Args:
        text (str): Review text. Non-strings and blank strings give `'unknown'`.

    Returns:
        str: Lowercase language name (for example `'english'`), or `'unknown'`
            when the detector cannot decide.
    """
```

Rules:

- The first line is a one-sentence summary in the imperative mood, ending with a period.
- Add `Args`, `Returns` and `Raises` sections whenever the function has parameters, a return value or a deliberate exception. Skip a section that would be empty.
- Wrap continuation lines of a description with a four-space indent.

### Notebooks

- Markdown cells and code comments are written in English.
- Each notebook opens with a title cell that lists its phases, and each section starts with a heading that says what the section answers.
- Findings that quote numbers sit in a markdown cell written after the run, so rerun the notebook and recheck them when the data or preprocessing changes.

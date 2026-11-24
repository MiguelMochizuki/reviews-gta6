"""Text preparation for topic modeling: language, normalization, tokens.

The rules are those of `notebooks/reviews_gta6.ipynb` (sections 2.5 and 3), reimplemented here
so the pipeline does not import the notebook.
"""

import re
from collections.abc import Callable
from functools import cache

MIN_TOKENS = 10  # a review needs this many tokens to get a topic
MIN_BOW = 5  # and this many distinct in-vocabulary words

_NEGATIONS = {"not", "no", "never", "none", "nor", "cannot"}
_NEGATABLE_STOPWORDS = {"good"}  # a stopword alone, but "not good" is a strong signal
_PROTECTED_WORDS = {"woke", "based", "cringe", "goated"}  # slang the lemmatizer would damage
_CUSTOM_STOPWORDS = {
    "game", "games", "play", "played", "playing", "player", "players",
    "like", "really", "would", "could", "get", "got", "one", "also",
    "much", "even", "still", "think", "make", "made", "want", "know",
    "see", "going", "time", "thing", "way", "lot", "good",
    "gta", "rockstar",
}  # fmt: skip


def normalize_text(text: str) -> str:
    """Lowercase a text and collapse its whitespace, the key used to spot repeated texts.

    Args:
        text: Review text.

    Returns:
        str: Normalized text.
    """
    return re.sub(r"\s+", " ", text).strip().lower()


@cache
def _detector():
    from lingua import Language, LanguageDetectorBuilder

    languages = (
        Language.ENGLISH, Language.PORTUGUESE, Language.SPANISH, Language.ARABIC,
        Language.FRENCH, Language.GERMAN, Language.RUSSIAN, Language.CHINESE,
        Language.JAPANESE, Language.KOREAN,
    )  # fmt: skip
    return LanguageDetectorBuilder.from_languages(*languages).with_minimum_relative_distance(0.1).build()


def detect_language(text: str | None) -> str:
    """Detect the language of a review from its first 200 characters.

    Args:
        text: Review text.

    Returns:
        str: Lowercase language name such as `"english"`, or `"unknown"` for blank text or
            when the detector cannot decide.
    """
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    language = _detector().detect_language_of(text[:200])
    return language.name.lower() if language else "unknown"


@cache
def _stopwords() -> frozenset[str]:
    import nltk
    from nltk.corpus import stopwords

    nltk.download("stopwords", quiet=True)
    words = set(stopwords.words("english")) - {"not", "no", "nor", "never", "none", "n't"}
    return frozenset(words | _CUSTOM_STOPWORDS)


def _wordnet_pos(tag: str) -> str:
    from nltk.corpus import wordnet

    return {"J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV}.get(tag[:1], wordnet.NOUN)


@cache
def _nltk_tools():
    """Download the nltk data once and build the tagger and lemmatizer once.

    Rebuilding them per review made preprocessing about 40 ms a review, too slow to retrain
    daily on a launch week's volume.
    """
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.tag import PerceptronTagger

    for package in ("wordnet", "omw-1.4", "averaged_perceptron_tagger_eng"):
        nltk.download(package, quiet=True)
    return PerceptronTagger(), WordNetLemmatizer()


def _nltk_lemmatize(tokens: list[str]) -> list[str]:
    from nltk.corpus import wordnet

    tagger, lemmatizer = _nltk_tools()
    lemmas = []
    for word, tag in tagger.tag(tokens):
        if word in _PROTECTED_WORDS:
            lemmas.append(word)
            continue
        pos = _wordnet_pos(tag)
        if word.endswith("ed") and pos == wordnet.ADJ:
            verb = lemmatizer.lemmatize(word, wordnet.VERB)
            if verb != word:
                lemmas.append(verb)
                continue
        lemmas.append(lemmatizer.lemmatize(word, pos))
    return lemmas


def preprocess(
    text: str | None, lemmatize: Callable[[list[str]], list[str]] | None = None
) -> list[str]:
    """Clean, tokenize, collapse negations and lemmatize a review text.

    Args:
        text: Raw review text.
        lemmatize: Replaces the nltk POS-aware lemmatizer. Tests pass the identity so they
            do not need the nltk data.

    Returns:
        list[str]: Tokens, with a negation and the next word collapsed into `not_<word>`.
            Empty for blank text or when no token survives the stopwords.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    stop = _stopwords()
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"n['’]t\b", " not", text)
    text = text.replace("cannot", "can not")
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    collapsed, i = [], 0
    while i < len(tokens):
        if tokens[i] in _NEGATIONS and i + 1 < len(tokens):
            collapsed.append(f"not_{tokens[i + 1]}")
            i += 2
        else:
            collapsed.append(tokens[i])
            i += 1
    collapsed = [
        t for t in collapsed
        if not (t.startswith("not_") and t[4:] in stop and t[4:] not in _NEGATABLE_STOPWORDS)
    ]  # fmt: skip
    tokens = [t for t in collapsed if t not in stop and len(t) > 2]
    if not tokens:
        return []
    return (lemmatize or _nltk_lemmatize)(tokens)

import pytest

from reviews_pipeline.nlp.text import detect_language, normalize_text, preprocess


def prep(text):
    return preprocess(text, lemmatize=lambda tokens: tokens)


def test_negations_collapse_and_stopwords_go():
    tokens = prep("It's not good, I can't stand the 30 FPS and the bugs! Never again.")
    for wanted in ("not_good", "not_stand", "fps", "bugs"):
        assert wanted in tokens
    for unwanted in ("the", "and", "30", "not_again"):
        assert unwanted not in tokens


def test_good_is_a_stopword_but_not_good_is_kept():
    assert prep("good") == []
    assert prep("not good") == ["not_good"]


def test_urls_are_removed():
    assert not any("example" in t for t in prep("see https://x.example/a great story"))


def test_short_tokens_and_blank_input():
    assert "ok" not in prep("ok story")
    assert prep(None) == prep("") == prep("   ") == []


def test_normalize_text():
    assert normalize_text("  Same   TEXT\nhere ") == "same text here"


def test_detect_language():
    assert detect_language("This game is a masterpiece and the story is amazing") == "english"
    assert detect_language("Este jogo é uma obra-prima e a história é incrível") == "portuguese"
    assert detect_language("") == detect_language(None) == "unknown"


def test_real_lemmatizer_path():
    pytest.importorskip("nltk")
    tokens = preprocess("The players were running and crashed")
    assert "run" in tokens and "crash" in tokens

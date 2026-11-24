import hashlib
import hmac
import uuid

import pytest
from helpers import FakeClient, make_item

from reviews_pipeline.config import MAX_OFFSET, PAGE_LIMIT
from reviews_pipeline.sources import metacritic

L = PAGE_LIMIT


def pages(client, **kwargs):
    return list(metacritic.iter_pages(client, "playstation-5", "negative", **kwargs))


def test_iter_pages_walks_until_short_page():
    client = FakeClient(served=2 * L + 37)
    assert len(pages(client)) == 3
    assert client.offsets == [0, L, 2 * L]


def test_iter_pages_never_requests_offset_10000():
    client = FakeClient(served=50_000)
    pages(client)
    assert client.offsets[-1] == MAX_OFFSET
    assert 10_000 not in client.offsets


def test_iter_pages_stops_when_last_full_page_reaches_total():
    client = FakeClient(served=2 * L)
    assert len(pages(client)) == 2
    assert client.offsets == [0, L]


def test_iter_pages_stops_at_cap_exactly_ten_thousand():
    client = FakeClient(served=10_000)
    pages(client)
    assert client.offsets[-1] == MAX_OFFSET
    assert 10_000 not in client.offsets


def test_normalize_hashes_author_and_drops_name():
    row = metacritic.normalize(make_item(1, author="Alice"), "playstation-5", "salt")
    assert "author" not in row
    assert row["author_hash"] == hmac.new(b"salt", b"Alice", "sha256").hexdigest()
    again = metacritic.normalize(make_item(1, author="Alice"), "playstation-5", "salt")
    assert again["author_hash"] == row["author_hash"]


def test_normalize_requires_salt():
    with pytest.raises(ValueError):
        metacritic.normalize(make_item(1), "playstation-5", "")


def test_normalize_tolerates_empty_quote_and_missing_author():
    row = metacritic.normalize(make_item(1, quote=None, author=None), "playstation-5", "salt")
    assert row["text"] == ""
    assert row["author_hash"] is None


def test_normalize_maps_fields():
    row = metacritic.normalize(make_item(5, spoiler=True), "xbox-series-x", "salt")
    assert row["source"] == "metacritic"
    assert row["source_review_id"] == str(uuid.UUID(int=5))
    assert (row["platform"], row["score"], row["date"]) == ("xbox-series-x", 7, "2026-11-24")
    assert row["version_ms"] == 1795548626092 + 5
    assert row["spoiler"] is True


def test_content_hash_changes_with_score_or_text_not_version():
    base = metacritic.normalize(make_item(1), "playstation-5", "s")["content_hash"]
    assert base == hashlib.sha256(b"7\ntext 1").hexdigest()
    other_score = metacritic.normalize(make_item(1, score=3), "playstation-5", "s")
    other_text = metacritic.normalize(make_item(1, quote="edited"), "playstation-5", "s")
    other_version = metacritic.normalize(make_item(1, version=1), "playstation-5", "s")
    assert other_score["content_hash"] != base
    assert other_text["content_hash"] != base
    assert other_version["content_hash"] == base

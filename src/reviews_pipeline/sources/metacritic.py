"""Metacritic user-review adapter: paging and normalization."""

import hashlib
import hmac
from collections.abc import Callable, Iterator

from reviews_pipeline.config import BASE_URL, MAX_OFFSET, PAGE_LIMIT
from reviews_pipeline.http import PoliteClient

SOURCE_NAME = "metacritic"

_COMPONENT = {
    "componentName": "user-reviews",
    "componentDisplayName": "user Reviews",
    "componentType": "ReviewList",
}


def iter_pages(
    client: PoliteClient,
    platform: str,
    sentiment: str,
    sort: str = "date",
    is_known: Callable[[list[dict]], bool] = lambda page: False,
) -> Iterator[list[dict]]:
    """Yield raw item pages, newest first, never past the 10,000-result cap."""
    url = BASE_URL.format(platform=platform)
    for offset in range(0, MAX_OFFSET + 1, PAGE_LIMIT):
        params = {
            "offset": offset,
            "limit": PAGE_LIMIT,
            "filterBySentiment": sentiment,
            "sort": sort,
            **_COMPONENT,
        }
        data = client.get_json(url, params)["data"]
        items = data["items"]
        if not items:
            return
        yield items
        if len(items) < PAGE_LIMIT or offset + PAGE_LIMIT >= data["totalResults"] or is_known(items):
            return


def normalize(item: dict, platform: str, salt: str) -> dict:
    """Map a raw item to the silver schema, hashing the author name."""
    if not salt:
        raise ValueError("salt is required to hash author names")
    text = item.get("quote") or ""
    author = item.get("author")
    author_hash = (
        hmac.new(salt.encode(), author.encode(), hashlib.sha256).hexdigest() if author else None
    )
    return {
        "source": SOURCE_NAME,
        "source_review_id": item["id"],
        "platform": platform,
        "score": item["score"],
        "date": item["date"],
        "version_ms": item["version"],
        "author_hash": author_hash,
        "text": text,
        "spoiler": bool(item.get("spoiler")),
        "content_hash": hashlib.sha256(f"{item['score']}\n{text}".encode()).hexdigest(),
    }

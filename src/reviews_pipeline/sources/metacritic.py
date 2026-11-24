"""Metacritic user-review adapter: paging and normalization."""

import hashlib
import hmac
from collections.abc import Iterator

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
) -> Iterator[list[dict]]:
    """Yield raw item pages, newest first, never past the 10,000-result cap.

    Args:
        client: Client used for every request.
        platform: One of `config.PLATFORMS`.
        sentiment: `all`, `positive`, `neutral` or `negative`.
        sort: `date` or `score`.

    Yields:
        list[dict]: Raw items of one page. Stops after a short page, when the reported
            total is reached, or after the page at `config.MAX_OFFSET`.
    """
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
        if len(items) < PAGE_LIMIT or offset + PAGE_LIMIT >= data["totalResults"]:
            return


def normalize(item: dict, platform: str, salt: str) -> dict:
    """Map a raw item to the silver schema, hashing the author name.

    Args:
        item: Raw item as returned by the endpoint.
        platform: Platform the item was collected from.
        salt: Secret used as the HMAC key for the author name.

    Returns:
        dict: Row with the silver columns. The author name is never kept; `author_hash`
            is None when the item has no author.

    Raises:
        ValueError: If `salt` is empty.
    """
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

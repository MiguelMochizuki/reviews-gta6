"""One-off live checks of the Metacritic reviews endpoint.

Prints counts and booleans only, never review text or author names.
Run with: uv run python scripts/verify_endpoint.py
"""

import re
from datetime import UTC, datetime

from reviews_pipeline.config import BASE_URL
from reviews_pipeline.http import PoliteClient

PARAMS = {
    "componentName": "user-reviews",
    "componentDisplayName": "user Reviews",
    "componentType": "ReviewList",
}


def fetch(client, platform, sentiment, sort, limit, offset=0):
    params = {
        "offset": offset,
        "limit": limit,
        "filterBySentiment": sentiment,
        "sort": sort,
        **PARAMS,
    }
    return client.get_json(BASE_URL.format(platform=platform), params)


def main():
    client = PoliteClient()

    big = fetch(client, "playstation-5", "all", "date", 200)
    print(f"(a) limit=200 returned {len(big['data']['items'])} items")

    items = fetch(client, "playstation-5", "all", "date", 100)["data"]["items"]
    same_day = sum(
        datetime.fromtimestamp(i["version"] / 1000, UTC).date().isoformat() == i["date"]
        for i in items
    )
    print(f"(b) version as epoch-ms falls on the item's date for {same_day}/{len(items)} items")

    by_date = fetch(client, "playstation-5", "negative", "date", 100)["data"]["items"]
    by_score = fetch(client, "playstation-5", "negative", "score", 100)["data"]["items"]
    overlap = len({i["id"] for i in by_date} & {i["id"] for i in by_score})
    scores = [i["score"] for i in by_score]
    print(f"(c) negative page 0: date vs score overlap {overlap}/100; "
          f"score sort spans {min(scores)}..{max(scores)}")

    last = fetch(client, "playstation-5", "all", "date", 100)["links"]["last"]["href"]
    print(f"(d) links.last offset for PS5 all: {re.search(r'offset=(\d+)', last).group(1)}")


if __name__ == "__main__":
    main()

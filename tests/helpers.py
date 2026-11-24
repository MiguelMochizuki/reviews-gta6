"""Shared fakes for tests: synthetic review items and a fake client."""

import uuid

from reviews_pipeline.http import CircuitOpen


def make_item(i, **overrides):
    item = {
        "id": str(uuid.UUID(int=i)),
        "quote": f"text {i}",
        "score": 7,
        "date": "2026-11-24",
        "author": f"user{i}",
        "version": 1795548626092 + i,
        "spoiler": False,
        "platform": "PlayStation 5",
    }
    item.update(overrides)
    return item


class FakeClient:
    """Serves `served` items per query and reports `total` as totalResults.

    `item` builds the i-th item; `fail_at_offset` raises CircuitOpen at that offset.
    """

    def __init__(self, served, total=None, item=make_item, fail_at_offset=None, cap=None):
        self.served = served
        self.total = served if total is None else total
        self.item = item
        self.fail_at_offset = fail_at_offset
        self.cap = cap  # server-side page size cap, smaller than the requested limit
        self.offsets = []

    def get_json(self, url, params):
        offset = params["offset"]
        self.offsets.append(offset)
        if offset == self.fail_at_offset:
            raise CircuitOpen("503 after 5 attempts")
        limit = min(params["limit"], self.cap) if self.cap else params["limit"]
        items = [self.item(i) for i in range(offset, min(offset + limit, self.served))]
        return {"data": {"totalResults": self.total, "items": items}}


def utc_ms(year, month, day, hour=0):
    """Epoch milliseconds of a UTC time, for readable version_ms values in tests."""
    from datetime import UTC, datetime

    return int(datetime(year, month, day, hour, tzinfo=UTC).timestamp() * 1000)


def seeded_silver(rows, seen_at=None):
    """Silver connection seeded from (platform, score, date, version_ms, text) tuples."""
    from datetime import datetime

    import duckdb

    from reviews_pipeline import silver

    con = duckdb.connect()
    silver.init(con)
    batch = [
        {
            "source": "metacritic", "source_review_id": f"id-{i}", "platform": platform,
            "score": score, "date": date, "version_ms": version_ms,
            "author_hash": f"hash-{i}", "text": text, "spoiler": False,
            "content_hash": f"c-{i}",
        }
        for i, (platform, score, date, version_ms, text) in enumerate(rows)
    ]
    silver.upsert(con, batch, seen_at or datetime(2026, 11, 24, 12, 0))
    return con

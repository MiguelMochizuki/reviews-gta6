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

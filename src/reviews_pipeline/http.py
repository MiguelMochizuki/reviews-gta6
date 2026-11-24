"""Polite HTTP client: honest User-Agent, rate limit, retries with backoff."""

import time

import httpx

from reviews_pipeline.config import MIN_INTERVAL_S, USER_AGENT


class CircuitOpen(Exception):
    """Raised when the server keeps answering 429/5xx after all retries."""


class PoliteClient:
    def __init__(
        self,
        transport: httpx.BaseTransport | None = None,
        min_interval: float = MIN_INTERVAL_S,
        max_retries: int = 4,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        self._client = httpx.Client(
            transport=transport, headers={"User-Agent": USER_AGENT}, timeout=30
        )
        self._min_interval = min_interval
        self._max_retries = max_retries
        self._sleep = sleep
        self._clock = clock
        self._last: float | None = None

    def _pace(self) -> None:
        if self._last is not None:
            wait = self._min_interval - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._clock()

    def get_json(self, url: str, params: dict) -> dict:
        for attempt in range(self._max_retries + 1):
            self._pace()
            response = self._client.get(url, params=params)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == self._max_retries:
                    raise CircuitOpen(f"{response.status_code} after {attempt + 1} attempts")
                self._sleep(2 ** (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()
        raise AssertionError("unreachable")

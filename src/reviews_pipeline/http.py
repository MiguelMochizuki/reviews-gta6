"""Polite HTTP client: honest User-Agent, rate limit, retries with backoff."""

import time

import httpx

from reviews_pipeline.config import MIN_INTERVAL_S, USER_AGENT


class CircuitOpen(Exception):
    """Raised when the server keeps answering 429/5xx or the connection keeps failing."""


class PoliteClient:
    """HTTP client that paces requests, retries 429/5xx and identifies itself honestly."""

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
        """Fetch a URL and return its JSON body.

        Args:
            url: Endpoint to request.
            params: Query parameters.

        Returns:
            dict: Parsed JSON response.

        Raises:
            CircuitOpen: If the server still answers 429/5xx, or the connection keeps
                failing, after all retries.
            httpx.HTTPStatusError: For any other 4xx response (not retried).
        """
        for attempt in range(self._max_retries + 1):
            self._pace()
            try:
                response = self._client.get(url, params=params)
                failure = str(response.status_code) if (
                    response.status_code == 429 or response.status_code >= 500
                ) else None
            except httpx.TransportError as error:
                failure = type(error).__name__
            if failure:
                if attempt == self._max_retries:
                    raise CircuitOpen(f"{failure} after {attempt + 1} attempts")
                self._sleep(2 ** (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()
        raise AssertionError("unreachable")

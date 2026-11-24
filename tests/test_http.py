import httpx
import pytest

from reviews_pipeline.config import USER_AGENT
from reviews_pipeline.http import CircuitOpen, PoliteClient

URL = "https://example.test/api"


def make_client(handler, **kwargs):
    sleeps = []
    client = PoliteClient(
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        clock=lambda: 100.0,
        **kwargs,
    )
    return client, sleeps


def test_sends_honest_user_agent():
    seen = []

    def handler(request):
        seen.append(request.headers["user-agent"])
        return httpx.Response(200, json={})

    client, _ = make_client(handler)
    client.get_json(URL, {})
    assert seen == [USER_AGENT]


def test_waits_min_interval_between_calls():
    client, sleeps = make_client(lambda r: httpx.Response(200, json={}))
    client.get_json(URL, {})
    client.get_json(URL, {})
    assert sleeps == [3.0]


def test_retries_429_then_succeeds():
    answers = iter([429, 429, 200])
    client, sleeps = make_client(
        lambda r: httpx.Response(next(answers), json={"ok": True}), min_interval=0
    )
    assert client.get_json(URL, {}) == {"ok": True}
    assert sleeps == [2, 4]


def test_raises_circuit_open_when_retries_exhausted():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503)

    client, _ = make_client(handler, min_interval=0, max_retries=2)
    with pytest.raises(CircuitOpen):
        client.get_json(URL, {})
    assert len(calls) == 3


def test_other_4xx_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404)

    client, _ = make_client(handler, min_interval=0)
    with pytest.raises(httpx.HTTPStatusError):
        client.get_json(URL, {})
    assert len(calls) == 1


def test_retries_transport_errors_then_succeeds():
    outcomes = iter([httpx.ConnectTimeout("slow"), httpx.ReadError("reset"), None])

    def handler(request):
        outcome = next(outcomes)
        if outcome:
            raise outcome
        return httpx.Response(200, json={"ok": True})

    client, sleeps = make_client(handler, min_interval=0)
    assert client.get_json(URL, {}) == {"ok": True}
    assert sleeps == [2, 4]


def test_transport_errors_exhaust_into_circuit_open():
    def handler(request):
        raise httpx.ConnectError("down")

    client, _ = make_client(handler, min_interval=0, max_retries=2)
    with pytest.raises(CircuitOpen):
        client.get_json(URL, {})

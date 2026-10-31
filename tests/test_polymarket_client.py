"""Tests for ``polysignal.ingestion.polymarket.client``.

The HTTP layer is exercised via ``httpx.MockTransport`` so the suite
never touches the network. The (default) opt-in live integration test
lives in ``tests/test_polymarket_network.py`` and is skipped unless
``POLYSIGNAL_RUN_NETWORK_TESTS=1`` is set.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from polysignal.config import Settings
from polysignal.ingestion.polymarket.client import PolymarketAPIError, PolymarketClient

FIXTURES = Path(__file__).parent / "fixtures" / "polymarket"


def _settings() -> Settings:
    s = Settings()
    s.polymarket.request_timeout_seconds = 1.0
    return s


def _mock_client(handler) -> PolymarketClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, timeout=1.0)
    return PolymarketClient(settings=_settings(), http_client=http)


def test_fetch_markets_page_returns_list() -> None:
    payload = [json.loads((FIXTURES / "binary_market.json").read_text(encoding="utf-8"))]

    def handler(request: httpx.Request) -> httpx.Response:
        assert "markets" in str(request.url)
        assert request.url.params.get("limit") == "10"
        return httpx.Response(200, json=payload)

    with _mock_client(handler) as client:
        out = client.fetch_markets_page(limit=10, offset=0)
    assert isinstance(out, list)
    assert out[0]["conditionId"] == payload[0]["conditionId"]


def test_iter_markets_paginates_and_respects_limit() -> None:
    page1 = [{"id": str(i)} for i in range(3)]
    page2 = [{"id": str(i)} for i in range(3, 5)]
    pages = iter([page1, page2, []])

    def handler(request: httpx.Request) -> httpx.Response:
        next_page = next(pages)
        return httpx.Response(200, json=next_page)

    with _mock_client(handler) as client:
        collected = list(client.iter_markets(limit=5, page_size=3))
    flat = [m for _, p in collected for m in p]
    assert len(flat) == 5
    # Two pages: 3 then 2.
    assert [len(p) for _, p in collected] == [3, 2]


def test_iter_markets_respects_zero_limit() -> None:
    def handler(_: httpx.Request) -> httpx.Response:  # pragma: no cover — must not be called
        raise AssertionError("HTTP must not be called when limit=0")

    with _mock_client(handler) as client:
        assert list(client.iter_markets(limit=0)) == []


def test_fetch_events_page_returns_list_with_ordering_params() -> None:
    payload = [{"id": "evt-1", "markets": []}]

    def handler(request: httpx.Request) -> httpx.Response:
        assert "events" in str(request.url)
        assert request.url.params.get("limit") == "10"
        assert request.url.params.get("offset") == "5"
        assert request.url.params.get("order") == "volume_24hr"
        assert request.url.params.get("ascending") == "false"
        return httpx.Response(200, json=payload)

    with _mock_client(handler) as client:
        out = client.fetch_events_page(
            limit=10,
            offset=5,
            extra_params={"order": "volume_24hr", "ascending": "false"},
        )
    assert out == payload


def test_iter_events_paginates_and_respects_limit() -> None:
    page1 = [{"id": "evt-1"}, {"id": "evt-2"}]
    page2 = [{"id": "evt-3"}]
    pages = iter([page1, page2, []])

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(pages))

    with _mock_client(handler) as client:
        collected = list(client.iter_events(limit=3, page_size=2))
    assert [len(p) for _, p in collected] == [2, 1]
    assert [e["id"] for _, page in collected for e in page] == ["evt-1", "evt-2", "evt-3"]


def test_fetch_market_tries_multiple_lookup_params() -> None:
    seen_params: list[str] = []
    target = json.loads((FIXTURES / "binary_market.json").read_text(encoding="utf-8"))

    def handler(request: httpx.Request) -> httpx.Response:
        # First lookup param ("condition_ids") wins.
        keys = sorted(request.url.params.keys())
        seen_params.extend(keys)
        return httpx.Response(200, json=[target])

    with _mock_client(handler) as client:
        out = client.fetch_market(target["conditionId"])
    assert out == target
    assert "condition_ids" in seen_params


def test_fetch_market_returns_none_when_not_found() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    with _mock_client(handler) as client:
        assert client.fetch_market("nope") is None


def test_fetch_market_falls_through_soft_400_to_next_variant() -> None:
    """A 400 on one lookup variant should NOT mask a 200 on the next one."""
    target = json.loads((FIXTURES / "binary_market.json").read_text(encoding="utf-8"))
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        keys = set(request.url.params.keys())
        # Reject ``condition_ids``, accept ``slug``.
        if "condition_ids" in keys:
            return httpx.Response(400, json={"error": "unknown param"})
        if "slug" in keys:
            return httpx.Response(200, json=[target])
        return httpx.Response(404, json={"error": "no"})

    with _mock_client(handler) as client:
        out = client.fetch_market(target["slug"])
    assert out == target
    # Both variants must have been attempted.
    assert state["calls"] == 2


def test_fetch_market_raises_when_every_variant_returns_soft_error() -> None:
    """If Gamma never gives us a 200, we must NOT silently return None."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad lookup"})

    with _mock_client(handler) as client, pytest.raises(PolymarketAPIError) as exc_info:
        client.fetch_market("anything")
    assert exc_info.value.status_code == 400
    assert "every lookup variant rejected" in str(exc_info.value)


def test_fetch_market_propagates_5xx_without_swallowing() -> None:
    """A real outage must propagate immediately, not be hidden as 'not found'."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    with _mock_client(handler) as client, pytest.raises(PolymarketAPIError) as exc_info:
        client.fetch_market("anything")
    assert exc_info.value.status_code == 500


def test_fetch_market_propagates_401_without_swallowing() -> None:
    """Auth-style errors must surface — they are not 'bad lookup' problems."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    with _mock_client(handler) as client, pytest.raises(PolymarketAPIError) as exc_info:
        client.fetch_market("anything")
    assert exc_info.value.status_code == 401


def test_fetch_price_history_envelope() -> None:
    envelope = json.loads((FIXTURES / "price_history.json").read_text(encoding="utf-8"))

    def handler(request: httpx.Request) -> httpx.Response:
        assert "prices-history" in str(request.url)
        return httpx.Response(200, json=envelope)

    with _mock_client(handler) as client:
        out = client.fetch_price_history("token-1")
    assert "history" in out
    assert out["history"][0]["p"] == pytest.approx(0.62)


def test_get_4xx_raises_no_retry() -> None:
    call_count = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(404, json={"error": "not found"})

    with _mock_client(handler) as client, pytest.raises(PolymarketAPIError) as exc_info:
        client.fetch_markets_page(limit=1)
    assert exc_info.value.status_code == 404
    # No retry on 4xx — exactly one call.
    assert call_count["n"] == 1


def test_get_5xx_eventually_raises() -> None:
    call_count = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(500, json={"error": "boom"})

    with _mock_client(handler) as client, pytest.raises(PolymarketAPIError) as exc_info:
        client.fetch_markets_page(limit=1)
    assert exc_info.value.status_code == 500
    # tenacity stops after 4 attempts before the wrapper raises.
    assert call_count["n"] == 4


def test_get_recovers_after_transient_5xx() -> None:
    state = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] < 2:
            return httpx.Response(503, json={"error": "transient"})
        return httpx.Response(200, json=[{"id": "ok"}])

    with _mock_client(handler) as client:
        page = client.fetch_markets_page(limit=1)
    assert page == [{"id": "ok"}]
    assert state["n"] == 2

"""Tests for the chunked full-lifetime price fetch and corrected market lookup.

Both exercise the HTTP layer through ``httpx.MockTransport`` so the suite
never touches the network.
"""

from __future__ import annotations

import httpx

from polysignal.config import Settings
from polysignal.ingestion.polymarket.client import PolymarketClient


def _settings() -> Settings:
    s = Settings()
    s.polymarket.request_timeout_seconds = 1.0
    return s


def _mock_client(handler) -> PolymarketClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, timeout=1.0)
    return PolymarketClient(settings=_settings(), http_client=http)


def test_fetch_price_history_range_walks_chunks_and_dedupes() -> None:
    seen_windows: list[tuple[int, int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        start = int(params["startTs"])
        end = int(params["endTs"])
        seen_windows.append((start, end))
        # Return one point at the window start and one shared boundary point
        # (start) so the dedupe path is exercised across adjacent chunks.
        return httpx.Response(
            200,
            json={"history": [{"t": start, "p": 0.5}, {"t": start + 100, "p": 0.6}]},
        )

    chunk = 14 * 86400
    start_ts = 1_700_000_000
    end_ts = start_ts + 3 * chunk  # exactly three windows
    with _mock_client(handler) as client:
        envelope = client.fetch_price_history_range(
            "token-yes",
            start_ts=start_ts,
            end_ts=end_ts,
            fidelity=60,
            pause_seconds=0.0,
        )

    # Three windows requested, none longer than the chunk size.
    assert len(seen_windows) == 3
    assert all(end - start <= chunk for start, end in seen_windows)
    history = envelope["history"]
    # Sorted ascending and deduplicated by timestamp.
    timestamps = [row["t"] for row in history]
    assert timestamps == sorted(timestamps)
    assert len(timestamps) == len(set(timestamps))


def test_fetch_price_history_range_empty_window() -> None:
    def handler(_: httpx.Request) -> httpx.Response:  # pragma: no cover - must not be called
        raise AssertionError("no HTTP call expected for an empty window")

    with _mock_client(handler) as client:
        envelope = client.fetch_price_history_range(
            "token-yes", start_ts=100, end_ts=100, pause_seconds=0.0
        )
    assert envelope == {"history": []}


def test_fetch_market_verifies_match_and_retries_closed() -> None:
    cid = "0xabc"
    calls: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        calls.append(params)
        # First (no closed flag) returns the wrong market (Gamma ignored the
        # unknown filter and served the listing head); the closed retry
        # returns the right one.
        if params.get("closed") == "true":
            return httpx.Response(200, json=[{"conditionId": cid, "slug": "right"}])
        return httpx.Response(200, json=[{"conditionId": "0xother", "slug": "wrong"}])

    with _mock_client(handler) as client:
        market = client.fetch_market(cid)
    assert market is not None
    assert market["conditionId"] == cid
    # The non-matching first response must not be accepted.
    assert any(call.get("closed") == "true" for call in calls)


def test_fetch_market_returns_none_without_verified_match() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"conditionId": "0xother", "slug": "wrong"}])

    with _mock_client(handler) as client:
        assert client.fetch_market("0xnotpresent") is None

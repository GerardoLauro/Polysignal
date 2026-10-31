"""Integration-style tests for the Polymarket ingestion pipeline.

The HTTP layer is faked via ``httpx.MockTransport``. The pipeline runs
the real filter, normalizer, raw store, and Parquet writer against a
``tmp_path`` data root, so this exercise covers every link in the
ingestion chain without touching the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd

from polysignal.config import Settings
from polysignal.ingestion.polymarket.client import PolymarketClient
from polysignal.ingestion.polymarket.pipeline import (
    fetch_prices_for_market,
    run_market_ingestion,
)
from polysignal.storage.paths import get_data_paths

FIXTURES = Path(__file__).parent / "fixtures" / "polymarket"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _event(event_id: str, markets: list[dict]) -> dict:
    return {
        "id": event_id,
        "slug": f"event-{event_id}",
        "title": f"Event {event_id}",
        "active": True,
        "closed": False,
        "markets": markets,
    }


def _without_events(payload: dict) -> dict:
    out = dict(payload)
    out.pop("events", None)
    return out


def _build_client(handler) -> PolymarketClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, timeout=1.0)
    return PolymarketClient(settings=Settings(), http_client=http)


def test_run_market_ingestion_filters_and_persists(tmp_path: Path) -> None:
    binary = _load("binary_market.json")
    resolved = _load("resolved_binary_market.json")
    multi = _load("multi_outcome_market.json")
    non_yes_no = _load("non_yes_no_binary.json")
    price_history = _load("price_history.json")
    page = [binary, multi, non_yes_no, resolved]

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "events" in url:
            offset = int(request.url.params.get("offset", "0"))
            if offset == 0:
                return httpx.Response(200, json=[_event("evt-a", page)])
            return httpx.Response(200, json=[])
        if "prices-history" in url:
            return httpx.Response(200, json=price_history)
        return httpx.Response(404, json={"error": "unknown"})

    paths = get_data_paths(root=tmp_path)
    client = _build_client(handler)
    try:
        result = run_market_ingestion(
            limit=10,
            binary_only=True,
            save_raw=True,
            fetch_prices=True,
            page_size=10,
            max_markets_per_event=None,
            market_status="all",
            client=client,
            paths=paths,
        )
    finally:
        client.close()

    assert result.fetched == 4
    assert result.events_fetched == 1
    assert result.markets_discovered == 4
    assert result.kept == 2
    assert len(result.skipped) == 2
    # Skip reasons exist and are informative.
    skip_reasons = {s.market_ref: s.reason for s in result.skipped}
    assert any("4 outcomes" in r for r in skip_reasons.values())
    assert any("non-YES/NO" in r for r in skip_reasons.values())

    # Parquet files exist and are loadable.
    assert result.markets_parquet is not None and result.markets_parquet.exists()
    assert result.outcomes_parquet is not None and result.outcomes_parquet.exists()
    assert result.price_points_parquet is not None and result.price_points_parquet.exists()

    markets_df = pd.read_parquet(result.markets_parquet)
    assert len(markets_df) == 2
    assert set(markets_df["market_id"]) == {binary["conditionId"], resolved["conditionId"]}
    assert {"event_id", "volume", "liquidity", "is_closed"} <= set(markets_df.columns)

    outcomes_df = pd.read_parquet(result.outcomes_parquet)
    assert len(outcomes_df) == 4  # 2 markets * 2 outcomes each.
    assert "market_id" in outcomes_df.columns
    assert set(outcomes_df["label"]) >= {"Yes", "No", "YES", "NO"}

    prices_df = pd.read_parquet(result.price_points_parquet)
    # 2 binary markets * 2 outcomes * 4 history entries = 16 points.
    assert len(prices_df) == 16

    # Raw payloads were persisted.
    assert result.raw_events_dir is not None and result.raw_events_dir.exists()
    assert (result.raw_events_dir / "_manifest.json").exists()
    page_files = sorted(result.raw_events_dir.glob("page-*.json"))
    assert len(page_files) == 1
    assert result.raw_prices_dir is not None and (result.raw_prices_dir / "_manifest.json").exists()


def test_run_market_ingestion_dry_run_skips_parquet(tmp_path: Path) -> None:
    binary = _load("binary_market.json")

    def handler(request: httpx.Request) -> httpx.Response:
        if "events" in str(request.url):
            offset = int(request.url.params.get("offset", "0"))
            return httpx.Response(200, json=[_event("evt-a", [binary])] if offset == 0 else [])
        return httpx.Response(404)

    paths = get_data_paths(root=tmp_path)
    client = _build_client(handler)
    try:
        result = run_market_ingestion(
            limit=5,
            save_raw=False,
            dry_run=True,
            fetch_prices=False,
            client=client,
            paths=paths,
        )
    finally:
        client.close()
    assert result.kept == 1
    assert result.markets_parquet is None
    assert result.outcomes_parquet is None
    assert result.price_points_parquet is None
    assert not paths.polymarket_markets_parquet.exists()


def test_run_market_ingestion_no_save_raw(tmp_path: Path) -> None:
    binary = _load("binary_market.json")

    def handler(request: httpx.Request) -> httpx.Response:
        if "events" in str(request.url):
            offset = int(request.url.params.get("offset", "0"))
            return httpx.Response(200, json=[_event("evt-a", [binary])] if offset == 0 else [])
        return httpx.Response(404)

    paths = get_data_paths(root=tmp_path)
    client = _build_client(handler)
    try:
        result = run_market_ingestion(
            limit=5,
            save_raw=False,
            client=client,
            paths=paths,
        )
    finally:
        client.close()
    assert result.kept == 1
    assert result.raw_events_dir is None


def test_run_market_ingestion_uses_events_ordering_and_event_cap(tmp_path: Path) -> None:
    first = _without_events(_load("binary_market.json"))
    second = _without_events({**_load("binary_market.json"), "conditionId": "0x222", "id": "222"})
    third = _without_events({**_load("binary_market.json"), "conditionId": "0x333", "id": "333"})
    seen_query: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if "events" in str(request.url):
            seen_query["order"] = request.url.params.get("order")
            seen_query["ascending"] = request.url.params.get("ascending")
            offset = int(request.url.params.get("offset", "0"))
            if offset == 0:
                return httpx.Response(
                    200,
                    json=[
                        _event("evt-a", [first, third]),
                        _event("evt-b", [second]),
                    ],
                )
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    paths = get_data_paths(root=tmp_path)
    client = _build_client(handler)
    try:
        result = run_market_ingestion(
            limit=5,
            save_raw=False,
            fetch_prices=False,
            max_markets_per_event=1,
            order="volume_24hr",
            ascending=False,
            client=client,
            paths=paths,
        )
    finally:
        client.close()

    assert seen_query == {"order": "volume_24hr", "ascending": "false"}
    assert result.events_fetched == 2
    assert result.markets_discovered == 3
    assert result.kept == 2
    assert {m.event_id for m in result.markets} == {"evt-a", "evt-b"}


def test_run_market_ingestion_records_clob_failure_per_outcome(tmp_path: Path) -> None:
    """When CLOB returns 500 for a token, we record the reason and move on."""
    binary = _load("binary_market.json")

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "events" in url:
            offset = int(request.url.params.get("offset", "0"))
            return httpx.Response(200, json=[_event("evt-a", [binary])] if offset == 0 else [])
        if "prices-history" in url:
            return httpx.Response(503, json={"error": "boom"})
        return httpx.Response(404)

    paths = get_data_paths(root=tmp_path)
    client = _build_client(handler)
    try:
        result = run_market_ingestion(
            limit=1,
            save_raw=False,
            fetch_prices=True,
            client=client,
            paths=paths,
        )
    finally:
        client.close()

    assert result.kept == 1
    # The two outcomes each have a failed price-fetch attempt with a reason.
    assert len(result.price_attempts) == 2
    assert all(not a.fetched for a in result.price_attempts)
    assert all(a.reason and "clob" in a.reason.lower() for a in result.price_attempts)
    # No price-points parquet is written when nothing was fetched.
    assert result.price_points_parquet is None


def test_bulk_ingestion_overwrites_price_points_dataset(tmp_path: Path) -> None:
    first = _without_events(_load("binary_market.json"))
    second = _without_events({**_load("binary_market.json"), "conditionId": "0x222", "id": "222"})
    history = _load("price_history.json")
    state = {"markets": [first]}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "events" in url:
            offset = int(request.url.params.get("offset", "0"))
            return (
                httpx.Response(200, json=[_event("evt-a", state["markets"])])
                if offset == 0
                else httpx.Response(200, json=[])
            )
        if "prices-history" in url:
            return httpx.Response(200, json=history)
        return httpx.Response(404)

    paths = get_data_paths(root=tmp_path)
    client = _build_client(handler)
    try:
        run_market_ingestion(
            limit=1,
            save_raw=False,
            fetch_prices=True,
            client=client,
            paths=paths,
        )
        state["markets"] = [second]
        result = run_market_ingestion(
            limit=1,
            save_raw=False,
            fetch_prices=True,
            client=client,
            paths=paths,
        )
    finally:
        client.close()

    df = pd.read_parquet(result.price_points_parquet)
    assert set(df["market_id"]) == {"0x222"}


def test_fetch_prices_for_market_single_market(tmp_path: Path) -> None:
    binary = _load("binary_market.json")
    history = _load("price_history.json")

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "markets" in url:
            return httpx.Response(200, json=[binary])
        if "prices-history" in url:
            return httpx.Response(200, json=history)
        return httpx.Response(404)

    paths = get_data_paths(root=tmp_path)
    client = _build_client(handler)
    try:
        result = fetch_prices_for_market(binary["conditionId"], client=client, paths=paths)
    finally:
        client.close()

    assert result.kept == 1
    assert len(result.price_points) == 8  # 2 outcomes * 4 entries.
    assert result.price_points_parquet is not None


def test_fetch_prices_for_market_rejects_non_binary(tmp_path: Path) -> None:
    multi = _load("multi_outcome_market.json")

    def handler(request: httpx.Request) -> httpx.Response:
        if "markets" in str(request.url):
            return httpx.Response(200, json=[multi])
        return httpx.Response(404)

    paths = get_data_paths(root=tmp_path)
    client = _build_client(handler)
    try:
        result = fetch_prices_for_market(multi["conditionId"], client=client, paths=paths)
    finally:
        client.close()
    assert result.kept == 0
    assert len(result.skipped) == 1
    assert "4 outcomes" in result.skipped[0].reason


def test_price_points_dedupe_on_repeat_fetch(tmp_path: Path) -> None:
    """A second fetch of the same market should not duplicate price rows."""
    binary = _load("binary_market.json")
    history = _load("price_history.json")

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "markets" in url:
            return httpx.Response(200, json=[binary])
        if "prices-history" in url:
            return httpx.Response(200, json=history)
        return httpx.Response(404)

    paths = get_data_paths(root=tmp_path)
    client = _build_client(handler)
    try:
        fetch_prices_for_market(binary["conditionId"], client=client, paths=paths)
        result = fetch_prices_for_market(binary["conditionId"], client=client, paths=paths)
    finally:
        client.close()
    df = pd.read_parquet(result.price_points_parquet)
    # 2 outcomes * 4 timestamps = 8 unique rows even after two fetches.
    assert len(df) == 8

"""Live Polymarket integration tests.

Skipped by default. Set ``POLYSIGNAL_RUN_NETWORK_TESTS=1`` in the
environment to opt in. These tests exercise the real Gamma + CLOB
endpoints and require an internet connection. They are slow and may
flake on upstream maintenance — keep them out of CI unless you have
explicit network budget.
"""

from __future__ import annotations

import os

import pytest

from polysignal.ingestion.polymarket.binary_filter import split_binary
from polysignal.ingestion.polymarket.client import PolymarketClient

pytestmark = pytest.mark.skipif(
    os.environ.get("POLYSIGNAL_RUN_NETWORK_TESTS") != "1",
    reason="set POLYSIGNAL_RUN_NETWORK_TESTS=1 to run live Polymarket tests",
)


def test_gamma_returns_real_markets_and_filter_works() -> None:
    with PolymarketClient() as client:
        page = client.fetch_markets_page(limit=20, offset=0)
    assert isinstance(page, list)
    assert page, "Gamma API returned an empty page"
    kept, skipped = split_binary(page)
    # We do not assert specific counts (the Polymarket catalog changes
    # constantly), but we do assert that the function does not crash
    # and at least one outcome lands in either bucket.
    assert len(kept) + len(skipped) == len(page)


def test_clob_price_history_fetch() -> None:
    with PolymarketClient() as client:
        page = client.fetch_markets_page(limit=20, offset=0)
        kept, _ = split_binary(page)
        if not kept:
            pytest.skip("no binary markets in the first Gamma page right now")
        _, binary = kept[0]
        if not binary.yes_token_id:
            pytest.skip("kept binary market has no CLOB token id exposed")
        envelope = client.fetch_price_history(binary.yes_token_id)
    assert isinstance(envelope, dict)
    history = envelope.get("history")
    assert isinstance(history, list)

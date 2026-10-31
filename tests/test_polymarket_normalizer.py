"""Tests for ``polysignal.ingestion.polymarket.normalizer``."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from polysignal.domain import MarketDomain, ResolutionResult
from polysignal.ingestion.polymarket.binary_filter import classify_market
from polysignal.ingestion.polymarket.normalizer import (
    build_outcomes,
    normalize_market,
    normalize_price_history,
)

FIXTURES = Path(__file__).parent / "fixtures" / "polymarket"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# ------------------------------------------------------------------- markets


def test_normalize_binary_market_full_payload() -> None:
    payload = _load("binary_market.json")
    market = normalize_market(payload)

    assert market.market_id == payload["conditionId"]
    assert market.condition_id == payload["conditionId"]
    assert market.slug == payload["slug"]
    assert market.question == payload["question"]
    assert market.description == payload["description"]
    assert market.domain is MarketDomain.POLITICS
    assert market.is_closed is False
    assert market.is_resolved is False
    assert market.resolution is None
    assert market.volume == pytest.approx(payload["volumeNum"])
    assert market.liquidity == pytest.approx(payload["liquidityNum"])
    assert market.event_id == "evt-9001"
    assert market.source_url == "https://polymarket.com/market/" + payload["slug"]
    assert market.created_at == datetime(2025, 1, 15, 0, 0, tzinfo=timezone.utc)
    assert market.close_time == datetime(2025, 11, 4, 23, 59, tzinfo=timezone.utc)

    # Two outcomes — YES first, NO second.
    assert len(market.outcomes) == 2
    yes, no = market.outcomes
    assert yes.market_id == payload["conditionId"]
    assert no.market_id == payload["conditionId"]
    assert yes.index == 0 and yes.label == "Yes"
    assert no.index == 1 and no.label == "No"
    assert yes.current_price == pytest.approx(0.62)
    assert no.current_price == pytest.approx(0.38)
    assert yes.outcome_id != no.outcome_id


def test_normalize_resolved_market_sets_resolution() -> None:
    payload = _load("resolved_binary_market.json")
    market = normalize_market(payload)
    assert market.is_closed is True
    assert market.is_resolved is True
    assert market.resolution is ResolutionResult.YES
    assert market.resolved_at == datetime(2025, 3, 15, 17, 0, tzinfo=timezone.utc)
    assert market.domain is MarketDomain.CRYPTO


def test_normalize_market_rejects_non_binary() -> None:
    payload = _load("multi_outcome_market.json")
    with pytest.raises(ValueError, match="not a binary"):
        normalize_market(payload)


def test_normalize_market_accepts_explicit_binary_arg() -> None:
    payload = _load("binary_market.json")
    classification = classify_market(payload)
    assert classification.market is not None
    market = normalize_market(payload, binary=classification.market)
    assert market.outcomes[0].label == "Yes"


def test_build_outcomes_uses_fallback_when_token_ids_missing() -> None:
    payload = {
        "id": "abc",
        "conditionId": "0xfeed",
        "slug": "demo",
        "outcomes": '["Yes", "No"]',
    }
    classification = classify_market(payload)
    assert classification.market is not None
    outcomes = build_outcomes(payload, classification.market)
    assert outcomes[0].outcome_id == "0xfeed-yes"
    assert outcomes[1].outcome_id == "0xfeed-no"


def test_normalize_market_falls_back_to_id_when_no_condition_id() -> None:
    payload = {
        "id": "12345",
        "slug": "demo",
        "question": "Will this happen?",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.5", "0.5"]',
    }
    market = normalize_market(payload)
    assert market.market_id == "12345"
    assert market.condition_id is None


def test_normalize_market_falls_back_to_now_when_dates_missing() -> None:
    payload = {
        "conditionId": "0xabc",
        "slug": "demo",
        "question": "Will this happen?",
        "outcomes": '["Yes", "No"]',
    }
    market = normalize_market(payload)
    # ``created_at`` must be timezone-aware UTC.
    assert market.created_at.tzinfo is not None
    assert market.created_at.utcoffset().total_seconds() == 0


# ----------------------------------------------------------------- prices


def test_normalize_price_history_from_envelope() -> None:
    history = _load("price_history.json")
    points = normalize_price_history(
        market_id="m1",
        outcome_id="o1",
        history=history,
    )
    assert len(points) == 4
    first = points[0]
    assert first.market_id == "m1"
    assert first.outcome_id == "o1"
    assert first.timestamp.tzinfo is not None
    assert first.timestamp.utcoffset().total_seconds() == 0
    assert 0.0 <= first.price <= 1.0


def test_normalize_price_history_accepts_bare_list() -> None:
    bare = [{"t": 1735689600, "p": 0.5}]
    points = normalize_price_history(market_id="m", outcome_id="o", history=bare)
    assert len(points) == 1


def test_normalize_price_history_drops_out_of_range_prices(caplog) -> None:
    history = {"history": [{"t": 1735689600, "p": 1.5}, {"t": 1735776000, "p": 0.4}]}
    with caplog.at_level("DEBUG"):
        points = normalize_price_history(market_id="m", outcome_id="o", history=history)
    assert len(points) == 1
    assert points[0].price == pytest.approx(0.4)


def test_normalize_price_history_handles_empty_payload() -> None:
    assert normalize_price_history(market_id="m", outcome_id="o", history={}) == []
    assert normalize_price_history(market_id="m", outcome_id="o", history=[]) == []
    assert normalize_price_history(market_id="m", outcome_id="o", history={"history": []}) == []


def test_normalize_price_history_skips_malformed_entries() -> None:
    history = {
        "history": [
            {"t": "not-a-number", "p": 0.5},
            {"p": 0.5},
            {"t": 1735689600},
            "garbage",
            {"t": 1735776000, "p": "0.42"},
        ]
    }
    points = normalize_price_history(market_id="m", outcome_id="o", history=history)
    assert len(points) == 1
    assert points[0].price == pytest.approx(0.42)

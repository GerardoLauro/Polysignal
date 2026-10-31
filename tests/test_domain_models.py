"""Tests for ``polysignal.domain.models``."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from polysignal.domain import (
    Driver,
    Market,
    MarketDomain,
    Outcome,
    PricePoint,
    SocialPressureRecord,
)


def test_market_requires_at_least_two_outcomes():
    with pytest.raises(ValidationError):
        Market(
            market_id="m",
            slug="s",
            question="q",
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            outcomes=[Outcome(outcome_id="yes", label="Yes", index=0)],
        )


def test_market_rejects_duplicate_outcome_ids():
    with pytest.raises(ValidationError):
        Market(
            market_id="m",
            slug="s",
            question="q",
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            outcomes=[
                Outcome(outcome_id="x", label="X1", index=0),
                Outcome(outcome_id="x", label="X2", index=1),
            ],
        )


def test_market_with_two_outcomes_is_valid():
    m = Market(
        market_id="m",
        slug="s",
        question="q",
        domain=MarketDomain.POLITICS,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        outcomes=[
            Outcome(outcome_id="yes", label="Yes", index=0),
            Outcome(outcome_id="no", label="No", index=1),
        ],
    )
    assert m.domain is MarketDomain.POLITICS


def test_price_point_clamps_range():
    with pytest.raises(ValidationError):
        PricePoint(
            market_id="m",
            outcome_id="yes",
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            price=1.2,
        )


def test_social_pressure_record_validates_window_order():
    with pytest.raises(ValidationError):
        SocialPressureRecord(
            market_id="m",
            outcome_id="yes",
            window_start=datetime(2025, 2, 1, tzinfo=timezone.utc),
            window_end=datetime(2025, 1, 1, tzinfo=timezone.utc),
            index_value=0.5,
            n_posts=0,
            drivers=[Driver(name="x", contribution=0.0, weight=1.0)],
        )

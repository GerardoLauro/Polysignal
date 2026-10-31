"""Tests for ``polysignal.forecasting.baseline``."""

from __future__ import annotations

from datetime import datetime, timezone

from polysignal.domain import ConfidenceBand
from polysignal.forecasting.baseline import BaselineBlender, BlenderConfig, ForecastInputs


def _ts() -> datetime:
    return datetime(2025, 3, 1, 12, tzinfo=timezone.utc)


def test_no_social_signal_falls_back_to_market_only():
    blender = BaselineBlender()
    out = blender.forecast(
        ForecastInputs(
            market_id="m1",
            outcome_id="yes",
            timestamp=_ts(),
            market_probability=0.6,
            social_probability=None,
            n_supporting_posts=0,
        )
    )
    assert out.adjusted_probability == 0.6
    assert out.social_probability is None
    assert out.delta == 0.0
    assert out.confidence is ConfidenceBand.LOW
    assert any("fallback" in line.lower() for line in out.explanation)


def test_small_blend_when_signals_agree():
    blender = BaselineBlender()
    out = blender.forecast(
        ForecastInputs(
            market_id="m1",
            outcome_id="yes",
            timestamp=_ts(),
            market_probability=0.6,
            social_probability=0.62,
            n_supporting_posts=20,
        )
    )
    assert 0.6 <= out.adjusted_probability <= 0.62
    assert abs(out.delta) < 0.02
    assert out.confidence is ConfidenceBand.MEDIUM


def test_large_disagreement_caps_social_weight():
    blender = BaselineBlender(BlenderConfig(max_social_weight=0.4))
    out = blender.forecast(
        ForecastInputs(
            market_id="m1",
            outcome_id="yes",
            timestamp=_ts(),
            market_probability=0.6,
            social_probability=0.05,
            n_supporting_posts=200,
        )
    )
    # With a 55-point disagreement, weight should be capped well below max_social_weight.
    # Empirically: w ≈ 0.4 * 1.0 * (1/3) * sanity → at most ~0.13.
    moved = out.market_probability - out.adjusted_probability
    assert 0.0 < moved < 0.4 * 0.6  # never more than would happen at max weight


def test_high_volume_high_confidence_band():
    blender = BaselineBlender()
    out = blender.forecast(
        ForecastInputs(
            market_id="m1",
            outcome_id="yes",
            timestamp=_ts(),
            market_probability=0.5,
            social_probability=0.55,
            n_supporting_posts=100,
        )
    )
    assert out.confidence is ConfidenceBand.HIGH


def test_adjusted_probability_stays_in_unit_interval():
    blender = BaselineBlender()
    for p_m, p_s in [(0.01, 0.99), (0.99, 0.01), (0.0, 0.5), (0.5, 1.0)]:
        out = blender.forecast(
            ForecastInputs(
                market_id="m1",
                outcome_id="yes",
                timestamp=_ts(),
                market_probability=p_m,
                social_probability=p_s,
                n_supporting_posts=10,
            )
        )
        assert 0.0 <= out.adjusted_probability <= 1.0

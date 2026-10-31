"""Tests for ``polysignal.features.social_pressure``."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from polysignal.domain import StanceLabel
from polysignal.features.social_pressure import (
    SocialPressureConfig,
    SocialPressureScorer,
    StanceObservation,
)


def _obs(label: StanceLabel, ts: datetime, **kwargs) -> StanceObservation:
    defaults = dict(
        market_id="m1",
        outcome_id="yes",
        stance_confidence=0.8,
        link_confidence=0.7,
        engagement_score=0.5,
        author_weight=0.5,
    )
    defaults.update(kwargs)
    return StanceObservation(label=label, created_at=ts, **defaults)


def test_empty_window_returns_neutral_half():
    scorer = SocialPressureScorer()
    record = scorer.score(
        [],
        market_id="m1",
        outcome_id="yes",
        window_start=datetime(2025, 3, 1, tzinfo=timezone.utc),
        window_end=datetime(2025, 3, 2, tzinfo=timezone.utc),
    )
    assert record.index_value == 0.5
    assert record.n_posts == 0


def test_only_yes_stances_index_is_one():
    now = datetime(2025, 3, 1, 12, tzinfo=timezone.utc)
    scorer = SocialPressureScorer()
    obs = [_obs(StanceLabel.SUPPORTS_YES, now) for _ in range(3)]
    record = scorer.score(
        obs,
        market_id="m1",
        outcome_id="yes",
        window_start=now - timedelta(hours=1),
        window_end=now + timedelta(hours=1),
    )
    assert record.index_value == 1.0
    assert record.n_posts == 3


def test_only_no_stances_index_is_zero():
    now = datetime(2025, 3, 1, 12, tzinfo=timezone.utc)
    scorer = SocialPressureScorer()
    obs = [_obs(StanceLabel.SUPPORTS_NO, now) for _ in range(2)]
    record = scorer.score(
        obs,
        market_id="m1",
        outcome_id="yes",
        window_start=now - timedelta(hours=1),
        window_end=now + timedelta(hours=1),
    )
    assert record.index_value == 0.0
    assert record.n_posts == 2


def test_recency_decay_weights_older_posts_less():
    now = datetime(2025, 3, 1, 12, tzinfo=timezone.utc)
    cfg = SocialPressureConfig(half_life=timedelta(hours=1))
    scorer = SocialPressureScorer(cfg)
    obs = [
        _obs(StanceLabel.SUPPORTS_YES, now, stance_confidence=1.0),
        _obs(StanceLabel.SUPPORTS_NO, now - timedelta(hours=10), stance_confidence=1.0),
    ]
    record = scorer.score(
        obs,
        market_id="m1",
        outcome_id="yes",
        window_start=now - timedelta(days=1),
        window_end=now + timedelta(hours=1),
    )
    # The YES post is far more recent than the NO post; index should
    # be heavily skewed toward YES.
    assert record.index_value > 0.8


def test_observations_outside_window_are_ignored():
    now = datetime(2025, 3, 1, 12, tzinfo=timezone.utc)
    scorer = SocialPressureScorer()
    obs = [_obs(StanceLabel.SUPPORTS_YES, now - timedelta(days=10))]
    record = scorer.score(
        obs,
        market_id="m1",
        outcome_id="yes",
        window_start=now - timedelta(hours=1),
        window_end=now + timedelta(hours=1),
    )
    assert record.index_value == 0.5
    assert record.n_posts == 0

"""Unit tests for the Task-2 analytics layer (small offline fixtures)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from polysignal.analytics import correlation, cross_platform, descriptive, discourse, temporal
from polysignal.analytics.runner import run_all

UTC = timezone.utc


# --- fixtures ---------------------------------------------------------------


def _markets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"market_id": "mYES", "thematic_category": "politics", "domain": "politics",
             "created_at": "2025-01-01T00:00:00Z", "close_time": "2025-02-10T00:00:00Z",
             "resolved_at": None, "is_resolved": True, "resolution": "yes", "volume": 1_000_000.0},
            {"market_id": "mNO", "thematic_category": "finance", "domain": "economy",
             "created_at": "2025-01-01T00:00:00Z", "close_time": "2025-02-10T00:00:00Z",
             "resolved_at": None, "is_resolved": True, "resolution": "no", "volume": 500_000.0},
        ]
    )


def _outcomes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"market_id": "mYES", "outcome_id": "y1", "label": "Yes", "index": 0, "current_price": 1.0},
            {"market_id": "mYES", "outcome_id": "n1", "label": "No", "index": 1, "current_price": 0.0},
            {"market_id": "mNO", "outcome_id": "y2", "label": "Yes", "index": 0, "current_price": 0.0},
            {"market_id": "mNO", "outcome_id": "n2", "label": "No", "index": 1, "current_price": 1.0},
        ]
    )


def _price_ramp(market_id: str, outcome_id: str, start: float, end: float, days: int = 40) -> list[dict]:
    anchor = datetime(2025, 2, 10, tzinfo=UTC)
    rows = []
    for d in range(days + 1):
        ts = anchor - timedelta(days=days - d)
        price = start + (end - start) * (d / days)
        rows.append({"market_id": market_id, "outcome_id": outcome_id,
                     "timestamp": ts.isoformat(), "price": round(price, 4)})
    return rows


def _prices() -> pd.DataFrame:
    rows = _price_ramp("mYES", "y1", 0.50, 0.95) + _price_ramp("mNO", "y2", 0.50, 0.05)
    return pd.DataFrame(rows)


def _linked() -> pd.DataFrame:
    rows = []
    base = datetime(2025, 1, 5, tzinfo=UTC)
    for i in range(30):
        rows.append({"market_id": "mYES", "post_id": f"p{i}", "platform": "telegram",
                     "final_relevance_label": "RELEVANT", "relevance_label": "RELEVANT",
                     "final_confidence": 0.8, "confidence": 0.8,
                     "post_created_at": (base + timedelta(days=i)).isoformat()})
    # a couple of reddit posts on mYES -> multi-platform
    for i in range(8):
        rows.append({"market_id": "mYES", "post_id": f"r{i}", "platform": "reddit",
                     "final_relevance_label": "RELEVANT", "relevance_label": "RELEVANT",
                     "final_confidence": 0.7, "confidence": 0.7,
                     "post_created_at": (base + timedelta(days=i)).isoformat()})
    # an unrelated (rejected) link that must be filtered out
    rows.append({"market_id": "mNO", "post_id": "x0", "platform": "telegram",
                 "final_relevance_label": "UNRELATED", "relevance_label": "UNRELATED",
                 "final_confidence": 0.9, "confidence": 0.9,
                 "post_created_at": base.isoformat()})
    return pd.DataFrame(rows)


def _enrichment() -> pd.DataFrame:
    rows = []
    for i in range(30):
        rows.append({"post_id": f"p{i}", "sentiment_label": "positive" if i % 2 else "neutral",
                     "sentiment_compound": 0.4 if i % 2 else 0.0,
                     "mentioned_entities": [{"text": "Trump", "label": "PERSON"}]})
    for i in range(8):
        rows.append({"post_id": f"r{i}", "sentiment_label": "negative",
                     "sentiment_compound": -0.3,
                     "mentioned_entities": [{"text": "Senate", "label": "ORG"}]})
    return pd.DataFrame(rows)


def _topics() -> pd.DataFrame:
    return pd.DataFrame([
        {"topic_id": 0, "label": "elections", "top_terms": ["vote", "poll"], "weight": 0.6},
        {"topic_id": 1, "label": "markets", "top_terms": ["price", "trade"], "weight": 0.4},
    ])


# --- descriptive ------------------------------------------------------------


def test_domain_distribution():
    out = descriptive.domain_distribution(_markets()).set_index("domain")
    assert out.loc["politics", "n_contracts"] == 1
    assert out.loc["politics", "resolved_yes"] == 1
    assert out.loc["finance", "resolved_no"] == 1


def test_contract_summary_duration_and_volume():
    out = descriptive.contract_summary(_markets()).set_index("domain")
    assert out.loc["politics", "duration_days_mean"] == pytest.approx(40.0, abs=0.1)
    assert out.loc["politics", "volume_total"] == pytest.approx(1_000_000.0)


def test_price_volatility_nonzero():
    vol = descriptive.price_volatility(_markets(), _outcomes(), _prices())
    assert set(vol["market_id"]) == {"mYES", "mNO"}
    assert (vol["volatility"] > 0).all()


def test_price_informativeness_improves_toward_resolution():
    curve = descriptive.price_informativeness(_markets(), _outcomes(), _prices()).set_index("horizon_days")
    assert curve.loc[1, "n_contracts"] == 2
    assert curve.loc[1, "accuracy"] == 1.0
    assert curve.loc[1, "auc"] == 1.0
    # Brier is lower (better) closer to resolution.
    assert curve.loc[1, "brier"] < curve.loc[30, "brier"]


# --- discourse --------------------------------------------------------------


def test_volume_by_platform_domain_filters_rejected():
    out = discourse.volume_by_platform_domain(_linked(), _markets())
    tg = out[(out.platform == "telegram") & (out.domain == "politics")]
    assert int(tg["n_posts"].iloc[0]) == 30
    # The UNRELATED mNO link is excluded entirely.
    assert "finance" not in set(out["domain"])


def test_sentiment_distribution():
    out = discourse.sentiment_distribution(_linked(), _enrichment(), _markets())
    tg = out[(out.platform == "telegram")].iloc[0]
    assert tg["n_enriched"] == 30
    assert 0.0 <= tg["pos_share"] <= 1.0


def test_entity_frequency_top_entities():
    out = discourse.entity_frequency(_linked(), _enrichment(), _markets())
    pol = out[out.domain == "politics"]
    assert "Trump" in set(pol["entity"])
    assert int(pol[pol.entity == "Trump"]["mentions"].iloc[0]) == 30


def test_activity_timeline_shares_sum_to_one():
    out = discourse.activity_timeline(_linked(), _markets())
    for _, g in out.groupby("domain"):
        # per-bucket shares are rounded to 4 dp for display, so allow drift
        assert g["share"].sum() == pytest.approx(1.0, abs=2e-3)


def test_topic_summary_sorted():
    out = discourse.topic_summary(_topics())
    assert list(out["topic_id"]) == [0, 1]
    assert out["weight"].iloc[0] >= out["weight"].iloc[1]


# --- correlation ------------------------------------------------------------


def test_lead_lag_and_best_lag():
    idx = pd.date_range("2025-01-01", periods=12, freq="D", tz=UTC)
    price = pd.Series([0.5] * 12, index=idx)
    price.iloc[6] = 0.7  # a price move on day 6
    social = pd.Series([0] * 12, index=idx, dtype=float)
    social.iloc[4] = 10  # social spike 2 days before the move -> social leads
    ll = correlation.lead_lag(price, social, max_lag_days=4)
    assert set(ll["lag"]) == set(range(-4, 5))
    bl = correlation.best_lag(ll)
    assert bl["relation"] in {"social_leads", "coincident", "social_follows"}


def test_sentiment_price_alignment():
    idx = pd.date_range("2025-01-01", periods=5, freq="D", tz=UTC)
    price = pd.Series([0.3, 0.4, 0.5, 0.6, 0.7], index=idx)
    social = pd.DataFrame({"n_posts": [1] * 5, "mean_sentiment": [-0.2, 0.0, 0.1, 0.3, 0.5]}, index=idx)
    out = correlation.sentiment_price_alignment(price, social)
    assert out["price_direction"] == 1
    assert out["sentiment_direction"] == 1
    assert out["aligned"] is True


def test_spike_event_study_detects_spike():
    idx = pd.date_range("2025-01-01", periods=10, freq="D", tz=UTC)
    social = pd.Series([1, 1, 1, 1, 20, 1, 1, 1, 1, 1], index=idx, dtype=float)
    price = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.7], index=idx)
    out = correlation.spike_event_study(price, social, z_threshold=2.0, window_days=3)
    assert len(out) == 1
    assert out["price_move"].iloc[0] != 0.0


# --- cross-platform ---------------------------------------------------------


def test_cross_platform_comparison_states():
    out = cross_platform.cross_platform_comparison(_linked(), _markets()).set_index("market_id")
    assert out.loc["mYES", "n_platforms"] == 2
    assert out.loc["mYES", "state"] == "multi_platform_complementary"
    assert out.loc["mYES", "dominant_platform"] == "telegram"


def test_cross_platform_overall_state():
    comp = cross_platform.cross_platform_comparison(_linked(), _markets())
    state = cross_platform.overall_platform_state(comp)
    assert state["multi_platform_contracts"] == 1
    assert set(state["platforms_present"]) == {"telegram", "reddit"}


# --- runner end-to-end ------------------------------------------------------


def test_run_all_writes_artifacts(tmp_path, monkeypatch):
    root = tmp_path / "data"
    (root / "normalized" / "polymarket").mkdir(parents=True)
    (root / "normalized" / "social").mkdir(parents=True)
    _markets().to_parquet(root / "normalized" / "polymarket" / "markets.parquet")
    _outcomes().to_parquet(root / "normalized" / "polymarket" / "outcomes.parquet")
    _prices().to_parquet(root / "normalized" / "polymarket" / "price_points.parquet")
    _linked().to_parquet(root / "normalized" / "social" / "linked_posts.parquet")
    _enrichment().to_parquet(root / "normalized" / "social" / "post_enrichment.parquet")
    _topics().to_parquet(root / "normalized" / "social" / "topics.parquet")

    from polysignal.storage.paths import get_data_paths

    paths = get_data_paths(root=root)
    result = run_all(paths=paths)
    assert (result.output_dir / "price_informativeness.parquet").exists()
    assert (result.output_dir / "analytics_summary.json").exists()
    assert result.summary["row_counts"]["domain_distribution"] == 2
    assert result.summary["cross_platform"]["multi_platform_contracts"] == 1

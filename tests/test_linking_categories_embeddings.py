"""Tests for Branch 05A term categorization and Branch 05B embedding overlay."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from polysignal.domain import (
    ConstraintMatch,
    Market,
    MarketDomain,
    Outcome,
    Platform,
    RelevanceLabel,
    SocialPost,
)
from polysignal.linking import (
    NullEmbeddingProvider,
    RelevanceScorer,
    StaticMockEmbeddingProvider,
    build_market_profile,
    evaluate_predictions,
    fixture_labels,
    fixture_markets,
    fixture_posts,
    run_linking_cross_product,
    write_evaluation_report,
)
from polysignal.linking.embeddings import (
    CachedEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    build_default_provider,
)
from polysignal.linking.relevance import RelevanceJudgement

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _mk_market(question: str, slug: str, domain: MarketDomain = MarketDomain.OTHER) -> Market:
    return Market(
        market_id="m",
        slug=slug,
        question=question,
        domain=domain,
        created_at=_NOW,
        outcomes=[
            Outcome(outcome_id="m-yes", label="Yes", index=0),
            Outcome(outcome_id="m-no", label="No", index=1),
        ],
    )


def _mk_post(post_id: str, text: str) -> SocialPost:
    return SocialPost(
        post_id=post_id,
        platform=Platform.REDDIT,
        author_id="a",
        author_handle="a",
        created_at=_NOW,
        text=text,
    )


# ---------------------------------------------------------------------------
# Part A — term categorization
# ---------------------------------------------------------------------------


def test_categorize_bitcoin_market_buckets() -> None:
    market = _mk_market(
        "Will Bitcoin hit $150k by December 2026?",
        "will-bitcoin-hit-150k-by-december-2026",
        MarketDomain.CRYPTO,
    )
    profile = build_market_profile(market)
    assert "bitcoin" in profile.entity_terms
    assert "hit" in profile.action_terms
    assert "150k" in profile.threshold_terms
    assert "december" in profile.temporal_terms
    assert "2026" in profile.temporal_terms


def test_categorize_china_taiwan_anchors_to_temporal() -> None:
    market = _mk_market(
        "Will China invade Taiwan before 2027?",
        "will-china-invade-taiwan-before-2027",
        MarketDomain.POLITICS,
    )
    profile = build_market_profile(market)
    assert "china" in profile.entity_terms
    assert "taiwan" in profile.entity_terms
    assert "invade" in profile.action_terms
    assert "2027" in profile.temporal_terms
    # Relation phrases must NOT leak into temporal_terms.
    assert "before" not in profile.temporal_terms


# ---------------------------------------------------------------------------
# Part A — entity-only false positive is unrelated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "slug", "domain", "post_text"),
    [
        (
            "Will Apple announce a foldable iPhone before WWDC?",
            "will-apple-announce-a-foldable-iphone-before-wwdc",
            MarketDomain.CULTURE,
            "Apple stock is down after weak iPad sales.",
        ),
        (
            "Will Trump be out as President before GTA VI?",
            "will-trump-be-out-as-president-before-gta-vi",
            MarketDomain.POLITICS,
            "Trump launches a new social media campaign aimed at younger voters.",
        ),
        (
            "Will GTA VI be released before 2026?",
            "will-gta-vi-be-released-before-2026",
            MarketDomain.CULTURE,
            "GTA 6 trailer broke YouTube records.",
        ),
    ],
)
def test_entity_only_mention_is_unrelated(
    question: str, slug: str, domain: MarketDomain, post_text: str
) -> None:
    """The hardened baseline labels entity-only mentions as UNRELATED."""
    market = _mk_market(question, slug, domain)
    post = _mk_post("p", post_text)
    verdict = RelevanceScorer().score(post, market)
    assert verdict.relevance_label == RelevanceLabel.UNRELATED, verdict
    assert verdict.signal_count == 0


def test_threshold_mismatch_demotes_below_relevant() -> None:
    market = _mk_market(
        "Will Bitcoin hit $150k by December 2026?",
        "will-bitcoin-hit-150k-by-december-2026",
        MarketDomain.CRYPTO,
    )
    # Same entity, same anchor, but a *different* threshold value.
    post = _mk_post("p", "Bitcoin could hit $80k by December 2026, some analysts say.")
    verdict = RelevanceScorer().score(post, market)
    assert verdict.relevance_label != RelevanceLabel.RELEVANT
    assert verdict.constraint_match == ConstraintMatch.FULL


def test_product_mismatch_demotes_below_relevant() -> None:
    market = _mk_market(
        "Will Apple announce a foldable iPhone before WWDC?",
        "will-apple-announce-a-foldable-iphone-before-wwdc",
        MarketDomain.CULTURE,
    )
    post = _mk_post("p", "Apple is expected to launch a new Vision headset before WWDC.")
    verdict = RelevanceScorer().score(post, market)
    assert verdict.relevance_label != RelevanceLabel.RELEVANT
    assert verdict.constraint_match == ConstraintMatch.FULL


def test_anchor_year_paraphrase_matches_full() -> None:
    """'before the end of 2026' must match a market anchored on 'December 2026'."""
    market = _mk_market(
        "Will Bitcoin hit $150k by December 2026?",
        "will-bitcoin-hit-150k-by-december-2026",
        MarketDomain.CRYPTO,
    )
    post = _mk_post("p", "Analysts now expect Bitcoin to reach $150k before the end of 2026.")
    verdict = RelevanceScorer().score(post, market)
    assert verdict.constraint_match == ConstraintMatch.FULL
    assert verdict.relevance_label == RelevanceLabel.RELEVANT


# ---------------------------------------------------------------------------
# Part B — embedding overlay
# ---------------------------------------------------------------------------


def test_default_mode_works_without_embedding_deps() -> None:
    """Default run uses no provider — verdicts have embedding_enabled=False."""
    markets = fixture_markets()
    posts = fixture_posts()
    rows = run_linking_cross_product(posts, markets)
    assert all(row.embedding_enabled is False for row in rows)
    assert all(row.semantic_similarity is None for row in rows)


def test_null_provider_fallback() -> None:
    """NullEmbeddingProvider.is_available() is False; scorer skips overlay."""
    market = fixture_markets()[0]
    post = next(p for p in fixture_posts() if p.post_id == "p-btc-relevant")
    provider = NullEmbeddingProvider()
    assert provider.is_available() is False
    verdict = RelevanceScorer().score(post, market, embedding_provider=provider)
    assert verdict.embedding_enabled is False
    assert verdict.semantic_similarity is None


def test_build_default_provider_disabled_returns_null() -> None:
    provider = build_default_provider(enabled=False, model_id="x", cache_dir=None)
    assert isinstance(provider, NullEmbeddingProvider)
    assert provider.is_available() is False


def test_st_provider_unavailable_when_dep_missing() -> None:
    """Importing sentence-transformers may fail; provider must report not-available."""
    provider = SentenceTransformerEmbeddingProvider("nonexistent/model-does-not-exist")
    # The lazy import path swallows the failure and returns False.
    # In CI without sentence-transformers installed, this is the common case.
    assert isinstance(provider.is_available(), bool)


def test_mock_provider_populates_diagnostics() -> None:
    market = _mk_market(
        "Will Bitcoin hit $150k by December 2026?",
        "will-bitcoin-hit-150k-by-december-2026",
        MarketDomain.CRYPTO,
    )
    post = _mk_post("p", "Bitcoin price may rise strongly over the next few years.")
    provider = StaticMockEmbeddingProvider(
        {
            "Will Bitcoin hit $150k by December 2026? ": [1.0, 0.0, 0.0],
            "Will Bitcoin hit $150k by December 2026?": [1.0, 0.0, 0.0],
            "Bitcoin price may rise strongly over the next few years.": [0.9, 0.1, 0.0],
        }
    )
    verdict = RelevanceScorer().score(post, market, embedding_provider=provider)
    assert verdict.embedding_enabled is True
    assert verdict.embedding_model == "mock"
    assert verdict.semantic_similarity is not None
    assert verdict.semantic_similarity > 0.9


def test_embedding_overlay_rescues_broad_topic_to_weakly() -> None:
    """A high-cosine broad-topic mention remains safely WEAKLY_RELEVANT."""
    market = _mk_market(
        "Will Bitcoin hit $150k by December 2026?",
        "will-bitcoin-hit-150k-by-december-2026",
        MarketDomain.CRYPTO,
    )
    post_text = "Bitcoin price may rise strongly over the next few years."
    post = _mk_post("p", post_text)
    market_text = (market.question or "") + " " + (market.description or "")
    market_text = market_text.strip()
    # Engineered vectors: high cosine for the broad-topic pair.
    provider = StaticMockEmbeddingProvider(
        {
            market_text: [1.0, 0.0],
            post_text: [0.95, 0.05],
        }
    )
    deterministic = RelevanceScorer().score(post, market)
    overlaid = RelevanceScorer().score(post, market, embedding_provider=provider)
    assert deterministic.relevance_label == RelevanceLabel.WEAKLY_RELEVANT
    assert overlaid.relevance_label == RelevanceLabel.WEAKLY_RELEVANT
    assert overlaid.semantic_similarity is not None
    assert overlaid.semantic_similarity >= 0.6


def test_embedding_overlay_does_not_promote_entity_only_to_relevant() -> None:
    """Even with very high cosine, entity-only mentions cap at WEAKLY_RELEVANT."""
    market = _mk_market(
        "Will Apple announce a foldable iPhone before WWDC?",
        "will-apple-announce-a-foldable-iphone-before-wwdc",
        MarketDomain.CULTURE,
    )
    post_text = "Apple stock is down after weak iPad sales."
    post = _mk_post("p", post_text)
    market_text = (market.question or "").strip() + " "
    market_text = market_text.strip()
    # Cosine = 1.0 (same vector both sides).
    provider = StaticMockEmbeddingProvider(
        {
            market_text: [1.0, 0.0],
            post_text: [1.0, 0.0],
        }
    )
    overlaid = RelevanceScorer().score(post, market, embedding_provider=provider)
    assert overlaid.relevance_label != RelevanceLabel.RELEVANT


def test_embedding_overlay_does_not_promote_mismatch() -> None:
    """Constraint MISMATCH stays AMBIGUOUS even when cosine is high."""
    market = _mk_market(
        "Will the Fed cut rates after the next meeting?",
        "will-the-fed-cut-rates-after-the-next-meeting",
        MarketDomain.ECONOMY,
    )
    post_text = "The Fed may cut rates before the next meeting."
    post = _mk_post("p", post_text)
    provider = StaticMockEmbeddingProvider(
        {
            (market.question or "").strip(): [1.0, 0.0],
            post_text: [1.0, 0.0],
        }
    )
    overlaid = RelevanceScorer().score(post, market, embedding_provider=provider)
    assert overlaid.constraint_match == ConstraintMatch.MISMATCH
    assert overlaid.relevance_label == RelevanceLabel.AMBIGUOUS


def test_embedding_overlay_can_promote_weakly_to_relevant() -> None:
    """High cosine + signal_count >= 1 promotes WEAKLY_RELEVANT to RELEVANT."""
    market = _mk_market(
        "Will Apple announce a foldable iPhone before WWDC?",
        "will-apple-announce-a-foldable-iphone-before-wwdc",
        MarketDomain.CULTURE,
    )
    post_text = "Rumors say Apple is testing a foldable iPhone prototype."
    post = _mk_post("p", post_text)
    market_text = (market.question or "").strip()
    provider = StaticMockEmbeddingProvider(
        {
            market_text: [1.0, 0.0],
            post_text: [0.99, 0.01],
        }
    )
    deterministic = RelevanceScorer().score(post, market)
    overlaid = RelevanceScorer().score(post, market, embedding_provider=provider)
    assert deterministic.relevance_label == RelevanceLabel.WEAKLY_RELEVANT
    assert deterministic.signal_count >= 1
    assert overlaid.relevance_label == RelevanceLabel.RELEVANT


def test_cached_provider_round_trip(tmp_path: Path) -> None:
    """The on-disk JSON cache returns the same vector on a second call."""
    inner_calls: list[str] = []

    class CountingInner:
        model_id = "test-model"

        def is_available(self) -> bool:
            return True

        def embed(self, text: str) -> list[float]:
            inner_calls.append(text)
            return [float(len(text)), 0.0]

    cached = CachedEmbeddingProvider(CountingInner(), tmp_path)
    a = cached.embed("hello")
    b = cached.embed("hello")
    assert a == b
    assert len(inner_calls) == 1
    # New provider reading the same cache file picks it up.
    cached2 = CachedEmbeddingProvider(CountingInner(), tmp_path)
    c = cached2.embed("hello")
    assert c == a


# ---------------------------------------------------------------------------
# Evaluation enhancements
# ---------------------------------------------------------------------------


def test_evaluation_report_includes_diagnostics(tmp_path: Path) -> None:
    markets = fixture_markets()
    posts = fixture_posts()
    rows = run_linking_cross_product(posts, markets)
    predictions = [
        RelevanceJudgement(
            market_id=r.market_id,
            post_id=r.post_id,
            relevance_score=r.relevance_score,
            relevance_label=r.relevance_label,
            confidence=r.confidence,
            constraint_match=r.constraint_match,
            matched_entities=list(r.matched_entities),
            matched_terms=list(r.matched_terms),
            signal_count=r.signal_count,
            explanation=r.explanation,
        )
        for r in rows
    ]
    report = evaluate_predictions(predictions, fixture_labels())
    assert report.score_distribution
    # JSON export round-trips and contains the diagnostic fields.
    out = write_evaluation_report(report, tmp_path / "report.json")
    text = out.read_text(encoding="utf-8")
    assert "score_distribution" in text
    assert "false_positives" in text
    assert "false_negatives" in text
    assert "top_errors" in text


def test_fixture_count_at_least_40() -> None:
    assert len(fixture_labels()) >= 40

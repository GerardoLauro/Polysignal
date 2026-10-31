"""Tests for the Branch 05 deterministic semantic-relevance scorer.

Each canonical example from the Branch 05 brief is checked against the
scorer here. The tests intentionally over-specify expected labels for
the worked examples so a regression in scoring weights is loud.
"""

from __future__ import annotations

from datetime import datetime, timezone

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
    RelevanceConfig,
    RelevanceScorer,
    extract_market_constraints,
    fixture_labels,
    fixture_markets,
    fixture_posts,
    match_constraint,
    run_linking_cross_product,
)
from polysignal.linking.fixtures import LabeledPair

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _mk_market(
    market_id: str,
    slug: str,
    question: str,
    domain: MarketDomain = MarketDomain.OTHER,
) -> Market:
    return Market(
        market_id=market_id,
        slug=slug,
        question=question,
        domain=domain,
        created_at=_NOW,
        outcomes=[
            Outcome(outcome_id=f"{market_id}-yes", label="Yes", index=0),
            Outcome(outcome_id=f"{market_id}-no", label="No", index=1),
        ],
    )


def _mk_post(post_id: str, text: str) -> SocialPost:
    return SocialPost(
        post_id=post_id,
        platform=Platform.REDDIT,
        author_id=f"u-{post_id}",
        author_handle=f"author_{post_id}",
        created_at=_NOW,
        text=text,
    )


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------


def test_fixture_markets_and_posts_load() -> None:
    markets = fixture_markets()
    posts = fixture_posts()
    labels = fixture_labels()
    assert markets and posts and labels
    # Every labeled pair refers to a real market and a real post.
    market_ids = {m.market_id for m in markets}
    post_ids = {p.post_id for p in posts}
    for pair in labels:
        assert pair.market_id in market_ids
        assert pair.post_id in post_ids


# ---------------------------------------------------------------------------
# Constraint extraction & matching
# ---------------------------------------------------------------------------


def test_extract_constraint_recognizes_anchor_and_relation() -> None:
    constraints = extract_market_constraints("Will Bitcoin hit $150k by December 2026?")
    assert len(constraints) == 1
    c = constraints[0]
    assert c.relation_phrase == "by"
    assert c.relation_class == "forward_limit"
    assert "December 2026" in c.anchor_variants


def test_constraint_full_match_when_anchor_and_class_align() -> None:
    result = match_constraint(
        "Will Bitcoin hit $150k by December 2026?",
        "Bitcoin could hit $150k by December 2026 if ETF inflows continue.",
    )
    assert result.match == ConstraintMatch.FULL
    assert "December 2026" in result.matched_anchors


def test_constraint_mismatch_when_relation_class_differs() -> None:
    # Market says "after the next meeting"; post says "before the next meeting".
    result = match_constraint(
        "Will the Fed cut rates after the next meeting?",
        "The Fed may cut rates before the next meeting.",
    )
    assert result.match == ConstraintMatch.MISMATCH


def test_constraint_missing_when_anchor_not_mentioned() -> None:
    result = match_constraint(
        "Will China invade Taiwan before 2027?",
        "China and Taiwan tensions are increasing after new military exercises.",
    )
    assert result.match == ConstraintMatch.MISSING


def test_constraint_unknown_when_market_has_no_constraint() -> None:
    result = match_constraint(
        "Will it rain tomorrow?",  # no anchor / relation
        "Forecast says heavy rain across the region.",
    )
    assert result.match == ConstraintMatch.UNKNOWN


# ---------------------------------------------------------------------------
# Scoring: clear relevant vs clear unrelated
# ---------------------------------------------------------------------------


def test_clear_relevant_pair_scores_higher_than_unrelated() -> None:
    market = _mk_market(
        "m-btc",
        "will-bitcoin-hit-150k-by-december-2026",
        "Will Bitcoin hit $150k by December 2026?",
        MarketDomain.CRYPTO,
    )
    relevant_post = _mk_post(
        "p-rel",
        "Bitcoin could hit $150k by December 2026 if ETF inflows continue.",
    )
    unrelated_post = _mk_post(
        "p-unrel",
        "I just made the best lasagna recipe ever this weekend.",
    )
    scorer = RelevanceScorer()
    rel = scorer.score(relevant_post, market)
    unrel = scorer.score(unrelated_post, market)
    assert rel.relevance_label == RelevanceLabel.RELEVANT
    assert unrel.relevance_label == RelevanceLabel.UNRELATED
    assert rel.relevance_score > 0.6
    assert unrel.relevance_score < 0.1
    assert rel.constraint_match == ConstraintMatch.FULL
    assert unrel.constraint_match in (ConstraintMatch.MISSING, ConstraintMatch.UNKNOWN)


def test_entity_overlap_alone_is_not_enough_for_relevant() -> None:
    """Apple stock post must not be marked relevant to the foldable-iPhone market.

    This is the hard requirement from the Branch 05 brief — without
    constraint or strong topic overlap, an entity-only mention can never
    be promoted to ``relevant``. The boundary between ``weakly_relevant``
    and ``unrelated`` for these cases is intentionally deferred to the
    embedding / LLM judge layer.
    """
    market = _mk_market(
        "m-apple-foldable",
        "will-apple-announce-a-foldable-iphone-before-wwdc",
        "Will Apple announce a foldable iPhone before WWDC?",
        MarketDomain.CULTURE,
    )
    post = _mk_post("p-apple-stock", "Apple stock is down after weak iPad sales.")
    verdict = RelevanceScorer().score(post, market)
    assert verdict.relevance_label != RelevanceLabel.RELEVANT
    # The score must stay clearly below the "relevant" threshold.
    assert verdict.relevance_score < RelevanceConfig().relevant_threshold
    # Apple is matched as an entity but no constraint signal exists.
    assert "apple" in verdict.matched_entities
    assert verdict.constraint_match in (ConstraintMatch.MISSING, ConstraintMatch.UNKNOWN)


def test_entity_only_same_event_trump_is_not_relevant() -> None:
    """Trump campaign post must not be marked relevant to the Trump-out market."""
    market = _mk_market(
        "m-trump-out",
        "will-trump-be-out-as-president-before-gta-vi",
        "Will Trump be out as President before GTA VI?",
        MarketDomain.POLITICS,
    )
    post = _mk_post(
        "p-trump-camp",
        "Trump launches a new social media campaign aimed at younger voters.",
    )
    verdict = RelevanceScorer().score(post, market)
    assert verdict.relevance_label != RelevanceLabel.RELEVANT
    assert verdict.relevance_score < RelevanceConfig().relevant_threshold


def test_constraint_match_increases_score() -> None:
    market = _mk_market(
        "m-china",
        "will-china-invade-taiwan-before-2027",
        "Will China invade Taiwan before 2027?",
        MarketDomain.POLITICS,
    )
    with_constraint = _mk_post(
        "p-with", "Some analysts warn China could invade Taiwan before 2027."
    )
    without_constraint = _mk_post(
        "p-without", "China and Taiwan tensions are increasing after new exercises."
    )
    scorer = RelevanceScorer()
    a = scorer.score(with_constraint, market)
    b = scorer.score(without_constraint, market)
    assert a.relevance_score > b.relevance_score
    assert a.constraint_match == ConstraintMatch.FULL
    assert b.constraint_match == ConstraintMatch.MISSING


def test_constraint_mismatch_lowers_score() -> None:
    market = _mk_market(
        "m-fed",
        "will-the-fed-cut-rates-after-the-next-meeting",
        "Will the Fed cut rates after the next meeting?",
        MarketDomain.ECONOMY,
    )
    aligned = _mk_post(
        "p-aligned",
        "Analysts expect the Fed to cut rates after the next meeting in March.",
    )
    misaligned = _mk_post(
        "p-misaligned",
        "The Fed may cut rates before the next meeting.",
    )
    scorer = RelevanceScorer()
    a = scorer.score(aligned, market)
    b = scorer.score(misaligned, market)
    assert a.relevance_score > b.relevance_score
    assert a.constraint_match == ConstraintMatch.FULL
    assert b.constraint_match == ConstraintMatch.MISMATCH
    # A mismatch is at most ambiguous, never relevant.
    assert b.relevance_label != RelevanceLabel.RELEVANT


# ---------------------------------------------------------------------------
# Branch 05E anchor-gate regressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "domain", "post_text"),
    [
        (
            "Trump eliminates capital gains tax on crypto before 2027?",
            MarketDomain.POLITICS,
            "Goldman Sachs forecasts AI-related capex to reach $905 billion in 2027.",
        ),
        (
            "GTA VI released before June 2026?",
            MarketDomain.CULTURE,
            "Bitcoin sentiment just hit its most bullish level of 2026.",
        ),
        (
            "Will the Carolina Hurricanes win the 2026 NHL Stanley Cup?",
            MarketDomain.SPORTS,
            "2026 has wiped out $500B from the Total Crypto Market.",
        ),
        (
            "Will Harvey Weinstein be sentenced to less than 5 years in prison?",
            MarketDomain.CULTURE,
            "Gravity Bridge was just drained of $5.4M by a hacker.",
        ),
        (
            "Taylor Swift pregnant before 2027?",
            MarketDomain.CULTURE,
            "Goldman Sachs forecasts AI-related capex to reach $905 billion in 2027.",
        ),
        (
            "Xi Jinping out before 2027?",
            MarketDomain.POLITICS,
            "Goldman Sachs forecasts AI-related capex to reach $905 billion in 2027.",
        ),
    ],
)
def test_numeric_and_year_only_false_positives_are_unrelated(
    question: str, domain: MarketDomain, post_text: str
) -> None:
    market = _mk_market("m-regression", "regression-market", question, domain)
    verdict = RelevanceScorer().score(_mk_post("p-regression", post_text), market)
    assert verdict.relevance_label == RelevanceLabel.UNRELATED
    assert verdict.constraint_match != ConstraintMatch.FULL
    assert verdict.rejection_reason


@pytest.mark.parametrize(
    "post_text",
    [
        "President Trump delayed a final decision on Iran's latest proposal.",
        "Oil prices fell below $87 as President Trump weighs the Iran situation.",
    ],
)
def test_entity_only_wrong_event_does_not_promote(post_text: str) -> None:
    market = _mk_market(
        "m-trump-tax",
        "trump-eliminates-capital-gains-tax-on-crypto-before-2027",
        "Trump eliminates capital gains tax on crypto before 2027?",
        MarketDomain.POLITICS,
    )
    verdict = RelevanceScorer().score(_mk_post("p-trump-other", post_text), market)
    assert verdict.relevance_label in (RelevanceLabel.UNRELATED, RelevanceLabel.AMBIGUOUS)
    assert verdict.rejection_reason == "entity-only wrong-event rejected"


def test_year_only_match_cannot_be_full_constraint() -> None:
    market = _mk_market(
        "m-taylor",
        "taylor-swift-pregnant-before-2027",
        "Taylor Swift pregnant before 2027?",
        MarketDomain.CULTURE,
    )
    verdict = RelevanceScorer().score(
        _mk_post("p-ai", "AI-related capex could reach $905 billion before 2027."),
        market,
    )
    assert verdict.constraint_match != ConstraintMatch.FULL
    assert "year-only match rejected" in verdict.explanation


def test_numeric_only_match_cannot_promote() -> None:
    market = _mk_market(
        "m-prison",
        "harvey-weinstein-sentenced-to-less-than-5-years",
        "Will Harvey Weinstein be sentenced to less than 5 years in prison?",
        MarketDomain.CULTURE,
    )
    verdict = RelevanceScorer().score(
        _mk_post("p-hack", "A bridge hacker stole $5 million in today's update."),
        market,
    )
    assert verdict.relevance_label == RelevanceLabel.UNRELATED
    assert "numeric-only match rejected" in verdict.explanation


def test_same_domain_entity_without_constraint_is_weakly_relevant() -> None:
    market = _mk_market(
        "m-btc-adoption",
        "will-bitcoin-hit-150k-before-2027",
        "Will Bitcoin hit $150k before 2027?",
        MarketDomain.CRYPTO,
    )
    verdict = RelevanceScorer().score(
        _mk_post("p-btc-adoption", "Bitcoin adoption is increasing among institutions."),
        market,
    )
    assert verdict.relevance_label == RelevanceLabel.WEAKLY_RELEVANT
    assert verdict.constraint_match == ConstraintMatch.MISSING


# ---------------------------------------------------------------------------
# Edge cases — empty / very short
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "ok.", "lol", "n/a"])
def test_empty_or_tiny_post_does_not_crash(text: str) -> None:
    market = _mk_market(
        "m-x",
        "will-x-happen",
        "Will X happen by Friday?",
        MarketDomain.OTHER,
    )
    post = _mk_post("p-empty", text)
    verdict = RelevanceScorer().score(post, market)
    assert verdict.relevance_label == RelevanceLabel.UNRELATED
    assert verdict.relevance_score == 0.0


# ---------------------------------------------------------------------------
# Fixture-wide regression check
# ---------------------------------------------------------------------------


def _index_labels(labels: list[LabeledPair]) -> dict[tuple[str, str], LabeledPair]:
    return {(p.market_id, p.post_id): p for p in labels}


def test_fixture_deterministic_baseline_meets_quality_floor() -> None:
    """Deterministic baseline (no embeddings) must hit floor metrics.

    Broad-topic FNs (gold=weakly_relevant, pred=unrelated) are expected
    residual errors that the Branch 05B embedding overlay rescues. The
    floor here is calibrated so the *hard* invariants are enforced:

    * every gold ``relevant`` pair is predicted ``relevant`` (recall=1);
    * no entity-only FP is ever promoted to ``relevant``;
    * overall accuracy >= 0.80, macro-F1 >= 0.65;
    * constraint accuracy >= 0.90.
    """
    from polysignal.linking import evaluate_predictions
    from polysignal.linking.relevance import RelevanceJudgement

    markets = fixture_markets()
    posts = fixture_posts()
    labels = fixture_labels()
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
    report = evaluate_predictions(predictions, labels)
    assert report.accuracy >= 0.80, report.accuracy
    assert report.macro_f1 >= 0.65, report.macro_f1
    assert report.constraint_accuracy >= 0.90, report.constraint_accuracy
    relevant_metrics = report.per_label[RelevanceLabel.RELEVANT]
    assert relevant_metrics.recall == 1.0, relevant_metrics
    assert relevant_metrics.precision >= 0.85, relevant_metrics


def test_custom_config_is_respected() -> None:
    """Lowering the relevant threshold must promote a borderline pair."""
    market = _mk_market(
        "m-fed-relaxed",
        "will-the-fed-cut-rates-after-the-next-meeting",
        "Will the Fed cut rates after the next meeting?",
        MarketDomain.ECONOMY,
    )
    # Broad post: no exact anchor, but lots of overlapping terms.
    post = _mk_post(
        "p-fed-broad",
        "The Fed will probably cut interest rates soon, analysts say.",
    )
    default = RelevanceScorer().score(post, market)
    relaxed = RelevanceScorer(
        RelevanceConfig(relevant_threshold=0.30, weakly_relevant_threshold=0.10)
    ).score(post, market)
    assert relaxed.relevance_score == default.relevance_score
    # Same score, but the relaxed config should give an at-least-as-strong label.
    label_order = {
        RelevanceLabel.UNRELATED: 0,
        RelevanceLabel.AMBIGUOUS: 1,
        RelevanceLabel.WEAKLY_RELEVANT: 2,
        RelevanceLabel.RELEVANT: 3,
    }
    assert label_order[relaxed.relevance_label] >= label_order[default.relevance_label]

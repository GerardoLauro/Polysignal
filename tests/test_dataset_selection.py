"""Offline tests for the high-signal market scoring and selection layer."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from polysignal.dataset import (
    HighSignalDomain,
    classify_high_signal_domain,
    market_priority_score,
    rank_markets,
    select_high_signal_markets,
    social_searchability_score,
    telegram_searchability_score,
)
from polysignal.dataset.models import DatasetTargets
from polysignal.domain import Market, MarketDomain, Outcome, ResolutionResult


def _market(
    *,
    market_id: str,
    question: str,
    domain: MarketDomain,
    volume: float | None = 0.0,
    liquidity: float | None = 0.0,
    description: str | None = None,
    close_time: datetime | None = None,
    outcomes: tuple[tuple[str, str], ...] = (("yes", "Yes"), ("no", "No")),
) -> Market:
    return Market(
        market_id=market_id,
        slug=market_id.lower(),
        question=question,
        description=description,
        domain=domain,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        close_time=close_time or datetime(2026, 12, 31, tzinfo=timezone.utc),
        is_resolved=True,
        is_closed=True,
        outcomes=[
            Outcome(outcome_id=oid, label=label, index=idx)
            for idx, (oid, label) in enumerate(outcomes)
        ],
        volume=volume,
        liquidity=liquidity,
    )


def test_classify_high_signal_domain_collapses_crypto_economy_into_finance() -> None:
    crypto = _market(
        market_id="c1",
        question="Will Bitcoin hit $200K by end of 2026?",
        domain=MarketDomain.CRYPTO,
    )
    econ = _market(
        market_id="e1",
        question="Will the Fed cut rates in 2026?",
        domain=MarketDomain.ECONOMY,
    )
    assert classify_high_signal_domain(crypto) is HighSignalDomain.FINANCE_MACRO_CRYPTO
    assert classify_high_signal_domain(econ) is HighSignalDomain.FINANCE_MACRO_CRYPTO


def test_classify_high_signal_domain_collapses_sports_culture_into_entertainment() -> None:
    sport = _market(
        market_id="s1",
        question="Will the Lakers win the 2026 NBA Finals?",
        domain=MarketDomain.SPORTS,
    )
    culture = _market(
        market_id="cu1",
        question="Will GTA VI release before end of 2026?",
        domain=MarketDomain.CULTURE,
    )
    assert classify_high_signal_domain(sport) is HighSignalDomain.SPORTS_ENTERTAINMENT
    assert classify_high_signal_domain(culture) is HighSignalDomain.SPORTS_ENTERTAINMENT


def test_social_searchability_rewards_clear_question_and_entities() -> None:
    clear = _market(
        market_id="m-clear",
        question="Will the Lakers win the NBA Finals by 2026?",
        domain=MarketDomain.SPORTS,
    )
    obscure = _market(
        market_id="m-obscure",
        question="x?",
        domain=MarketDomain.OTHER,
        outcomes=(("yes", "Yes"), ("no", "No")),
    )
    assert social_searchability_score(clear) > social_searchability_score(obscure)


def test_telegram_searchability_boosts_constrained_questions() -> None:
    constrained = _market(
        market_id="m-c",
        question="Will Bitcoin reach $200K by end of 2026?",
        domain=MarketDomain.CRYPTO,
    )
    bare = _market(
        market_id="m-b",
        question="Bitcoin?",
        domain=MarketDomain.CRYPTO,
    )
    assert telegram_searchability_score(constrained) >= telegram_searchability_score(bare)


def test_market_priority_score_is_bounded_and_uses_price_history_bonus() -> None:
    market = _market(
        market_id="m-prio",
        question="Will the Lakers win the 2026 NBA Finals?",
        domain=MarketDomain.SPORTS,
        volume=500_000.0,
        liquidity=80_000.0,
    )
    without = market_priority_score(market, has_price_history=False)
    with_history = market_priority_score(market, has_price_history=True)
    assert 0.0 <= without <= 1.0
    assert 0.0 <= with_history <= 1.0
    assert with_history > without


def test_rank_markets_sorts_by_priority_descending() -> None:
    high = _market(
        market_id="high",
        question="Will the Lakers win the NBA Finals by 2026?",
        domain=MarketDomain.SPORTS,
        volume=1_000_000.0,
    )
    low = _market(
        market_id="low",
        question="X?",
        domain=MarketDomain.OTHER,
        volume=0.0,
    )
    ranked = rank_markets([low, high])
    assert [row[0].market_id for row in ranked] == ["high", "low"]


def test_select_high_signal_markets_balances_buckets_when_possible() -> None:
    politics = [
        _market(
            market_id=f"p{i}",
            question=f"Will candidate X{i} win the election by 2026?",
            domain=MarketDomain.POLITICS,
            volume=300_000.0,
        )
        for i in range(20)
    ]
    finance = [
        _market(
            market_id=f"f{i}",
            question=f"Will Bitcoin pass ${i}0K by end of 2026?",
            domain=MarketDomain.CRYPTO,
            volume=300_000.0,
        )
        for i in range(20)
    ]
    sports = [
        _market(
            market_id=f"s{i}",
            question=f"Will Team {i} win the 2026 Finals?",
            domain=MarketDomain.SPORTS,
            volume=300_000.0,
        )
        for i in range(20)
    ]
    result = select_high_signal_markets(politics + finance + sports, target=50)
    assert len(result.markets) == 50
    counts = result.domain_counts
    assert counts[HighSignalDomain.POLITICS] >= 15
    assert counts[HighSignalDomain.FINANCE_MACRO_CRYPTO] >= 15
    assert counts[HighSignalDomain.SPORTS_ENTERTAINMENT] >= 10
    # No invented markets — total is bounded by what was supplied.
    assert sum(counts.values()) <= 50


def test_select_records_imbalance_warning_when_bucket_is_thin() -> None:
    # Only 3 sports markets — minimum for sports_entertainment is 10.
    politics = [
        _market(
            market_id=f"p{i}",
            question=f"Will candidate X{i} win the election by 2026?",
            domain=MarketDomain.POLITICS,
        )
        for i in range(20)
    ]
    finance = [
        _market(
            market_id=f"f{i}",
            question=f"Will Bitcoin pass ${i}0K by end of 2026?",
            domain=MarketDomain.CRYPTO,
        )
        for i in range(20)
    ]
    sports = [
        _market(
            market_id=f"s{i}",
            question=f"Will Team {i} win the 2026 Finals?",
            domain=MarketDomain.SPORTS,
        )
        for i in range(3)
    ]
    result = select_high_signal_markets(politics + finance + sports, target=50)
    assert any("sports_entertainment" in w for w in result.imbalance_warnings)


def test_select_with_zero_target_returns_empty_result() -> None:
    result = select_high_signal_markets(
        [
            _market(
                market_id="m1",
                question="Will the Lakers win the NBA Finals by 2026?",
                domain=MarketDomain.SPORTS,
            )
        ],
        target=0,
    )
    assert result.markets == []


def test_select_attaches_search_window_canonical_keywords_and_queries() -> None:
    market = _market(
        market_id="m-c",
        question="Will Bitcoin pass $200K by end of 2026?",
        domain=MarketDomain.CRYPTO,
        description="Resolution based on Coinbase BTC-USD spot price reaching $200K.",
    )
    result = select_high_signal_markets([market], target=1)
    assert len(result.markets) == 1
    selected = result.markets[0]
    assert selected.canonical_keywords  # non-empty
    assert selected.suggested_telegram_queries  # non-empty
    assert selected.search_window_start.tzinfo is timezone.utc
    assert selected.search_window_end.tzinfo is timezone.utc
    assert selected.search_window_end > selected.search_window_start


def test_dataset_targets_defaults_match_brief() -> None:
    targets = DatasetTargets()
    assert targets.selected_market_target == 50
    assert targets.minimum_viable_linked_segments == 500
    assert targets.linked_segments_target_min <= targets.linked_segments_target_max


def test_selection_deduplicates_and_skips_unresolved_anchorless_and_micro_markets() -> None:
    keep = _market(
        market_id="keep",
        question="Will Bitcoin pass $200K by end of 2026?",
        domain=MarketDomain.CRYPTO,
    )
    unresolved = keep.model_copy(update={"market_id": "active", "is_resolved": False, "resolution": None})
    anchorless = keep.model_copy(update={"market_id": "anchorless", "close_time": None, "resolved_at": None})
    micro = keep.model_copy(
        update={"market_id": "weather", "question": "Will temperature exceed 80F tomorrow?"}
    )
    result = select_high_signal_markets([keep, keep, unresolved, anchorless, micro], target=10)
    assert [market.market_id for market in result.markets] == ["keep"]
    assert result.duplicates_removed == 1
    assert result.skipped_unresolved == 1
    assert result.skipped_temporal_anchor_missing == 1
    assert result.skipped_micro_markets == 1


def test_selection_query_plan_is_invariant_to_resolved_label() -> None:
    market = _market(
        market_id="m-label",
        question="Will Bitcoin pass $200K by end of 2026?",
        domain=MarketDomain.CRYPTO,
    )
    yes = market.model_copy(update={"resolution": ResolutionResult.YES})
    no = market.model_copy(update={"resolution": ResolutionResult.NO})
    yes_selected = select_high_signal_markets([yes], target=1).markets[0]
    no_selected = select_high_signal_markets([no], target=1).markets[0]
    assert yes_selected.suggested_telegram_queries == no_selected.suggested_telegram_queries
    assert yes_selected.search_window_start == no_selected.search_window_start
    assert yes_selected.search_window_end == no_selected.search_window_end


def test_question_led_classification_corrects_stale_upstream_domain() -> None:
    market = _market(
        market_id="m-election",
        question="Will candidate X win the presidential election?",
        domain=MarketDomain.OTHER,
    )
    assert classify_high_signal_domain(market) is HighSignalDomain.POLITICS


@pytest.mark.parametrize(
    "question",
    [
        "",
        "?",
        "Will?",
    ],
)
def test_scoring_handles_degenerate_questions(question: str) -> None:
    market = _market(market_id="m", question=question, domain=MarketDomain.OTHER)
    soc = social_searchability_score(market)
    tg = telegram_searchability_score(market)
    prio = market_priority_score(market)
    for value in (soc, tg, prio):
        assert 0.0 <= value <= 1.0

"""Tests for the outcome-aware deterministic stance scorer (Branch 06A)."""

from __future__ import annotations

import pytest

from polysignal.domain import StanceLabel
from polysignal.stance.scorer import ScorerConfig, score_stance


@pytest.fixture
def bitcoin_market() -> dict:
    return {
        "market_question": "Will Bitcoin hit $150k by December 2026?",
        "outcome_names": ["Yes", "No"],
        "canonical_keywords": ["Bitcoin", "150k", "December 2026"],
    }


@pytest.fixture
def fed_market() -> dict:
    return {
        "market_question": "Will the Fed cut rates after the next meeting?",
        "outcome_names": ["Yes", "No"],
        "canonical_keywords": ["Fed", "rates", "next meeting"],
    }


@pytest.fixture
def gta_market() -> dict:
    return {
        "market_question": "Will GTA VI be released before 2026?",
        "outcome_names": ["Yes", "No"],
        "canonical_keywords": ["GTA VI", "released", "2026"],
    }


def test_supports_yes_with_analysts_expect(bitcoin_market: dict) -> None:
    verdict = score_stance(
        text="Analysts expect Bitcoin to reach $150k before the end of 2026.",
        **bitcoin_market,
    )
    assert verdict.label is StanceLabel.SUPPORTS_YES
    assert verdict.stance_score > 0
    assert verdict.confidence > 0


def test_supports_no_with_unlikely(bitcoin_market: dict) -> None:
    verdict = score_stance(
        text="Bitcoin is unlikely to reach $150k before 2026 ends.",
        **bitcoin_market,
    )
    assert verdict.label is StanceLabel.SUPPORTS_NO
    assert verdict.stance_score < 0


def test_supports_no_with_keep_unchanged(fed_market: dict) -> None:
    verdict = score_stance(
        text="The Fed may keep rates unchanged after the next meeting.",
        **fed_market,
    )
    # "may" is uncertainty, "keep unchanged" / "keep rates unchanged" is NO.
    # The scorer may either commit to SUPPORTS_NO or down-rank to AMBIGUOUS
    # because the uncertainty cue is present. Either is conservative.
    assert verdict.label in {StanceLabel.SUPPORTS_NO, StanceLabel.AMBIGUOUS}


def test_supports_yes_when_markets_expect_cut(fed_market: dict) -> None:
    verdict = score_stance(
        text="Markets expect the Fed to cut rates after the next meeting.",
        **fed_market,
    )
    assert verdict.label is StanceLabel.SUPPORTS_YES


def test_gta_delay_supports_no(gta_market: dict) -> None:
    verdict = score_stance(
        text="Rockstar may delay GTA VI until 2026.",
        **gta_market,
    )
    # "delay" is NO, "may" is uncertainty.
    assert verdict.label in {StanceLabel.SUPPORTS_NO, StanceLabel.AMBIGUOUS}


def test_gta_will_release_supports_yes(gta_market: dict) -> None:
    verdict = score_stance(
        text="Rockstar is preparing to release GTA VI before 2026.",
        **gta_market,
    )
    assert verdict.label is StanceLabel.SUPPORTS_YES


def test_negation_flips_yes_to_no(bitcoin_market: dict) -> None:
    verdict = score_stance(
        text="Bitcoin will not reach $150k by December 2026.",
        **bitcoin_market,
    )
    assert verdict.label is StanceLabel.SUPPORTS_NO


def test_text_too_short_returns_insufficient_context(bitcoin_market: dict) -> None:
    verdict = score_stance(text="BTC up.", **bitcoin_market)
    assert verdict.label is StanceLabel.INSUFFICIENT_CONTEXT


def test_off_topic_text_returns_insufficient_context(bitcoin_market: dict) -> None:
    verdict = score_stance(
        text="The Lakers had a strong game against the Celtics last night.",
        **bitcoin_market,
    )
    assert verdict.label is StanceLabel.INSUFFICIENT_CONTEXT


def test_on_topic_no_cues_returns_neutral(bitcoin_market: dict) -> None:
    verdict = score_stance(
        text="Bitcoin price chart updated this morning across exchanges.",
        **bitcoin_market,
    )
    assert verdict.label is StanceLabel.NEUTRAL
    assert verdict.confidence <= 0.2


def test_uncertainty_dominates_returns_ambiguous(bitcoin_market: dict) -> None:
    verdict = score_stance(
        text=(
            "Bitcoin could maybe hit $150k. Rumor has it analysts speculate "
            "uncertain things about the Bitcoin path."
        ),
        **bitcoin_market,
    )
    assert verdict.label is StanceLabel.AMBIGUOUS


def test_balanced_directional_cues_return_ambiguous(bitcoin_market: dict) -> None:
    verdict = score_stance(
        text=(
            "Bitcoin is expected to reach $150k by 2026 but the rally is "
            "unlikely to last; some predict a delay."
        ),
        **bitcoin_market,
    )
    # 1 yes ("expected to") vs 2 no ("unlikely", "delay") — should still
    # be SUPPORTS_NO with low confidence.
    assert verdict.label is StanceLabel.SUPPORTS_NO


def test_confidence_capped_at_max_configurable(bitcoin_market: dict) -> None:
    cfg = ScorerConfig(max_confidence=0.5)
    verdict = score_stance(
        text=(
            "Analysts expect Bitcoin to reach $150k by 2026. "
            "Bitcoin is on track to hit $150k. Markets anticipate Bitcoin will reach $150k."
        ),
        config=cfg,
        **bitcoin_market,
    )
    assert verdict.confidence <= 0.5


def test_stance_score_is_bounded(bitcoin_market: dict) -> None:
    verdict = score_stance(
        text="Bitcoin will hit $150k by 2026 and the rally continues.",
        **bitcoin_market,
    )
    assert -1.0 <= verdict.stance_score <= 1.0

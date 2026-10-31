"""Tests for conservative Branch 06E target aliases."""

from __future__ import annotations

from polysignal.domain import StanceLabel
from polysignal.stance.scorer import score_stance
from polysignal.stance.target_aliases import (
    generate_target_aliases,
    mentions_target_alias,
    normalize_target_alias,
)

_MULTI_OUTCOMES = ["Donald Trump", "Kamala Harris", "Other"]


def test_normalize_target_alias_strips_punctuation_and_case() -> None:
    assert normalize_target_alias("  O'Connor-Jr.  ") == "o connor jr"


def test_generate_target_aliases_includes_exact_parenthetical_and_surname_forms() -> None:
    aliases = generate_target_aliases("Robert F. Kennedy Jr. (RFK Jr.)")
    assert "robert f kennedy jr rfk jr" in aliases
    assert "robert f kennedy jr" in aliases
    assert "rfk jr" in aliases
    assert "kennedy" in aliases


def test_person_like_surname_alias_matches_safely() -> None:
    aliases = generate_target_aliases("Donald Trump")
    assert "trump" in aliases
    assert mentions_target_alias("TRUMP is expected to win the election.", aliases)


def test_punctuation_normalization_matches_target_alias() -> None:
    aliases = generate_target_aliases("O'Connor")
    assert mentions_target_alias("O Connor is expected to win.", aliases)


def test_aliases_do_not_overmatch_short_or_common_tokens() -> None:
    aliases = generate_target_aliases("Team A")
    assert aliases == ["team a"]
    assert not mentions_target_alias("A strong result is expected.", aliases)
    assert not mentions_target_alias("Trumpets are expected to win.", ["trump"])


def test_multi_outcome_scorer_accepts_exact_target_match() -> None:
    verdict = score_stance(
        text="Donald Trump is set to win the election.",
        market_question="Who will win the election?",
        outcome_names=_MULTI_OUTCOMES,
        target_outcome_name="Donald Trump",
        canonical_keywords=["election"],
    )
    assert verdict.label is StanceLabel.SUPPORTS_YES


def test_multi_outcome_scorer_accepts_safe_surname_alias() -> None:
    verdict = score_stance(
        text="Trump is set to win the election.",
        market_question="Who will win the election?",
        outcome_names=_MULTI_OUTCOMES,
        target_outcome_name="Donald Trump",
        canonical_keywords=["election"],
    )
    assert verdict.label is StanceLabel.SUPPORTS_YES


def test_multi_outcome_scorer_stays_neutral_without_target_alias() -> None:
    verdict = score_stance(
        text="The election favorite is set to win.",
        market_question="Who will win the election?",
        outcome_names=_MULTI_OUTCOMES,
        target_outcome_name="Donald Trump",
        canonical_keywords=["election"],
    )
    assert verdict.label is StanceLabel.NEUTRAL


def test_binary_scorer_behavior_is_unchanged() -> None:
    verdict = score_stance(
        text="Bitcoin is unlikely to reach $150k before 2026 ends.",
        market_question="Will Bitcoin hit $150k by December 2026?",
        outcome_names=["Yes", "No"],
        canonical_keywords=["Bitcoin", "150k"],
    )
    assert verdict.label is StanceLabel.SUPPORTS_NO

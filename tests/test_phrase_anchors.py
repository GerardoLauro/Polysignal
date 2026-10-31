"""Tests for the Branch 05G phrase-anchor profile and matching."""

from __future__ import annotations

from datetime import datetime, timezone

from polysignal.domain import Market, MarketDomain, Outcome
from polysignal.linking.phrase_anchors import build_phrase_profile, phrase_matches

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _market(market_id: str, slug: str, question: str) -> Market:
    return Market(
        market_id=market_id,
        slug=slug,
        question=question,
        domain=MarketDomain.OTHER,
        created_at=_NOW,
        outcomes=[
            Outcome(outcome_id=f"{market_id}-yes", label="Yes", index=0),
            Outcome(outcome_id=f"{market_id}-no", label="No", index=1),
        ],
    )


def test_known_phrase_extracted_from_question() -> None:
    m = _market(
        "m1",
        "will-trump-eliminate-capital-gains-tax-on-crypto-before-2027",
        "Will Trump eliminate capital gains tax on crypto before 2027?",
    )
    profile = build_phrase_profile(m)
    assert "capital gains tax" in profile.canonical_phrases
    assert profile.has_phrase


def test_stanley_cup_requires_full_phrase() -> None:
    m = _market(
        "m2",
        "will-the-edmonton-oilers-win-the-stanley-cup",
        "Will the Edmonton Oilers win the Stanley Cup?",
    )
    profile = build_phrase_profile(m)
    assert "stanley cup" in profile.canonical_phrases
    assert phrase_matches(profile, "The Stanley Cup finals start tonight.") == ["stanley cup"]
    # Stanley alone is not enough.
    assert phrase_matches(profile, "Stanley Tucci stars in the new film.") == []


def test_kraken_ipo_requires_kraken_and_ipo_together() -> None:
    m = _market(
        "m3", "will-kraken-ipo-by-december-31-2026", "Will Kraken IPO by December 31, 2026?"
    )
    profile = build_phrase_profile(m)
    assert "kraken ipo" in profile.canonical_phrases
    assert phrase_matches(profile, "Kraken IPO is expected later this year.") == ["kraken ipo"]
    # IPO alone (SpaceX IPO news) is not enough.
    assert phrase_matches(profile, "SpaceX has officially filed for an IPO.") == []


def test_gta_vi_alias_matches_gta_6() -> None:
    m = _market("m4", "will-gta-vi-be-released-before-2026", "Will GTA VI be released before 2026?")
    profile = build_phrase_profile(m)
    assert "gta vi" in profile.canonical_phrases
    # Both surface forms match the canonical phrase.
    assert phrase_matches(profile, "GTA VI release date leaked!") == ["gta vi"]
    assert phrase_matches(profile, "GTA 6 trailer broke YouTube records.") == ["gta vi"]


def test_federal_reserve_alias_matches_fed() -> None:
    m = _market(
        "m5",
        "will-federal-reserve-cut-rates",
        "Will the Federal Reserve cut rates after the next meeting?",
    )
    profile = build_phrase_profile(m)
    assert "federal reserve" in profile.canonical_phrases
    assert phrase_matches(profile, "The Fed signaled a cut.") == ["federal reserve"]


def test_year_only_post_yields_no_phrase_match() -> None:
    m = _market(
        "m6", "will-china-invade-taiwan-before-2027", "Will China invade Taiwan before 2027?"
    )
    profile = build_phrase_profile(m)
    # Year-only post: no phrase hit, even if 2027 is mentioned.
    assert phrase_matches(profile, "Big things expected in 2027.") == []


def test_market_without_multi_word_phrases_has_no_hard_phrases() -> None:
    m = _market("m7", "will-x-do-y", "Will X do Y?")
    profile = build_phrase_profile(m)
    assert not profile.has_phrase

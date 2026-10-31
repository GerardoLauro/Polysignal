"""Offline fixtures for Branch 05 demonstration and evaluation.

Fixture categories (Branch 05A hardening)
-----------------------------------------
The labeled set targets >= 40 pairs covering:

* exact contract-specific match (gold ``relevant``);
* broad topic relevance (gold ``weakly_relevant``);
* entity-only false positives (gold ``unrelated``);
* same entity but wrong event (gold ``unrelated``);
* same event but wrong time (gold ``unrelated`` or ``ambiguous``);
* same topic missing threshold (gold ``weakly_relevant``);
* same product wrong action (gold ``weakly_relevant`` / ``ambiguous``);
* macro-related but not contract-specific (gold ``weakly_relevant``);
* short / low-information posts (gold ``unrelated``);
* noisy posts (gold ``unrelated``);
* relevant posts with opposite implication (gold ``relevant``);
* constraint mismatch (gold ``ambiguous``).

Some categories — notably broad topic relevance — are the documented
known-weakness of the deterministic baseline. The fixture labels reflect
the *ideal* gold label; the deterministic-only evaluation will show
those rows as residual errors. The embedding overlay (Branch 05B) is
expected to recover them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from polysignal.domain import (
    ConstraintMatch,
    Market,
    MarketDomain,
    Outcome,
    Platform,
    RelevanceLabel,
    SocialPost,
)


@dataclass(frozen=True)
class LabeledPair:
    market_id: str
    post_id: str
    expected_label: RelevanceLabel
    expected_constraint_match: ConstraintMatch
    reason: str


_FIXED_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _market(
    market_id: str,
    slug: str,
    question: str,
    domain: MarketDomain,
    *,
    description: str | None = None,
    tags: list[str] | None = None,
) -> Market:
    return Market(
        market_id=market_id,
        slug=slug,
        question=question,
        description=description,
        domain=domain,
        created_at=_FIXED_TIME,
        outcomes=[
            Outcome(outcome_id=f"{market_id}-yes", label="Yes", index=0),
            Outcome(outcome_id=f"{market_id}-no", label="No", index=1),
        ],
        tags=tags or [],
    )


def _post(post_id: str, text: str, *, platform: Platform = Platform.REDDIT) -> SocialPost:
    return SocialPost(
        post_id=post_id,
        platform=platform,
        author_id=f"u-{post_id}",
        author_handle=f"author_{post_id}",
        created_at=_FIXED_TIME,
        text=text,
    )


def fixture_markets() -> list[Market]:
    return [
        _market(
            "m-btc-150k",
            "will-bitcoin-hit-150k-by-december-2026",
            "Will Bitcoin hit $150k by December 2026?",
            MarketDomain.CRYPTO,
            tags=["crypto", "bitcoin"],
        ),
        _market(
            "m-apple-foldable",
            "will-apple-announce-a-foldable-iphone-before-wwdc",
            "Will Apple announce a foldable iPhone before WWDC?",
            MarketDomain.CULTURE,
            tags=["apple", "tech"],
        ),
        _market(
            "m-fed-rates",
            "will-the-fed-cut-rates-after-the-next-meeting",
            "Will the Fed cut rates after the next meeting?",
            MarketDomain.ECONOMY,
            tags=["economy", "fed"],
        ),
        _market(
            "m-china-taiwan-2027",
            "will-china-invade-taiwan-before-2027",
            "Will China invade Taiwan before 2027?",
            MarketDomain.POLITICS,
            tags=["politics", "china"],
        ),
        _market(
            "m-trump-out-gtavi",
            "will-trump-be-out-as-president-before-gta-vi",
            "Will Trump be out as President before GTA VI?",
            MarketDomain.POLITICS,
            tags=["politics", "gta"],
        ),
        _market(
            "m-ceasefire-30d",
            "will-a-ceasefire-happen-within-30-days",
            "Will a ceasefire happen within 30 days?",
            MarketDomain.POLITICS,
            tags=["politics", "conflict"],
        ),
        _market(
            "m-trump-2024",
            "will-trump-win-the-2024-us-presidential-election",
            "Will Trump win the 2024 U.S. presidential election?",
            MarketDomain.POLITICS,
            tags=["politics", "election"],
        ),
        _market(
            "m-gta-2026",
            "will-gta-vi-be-released-before-2026",
            "Will GTA VI be released before 2026?",
            MarketDomain.CULTURE,
            tags=["gaming", "rockstar"],
        ),
    ]


def fixture_posts() -> list[SocialPost]:
    return [
        # --- Bitcoin --------------------------------------------------
        _post(
            "p-btc-relevant",
            "Bitcoin could hit $150k by December 2026 if ETF inflows continue.",
        ),
        _post(
            "p-btc-end2026",
            "Analysts now expect Bitcoin to reach $150k before the end of 2026.",
        ),
        _post(
            "p-btc-broad",
            "Bitcoin price may rise strongly over the next few years.",
        ),
        _post(
            "p-btc-adoption",
            "Bitcoin adoption is increasing among institutions.",
        ),
        _post(
            "p-btc-wrong-threshold",
            "Bitcoin could hit $80k by December 2026, some analysts say.",
        ),
        # --- Apple ----------------------------------------------------
        _post(
            "p-apple-relevant",
            "Apple may reveal a foldable iPhone before WWDC according to supply chain rumors.",
        ),
        _post(
            "p-apple-testing",
            "Rumors say Apple is testing a foldable iPhone prototype.",
        ),
        _post(
            "p-apple-stock",
            "Apple stock is down after weak iPad sales.",
        ),
        _post(
            "p-apple-wrong-product",
            "Apple is expected to launch a new Vision headset before WWDC.",
        ),
        # --- Fed rates ------------------------------------------------
        _post(
            "p-fed-relevant",
            "Analysts expect the Fed to cut rates after the next meeting in March.",
        ),
        _post(
            "p-fed-unchanged",
            "The Fed may keep rates unchanged after the next meeting.",
        ),
        _post(
            "p-fed-mismatch",
            "The Fed may cut rates before the next meeting.",
        ),
        _post(
            "p-mortgage-rates",
            "Mortgage rates declined this week.",
        ),
        # --- China / Taiwan -------------------------------------------
        _post(
            "p-china-relevant",
            "Some analysts warn China could invade Taiwan before 2027.",
        ),
        _post(
            "p-china-broad",
            "China and Taiwan tensions are increasing after new military exercises.",
        ),
        _post(
            "p-china-2030",
            "Analysts say China may invade Taiwan after 2030 if tensions continue.",
        ),
        # --- Trump ----------------------------------------------------
        _post(
            "p-trump-unrelated",
            "Trump launches a new social media campaign aimed at younger voters.",
        ),
        _post(
            "p-trump-2024-polls",
            "New swing-state polls show Trump leading Biden in the 2024 election.",
        ),
        _post(
            "p-trump-fundraising",
            "Trump campaign launches a new fundraising website.",
        ),
        # --- Ceasefire ------------------------------------------------
        _post(
            "p-ceasefire-relevant",
            "A ceasefire could happen within 30 days according to mediators.",
        ),
        _post(
            "p-ceasefire-broad",
            "Diplomats meet today to discuss next steps in the conflict.",
        ),
        # --- GTA ------------------------------------------------------
        _post(
            "p-gta-relevant",
            "Rockstar may delay GTA VI until 2026.",
        ),
        _post(
            "p-gta-trailer",
            "GTA 6 trailer broke YouTube records.",
        ),
        _post(
            "p-gta-merch",
            "Rockstar opens a new merch store for GTA fans.",
        ),
        # --- noisy / edge --------------------------------------------
        _post("p-empty", ""),
        _post("p-tiny", "ok."),
        _post(
            "p-noisy",
            "lol lmao buy stonks now to the moon #gainz pump it",
        ),
        _post(
            "p-numbers",
            "1 2 3 4 5 6 7 8 9 10",
        ),
    ]


def fixture_labels() -> list[LabeledPair]:
    return [
        # Bitcoin ------------------------------------------------------
        LabeledPair(
            "m-btc-150k",
            "p-btc-relevant",
            RelevanceLabel.RELEVANT,
            ConstraintMatch.FULL,
            "exact match: bitcoin + $150k + December 2026",
        ),
        LabeledPair(
            "m-btc-150k",
            "p-btc-end2026",
            RelevanceLabel.RELEVANT,
            ConstraintMatch.FULL,
            "paraphrased anchor: 'end of 2026' matches December 2026",
        ),
        LabeledPair(
            "m-btc-150k",
            "p-btc-broad",
            RelevanceLabel.WEAKLY_RELEVANT,
            ConstraintMatch.MISSING,
            "broad bitcoin topic — needs embedding overlay (05A FN)",
        ),
        LabeledPair(
            "m-btc-150k",
            "p-btc-adoption",
            RelevanceLabel.WEAKLY_RELEVANT,
            ConstraintMatch.MISSING,
            "macro-related, no threshold/date — 05A may miss",
        ),
        LabeledPair(
            "m-btc-150k",
            "p-btc-wrong-threshold",
            RelevanceLabel.WEAKLY_RELEVANT,
            ConstraintMatch.FULL,
            "wrong threshold ($80k) but same date — contract-adjacent",
        ),
        # Apple --------------------------------------------------------
        LabeledPair(
            "m-apple-foldable",
            "p-apple-relevant",
            RelevanceLabel.RELEVANT,
            ConstraintMatch.FULL,
            "exact entity + product + constraint match",
        ),
        LabeledPair(
            "m-apple-foldable",
            "p-apple-testing",
            RelevanceLabel.WEAKLY_RELEVANT,
            ConstraintMatch.MISSING,
            "product + entity, no WWDC anchor",
        ),
        LabeledPair(
            "m-apple-foldable",
            "p-apple-stock",
            RelevanceLabel.UNRELATED,
            ConstraintMatch.MISSING,
            "entity-only false positive (different product axis)",
        ),
        LabeledPair(
            "m-apple-foldable",
            "p-apple-wrong-product",
            RelevanceLabel.WEAKLY_RELEVANT,
            ConstraintMatch.FULL,
            "same entity, same constraint, different product",
        ),
        # Fed ----------------------------------------------------------
        LabeledPair(
            "m-fed-rates",
            "p-fed-relevant",
            RelevanceLabel.RELEVANT,
            ConstraintMatch.FULL,
            "exact entity + action + constraint match",
        ),
        LabeledPair(
            "m-fed-rates",
            "p-fed-unchanged",
            RelevanceLabel.RELEVANT,
            ConstraintMatch.FULL,
            "opposite implication is still contract-relevant (stance is B06)",
        ),
        LabeledPair(
            "m-fed-rates",
            "p-fed-mismatch",
            RelevanceLabel.AMBIGUOUS,
            ConstraintMatch.MISMATCH,
            "same anchor, opposite relation (before vs after)",
        ),
        LabeledPair(
            "m-fed-rates",
            "p-mortgage-rates",
            RelevanceLabel.WEAKLY_RELEVANT,
            ConstraintMatch.MISSING,
            "topic-adjacent (rates), no Fed/constraint",
        ),
        # China --------------------------------------------------------
        LabeledPair(
            "m-china-taiwan-2027",
            "p-china-relevant",
            RelevanceLabel.RELEVANT,
            ConstraintMatch.FULL,
            "exact entity + action + constraint match",
        ),
        LabeledPair(
            "m-china-taiwan-2027",
            "p-china-broad",
            RelevanceLabel.WEAKLY_RELEVANT,
            ConstraintMatch.MISSING,
            "broad topic, no 2027 anchor",
        ),
        LabeledPair(
            "m-china-taiwan-2027",
            "p-china-2030",
            RelevanceLabel.AMBIGUOUS,
            ConstraintMatch.MISMATCH,
            "different anchor and opposite relation",
        ),
        # Trump-out-GTA ------------------------------------------------
        LabeledPair(
            "m-trump-out-gtavi",
            "p-trump-unrelated",
            RelevanceLabel.UNRELATED,
            ConstraintMatch.MISSING,
            "entity-only false positive",
        ),
        LabeledPair(
            "m-trump-out-gtavi",
            "p-trump-fundraising",
            RelevanceLabel.UNRELATED,
            ConstraintMatch.MISSING,
            "campaign-related but wrong event axis",
        ),
        # Ceasefire ---------------------------------------------------
        LabeledPair(
            "m-ceasefire-30d",
            "p-ceasefire-relevant",
            RelevanceLabel.RELEVANT,
            ConstraintMatch.FULL,
            "exact event + within-30-days constraint",
        ),
        LabeledPair(
            "m-ceasefire-30d",
            "p-ceasefire-broad",
            RelevanceLabel.WEAKLY_RELEVANT,
            ConstraintMatch.MISSING,
            "broad diplomacy topic, no anchor",
        ),
        # Trump 2024 ---------------------------------------------------
        LabeledPair(
            "m-trump-2024",
            "p-trump-2024-polls",
            RelevanceLabel.RELEVANT,
            ConstraintMatch.UNKNOWN,
            "Trump + 2024 + election match",
        ),
        LabeledPair(
            "m-trump-2024",
            "p-trump-fundraising",
            RelevanceLabel.UNRELATED,
            ConstraintMatch.UNKNOWN,
            "entity-only — fundraising is not the election event",
        ),
        LabeledPair(
            "m-trump-2024",
            "p-trump-unrelated",
            RelevanceLabel.UNRELATED,
            ConstraintMatch.UNKNOWN,
            "entity-only — different event axis",
        ),
        # GTA-release --------------------------------------------------
        LabeledPair(
            "m-gta-2026",
            "p-gta-relevant",
            RelevanceLabel.RELEVANT,
            ConstraintMatch.FULL,
            "exact entity + temporal + constraint match",
        ),
        LabeledPair(
            "m-gta-2026",
            "p-gta-trailer",
            RelevanceLabel.UNRELATED,
            ConstraintMatch.MISSING,
            "entity-only / different event axis (trailer vs release)",
        ),
        LabeledPair(
            "m-gta-2026",
            "p-gta-merch",
            RelevanceLabel.UNRELATED,
            ConstraintMatch.MISSING,
            "entity-only (Rockstar/GTA) — merch is unrelated",
        ),
        # noisy / empty -----------------------------------------------
        LabeledPair(
            "m-btc-150k",
            "p-empty",
            RelevanceLabel.UNRELATED,
            ConstraintMatch.UNKNOWN,
            "empty post must not crash",
        ),
        LabeledPair(
            "m-apple-foldable",
            "p-tiny",
            RelevanceLabel.UNRELATED,
            ConstraintMatch.UNKNOWN,
            "near-empty post",
        ),
        LabeledPair(
            "m-btc-150k",
            "p-noisy",
            RelevanceLabel.UNRELATED,
            ConstraintMatch.MISSING,
            "noisy / spam-style content",
        ),
        LabeledPair(
            "m-apple-foldable",
            "p-noisy",
            RelevanceLabel.UNRELATED,
            ConstraintMatch.MISSING,
            "noisy / spam-style content",
        ),
        LabeledPair(
            "m-fed-rates",
            "p-numbers",
            RelevanceLabel.UNRELATED,
            ConstraintMatch.MISSING,
            "low-information numeric noise",
        ),
        LabeledPair(
            "m-fed-rates",
            "p-noisy",
            RelevanceLabel.UNRELATED,
            ConstraintMatch.MISSING,
            "noisy / spam-style content",
        ),
        # Cross-pairs: same fixtures, different markets ---------------
        LabeledPair(
            "m-btc-150k",
            "p-apple-stock",
            RelevanceLabel.UNRELATED,
            ConstraintMatch.MISSING,
            "different domain entirely",
        ),
        LabeledPair(
            "m-apple-foldable",
            "p-btc-broad",
            RelevanceLabel.UNRELATED,
            ConstraintMatch.MISSING,
            "different domain entirely",
        ),
        LabeledPair(
            "m-fed-rates",
            "p-btc-broad",
            RelevanceLabel.UNRELATED,
            ConstraintMatch.MISSING,
            "different domain",
        ),
        LabeledPair(
            "m-china-taiwan-2027",
            "p-apple-stock",
            RelevanceLabel.UNRELATED,
            ConstraintMatch.MISSING,
            "different domain",
        ),
        LabeledPair(
            "m-trump-out-gtavi",
            "p-fed-relevant",
            RelevanceLabel.UNRELATED,
            ConstraintMatch.MISSING,
            "different topic axis",
        ),
        LabeledPair(
            "m-ceasefire-30d",
            "p-fed-relevant",
            RelevanceLabel.UNRELATED,
            ConstraintMatch.MISSING,
            "different topic axis",
        ),
        LabeledPair(
            "m-gta-2026",
            "p-apple-stock",
            RelevanceLabel.UNRELATED,
            ConstraintMatch.MISSING,
            "different domain",
        ),
        LabeledPair(
            "m-trump-2024",
            "p-mortgage-rates",
            RelevanceLabel.UNRELATED,
            ConstraintMatch.UNKNOWN,
            "different topic axis",
        ),
        LabeledPair(
            "m-btc-150k",
            "p-fed-relevant",
            RelevanceLabel.UNRELATED,
            ConstraintMatch.MISSING,
            "different topic axis",
        ),
        LabeledPair(
            "m-apple-foldable",
            "p-fed-relevant",
            RelevanceLabel.UNRELATED,
            ConstraintMatch.MISSING,
            "different topic axis",
        ),
    ]


__all__ = [
    "LabeledPair",
    "fixture_labels",
    "fixture_markets",
    "fixture_posts",
]

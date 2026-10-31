"""A deterministic, transparent stance baseline.

This baseline is intentionally simple. It is **not** the final
solution — it is a verifiable starting point that the rest of the
pipeline can rely on and that the test suite can pin.

Approach
--------
Given a post and a market, the classifier:

1. Computes the set of normalized tokens of the post and the market
   question.
2. Counts how often the post contains tokens associated with the
   YES side of the market (``"yes"``, the YES outcome label, words
   like ``"win"``, ``"approve"``, ...) vs. the NO side.
3. Produces a label and a confidence based on the imbalance, capped
   to a low confidence band since this is a heuristic.

Limitations are explicit:

* It does not understand negation (e.g. ``"not winning"``).
* It does not understand sarcasm.
* It does not handle multi-outcome markets beyond the first two
  outcomes — those will be promoted to a future model.

The next iteration replaces this classifier with an LLM-assisted one,
keeping the same ``StanceClassifier`` interface.
"""

from __future__ import annotations

from polysignal.domain import (
    Market,
    Platform,
    SocialPost,
    Stance,
    StanceLabel,
    StanceMethod,
)
from polysignal.nlp.normalize import token_set
from polysignal.nlp.stance.base import StanceClassifier

_YES_LEXICON = frozenset(
    {
        "yes",
        "win",
        "wins",
        "winning",
        "won",
        "approve",
        "approved",
        "pass",
        "passes",
        "passed",
        "success",
        "successful",
        "happen",
        "happens",
        "happened",
        "confirmed",
        "true",
    }
)

_NO_LEXICON = frozenset(
    {
        "no",
        "lose",
        "loses",
        "losing",
        "lost",
        "loss",
        "losses",
        "reject",
        "rejected",
        "fail",
        "fails",
        "failed",
        "failure",
        "collapse",
        "collapses",
        "collapsed",
        "collapsing",
        "false",
        "deny",
        "denies",
        "denied",
        "wont",
        "wouldnt",
        "doomed",
        "defeat",
        "defeats",
        "defeated",
    }
)


class HeuristicStanceClassifier(StanceClassifier):
    """Deterministic, lexicon-based stance baseline."""

    def __init__(
        self,
        yes_extras: set[str] | None = None,
        no_extras: set[str] | None = None,
    ) -> None:
        self.yes_lexicon = set(_YES_LEXICON) | (yes_extras or set())
        self.no_lexicon = set(_NO_LEXICON) | (no_extras or set())

    def classify(self, post: SocialPost, market: Market) -> Stance:
        post_tokens = token_set(post.text_normalized or post.text)
        question_tokens = token_set(market.question)
        # If the post does not mention any token from the market question
        # at all, we cannot make a claim about its stance.
        if not (post_tokens & question_tokens):
            return Stance(
                post_id=post.post_id,
                platform=post.platform,
                market_id=market.market_id,
                outcome_id=None,
                label=StanceLabel.UNRELATED,
                confidence=0.0,
                method=StanceMethod.HEURISTIC,
                rationale="no overlap between post tokens and market question",
            )

        yes_outcome, no_outcome = _yes_no_outcomes(market)
        yes_tokens = (
            self.yes_lexicon | {yes_outcome.label.lower()} if yes_outcome else self.yes_lexicon
        )
        no_tokens = self.no_lexicon | {no_outcome.label.lower()} if no_outcome else self.no_lexicon

        yes_hits = len(post_tokens & yes_tokens)
        no_hits = len(post_tokens & no_tokens)

        if yes_hits == 0 and no_hits == 0:
            return Stance(
                post_id=post.post_id,
                platform=post.platform,
                market_id=market.market_id,
                outcome_id=None,
                label=StanceLabel.NEUTRAL,
                confidence=0.1,
                method=StanceMethod.HEURISTIC,
                rationale="post mentions the market but contains no YES/NO lexicon hits",
            )

        total = yes_hits + no_hits
        if yes_hits == no_hits:
            return Stance(
                post_id=post.post_id,
                platform=post.platform,
                market_id=market.market_id,
                outcome_id=None,
                label=StanceLabel.AMBIGUOUS,
                confidence=0.2,
                method=StanceMethod.HEURISTIC,
                rationale=f"equal YES/NO lexicon hits ({yes_hits}={no_hits})",
            )

        # Confidence is capped: the heuristic is not trustworthy enough
        # to ever return very high values.
        margin = abs(yes_hits - no_hits) / total
        confidence = min(0.6, 0.3 + 0.3 * margin)
        if yes_hits > no_hits:
            return Stance(
                post_id=post.post_id,
                platform=post.platform,
                market_id=market.market_id,
                outcome_id=yes_outcome.outcome_id if yes_outcome else None,
                label=StanceLabel.SUPPORTS_YES,
                confidence=confidence,
                method=StanceMethod.HEURISTIC,
                rationale=f"YES hits ({yes_hits}) > NO hits ({no_hits})",
            )
        return Stance(
            post_id=post.post_id,
            platform=post.platform,
            market_id=market.market_id,
            outcome_id=no_outcome.outcome_id if no_outcome else None,
            label=StanceLabel.SUPPORTS_NO,
            confidence=confidence,
            method=StanceMethod.HEURISTIC,
            rationale=f"NO hits ({no_hits}) > YES hits ({yes_hits})",
        )


def _yes_no_outcomes(market: Market):
    """Find the (yes, no) outcomes of a binary market by label, if present."""
    yes_outcome = None
    no_outcome = None
    for outcome in market.outcomes:
        label = outcome.label.strip().lower()
        if label in {"yes", "y", "true"}:
            yes_outcome = outcome
        elif label in {"no", "n", "false"}:
            no_outcome = outcome
    if yes_outcome is None and no_outcome is None and len(market.outcomes) == 2:
        # Fall back to index order so the heuristic still has something
        # to attach SUPPORTS_YES/NO to in non-binary markets.
        yes_outcome = market.outcomes[0]
        no_outcome = market.outcomes[1]
    return yes_outcome, no_outcome


# Silence "unused" lint warnings for Platform: imported only to keep
# the public type surface obvious when reading this module.
_ = Platform


__all__ = ["HeuristicStanceClassifier"]

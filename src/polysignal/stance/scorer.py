"""Outcome-aware deterministic stance scorer (Branch 06A).

The scorer is a pure function — no I/O, no global state. It takes a
text snippet (usually a segment) plus the market question and outcome
labels and returns a :class:`StanceVerdict`. The pipeline layer turns
that verdict into a :class:`StanceRecord` with full provenance.

Conservative-by-design rule
---------------------------

When the text does not carry a clear directional signal, the scorer
prefers ``neutral`` / ``ambiguous`` / ``insufficient_context`` over
guessing YES or NO. The brief is explicit: "do not classify positive
tone as supports_yes".

The scorer therefore requires:

* the text to *mention* the market (entity / outcome / canonical
  keyword overlap), otherwise it returns ``insufficient_context``;
* multi-outcome text to mention the target label or a conservative
  generated alias before directional cues are applied to that target;
* at least one directional cue, otherwise ``neutral``;
* cue balance and uncertainty to lean toward ``ambiguous`` instead of
  picking a side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from polysignal.domain.enums import StanceLabel
from polysignal.stance.cues import extract_cues, no_cues, yes_cues
from polysignal.stance.target_aliases import (
    generate_target_aliases,
    mentions_target_alias,
    normalize_target_alias,
)

# Maximum confidence the deterministic baseline will ever emit — the
# downstream LLM judge is expected to push past this band.
_MAX_CONFIDENCE = 0.7

# Very short snippets cannot carry enough signal for a confident
# directional verdict.
_MIN_TEXT_LEN = 18


@dataclass(frozen=True)
class ScorerConfig:
    """Tunable knobs for the stance scorer.

    The defaults match the conservative behaviour described above. Tests
    can override individual fields to make corner cases observable.
    """

    min_text_len: int = _MIN_TEXT_LEN
    max_confidence: float = _MAX_CONFIDENCE
    # The fraction of cue evidence that must point one way before the
    # scorer commits to a SUPPORTS_YES / SUPPORTS_NO verdict.
    decision_margin: float = 0.2
    # If uncertainty cues outnumber directional cues, downgrade to
    # ``ambiguous``.
    uncertainty_override_ratio: float = 1.0


@dataclass
class StanceVerdict:
    """The scorer's verdict for one (text, market) pair."""

    label: StanceLabel
    stance_score: float
    confidence: float
    explanation: str
    matched_yes_cues: list[str] = field(default_factory=list)
    matched_no_cues: list[str] = field(default_factory=list)
    matched_uncertainty_cues: list[str] = field(default_factory=list)
    negated_cues: list[str] = field(default_factory=list)


def _tokens(text: str) -> set[str]:
    return set(normalize_target_alias(text).split())


def _market_anchors(
    *,
    market_question: str,
    outcome_names: list[str],
    canonical_keywords: list[str],
) -> set[str]:
    """Tokens we expect the post to mention to be considered on-topic.

    Outcome labels like "Yes"/"No" are filtered because they cannot
    establish topical relevance on their own.
    """
    anchors: set[str] = set()
    for keyword in canonical_keywords:
        anchors |= _tokens(keyword)
    anchors |= _tokens(market_question)
    for outcome in outcome_names:
        low = outcome.strip().lower()
        if low in {"yes", "no", "y", "n", "true", "false"}:
            continue
        anchors |= _tokens(outcome)
    # Drop very common short tokens that would always match.
    return {token for token in anchors if len(token) >= 3}


def _bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if value <= low:
        return low
    if value >= high:
        return high
    return value


def score_stance(
    *,
    text: str,
    market_question: str,
    outcome_names: list[str],
    target_outcome_name: str | None = None,
    target_aliases: list[str] | None = None,
    canonical_keywords: list[str] | None = None,
    config: ScorerConfig | None = None,
) -> StanceVerdict:
    """Return a :class:`StanceVerdict` for one (text, market) pair.

    The function deliberately treats the market question as the unit of
    interpretation: every YES cue is read as "supports the YES outcome
    of *this* market", which is why we require at least one anchor
    overlap before emitting a directional verdict. Without that
    constraint, "Bitcoin is set to rally" applied to a market about
    GTA VI would falsely score as SUPPORTS_YES.
    """
    cfg = config or ScorerConfig()
    snippet = (text or "").strip()
    if len(snippet) < cfg.min_text_len:
        return StanceVerdict(
            label=StanceLabel.INSUFFICIENT_CONTEXT,
            stance_score=0.0,
            confidence=0.0,
            explanation=f"text too short for stance ({len(snippet)} chars)",
        )

    resolved_target_aliases = (
        list(target_aliases)
        if target_aliases is not None
        else generate_target_aliases(target_outcome_name)
    )
    if target_outcome_name is not None and not mentions_target_alias(
        snippet, resolved_target_aliases
    ):
        mentioned_competitors = [
            outcome
            for outcome in outcome_names
            if outcome != target_outcome_name
            and mentions_target_alias(snippet, generate_target_aliases(outcome))
        ]
        explanation = f"target outcome {target_outcome_name!r} aliases are not mentioned"
        if mentioned_competitors:
            explanation += f"; mentioned alternatives: {', '.join(mentioned_competitors)}"
        return StanceVerdict(
            label=StanceLabel.NEUTRAL,
            stance_score=0.0,
            confidence=0.1 if mentioned_competitors else 0.05,
            explanation=explanation,
        )

    anchors = _market_anchors(
        market_question=market_question,
        outcome_names=outcome_names,
        canonical_keywords=canonical_keywords or [],
    )
    snippet_tokens = _tokens(snippet)
    overlap = snippet_tokens & anchors
    if not overlap and anchors:
        return StanceVerdict(
            label=StanceLabel.INSUFFICIENT_CONTEXT,
            stance_score=0.0,
            confidence=0.05,
            explanation="text does not mention any market anchor",
        )

    cues = extract_cues(snippet)
    yes_matches = yes_cues(cues)
    no_matches = no_cues(cues)
    uncertainty_matches = [c for c in cues if c.category == "uncertainty"]
    negated_yes = [c for c in cues if c.category == "yes" and c.negated]

    yes_score = float(len(yes_matches))
    no_score = float(len(no_matches))
    uncertainty_score = float(len(uncertainty_matches))
    total_directional = yes_score + no_score

    if total_directional == 0:
        # On-topic but no directional cue → neutral mention.
        return StanceVerdict(
            label=StanceLabel.NEUTRAL,
            stance_score=0.0,
            confidence=0.1 if overlap else 0.05,
            explanation="text mentions the market but contains no directional cues",
            matched_uncertainty_cues=[c.phrase for c in uncertainty_matches],
        )

    # Uncertainty dominates → ambiguous, regardless of which side has
    # more cues.
    if (
        uncertainty_score > 0
        and uncertainty_score >= total_directional * cfg.uncertainty_override_ratio
    ):
        return StanceVerdict(
            label=StanceLabel.AMBIGUOUS,
            stance_score=0.0,
            confidence=0.15,
            explanation=(
                f"uncertainty cues ({int(uncertainty_score)}) dominate "
                f"directional cues ({int(total_directional)})"
            ),
            matched_yes_cues=[c.phrase for c in yes_matches],
            matched_no_cues=[c.phrase for c in no_matches],
            matched_uncertainty_cues=[c.phrase for c in uncertainty_matches],
            negated_cues=[c.phrase for c in negated_yes],
        )

    margin = (yes_score - no_score) / total_directional
    if abs(margin) < cfg.decision_margin:
        return StanceVerdict(
            label=StanceLabel.AMBIGUOUS,
            stance_score=margin,
            confidence=0.2,
            explanation=(
                f"directional cues are balanced (yes={int(yes_score)}, "
                f"no={int(no_score)})"
            ),
            matched_yes_cues=[c.phrase for c in yes_matches],
            matched_no_cues=[c.phrase for c in no_matches],
            matched_uncertainty_cues=[c.phrase for c in uncertainty_matches],
            negated_cues=[c.phrase for c in negated_yes],
        )

    # Confidence floor + linear scaling on margin, capped at the
    # configured maximum so the deterministic baseline never overstates
    # its certainty.
    base_confidence = 0.3 + (0.4 * abs(margin))
    if uncertainty_score > 0:
        # Each uncertainty cue trims a small amount of confidence.
        base_confidence -= 0.05 * uncertainty_score
    confidence = _bounded(base_confidence, low=0.15, high=cfg.max_confidence)

    if margin > 0:
        return StanceVerdict(
            label=StanceLabel.SUPPORTS_YES,
            stance_score=_bounded(margin, low=-1.0, high=1.0),
            confidence=confidence,
            explanation=(
                f"YES cues ({int(yes_score)}) outweigh NO cues "
                f"({int(no_score)}); margin={margin:.2f}"
            ),
            matched_yes_cues=[c.phrase for c in yes_matches],
            matched_no_cues=[c.phrase for c in no_matches],
            matched_uncertainty_cues=[c.phrase for c in uncertainty_matches],
            negated_cues=[c.phrase for c in negated_yes],
        )
    return StanceVerdict(
        label=StanceLabel.SUPPORTS_NO,
        stance_score=_bounded(margin, low=-1.0, high=1.0),
        confidence=confidence,
        explanation=(
            f"NO cues ({int(no_score)}) outweigh YES cues "
            f"({int(yes_score)}); margin={margin:.2f}"
        ),
        matched_yes_cues=[c.phrase for c in yes_matches],
        matched_no_cues=[c.phrase for c in no_matches],
        matched_uncertainty_cues=[c.phrase for c in uncertainty_matches],
        negated_cues=[c.phrase for c in negated_yes],
    )


__all__ = ["ScorerConfig", "StanceVerdict", "score_stance"]

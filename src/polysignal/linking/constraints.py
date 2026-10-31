"""Constraint extraction and matching for Branch 05 semantic relevance.

A Polymarket question almost always carries one or more *constraints*
that anchor the prediction to a specific event, date, or condition:

* ``"Will Bitcoin hit $150k by December 2026?"`` -> ``by December 2026``
* ``"Will Apple announce a foldable iPhone before WWDC?"`` -> ``before WWDC``
* ``"Will China invade Taiwan before 2027?"`` -> ``before 2027``

A post that mentions the same entities but ignores (or contradicts) the
constraint is at best *weakly relevant*. This module is the deterministic
engine that answers "does the post mention the same constraint?".

The output is a :class:`ConstraintMatchResult` whose ``match`` field is
one of :class:`ConstraintMatch`:

* ``FULL``     — same anchor + same relation class (forward / backward / point).
* ``PARTIAL``  — same anchor, no recognizable relation phrase.
* ``MISSING``  — anchor not mentioned, no recognizable constraint at all.
* ``MISMATCH`` — same anchor but incompatible relation (``before`` vs ``after``),
                 or a different anchor that contradicts the market's anchor.
* ``UNKNOWN``  — the market itself carries no extractable constraint.

The matcher is intentionally string-level and rule-based. It does not
call an LLM and does not require embeddings. Future layers (Branch 05B
embedding similarity, Branch 05C LLM judge) can refine this; the
baseline must remain fully deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from polysignal.domain import ConstraintMatch
from polysignal.ingestion.social.query_gen import (
    _anchor_variants,
    decompose_question,
)

# ---------------------------------------------------------------------------
# Relation phrase taxonomy
# ---------------------------------------------------------------------------

# Relation phrases that bound a future event from above (deadline-style):
# "the event must happen on or before this anchor". They form an
# equivalence class — a post that uses any of them is compatible with a
# market that uses any of the others.
_FORWARD_LIMIT_PHRASES: tuple[str, ...] = (
    "before the end of",
    "by the end of",
    "by end of",
    "on or before",
    "no later than",
    "prior to",
    "ahead of",
    "before",
    "until",
    "by",
)

# Relation phrases that put the event *after* the anchor.
_BACKWARD_PHRASES: tuple[str, ...] = ("after",)

# Relation phrases that pin the event to a point or window.
_POINT_PHRASES: tuple[str, ...] = (
    "during",
    "within",
    "in",
    "on",
    "at",
    "through",
)

_RELATION_CLASSES: dict[str, str] = {
    **{p: "forward_limit" for p in _FORWARD_LIMIT_PHRASES},
    **{p: "backward" for p in _BACKWARD_PHRASES},
    **{p: "point" for p in _POINT_PHRASES},
}

# Order phrases longest-first when scanning, so "by the end of" wins over
# "by" and "before the end of" wins over "before".
_ALL_PHRASES_ORDERED: tuple[str, ...] = tuple(
    sorted(_RELATION_CLASSES.keys(), key=lambda p: (-len(p), p))
)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketConstraint:
    """The market-side view of one constraint.

    ``relation_class`` is one of ``forward_limit``, ``backward``,
    ``point``, or ``"none"`` (when the market question has no extractable
    constraint).
    """

    relation_phrase: str
    relation_class: str
    anchor: str
    anchor_variants: tuple[str, ...]
    raw: str  # the full original constraint phrase, e.g. "before WWDC"


@dataclass(frozen=True)
class ConstraintMatchResult:
    match: ConstraintMatch
    matched_anchors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Extraction (market side)
# ---------------------------------------------------------------------------


def _expand_anchor_variants(anchor: str) -> tuple[str, ...]:
    """Return ``_anchor_variants`` plus year/numeric sub-fragments.

    "December 2026" -> ("December 2026", "2026"). This lets the post-side
    matcher recognize "before the end of 2026" as matching a market
    anchored on "December 2026" — same year, same forward-limit class.
    """
    base = list(_anchor_variants(anchor))
    extras: list[str] = []
    seen = {v.lower() for v in base}
    for variant in base:
        for token in variant.split():
            cleaned = token.strip(",.;:!?()[]{}\"'")
            if not cleaned:
                continue
            if cleaned.isdigit() and len(cleaned) == 4 and 1900 <= int(cleaned) <= 2100:
                low = cleaned.lower()
                if low not in seen:
                    extras.append(cleaned)
                    seen.add(low)
    return tuple(base + extras)


@lru_cache(maxsize=4096)
def _extract_market_constraints_cached(question: str) -> tuple[MarketConstraint, ...]:
    decomposition = decompose_question(question)
    if not decomposition.constraints:
        return ()

    constraints: list[MarketConstraint] = []
    for raw_constraint in decomposition.constraints:
        phrase, anchor = _split_constraint_phrase(raw_constraint)
        if not anchor:
            continue
        relation_class = _RELATION_CLASSES.get(phrase.lower(), "none")
        variants = _expand_anchor_variants(anchor)
        constraints.append(
            MarketConstraint(
                relation_phrase=phrase,
                relation_class=relation_class,
                anchor=anchor,
                anchor_variants=variants,
                raw=raw_constraint,
            )
        )
    return tuple(constraints)


def extract_market_constraints(question: str) -> list[MarketConstraint]:
    """Pull the temporal / contextual constraints from a market question.

    Reuses the same decomposition as the Branch 04 query generator so the
    two halves of the project see exactly the same anchors. Memoized by
    question string: the per-pair scorer calls this once per (post, market)
    pair via ``match_constraint``, but the result depends only on the market
    question, so at 1.35M-post scale this avoids millions of redundant
    ``decompose_question`` calls. A fresh list is returned each call so
    callers can never mutate the cached tuple.
    """
    return list(_extract_market_constraints_cached(question))


def _split_constraint_phrase(raw: str) -> tuple[str, str]:
    """Split ``"before December 2026"`` into ``("before", "December 2026")``."""
    text = raw.strip()
    for phrase in _ALL_PHRASES_ORDERED:
        if text.lower().startswith(phrase + " "):
            return phrase, text[len(phrase) :].strip()
    # No recognized phrase — return the whole string as the anchor.
    return "", text


# ---------------------------------------------------------------------------
# Extraction (post side)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PostConstraintMention:
    """One ``<relation> <anchor>`` pair detected in the post text."""

    relation_phrase: str
    relation_class: str
    anchor_text: str
    span_start: int
    span_end: int


def _word_boundary_regex(phrase: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)


def _anchor_search_regex(anchor: str) -> re.Pattern[str]:
    # The anchor may contain punctuation (``$150k``, ``$150,000``).
    # Use a permissive boundary on each side so we still match the anchor
    # when it abuts punctuation or sentence breaks.
    escaped = re.escape(anchor.strip())
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)


def find_post_constraint_mentions(
    post_text: str,
    anchors: list[str],
) -> list[PostConstraintMention]:
    """Find ``<relation> <anchor>`` pairs in the post for any of the anchors.

    A mention is recorded whenever the post contains both a relation
    phrase and (within a short window after it) one of the anchors.
    """
    text = (post_text or "").strip()
    if not text or not anchors:
        return []

    mentions: list[PostConstraintMention] = []
    for phrase in _ALL_PHRASES_ORDERED:
        relation_class = _RELATION_CLASSES[phrase]
        for relation_match in _word_boundary_regex(phrase).finditer(text):
            window_start = relation_match.end()
            # Look ahead up to ~40 chars (a short clause) for an anchor.
            window = text[window_start : window_start + 60]
            for anchor in anchors:
                anchor_re = _anchor_search_regex(anchor)
                anchor_match = anchor_re.search(window)
                if anchor_match is None:
                    continue
                mentions.append(
                    PostConstraintMention(
                        relation_phrase=phrase,
                        relation_class=relation_class,
                        anchor_text=anchor,
                        span_start=relation_match.start(),
                        span_end=window_start + anchor_match.end(),
                    )
                )
                break  # one anchor per relation occurrence is enough
    return mentions


def find_bare_anchor_mentions(post_text: str, anchors: list[str]) -> list[str]:
    """Return the subset of ``anchors`` that appear in the post."""
    text = post_text or ""
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for anchor in anchors:
        key = anchor.lower()
        if key in seen:
            continue
        if _anchor_search_regex(anchor).search(text):
            found.append(anchor)
            seen.add(key)
    return found


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def match_constraint(
    market_question: str,
    post_text: str,
) -> ConstraintMatchResult:
    """Decide how the post matches the market question's constraint.

    The function combines the market-side extraction with the post-side
    scan and applies a small rule table:

    +------------------------------+--------------------------------+----------+
    | post says ...                | vs market relation_class       | result   |
    +==============================+================================+==========+
    | same anchor, same class      | any                            | FULL     |
    | same anchor, diff class      | known                          | MISMATCH |
    | bare anchor, no relation     | any                            | PARTIAL  |
    | no anchor at all             | known                          | MISSING  |
    | market has no constraint     | none                           | UNKNOWN  |
    +------------------------------+--------------------------------+----------+
    """
    constraints = extract_market_constraints(market_question)
    if not constraints:
        return ConstraintMatchResult(match=ConstraintMatch.UNKNOWN)
    if not (post_text or "").strip():
        return ConstraintMatchResult(match=ConstraintMatch.MISSING)

    # Collect every anchor variant from every market constraint.
    all_anchors: list[str] = []
    seen_anchors: set[str] = set()
    for c in constraints:
        for variant in c.anchor_variants:
            key = variant.lower()
            if key not in seen_anchors:
                all_anchors.append(variant)
                seen_anchors.add(key)

    post_mentions = find_post_constraint_mentions(post_text, all_anchors)
    bare_anchors = find_bare_anchor_mentions(post_text, all_anchors)

    # Resolve against each market constraint, then pick the strongest
    # signal across them (FULL > PARTIAL > MISMATCH > MISSING > UNKNOWN).
    rankings: dict[ConstraintMatch, int] = {
        ConstraintMatch.FULL: 4,
        ConstraintMatch.PARTIAL: 3,
        ConstraintMatch.MISMATCH: 2,
        ConstraintMatch.MISSING: 1,
        ConstraintMatch.UNKNOWN: 0,
    }
    best = ConstraintMatch.MISSING
    matched_anchors: list[str] = []
    notes: list[str] = []

    for constraint in constraints:
        # Does the post mention the same anchor with the same relation class?
        full_hit = next(
            (
                m
                for m in post_mentions
                if _anchor_matches_constraint(m.anchor_text, constraint)
                and m.relation_class == constraint.relation_class
                and constraint.relation_class != "none"
            ),
            None,
        )
        if full_hit is not None:
            if rankings[ConstraintMatch.FULL] > rankings[best]:
                best = ConstraintMatch.FULL
            if full_hit.anchor_text not in matched_anchors:
                matched_anchors.append(full_hit.anchor_text)
            notes.append(
                f"FULL: '{full_hit.relation_phrase} {full_hit.anchor_text}' "
                f"matches market constraint '{constraint.raw}'"
            )
            continue

        # Does the post mention the same anchor with a *different* class?
        mismatch_hit = next(
            (
                m
                for m in post_mentions
                if _anchor_matches_constraint(m.anchor_text, constraint)
                and m.relation_class != constraint.relation_class
                and constraint.relation_class != "none"
            ),
            None,
        )
        if mismatch_hit is not None:
            if rankings[ConstraintMatch.MISMATCH] > rankings[best]:
                best = ConstraintMatch.MISMATCH
            if mismatch_hit.anchor_text not in matched_anchors:
                matched_anchors.append(mismatch_hit.anchor_text)
            notes.append(
                f"MISMATCH: '{mismatch_hit.relation_phrase} {mismatch_hit.anchor_text}' "
                f"clashes with market constraint '{constraint.raw}'"
            )
            continue

        # Bare anchor mention, no recognizable relation?
        bare_hit = next(
            (a for a in bare_anchors if _anchor_matches_constraint(a, constraint)),
            None,
        )
        if bare_hit is not None:
            if rankings[ConstraintMatch.PARTIAL] > rankings[best]:
                best = ConstraintMatch.PARTIAL
            if bare_hit not in matched_anchors:
                matched_anchors.append(bare_hit)
            notes.append(
                f"PARTIAL: anchor '{bare_hit}' mentioned without relation, "
                f"market constraint '{constraint.raw}'"
            )
            continue

        # Otherwise the post simply does not address this constraint.
        if rankings[ConstraintMatch.MISSING] > rankings[best]:
            best = ConstraintMatch.MISSING

    return ConstraintMatchResult(match=best, matched_anchors=matched_anchors, notes=notes)


def _anchor_matches_constraint(text: str, constraint: MarketConstraint) -> bool:
    """Case-insensitive check that ``text`` is one of the constraint's variants."""
    needle = text.strip().lower()
    return any(needle == v.strip().lower() for v in constraint.anchor_variants)


__all__ = [
    "ConstraintMatchResult",
    "MarketConstraint",
    "PostConstraintMention",
    "extract_market_constraints",
    "find_bare_anchor_mentions",
    "find_post_constraint_mentions",
    "match_constraint",
]

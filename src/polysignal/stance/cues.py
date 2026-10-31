"""Lexicons of YES / NO / uncertainty cues used by the stance scorer.

The cues live in their own module so they can be:

* unit-tested independently of the scoring logic;
* extended without changing the scorer plumbing;
* introspected from the evaluation report ("which cues matched?").

Cues are stored as plain lower-cased phrases. Tokenization is the
caller's job. Each phrase is matched against the *windowed* text — the
scorer chooses the window — so longer multi-word phrases are matched
first to avoid accidental sub-string matches (``"will not"`` must beat
``"will"``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# YES cues — phrases that, on their own, tend to point at the YES outcome
# of a binary market question. Verbs of occurrence, confirmation, and
# directional ascent dominate this list.
# ---------------------------------------------------------------------------

_YES_PHRASES: tuple[str, ...] = (
    "will happen",
    "is set to",
    "set to",
    "on track to",
    "on track for",
    "expected to",
    "expected for",
    "expected by",
    "likely to",
    "likely will",
    "likely",
    "is preparing to",
    "preparing to",
    "preparing for",
    "is going to",
    "going to",
    "will be released",
    "to be released",
    "will release",
    "will reach",
    "will hit",
    "will pass",
    "will exceed",
    "will surpass",
    "will cut",
    "will rise",
    "will rally",
    "will approve",
    "will win",
    "will be confirmed",
    "is confirmed",
    "confirmed",
    "set a new record",
    "all time high",
    "rallies to",
    "rallies toward",
    "ahead of schedule",
    "ahead of plan",
    "before the deadline",
    "ahead of the deadline",
    "is happening",
    "is delivered",
    "has been delivered",
    "is approved",
    "approves",
    "approved",
    "passed",
    "passes",
    "wins",
    "won",
    "is winning",
    "expects to",
    "analysts expect",
    "markets expect",
    "investors expect",
    "traders expect",
    "investors anticipate",
    "anticipate",
    "anticipates",
    "is reaching",
    "reaches",
    "reached",
    "hits",
    "hit",
    # --- Event-occurrence cues for the demo domains (geopolitics / macro /
    # crypto / elections). YES = the market's "will X happen?" event occurs.
    # Polarity-ambiguous leadership phrasings (resigns / collapses / stays in
    # power) are intentionally left to the LLM judge, which reads them against
    # the specific market question. ---
    "ceasefire reached",
    "ceasefire agreed",
    "agreed to a ceasefire",
    "ceasefire deal",
    "peace deal",
    "peace agreement",
    "peace deal signed",
    "deal reached",
    "agreement reached",
    "agreement signed",
    "signed a deal",
    "signs a deal",
    "truce",
    "rate cut",
    "rate cuts",
    "cuts rates",
    "cut rates",
    "cuts interest rates",
    "basis point cut",
    "50 basis points",
    "slashes rates",
    "lowers rates",
    "record high",
    "all-time high",
    "breaks above",
    "surges past",
    "soars past",
    "climbs above",
    "rallies past",
    "wins the election",
    "wins election",
    "won the election",
    "election victory",
    "re-elected",
    "reelected",
    "sworn in",
    "takes office",
    "becomes prime minister",
    "becomes president",
)

# ---------------------------------------------------------------------------
# NO cues — phrases that suggest the YES outcome will NOT occur. These
# include negations of YES phrases ("won't reach", "is unlikely to ...")
# as well as standalone NO indicators ("delay", "cancellation").
# ---------------------------------------------------------------------------

_NO_PHRASES: tuple[str, ...] = (
    "will not",
    "won't",
    "wont",
    "is unlikely",
    "unlikely to",
    "is not likely",
    "not likely",
    "no chance",
    "zero chance",
    "low chance",
    "fails to",
    "fail to",
    "failed to",
    "delay",
    "delayed",
    "delays",
    "delay until",
    "delay to",
    "postpone",
    "postponed",
    "postpones",
    "push back",
    "pushed back",
    "miss the deadline",
    "miss deadline",
    "misses the deadline",
    "missed the deadline",
    "cancellation",
    "cancel",
    "cancelled",
    "cancels",
    "canceled",
    "fall short",
    "falls short",
    "fell short",
    "is rejected",
    "rejected",
    "is denied",
    "denied",
    "ruled out",
    "rules out",
    "keep unchanged",
    "keep rates unchanged",
    "kept unchanged",
    "remain unchanged",
    "remains unchanged",
    "hold rates",
    "holds rates",
    "held rates",
    "loses",
    "lost",
    "is losing",
    "defeated",
    "defeat",
    "collapse",
    "collapses",
    "collapsing",
    "collapsed",
    "no way",
    "not going to",
    "is not going to",
    "is not on track",
    "off track",
    "not happen",
    "won't happen",
    "doesn't happen",
    # --- Event-does-not-occur cues for the demo domains. ---
    "no ceasefire",
    "no peace deal",
    "no deal",
    "talks collapse",
    "talks collapsed",
    "talks fail",
    "talks failed",
    "negotiations collapse",
    "negotiations fail",
    "ceasefire fails",
    "ceasefire collapsed",
    "rejects ceasefire",
    "no agreement",
    "no rate cut",
    "rates unchanged",
    "rate unchanged",
    "no change in rates",
    "pause rate cuts",
    "loses the election",
    "loses election",
    "lost the election",
    "concedes",
    "falls short of",
    "fails to reach",
    "fell short of",
    "drops below",
    "plunges below",
    "crashes below",
    "rejected at",
)

# ---------------------------------------------------------------------------
# Uncertainty / hedging cues — when these dominate the text, downgrade
# confidence and prefer ``ambiguous`` over a directional verdict.
# ---------------------------------------------------------------------------

_UNCERTAINTY_PHRASES: tuple[str, ...] = (
    "rumor",
    "rumors",
    "rumour",
    "rumours",
    "unclear",
    "uncertain",
    "uncertainty",
    "maybe",
    "could",
    "might",
    "may",
    "perhaps",
    "possibly",
    "reportedly",
    "allegedly",
    "speculation",
    "speculate",
    "speculative",
    "no decision",
    "tbd",
    "to be determined",
    "no confirmation",
    "still pending",
    "still under review",
    "we'll see",
    "remains to be seen",
)

# ---------------------------------------------------------------------------
# Negators — words that flip the polarity of an adjacent YES cue.
# ---------------------------------------------------------------------------

_NEGATORS: tuple[str, ...] = (
    "not",
    "no",
    "never",
    "without",
    "neither",
    "nor",
    "don't",
    "doesn't",
    "didn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "won't",
    "shouldn't",
    "wouldn't",
    "couldn't",
)

# Number of preceding tokens we consider when checking whether a YES cue
# is locally negated. Three is a balance between "not going to happen"
# (negator 2 tokens before) and avoiding spurious negation across
# clause boundaries.
_NEGATION_WINDOW = 3


@dataclass(frozen=True)
class CueMatch:
    """One cue occurrence with its surface form and its character span."""

    phrase: str
    category: str  # "yes" | "no" | "uncertainty"
    start: int
    end: int
    negated: bool = False


def _phrase_regex(phrases: tuple[str, ...]) -> re.Pattern[str]:
    """Compile a regex that matches the longest phrase first.

    Phrases are sorted by length (descending) so "will not" beats
    "will". Word-boundary anchors keep "will" from matching inside
    "willing".
    """
    ordered = sorted(set(phrases), key=len, reverse=True)
    escaped = [re.escape(p) for p in ordered]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


_YES_RE = _phrase_regex(_YES_PHRASES)
_NO_RE = _phrase_regex(_NO_PHRASES)
_UNCERTAINTY_RE = _phrase_regex(_UNCERTAINTY_PHRASES)
_NEGATOR_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(n) for n in _NEGATORS) + r")\b",
    re.IGNORECASE,
)


def _is_negated(text: str, match_start: int) -> bool:
    """Return True if a negator appears in the ``_NEGATION_WINDOW`` preceding tokens."""
    if match_start <= 0:
        return False
    prefix = text[:match_start]
    tokens = re.findall(r"\b[A-Za-z']+\b", prefix.lower())
    if not tokens:
        return False
    window = tokens[-_NEGATION_WINDOW:]
    return any(token in window for token in (n.lower() for n in _NEGATORS))


def _find(text: str, pattern: re.Pattern[str], category: str) -> list[CueMatch]:
    out: list[CueMatch] = []
    for match in pattern.finditer(text):
        out.append(
            CueMatch(
                phrase=match.group(0).lower(),
                category=category,
                start=match.start(),
                end=match.end(),
                negated=_is_negated(text, match.start()),
            )
        )
    return out


def extract_cues(text: str) -> list[CueMatch]:
    """Return every cue match in ``text``, tagged with category and negation."""
    if not text:
        return []
    matches = _find(text, _YES_RE, "yes")
    matches.extend(_find(text, _NO_RE, "no"))
    matches.extend(_find(text, _UNCERTAINTY_RE, "uncertainty"))
    matches.sort(key=lambda m: (m.start, -len(m.phrase)))
    # Deduplicate overlapping spans — once a phrase has been matched,
    # do not also emit any shorter cue that lives entirely inside it.
    deduped: list[CueMatch] = []
    last_end = -1
    for cue in matches:
        if cue.start < last_end:
            continue
        deduped.append(cue)
        last_end = cue.end
    return deduped


def yes_cues(cues: list[CueMatch]) -> list[CueMatch]:
    """Return only YES-category cues (negated → counted as NO upstream)."""
    return [c for c in cues if c.category == "yes" and not c.negated]


def no_cues(cues: list[CueMatch]) -> list[CueMatch]:
    """Return NO-category cues plus any negated YES cues."""
    out = [c for c in cues if c.category == "no"]
    out.extend(c for c in cues if c.category == "yes" and c.negated)
    return out


__all__ = [
    "CueMatch",
    "extract_cues",
    "no_cues",
    "yes_cues",
]

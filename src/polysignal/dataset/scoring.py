"""Deterministic scoring functions for high-signal market selection.

Three scores in ``[0, 1]``:

* :func:`social_searchability_score` — how easy is this market to search
  on social media in general (clear question, recognizable entities,
  unambiguous outcomes, time window present).
* :func:`telegram_searchability_score` — Telegram-specific overlay: very
  short questions, generic single-word topics, and obscure niches all
  score lower because Telegram public channels are noisier than Reddit.
* :func:`market_priority_score` — combined ranking score that mixes
  social searchability, Telegram searchability, and observable market
  signals (volume, liquidity, price history availability).

The scores are pure functions of :class:`polysignal.domain.Market` plus
explicit booleans so the caller controls all I/O.
"""

from __future__ import annotations

import math
import re

from polysignal.domain import Market
from polysignal.ingestion.social.query_gen import decompose_question

_ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\b")
# Tokens of these lengths or longer count as "named" — single letters
# and very short acronyms (``X``) carry too little signal alone.
_MIN_ENTITY_LEN = 3
# Generic single-word questions ("Bitcoin?") are penalised because they
# yield massive but unfocused Telegram noise.
_GENERIC_TOPICS = frozenset(
    {
        "bitcoin",
        "ethereum",
        "crypto",
        "market",
        "election",
        "stock",
        "fed",
        "war",
        "ai",
    }
)

_BINARY_LABELS = frozenset({"yes", "no", "true", "false", "win", "lose"})


def _bounded(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _entity_candidates(market: Market) -> list[str]:
    text = f"{market.question} {market.description or ''}"
    seen: set[str] = set()
    out: list[str] = []
    for match in _ENTITY_PATTERN.finditer(text):
        candidate = match.group(0).strip()
        if len(candidate) < _MIN_ENTITY_LEN:
            continue
        low = candidate.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(candidate)
    return out


def _question_length_score(question: str) -> float:
    length = len(question)
    if length < 12:
        return 0.2
    if length < 24:
        return 0.6
    if length <= 120:
        return 1.0
    if length <= 200:
        return 0.7
    return 0.4


def _outcome_clarity_score(market: Market) -> float:
    labels = {o.label.strip().lower() for o in market.outcomes if o.label}
    if not labels:
        return 0.0
    if labels & _BINARY_LABELS and len(labels) <= 4:
        return 1.0
    if len(labels) <= 2:
        return 0.85
    if len(labels) <= 4:
        return 0.6
    return 0.4


def _time_window_score(market: Market) -> float:
    if market.close_time is None:
        return 0.4
    if market.is_resolved or market.is_closed:
        # Resolved markets still have an explicit window — useful for
        # backtesting Telegram chatter against a known outcome.
        return 1.0
    return 0.85


def _generic_topic_penalty(question: str) -> float:
    """Soft penalty for one-word questions whose only token is generic."""
    stripped = re.sub(r"[?!.,;:]", "", question).strip()
    tokens = [t for t in stripped.split() if t]
    if len(tokens) <= 2 and any(t.lower() in _GENERIC_TOPICS for t in tokens):
        return 0.65
    return 1.0


def social_searchability_score(market: Market) -> float:
    """Score how searchable this market is on social media in general.

    Components (each in ``[0, 1]``):

    1. Question length / clarity.
    2. Outcome clarity (binary YES/NO scores highest).
    3. Presence of recognizable entities (capitalised proper nouns).
    4. Presence of a search-window (close_time / resolution).
    5. Mild penalty for ultra-generic single-word topics.
    """
    components = [
        _question_length_score(market.question),
        _outcome_clarity_score(market),
        min(1.0, len(_entity_candidates(market)) / 3.0),
        _time_window_score(market),
    ]
    base = sum(components) / len(components)
    return _bounded(base * _generic_topic_penalty(market.question))


def telegram_searchability_score(market: Market) -> float:
    """Telegram-specific overlay on top of the social searchability score.

    Telegram public channels are noisier than Reddit: short, generic
    queries pull thousands of unrelated messages, and very obscure
    niches yield nothing. We therefore down-weight markets whose query
    families contain only one short query.
    """
    base = social_searchability_score(market)
    decomposition = decompose_question(market.question)
    anchors = decomposition.anchors
    constraints = decomposition.constraints
    # A question with at least one constraint anchor ("by 2026",
    # "before GTA VI") gets a Telegram boost — constraints help filter
    # the firehose to messages plausibly about *this* market.
    constraint_boost = 0.0
    if constraints:
        constraint_boost = 0.10
    if anchors:
        constraint_boost += 0.05
    # Markets with rich descriptions (resolution language, sources) help
    # the searchability via canonical keywords.
    description = (market.description or "").strip()
    description_boost = 0.05 if len(description) >= 80 else 0.0
    return _bounded(base + constraint_boost + description_boost)


def market_priority_score(
    market: Market,
    *,
    has_price_history: bool = False,
    volume_cap: float = 1_000_000.0,
    liquidity_cap: float = 250_000.0,
) -> float:
    """Combined priority score in ``[0, 1]`` for ranking the selected set.

    Mixes:

    * ``social_searchability_score`` * 0.40
    * ``telegram_searchability_score`` * 0.30
    * a saturating volume term * 0.15
    * a saturating liquidity term * 0.10
    * a flat 0.05 bonus when CLOB price history is available.

    The saturating terms use a log-scale so a market with $10M volume
    is not 100x heavier than one with $100k — we want a balanced
    selection, not a heavy tail of mega-markets.
    """
    soc = social_searchability_score(market)
    tg = telegram_searchability_score(market)
    volume = float(market.volume or 0.0)
    liquidity = float(market.liquidity or 0.0)
    vol_term = math.log10(volume + 1.0) / math.log10(volume_cap + 1.0) if volume > 0 else 0.0
    liq_term = (
        math.log10(liquidity + 1.0) / math.log10(liquidity_cap + 1.0) if liquidity > 0 else 0.0
    )
    score = (
        0.40 * soc
        + 0.30 * tg
        + 0.15 * _bounded(vol_term)
        + 0.10 * _bounded(liq_term)
        + (0.05 if has_price_history else 0.0)
    )
    return _bounded(score)


__all__ = [
    "market_priority_score",
    "social_searchability_score",
    "telegram_searchability_score",
]

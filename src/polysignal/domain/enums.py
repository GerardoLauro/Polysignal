"""Enumerations shared across the PolySignal domain model."""

from __future__ import annotations

from enum import Enum


class Platform(str, Enum):
    REDDIT = "reddit"
    TWITTER = "twitter"
    TELEGRAM = "telegram"
    OTHER = "other"


class MarketDomain(str, Enum):
    POLITICS = "politics"
    SPORTS = "sports"
    CRYPTO = "crypto"
    ECONOMY = "economy"
    CULTURE = "culture"
    SCIENCE = "science"
    OTHER = "other"


class ThematicCategory(str, Enum):
    """The three thematic domains Traccia_2 requires (≥100 contracts each).

    PolySignal classifies markets into a finer 6-value :class:`MarketDomain`;
    this coarser category is the project↔assignment contract used for the
    "100 resolved contracts per thematic domain" coverage requirement and
    for the dashboard's browse-by-domain filter.

      * ``POLITICS`` ← politics / geopolitics
      * ``FINANCE``  ← crypto + economy / macro
      * ``SPORTS``   ← sports + culture / entertainment
    """

    POLITICS = "politics"
    FINANCE = "finance"
    SPORTS = "sports"


_THEMATIC_BY_DOMAIN: dict[MarketDomain, ThematicCategory] = {
    MarketDomain.POLITICS: ThematicCategory.POLITICS,
    MarketDomain.CRYPTO: ThematicCategory.FINANCE,
    MarketDomain.ECONOMY: ThematicCategory.FINANCE,
    MarketDomain.SPORTS: ThematicCategory.SPORTS,
    MarketDomain.CULTURE: ThematicCategory.SPORTS,
    # SCIENCE / OTHER intentionally have no thematic category — they are
    # not one of the three target domains and are excluded from the
    # balanced 300-contract evaluation set.
}


def thematic_category(domain: MarketDomain) -> ThematicCategory | None:
    """Map a fine :class:`MarketDomain` onto a Traccia_2 thematic bucket.

    Returns ``None`` for SCIENCE / OTHER, which are not target domains.
    """
    return _THEMATIC_BY_DOMAIN.get(domain)


class ResolutionResult(str, Enum):
    YES = "yes"
    NO = "no"
    INVALID = "invalid"
    MULTI_OUTCOME = "multi_outcome"


class EntityType(str, Enum):
    PERSON = "person"
    ORG = "org"
    LOC = "loc"
    EVENT = "event"
    WORK_OF_ART = "work_of_art"
    PRODUCT = "product"
    MISC = "misc"


class LinkMethod(str, Enum):
    HEURISTIC = "heuristic"
    LLM = "llm"
    MANUAL = "manual"
    HYBRID = "hybrid"
    DETERMINISTIC_BASELINE = "deterministic_baseline"


class RelevanceLabel(str, Enum):
    """Semantic-relevance verdict for one (market, post) pair (Branch 05)."""

    RELEVANT = "relevant"
    WEAKLY_RELEVANT = "weakly_relevant"
    AMBIGUOUS = "ambiguous"
    UNRELATED = "unrelated"


class ConstraintMatch(str, Enum):
    """How the post matches the market's temporal / contextual constraint."""

    FULL = "full"
    PARTIAL = "partial"
    MISSING = "missing"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"


class StanceLabel(str, Enum):
    SUPPORTS_YES = "supports_yes"
    SUPPORTS_NO = "supports_no"
    NEUTRAL = "neutral"
    AMBIGUOUS = "ambiguous"
    UNRELATED = "unrelated"
    # Branch 06A — explicit signal that the segment is on-topic but the
    # text does not carry enough cues to decide YES vs NO. The
    # deterministic baseline emits this when relevance is confirmed
    # upstream but the local snippet alone is uninformative (e.g. a
    # one-line preview of a digest item that names the entity without
    # asserting anything about the outcome).
    INSUFFICIENT_CONTEXT = "insufficient_context"


class StanceMethod(str, Enum):
    HEURISTIC = "heuristic"
    LLM = "llm"
    HYBRID = "hybrid"


class SentimentLabel(str, Enum):
    """Polarity of a post's text, independent of the market outcome.

    Sentiment is *not* stance: it measures whether the language is
    positive / negative / neutral in tone, whereas stance (see
    :class:`StanceLabel`) is interpreted relative to the market question
    and its YES/NO outcomes. Both signals are kept because Traccia_2
    explicitly requires sentiment analysis (Task 2) and sentiment
    features (Task 3) in addition to the outcome-aware stance layer.
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ConfidenceBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ForecastMethod(str, Enum):
    BASELINE_BLEND = "baseline_blend"
    MARKET_ONLY = "market_only"
    SOCIAL_ONLY = "social_only"
    MODEL_LEARNED = "model_learned"

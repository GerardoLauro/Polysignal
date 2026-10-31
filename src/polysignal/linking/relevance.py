"""Deterministic semantic-relevance scorer (Branch 05, Layer 1 — hardened).

Branch 05A produces a fully-deterministic verdict for one (market, post)
pair. Branch 05B adds an optional embedding-similarity overlay on top —
see :mod:`polysignal.linking.embeddings` and the ``embedding_provider``
parameter on :class:`RelevanceScorer.score`.

Term categories (Branch 05A hardening)
--------------------------------------
The market is decomposed into a :class:`MarketProfile` with eight
non-overlapping term sets:

* ``entity_terms``     — proper nouns (Bitcoin, Apple, Trump, China, ...)
* ``event_terms``      — event nouns (election, invasion, release, ...)
* ``action_terms``     — action verbs (invade, win, announce, cut, hit, ...)
* ``product_terms``    — product hints (iPhone, GTA VI, trailer, ...)
* ``threshold_terms``  — numeric / dollar / percent tokens ($150k, 5%)
* ``temporal_terms``   — date / deadline tokens (2026, WWDC, December)
* ``domain_terms``     — domain keywords (crypto, politics, fed, election)
* ``generic_terms``    — remaining content tokens

The score is a weighted blend of token Jaccard, anchor-term overlap,
entity overlap, query overlap, plus gated category and constraint bonuses. The
**label rule** now uses ``signal_count`` — the number of distinct
contract-specific categories matched. Thresholds, dates, amounts, and
percentages are constraint evidence, not anchors. Labels can only be
promoted after meaningful anchor evidence exists.

Embedding overlay (Branch 05B)
------------------------------
When an ``embedding_provider`` is supplied the scorer additionally
computes ``semantic_similarity`` (cosine) between the market question
and the post text and uses it to *upgrade borderline labels by at most
one tier*:

* ``unrelated`` + cosine >= 0.60 + accepted anchor evidence -> ``weakly_relevant``;
* ``weakly_relevant`` + cosine >= 0.75 + signal_count >= 1     -> ``relevant``;
* MISMATCH cases are NEVER upgraded by embeddings;
* rejected year-only, numeric-only, wrong-event, and domain-mismatch pairs
  are NEVER upgraded by embeddings.

The overlay is documented in ``docs/BRANCH_05_LINKING.md``. Embeddings
are disabled by default; tests use a static mock provider.

Branch 05C wraps this scorer with an optional cached structured LLM
relevance judge for borderline rows; this baseline remains unchanged.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from polysignal.domain import (
    ConstraintMatch,
    Market,
    MarketDomain,
    RelevanceLabel,
    SocialPost,
)
from polysignal.ingestion.social.query_gen import (
    _anchor_variants,
    decompose_question,
)
from polysignal.linking.constraints import extract_market_constraints, match_constraint
from polysignal.linking.phrase_anchors import (
    MarketPhraseProfile,
    build_phrase_profile,
    phrase_matches,
)
from polysignal.nlp.normalize import jaccard, normalize_text, token_set

if TYPE_CHECKING:  # pragma: no cover
    from polysignal.linking.embeddings import EmbeddingProvider


# ---------------------------------------------------------------------------
# Public configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RelevanceConfig:
    """Tunable thresholds for the deterministic scorer."""

    min_post_tokens: int = 3

    weight_token_jaccard: float = 0.25
    weight_term_overlap: float = 0.35
    weight_entity_overlap: float = 0.20
    weight_query_overlap: float = 0.20

    bonus_constraint_full: float = 0.25
    bonus_constraint_partial: float = 0.10
    penalty_constraint_mismatch: float = 0.15
    # Branch 05A hardening: penalize a post whose threshold value differs
    # from the market's, or whose product hint differs from the market's.
    penalty_threshold_mismatch: float = 0.20
    penalty_product_mismatch: float = 0.15
    penalty_domain_mismatch: float = 0.25

    bonus_domain_keyword: float = 0.05
    bonus_per_category: float = 0.05  # added per matched anchor category

    # Constraint evidence may only promote labels after meaningful
    # contract-specific anchors exist.
    min_anchor_score: float = 0.35
    strong_anchor_score: float = 0.50

    relevant_threshold: float = 0.55
    relevant_threshold_strong: float = 0.45  # used when signal_count >= 2
    relevant_threshold_very_strong: float = 0.40  # signal_count >= 3
    weakly_relevant_threshold: float = 0.20
    ambiguous_threshold: float = 0.15

    # Embedding overlay (Branch 05B).
    embedding_promote_to_weak: float = 0.60
    embedding_promote_to_relevant: float = 0.75


# ---------------------------------------------------------------------------
# Term-category profile (Branch 05A hardening)
# ---------------------------------------------------------------------------


_EVENT_NOUNS: frozenset[str] = frozenset(
    {
        "election",
        "vote",
        "voting",
        "primary",
        "summit",
        "meeting",
        "war",
        "invasion",
        "attack",
        "ceasefire",
        "strike",
        "announcement",
        "reveal",
        "release",
        "launch",
        "premiere",
        "debut",
        "approval",
        "ipo",
        "merger",
        "acquisition",
        "ban",
        "lawsuit",
        "verdict",
        "ruling",
        "settlement",
        "deal",
        "agreement",
        "treaty",
        "earnings",
        "report",
        "results",
        "decision",
        "rate",
        "rates",
        "cut",
        "hike",
        "rally",
        "crash",
        "delay",
        "postpone",
        "cancellation",
    }
)


_ACTION_VERBS_EXT: frozenset[str] = frozenset(
    {
        "invade",
        "invades",
        "invaded",
        "win",
        "wins",
        "won",
        "lose",
        "loses",
        "lost",
        "resign",
        "resigns",
        "resigned",
        "announce",
        "announces",
        "announced",
        "approve",
        "approves",
        "approved",
        "release",
        "releases",
        "released",
        "ban",
        "bans",
        "banned",
        "launch",
        "launches",
        "launched",
        "sell",
        "sells",
        "sold",
        "buy",
        "buys",
        "bought",
        "elect",
        "elects",
        "elected",
        "cut",
        "cuts",
        "hit",
        "hits",
        "reach",
        "reaches",
        "reached",
        "rise",
        "rises",
        "rose",
        "fall",
        "falls",
        "fell",
        "drop",
        "drops",
        "dropped",
        "raise",
        "raises",
        "raised",
        "delay",
        "delays",
        "delayed",
        "postpone",
        "cancel",
        "cancels",
        "canceled",
        "reveal",
        "reveals",
        "revealed",
        "leak",
        "leaks",
        "leaked",
        "halt",
        "halts",
        "default",
        "merge",
        "halve",
        "halving",
    }
)


_PRODUCT_HINTS: frozenset[str] = frozenset(
    {
        "iphone",
        "ipad",
        "macbook",
        "model",
        "version",
        "edition",
        "trailer",
        "console",
        "ps5",
        "xbox",
        "switch",
        "vision",
        "tesla",
        "model3",
        "modely",
    }
)


_MONTHS: frozenset[str] = frozenset(
    {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
    }
)


_DOMAIN_KEYWORDS: dict[MarketDomain, frozenset[str]] = {
    MarketDomain.POLITICS: frozenset({"election", "vote", "campaign", "president", "senator"}),
    MarketDomain.SPORTS: frozenset({"game", "match", "score", "championship", "league"}),
    MarketDomain.CRYPTO: frozenset(
        {"crypto", "cryptocurrency", "bitcoin", "btc", "ethereum", "eth", "blockchain"}
    ),
    MarketDomain.ECONOMY: frozenset({"inflation", "gdp", "fed", "rates", "economy", "recession"}),
    MarketDomain.CULTURE: frozenset({"movie", "film", "album", "release", "premiere"}),
    MarketDomain.SCIENCE: frozenset({"science", "research", "study", "discovery"}),
    MarketDomain.OTHER: frozenset(),
}

_DOMAIN_EVIDENCE: dict[MarketDomain, frozenset[str]] = {
    MarketDomain.POLITICS: frozenset(
        {"campaign", "election", "iran", "nato", "president", "senator", "trump", "ukraine"}
    ),
    MarketDomain.SPORTS: frozenset(
        {"championship", "cup", "fifa", "finals", "league", "nba", "nhl", "stanley"}
    ),
    MarketDomain.CRYPTO: frozenset(
        {
            "airdrop",
            "bitcoin",
            "blockchain",
            "bridge",
            "btc",
            "crypto",
            "cryptocurrency",
            "defi",
            "eth",
            "ethereum",
            "token",
        }
    ),
    MarketDomain.ECONOMY: frozenset(
        {"capex", "economy", "fed", "gdp", "inflation", "oil", "rates", "recession"}
    ),
    MarketDomain.CULTURE: frozenset(
        {"album", "film", "gta", "movie", "premiere", "release", "swift", "trailer"}
    ),
    MarketDomain.SCIENCE: frozenset({"ai", "hardware", "openai", "research", "science", "study"}),
    MarketDomain.OTHER: frozenset(),
}

_TOKEN_ALIASES: dict[str, str] = {
    "btc": "bitcoin",
    "eth": "ethereum",
}

_CONTEXT_NOISE: frozenset[str] = frozenset(
    {
        "after",
        "before",
        "breaking",
        "daily",
        "latest",
        "market",
        "markets",
        "news",
        "price",
        "today",
        "tomorrow",
        "update",
        "will",
        "win",
    }
)

_ENTITY_NOISE: frozenset[str] = frozenset({"new"})


@dataclass(frozen=True)
class MarketProfile:
    """Categorized view of one market's important terms."""

    entity_terms: frozenset[str]
    event_terms: frozenset[str]
    action_terms: frozenset[str]
    product_terms: frozenset[str]
    threshold_terms: frozenset[str]
    temporal_terms: frozenset[str]
    domain_terms: frozenset[str]
    generic_terms: frozenset[str]

    @property
    def all_terms(self) -> frozenset[str]:
        return (
            self.entity_terms
            | self.event_terms
            | self.action_terms
            | self.product_terms
            | self.threshold_terms
            | self.temporal_terms
            | self.generic_terms
        )

    @property
    def non_entity_terms(self) -> frozenset[str]:
        return (
            self.event_terms
            | self.action_terms
            | self.product_terms
            | self.threshold_terms
            | self.temporal_terms
            | self.generic_terms
        )


def _norm_token(raw: str) -> str:
    """Strip punctuation, return ``normalize_text`` lowercase."""
    stripped = raw.strip(",.;:!?()[]{}\"'`’—")
    if not stripped:
        return ""
    return normalize_text(stripped).strip()


def _classify_token(token: str, low: str, buckets: dict[str, set[str]]) -> None:
    """Assign ``low`` to one of the buckets based on ``token``/``low``.

    ``token`` is the original (case-preserving) form; ``low`` is its
    normalized lowercase form.
    """
    if not low:
        return
    if any(ch.isdigit() for ch in low):
        if low.isdigit() and len(low) == 4 and 1900 <= int(low) <= 2100:
            buckets["temporal"].add(low)
        else:
            buckets["threshold"].add(low)
        return
    if low in _ACTION_VERBS_EXT:
        buckets["action"].add(low)
        return
    if low in _EVENT_NOUNS:
        buckets["event"].add(low)
        return
    if low in _PRODUCT_HINTS:
        buckets["product"].add(low)
        return
    if low in _MONTHS:
        buckets["temporal"].add(low)
        return
    if token and (token[0].isupper() or token.isupper()):
        buckets["entity"].add(low)
        return
    if len(low) >= 4:
        buckets["generic"].add(low)


# Leadership-ouster / removal markets ("X out as", "regime fall", "X resign /
# step down") phrase the YES event in many surface forms in the wild ("not
# alive", "successor", "overthrown", "removed as", ...). The bare question word
# ("out", "fall") classifies as no event/action term, so the scorer would
# otherwise reject every leader mention as "entity-only wrong-event". These
# markets are detected by question pattern and their action axis is augmented
# with the removal-synonym set, so a post about the leader's death / succession
# / removal matches the market's event. Non-ouster markets are untouched.
_OUSTER_QUESTION_RE = re.compile(
    r"\bout as\b|\bout by\b|\bstep(?:s|ped)? down\b|\bresign|\boust|"
    r"\bremoved?\s+(?:as|from)\b|\bno longer\b|\bregime\s+(?:fall|change|collapse)\b|"
    r"\bfall of\b|\bfall by\b|\bsupreme leader\b",
    re.IGNORECASE,
)
_OUSTER_EVENT_TERMS: frozenset[str] = frozenset(
    {
        "out", "ousted", "oust", "removed", "remove", "removal", "resign",
        "resigns", "resignation", "successor", "succession", "succeed",
        "replaced", "replace", "overthrown", "overthrow", "toppled", "topple",
        "coup", "dead", "dies", "died", "death", "killed", "assassinated",
        "quit", "quits", "step", "steps", "stepped", "fall", "fallen", "fell",
        "collapse", "collapsed", "gone",
    }
)


_MARKET_PROFILE_CACHE: dict[tuple[str, str, str, object], MarketProfile] = {}


def build_market_profile(market: Market) -> MarketProfile:
    """Categorized view of one market's terms, memoized per market.

    The profile is a pure function of the market (question / slug / domain /
    tags / description) and ``MarketProfile`` is immutable (frozensets), so it
    is safe to share. At 1.35M posts x N markets the candidate prefilter and
    the per-pair scorer would otherwise re-decompose the same question
    millions of times; caching collapses that to one build per market.
    """
    key = (market.market_id, market.question or "", market.slug or "", market.domain)
    cached = _MARKET_PROFILE_CACHE.get(key)
    if cached is not None:
        return cached
    profile = _build_market_profile_uncached(market)
    _MARKET_PROFILE_CACHE[key] = profile
    return profile


def _build_market_profile_uncached(market: Market) -> MarketProfile:
    """Categorize the market's important terms into the eight buckets."""
    decomposition = decompose_question(market.question or "")
    event_core = decomposition.event_core or (market.question or "")

    buckets: dict[str, set[str]] = {
        "entity": set(),
        "event": set(),
        "action": set(),
        "product": set(),
        "threshold": set(),
        "temporal": set(),
        "generic": set(),
    }

    # Pass 1: event_core tokens (case preserved).
    for raw in event_core.split():
        low = _norm_token(raw)
        _classify_token(raw, low, buckets)

    # Pass 2: slug tokens fill in fragments dropped from the question.
    for raw in market.slug.replace("-", " ").replace("_", " ").split():
        low = _norm_token(raw)
        if not low or low in buckets["entity"] or low in buckets["action"]:
            continue
        # Slug is lower-case in Polymarket; treat capitalization heuristic
        # carefully — only the EVENT/ACTION/PRODUCT keyword buckets apply.
        if low.isdigit() and len(low) == 4 and 1900 <= int(low) <= 2100:
            buckets["temporal"].add(low)
            continue
        if any(ch.isdigit() for ch in low):
            buckets["threshold"].add(low)
            continue
        if low in _ACTION_VERBS_EXT:
            buckets["action"].add(low)
            continue
        if low in _EVENT_NOUNS:
            buckets["event"].add(low)
            continue
        if low in _PRODUCT_HINTS:
            buckets["product"].add(low)
            continue
        if low in _MONTHS:
            buckets["temporal"].add(low)
            continue

    # Pass 3: constraint anchors -> temporal_terms. We use the parsed
    # constraint anchors (so the relation phrase "before", "after", "by"
    # is excluded) and drop common stopwords.
    constraints = extract_market_constraints(market.question or "")
    temporal_noise: frozenset[str] = frozenset(
        {
            "the",
            "a",
            "an",
            "and",
            "or",
            "of",
            "for",
            "to",
            "with",
            "from",
            "by",
            "on",
            "at",
            "in",
            "be",
            "is",
            "are",
            "was",
            "were",
            "this",
            "that",
            "next",  # too generic to act as a temporal anchor on its own
        }
    )
    for constraint in constraints:
        for variant in constraint.anchor_variants:
            for raw in variant.split():
                low = _norm_token(raw)
                if not low or low in temporal_noise:
                    continue
                if (low.isdigit() and len(low) == 4 and 1900 <= int(low) <= 2100) or low in _MONTHS:
                    buckets["temporal"].add(low)
                elif any(ch.isdigit() for ch in low):
                    buckets["threshold"].add(low)
                elif len(low) >= 3:
                    buckets["temporal"].add(low)

    # Pass 4: expand entity variants (e.g. "VI" -> "6").
    for entity in list(buckets["entity"]):
        for variant in _anchor_variants(entity):
            low = _norm_token(variant)
            if not low:
                continue
            if low.isdigit():
                buckets["entity"].add(low)
            else:
                buckets["entity"].add(low)

    # Domain terms — entries from the domain keyword table that *also*
    # appear in the market's broader token set.
    market_tokens = token_set(market.question or "")
    market_tokens |= token_set(market.slug.replace("-", " "))
    for tag in market.tags or []:
        market_tokens |= token_set(tag)
    if market.description:
        market_tokens |= token_set(market.description)
    domain_keywords = _DOMAIN_KEYWORDS.get(market.domain, frozenset())

    # Ouster/leadership-removal markets: augment the action axis so the many
    # real-world surface forms of "the leader is out" (death, succession,
    # overthrow, removal) match the market's event instead of being rejected as
    # entity-only. Gated by question pattern, so other markets are unaffected.
    if _OUSTER_QUESTION_RE.search(market.question or ""):
        buckets["action"].update(_OUSTER_EVENT_TERMS)

    return MarketProfile(
        entity_terms=frozenset(
            token
            for token in buckets["entity"]
            if token not in _ENTITY_NOISE and not any(char.isdigit() for char in token)
        ),
        event_terms=frozenset(buckets["event"]),
        action_terms=frozenset(buckets["action"]),
        product_terms=frozenset(buckets["product"]),
        threshold_terms=frozenset(buckets["threshold"]),
        temporal_terms=frozenset(buckets["temporal"]),
        domain_terms=frozenset(domain_keywords),
        generic_terms=frozenset(buckets["generic"]),
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RelevanceJudgement:
    """Per-(post, market) verdict produced by :class:`RelevanceScorer`."""

    market_id: str
    post_id: str
    relevance_score: float
    relevance_label: RelevanceLabel
    confidence: float
    constraint_match: ConstraintMatch
    matched_entities: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    explanation: str = ""
    signal_count: int = 0
    matched_categories: dict[str, list[str]] = field(default_factory=dict)
    anchor_score: float = 0.0
    constraint_score: float = 0.0
    domain_mismatch: bool = False
    rejection_reason: str | None = None
    matched_phrases: list[str] = field(default_factory=list)
    market_phrase_count: int = 0
    semantic_similarity: float | None = None
    embedding_model: str | None = None
    embedding_enabled: bool = False

    @property
    def is_relevant(self) -> bool:
        return self.relevance_label in (
            RelevanceLabel.RELEVANT,
            RelevanceLabel.WEAKLY_RELEVANT,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[a-z0-9$%]+")


def _post_lower_tokens(text: str) -> set[str]:
    """Lowercase token set tolerant of $ and % so thresholds match."""
    if not text:
        return set()
    base = {t.lower() for t in token_set(text)}
    # Also pick up $150k / 5% raw, since normalize_text drops $ and %.
    for match in _TOKEN_RE.finditer(text.lower()):
        tok = match.group(0)
        if len(tok) >= 2:
            base.add(tok)
            # Strip $ and % to align with how thresholds get categorized.
            cleaned = tok.replace("$", "").replace("%", "")
            if cleaned:
                base.add(cleaned)
                alias = _TOKEN_ALIASES.get(cleaned)
                if alias:
                    base.add(alias)
    return base


def _is_numeric_token(token: str) -> bool:
    return any(char.isdigit() for char in token) or token.startswith("$") or token.endswith("%")


def _meaningful_context(tokens: set[str] | frozenset[str]) -> set[str]:
    return {
        token
        for token in tokens
        if len(token) >= 3 and token not in _CONTEXT_NOISE and not _is_numeric_token(token)
    }


def _post_domains(post_tokens: set[str]) -> set[MarketDomain]:
    return {
        domain
        for domain, keywords in _DOMAIN_EVIDENCE.items()
        if domain != MarketDomain.OTHER and keywords & post_tokens
    }


def _constraint_score(match: ConstraintMatch) -> float:
    return {
        ConstraintMatch.FULL: 1.0,
        ConstraintMatch.PARTIAL: 0.5,
        ConstraintMatch.MISSING: 0.0,
        ConstraintMatch.MISMATCH: -0.5,
        ConstraintMatch.UNKNOWN: 0.0,
    }[match]


def _short_explanation(
    label: RelevanceLabel,
    constraint_match: ConstraintMatch,
    matched_entities: list[str],
    matched_terms: list[str],
    constraint_notes: list[str],
    signal_count: int,
    semantic_similarity: float | None,
    anchor_score: float,
    constraint_score: float,
    domain_mismatch: bool,
    rejection_reason: str | None,
    matched_phrases: list[str] | None = None,
) -> str:
    parts = [
        f"label={label.value}",
        f"constraint={constraint_match.value}",
        f"signals={signal_count}",
        f"anchor={anchor_score:.2f}",
        f"constraint_score={constraint_score:.2f}",
    ]
    if domain_mismatch:
        parts.append("domain mismatch")
    if rejection_reason and rejection_reason != "domain mismatch":
        parts.append(rejection_reason)
    if matched_entities:
        parts.append("entities=" + ",".join(matched_entities[:5]))
    if matched_terms:
        parts.append("terms=" + ",".join(matched_terms[:5]))
    if matched_phrases:
        parts.append("phrases=" + ",".join(matched_phrases[:3]))
    if semantic_similarity is not None:
        parts.append(f"cos={semantic_similarity:.2f}")
    if constraint_notes:
        parts.append(constraint_notes[0])
    return " | ".join(parts)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class RelevanceScorer:
    """Hardened deterministic scorer (Branch 05A) + optional embedding overlay
    (Branch 05B)."""

    def __init__(self, config: RelevanceConfig | None = None) -> None:
        self.config = config or RelevanceConfig()

    def score(
        self,
        post: SocialPost,
        market: Market,
        *,
        query: str | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> RelevanceJudgement:
        cfg = self.config
        raw_text = (post.text_normalized or post.text or "").strip()
        normalized_text = normalize_text(raw_text)
        post_tokens = token_set(raw_text)

        if not normalized_text or len(post_tokens) < cfg.min_post_tokens:
            return RelevanceJudgement(
                market_id=market.market_id,
                post_id=post.post_id,
                relevance_score=0.0,
                relevance_label=RelevanceLabel.UNRELATED,
                confidence=0.2,
                constraint_match=ConstraintMatch.UNKNOWN,
                explanation="post text too short or empty",
            )

        profile = build_market_profile(market)
        phrase_profile = build_phrase_profile(market)
        matched_phrases = phrase_matches(phrase_profile, raw_text)
        post_lower = _post_lower_tokens(raw_text)

        # Per-category overlaps.
        m_entity = profile.entity_terms & post_lower
        m_event = profile.event_terms & post_lower
        m_action = profile.action_terms & post_lower
        m_product = profile.product_terms & post_lower
        m_threshold = profile.threshold_terms & post_lower
        m_temporal = profile.temporal_terms & post_lower
        m_domain = profile.domain_terms & post_lower
        m_generic = profile.generic_terms & post_lower
        named_temporal = {
            token for token in m_temporal if not _is_numeric_token(token) and token not in _MONTHS
        }

        context_terms = _meaningful_context(set(profile.generic_terms)) & _meaningful_context(
            post_lower
        )
        query_terms = _meaningful_context(_post_lower_tokens(query or ""))
        query_context = query_terms & _meaningful_context(post_lower)

        post_domains = _post_domains(post_lower)
        domain_mismatch = bool(
            post_domains
            and market.domain != MarketDomain.OTHER
            and market.domain not in post_domains
        )
        domain_extra = m_domain - m_entity

        anchor_score = 0.0
        if m_entity:
            anchor_score += 0.50 if len(m_entity) >= 2 else 0.35
        if m_event:
            anchor_score += 0.50
        if m_action:
            anchor_score += 0.30
        if m_product:
            anchor_score += 0.35
        if m_domain:
            anchor_score += 0.15
        if named_temporal:
            anchor_score += 0.15
        anchor_score += min(0.20, 0.10 * len(context_terms))
        anchor_score += min(0.20, 0.10 * len(query_context))
        anchor_score = min(1.0, anchor_score)

        raw_constraint_result = match_constraint(market.question or "", raw_text)
        constraint_match = raw_constraint_result.match
        market_years = {
            token
            for token in profile.temporal_terms
            if token.isdigit() and len(token) == 4 and 1900 <= int(token) <= 2100
        }
        post_years = {
            token
            for token in post_lower
            if token.isdigit() and len(token) == 4 and 1900 <= int(token) <= 2100
        }
        if (
            constraint_match == ConstraintMatch.MISSING
            and market_years
            and post_years
            and not (market_years & post_years)
        ):
            constraint_match = ConstraintMatch.MISMATCH
        has_meaningful_anchor = bool(m_entity or m_event or m_action or m_product or context_terms)
        year_only = bool(m_temporal and not has_meaningful_anchor)
        numeric_only = bool(m_threshold and not has_meaningful_anchor and not m_temporal)
        rejection_reason: str | None = None
        if year_only:
            rejection_reason = "year-only match rejected"
        elif numeric_only:
            rejection_reason = "numeric-only match rejected"
        if (
            constraint_match
            in (ConstraintMatch.FULL, ConstraintMatch.PARTIAL, ConstraintMatch.MISMATCH)
            and anchor_score < cfg.strong_anchor_score
        ):
            constraint_match = ConstraintMatch.MISSING
            if rejection_reason is None:
                rejection_reason = "insufficient anchor evidence"

        # signal_count is diagnostic only. Constraint categories are counted
        # after the anchor gate, so a shared year cannot manufacture a FULL.
        signals: list[str] = []
        if m_event:
            signals.append("event")
        if m_action:
            signals.append("action")
        if m_product:
            signals.append("product")
        if m_threshold:
            signals.append("threshold")
        if m_temporal:
            signals.append("temporal")
        if constraint_match == ConstraintMatch.FULL:
            signals.append("constraint_full")
        elif constraint_match == ConstraintMatch.PARTIAL:
            signals.append("constraint_partial")
        signal_count = len(signals)

        # Token Jaccard against the full market token set.
        market_token_set = (
            profile.all_terms
            | profile.entity_terms
            | token_set(market.question or "")
            | token_set(market.slug.replace("-", " "))
        )
        token_j = jaccard(post_lower, market_token_set)

        # Thresholds and dates are constraints, not anchor terms. They can
        # contribute through the gated constraint bonus but not term overlap.
        important_non_entity = profile.event_terms | profile.action_terms | profile.product_terms
        matched_non_entity = m_event | m_action | m_product | m_threshold | m_temporal
        matched_anchor_terms = m_event | m_action | m_product
        term_overlap = (
            len(matched_anchor_terms) / len(important_non_entity) if important_non_entity else 0.0
        )
        entity_overlap = len(m_entity) / len(profile.entity_terms) if profile.entity_terms else 0.0

        query_overlap = 0.0
        if query_terms:
            query_overlap = jaccard(_meaningful_context(post_lower), query_terms)

        domain_bonus = cfg.bonus_domain_keyword if domain_extra else 0.0
        constraint_bonus = self._constraint_bonus(constraint_match)
        category_bonus = cfg.bonus_per_category * min(
            4, sum(bool(terms) for terms in (m_event, m_action, m_product))
        )

        # Threshold mismatch: post has a numeric (non-year) value that
        # doesn't match any of the market's threshold values. Example:
        # market says "$150k by December 2026", post says "$80k by 2026".
        post_thresholds = {
            tok
            for tok in post_lower
            if any(ch.isdigit() for ch in tok)
            and not (tok.isdigit() and len(tok) == 4 and 1900 <= int(tok) <= 2100)
        }
        threshold_mismatch = bool(
            profile.threshold_terms
            and post_thresholds
            and not (profile.threshold_terms & post_thresholds)
        )
        threshold_penalty = cfg.penalty_threshold_mismatch if threshold_mismatch else 0.0

        # Product mismatch: post mentions a product hint that isn't in
        # the market's product set. Example: market about "foldable
        # iPhone", post about "Vision headset".
        other_product_hints = (_PRODUCT_HINTS & post_lower) - profile.product_terms
        product_mismatch = bool(profile.product_terms and other_product_hints)
        product_penalty = cfg.penalty_product_mismatch if product_mismatch else 0.0
        domain_penalty = cfg.penalty_domain_mismatch if domain_mismatch else 0.0

        base = (
            cfg.weight_token_jaccard * token_j
            + cfg.weight_term_overlap * term_overlap
            + cfg.weight_entity_overlap * entity_overlap
            + cfg.weight_query_overlap * query_overlap
        )
        score = max(
            0.0,
            min(
                1.0,
                base
                + constraint_bonus
                + domain_bonus
                + category_bonus
                - threshold_penalty
                - product_penalty
                - domain_penalty,
            ),
        )

        matched_entities = sorted(m_entity)
        matched_terms = sorted(matched_non_entity)
        matched_categories = {
            "event": sorted(m_event),
            "action": sorted(m_action),
            "product": sorted(m_product),
            "threshold": sorted(m_threshold),
            "temporal": sorted(m_temporal),
            "domain": sorted(m_domain),
            "generic": sorted(m_generic),
            "entity": matched_entities,
        }

        post_event_axis = (_EVENT_NOUNS | _ACTION_VERBS_EXT | _PRODUCT_HINTS) & post_lower
        market_event_axis = profile.event_terms | profile.action_terms | profile.product_terms
        broad_entity_context = bool(m_entity & {"bitcoin", "ethereum"}) or (
            len(m_entity) >= 2 and not post_event_axis
        )
        entity_only_wrong_event = bool(
            m_entity
            and not matched_anchor_terms
            and constraint_match != ConstraintMatch.FULL
            and not broad_entity_context
            and (
                (post_event_axis and market_event_axis)
                or (not broad_entity_context and not context_terms)
            )
        )

        label, rejection_reason = self._label_for(
            score=score,
            constraint_match=constraint_match,
            has_entity=bool(matched_entities),
            signal_count=signal_count,
            anchor_score=anchor_score,
            has_anchor_terms=bool(matched_anchor_terms),
            matched_terms=matched_terms,
            domain_mismatch=domain_mismatch,
            entity_only_wrong_event=entity_only_wrong_event,
            rejection_reason=rejection_reason,
            threshold_mismatch=threshold_mismatch,
            product_mismatch=product_mismatch,
            phrase_profile=phrase_profile,
            matched_phrases=matched_phrases,
        )

        # Embedding overlay (Branch 05B).
        semantic_similarity: float | None = None
        embedding_model: str | None = None
        embedding_enabled = embedding_provider is not None and embedding_provider.is_available()
        if embedding_enabled:
            assert embedding_provider is not None
            market_text = " ".join(p for p in (market.question, market.description or "") if p)
            emb_m = embedding_provider.embed(market_text)
            emb_p = embedding_provider.embed(raw_text)
            semantic_similarity = _cosine(emb_m, emb_p)
            embedding_model = embedding_provider.model_id
            label = self._apply_embedding_overlay(
                label=label,
                cosine=semantic_similarity,
                has_entity=bool(matched_entities),
                signal_count=signal_count,
                constraint_match=constraint_match,
                anchor_score=anchor_score,
                rejected=bool(rejection_reason),
            )

        confidence = self._confidence_for(
            score=score,
            constraint_match=constraint_match,
            post_token_count=len(post_tokens),
            signal_count=signal_count,
            semantic_similarity=semantic_similarity,
        )

        return RelevanceJudgement(
            market_id=market.market_id,
            post_id=post.post_id,
            relevance_score=round(score, 4),
            relevance_label=label,
            confidence=round(confidence, 4),
            constraint_match=constraint_match,
            matched_entities=matched_entities,
            matched_terms=matched_terms,
            signal_count=signal_count,
            matched_categories=matched_categories,
            anchor_score=round(anchor_score, 4),
            constraint_score=_constraint_score(constraint_match),
            domain_mismatch=domain_mismatch,
            rejection_reason=rejection_reason,
            matched_phrases=matched_phrases,
            market_phrase_count=len(phrase_profile.canonical_phrases),
            semantic_similarity=(
                round(semantic_similarity, 4) if semantic_similarity is not None else None
            ),
            embedding_model=embedding_model,
            embedding_enabled=embedding_enabled,
            explanation=_short_explanation(
                label,
                constraint_match,
                matched_entities,
                matched_terms,
                raw_constraint_result.notes,
                signal_count,
                semantic_similarity,
                anchor_score,
                _constraint_score(constraint_match),
                domain_mismatch,
                rejection_reason,
                matched_phrases,
            ),
        )

    # ---- label rule ---------------------------------------------------

    def _label_for(
        self,
        *,
        score: float,
        constraint_match: ConstraintMatch,
        has_entity: bool,
        signal_count: int,
        anchor_score: float,
        has_anchor_terms: bool,
        matched_terms: list[str],
        domain_mismatch: bool,
        entity_only_wrong_event: bool,
        rejection_reason: str | None,
        threshold_mismatch: bool = False,
        product_mismatch: bool = False,
        phrase_profile: MarketPhraseProfile | None = None,
        matched_phrases: list[str] | None = None,
    ) -> tuple[RelevanceLabel, str | None]:
        cfg = self.config
        matched_phrases = matched_phrases or []

        if domain_mismatch and anchor_score < cfg.strong_anchor_score:
            return RelevanceLabel.UNRELATED, rejection_reason or "domain mismatch"
        if entity_only_wrong_event:
            return RelevanceLabel.UNRELATED, "entity-only wrong-event rejected"
        if anchor_score < cfg.min_anchor_score:
            return RelevanceLabel.UNRELATED, rejection_reason or "insufficient anchor evidence"
        if constraint_match == ConstraintMatch.MISMATCH:
            return RelevanceLabel.AMBIGUOUS, rejection_reason

        if signal_count >= 3:
            eff_relevant = cfg.relevant_threshold_very_strong
        elif signal_count >= 2:
            eff_relevant = cfg.relevant_threshold_strong
        else:
            eff_relevant = cfg.relevant_threshold
        if anchor_score >= 0.75 and signal_count >= 2:
            eff_relevant = min(eff_relevant, cfg.relevant_threshold_very_strong)

        candidate: RelevanceLabel
        if (
            score >= eff_relevant
            and anchor_score >= cfg.strong_anchor_score
            and (has_anchor_terms or constraint_match == ConstraintMatch.FULL)
        ):
            candidate = RelevanceLabel.RELEVANT
        elif score >= cfg.weakly_relevant_threshold:
            candidate = RelevanceLabel.WEAKLY_RELEVANT
        elif score >= cfg.ambiguous_threshold and (
            has_entity or matched_terms or constraint_match == ConstraintMatch.PARTIAL
        ):
            candidate = RelevanceLabel.AMBIGUOUS
        else:
            candidate = RelevanceLabel.UNRELATED

        # Phrase-anchor gate (Branch 05G): when the market has at least one
        # phrase anchor (e.g. "Stanley Cup", "Kraken IPO", "capital gains
        # tax"), a RELEVANT verdict requires a phrase hit or a clean
        # constraint FULL match. This prevents a post that mentions one
        # isolated token of a phrase (Stanley alone, IPO alone, Trump
        # alone, ...) from looking contract-specific just because some
        # other token coincided. Broad-topic ``weakly_relevant`` is left
        # alone — it is a documented soft-recall case for Branch 05B.
        if (
            phrase_profile is not None
            and phrase_profile.has_phrase
            and not matched_phrases
            and constraint_match != ConstraintMatch.FULL
            and candidate == RelevanceLabel.RELEVANT
        ):
            candidate = RelevanceLabel.WEAKLY_RELEVANT
            rejection_reason = rejection_reason or "missing phrase anchor"

        # Threshold / product mismatch caps the label at weakly_relevant:
        # the post is on the right entity and even the right constraint,
        # but talking about the wrong threshold value or wrong product.
        if (threshold_mismatch or product_mismatch) and candidate == RelevanceLabel.RELEVANT:
            return RelevanceLabel.WEAKLY_RELEVANT, rejection_reason
        return candidate, rejection_reason

    def _apply_embedding_overlay(
        self,
        *,
        label: RelevanceLabel,
        cosine: float,
        has_entity: bool,
        signal_count: int,
        constraint_match: ConstraintMatch,
        anchor_score: float,
        rejected: bool,
    ) -> RelevanceLabel:
        cfg = self.config

        # MISMATCH overrides any semantic signal — stance/direction is
        # wrong regardless of how close the embedding says they are.
        if constraint_match == ConstraintMatch.MISMATCH:
            return label
        if rejected or anchor_score < cfg.min_anchor_score:
            return label

        if label == RelevanceLabel.UNRELATED:
            if has_entity and cosine >= cfg.embedding_promote_to_weak:
                return RelevanceLabel.WEAKLY_RELEVANT
            return label
        if label == RelevanceLabel.AMBIGUOUS:
            if cosine >= cfg.embedding_promote_to_weak:
                return RelevanceLabel.WEAKLY_RELEVANT
            return label
        if label == RelevanceLabel.WEAKLY_RELEVANT:
            if cosine >= cfg.embedding_promote_to_relevant and signal_count >= 1:
                return RelevanceLabel.RELEVANT
            return label
        return label

    def _constraint_bonus(self, match: ConstraintMatch) -> float:
        cfg = self.config
        return {
            ConstraintMatch.FULL: cfg.bonus_constraint_full,
            ConstraintMatch.PARTIAL: cfg.bonus_constraint_partial,
            ConstraintMatch.MISSING: 0.0,
            ConstraintMatch.MISMATCH: -cfg.penalty_constraint_mismatch,
            ConstraintMatch.UNKNOWN: 0.0,
        }[match]

    def _confidence_for(
        self,
        *,
        score: float,
        constraint_match: ConstraintMatch,
        post_token_count: int,
        signal_count: int,
        semantic_similarity: float | None,
    ) -> float:
        if constraint_match == ConstraintMatch.FULL and score >= 0.55:
            base = 0.85
        elif score <= 0.05 and constraint_match in (
            ConstraintMatch.MISSING,
            ConstraintMatch.UNKNOWN,
        ):
            base = 0.80
        elif post_token_count < 8:
            base = min(0.45, 0.30 + score * 0.5)
        else:
            base = max(0.35, min(0.75, 0.30 + score * 0.6))

        # Embedding signal raises confidence a little when it agrees
        # strongly with the deterministic verdict.
        if semantic_similarity is not None and semantic_similarity >= 0.75 and signal_count >= 1:
            base = min(0.95, base + 0.05)
        return base


__all__ = [
    "MarketProfile",
    "RelevanceConfig",
    "RelevanceJudgement",
    "RelevanceScorer",
    "build_market_profile",
]

"""Build bounded candidate (market, post) pairs for the Branch 05 scorer.

Production records normally carry Branch 04 retrieval provenance. Direct
Telegram channel rows may not. In that case the default ``auto`` policy uses
a deterministic anchor prefilter instead of expanding to the full
post-by-market cross-product. ``full-scan`` remains available for debugging.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

from polysignal.domain import Engagement, Market, SocialPost, SocialPostRecord
from polysignal.linking.relevance import MarketProfile, build_market_profile
from polysignal.nlp.normalize import token_set


class CandidateMode(str, Enum):
    """Supported candidate-generation policies."""

    AUTO = "auto"
    PROVENANCE = "provenance"
    PREFILTER = "prefilter"
    FULL_SCAN = "full-scan"


@dataclass(frozen=True)
class CandidatePair:
    market: Market
    post: SocialPost
    query: str | None = None
    query_type: str | None = None
    raw_post_path: str | None = None
    inclusion_reason: str | None = None


@dataclass(frozen=True)
class CandidateGenerationDiagnostics:
    """Compact audit summary for one candidate-generation pass."""

    candidate_mode: CandidateMode
    posts: int
    markets: int
    candidate_rows_generated: int
    average_candidate_markets_per_post: float
    skipped_posts_no_candidate_anchors: int
    provenance_posts: int
    prefilter_posts: int
    full_scan_posts: int
    fallback_mode_used: str
    inclusion_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def top_inclusion_reason(self) -> str:
        if not self.inclusion_reasons:
            return "none"
        reason, count = max(self.inclusion_reasons.items(), key=lambda item: (item[1], item[0]))
        return f"{reason} ({count})"


@dataclass(frozen=True)
class CandidateGenerationResult:
    pairs: list[CandidatePair]
    diagnostics: CandidateGenerationDiagnostics


@dataclass(frozen=True)
class _PrefilterMatch:
    market: Market
    score: float
    reason: str


@dataclass(frozen=True)
class _MarketPrefilter:
    """Per-market prefilter inputs, precomputed once per run.

    Building these once (rather than re-deriving the market profile and its
    meaningful generic terms inside every (post, market) comparison) turns the
    prefilter from O(posts x markets) profile builds into O(markets).
    """

    market: Market
    profile: MarketProfile
    meaningful_generic: frozenset[str]


_TOKEN_RE = re.compile(r"[a-z0-9$%]+")
_TOKEN_ALIASES: dict[str, str] = {
    "btc": "bitcoin",
    "eth": "ethereum",
}
_PREFILTER_NOISE: frozenset[str] = frozenset(
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


def _record_to_post(record: SocialPostRecord) -> SocialPost:
    """Convert a Branch 04 record to a scorer-facing post."""
    extra: dict[str, object] = {}
    if record.title:
        extra["title"] = record.title
    if record.subreddit_or_channel:
        key = "channel" if record.platform.value == "telegram" else "subreddit"
        extra[key] = record.subreddit_or_channel
    if record.upvotes is not None:
        extra["score"] = record.upvotes
    return SocialPost(
        post_id=record.post_id,
        platform=record.platform,
        author_id=record.author_id,
        author_handle=record.author_name,
        created_at=record.created_at,
        text=record.text,
        text_normalized=None,
        language=record.language,
        url=record.url,
        engagement=Engagement(
            score=record.upvotes,
            replies=record.comments_count,
            reposts=record.shares_count,
            views=record.views_count,
            reactions=record.reactions_count,
        ),
        extra=extra,
    )


def _lower_tokens(text: str) -> set[str]:
    tokens = {token.lower() for token in token_set(text or "")}
    tokens.update(match.group(0).lower() for match in _TOKEN_RE.finditer(text or ""))
    for token in list(tokens):
        cleaned = token.replace("$", "").replace("%", "")
        if cleaned:
            tokens.add(cleaned)
        alias = _TOKEN_ALIASES.get(cleaned)
        if alias:
            tokens.add(alias)
    return tokens


def _is_constraint_or_noise(token: str) -> bool:
    return (
        token in _PREFILTER_NOISE
        or any(char.isdigit() for char in token)
        or token.startswith("$")
        or token.endswith("%")
    )


def _meaningful(tokens: set[str]) -> set[str]:
    return {token for token in tokens if len(token) >= 3 and not _is_constraint_or_noise(token)}


def _prefilter_match(post: SocialPost, market: Market, query: str | None) -> _PrefilterMatch | None:
    """Return a deterministic inclusion score only for plausible pairs."""
    profile = build_market_profile(market)
    post_tokens = _lower_tokens(post.text_normalized or post.text or "")
    meaningful_post = _meaningful(post_tokens)

    entities = profile.entity_terms & post_tokens
    events = profile.event_terms & post_tokens
    actions = profile.action_terms & post_tokens
    products = profile.product_terms & post_tokens
    domains = profile.domain_terms & post_tokens
    generic = _meaningful(set(profile.generic_terms)) & meaningful_post
    query_terms = _meaningful(_lower_tokens(query or ""))
    query_overlap = query_terms & meaningful_post

    if entities and (events or actions or products):
        return _PrefilterMatch(market, 1.0, "entity+event")
    if len(entities) >= 2:
        return _PrefilterMatch(market, 0.9, "multi-entity")
    if products and (events or actions or generic):
        return _PrefilterMatch(market, 0.85, "product-context")
    if entities:
        return _PrefilterMatch(market, 0.75, "strong-entity")
    if domains and (events or actions or products):
        return _PrefilterMatch(market, 0.65, "domain+event")
    if query_overlap and len(query_overlap) >= 2:
        return _PrefilterMatch(market, 0.60, "query-overlap")
    if domains and generic:
        return _PrefilterMatch(market, 0.55, "domain+context")
    if len(generic) >= 2:
        return _PrefilterMatch(market, 0.50, "title-overlap")
    return None


def _prefilter_match_pre(
    prefilter: _MarketPrefilter,
    post_tokens: set[str],
    meaningful_post: set[str],
    query_terms: set[str],
) -> _PrefilterMatch | None:
    """Same scoring as :func:`_prefilter_match` but with per-post terms and the
    per-market profile precomputed, so neither is rebuilt per (post, market)."""
    profile = prefilter.profile
    market = prefilter.market

    entities = profile.entity_terms & post_tokens
    events = profile.event_terms & post_tokens
    actions = profile.action_terms & post_tokens
    products = profile.product_terms & post_tokens
    domains = profile.domain_terms & post_tokens
    generic = prefilter.meaningful_generic & meaningful_post
    query_overlap = query_terms & meaningful_post

    if entities and (events or actions or products):
        return _PrefilterMatch(market, 1.0, "entity+event")
    if len(entities) >= 2:
        return _PrefilterMatch(market, 0.9, "multi-entity")
    if products and (events or actions or generic):
        return _PrefilterMatch(market, 0.85, "product-context")
    if entities:
        return _PrefilterMatch(market, 0.75, "strong-entity")
    if domains and (events or actions or products):
        return _PrefilterMatch(market, 0.65, "domain+event")
    if query_overlap and len(query_overlap) >= 2:
        return _PrefilterMatch(market, 0.60, "query-overlap")
    if domains and generic:
        return _PrefilterMatch(market, 0.55, "domain+context")
    if len(generic) >= 2:
        return _PrefilterMatch(market, 0.50, "title-overlap")
    return None


def _pair(record: SocialPostRecord, market: Market, reason: str) -> CandidatePair:
    return CandidatePair(
        market=market,
        post=_record_to_post(record),
        query=record.query,
        query_type=record.query_type,
        raw_post_path=record.raw_path,
        inclusion_reason=reason,
    )


def build_pairs_from_records_with_diagnostics(
    records: Iterable[SocialPostRecord],
    markets: Sequence[Market],
    *,
    include_duplicates: bool = False,
    candidate_mode: CandidateMode = CandidateMode.AUTO,
    max_candidate_markets_per_post: int = 10,
) -> CandidateGenerationResult:
    """Build bounded candidate pairs and return diagnostics.

    ``auto`` uses retrieval provenance per record when available and otherwise
    applies the prefilter. ``full-scan`` is the explicit O(posts * markets)
    debug mode.
    """
    if max_candidate_markets_per_post < 1:
        raise ValueError("max_candidate_markets_per_post must be at least 1")

    market_by_id = {market.market_id: market for market in markets}
    # Precompute each market's prefilter profile once (O(markets)) instead of
    # rebuilding it inside every (post, market) comparison (O(posts*markets)).
    market_prefilters: list[_MarketPrefilter] = []
    for market in markets:
        profile = build_market_profile(market)
        market_prefilters.append(
            _MarketPrefilter(
                market=market,
                profile=profile,
                meaningful_generic=frozenset(_meaningful(set(profile.generic_terms))),
            )
        )
    eligible = [
        record
        for record in records
        if (include_duplicates or not record.is_duplicate)
        and record.created_at is not None
        and record.temporal_valid_candidate is not False
    ]
    pairs: list[CandidatePair] = []
    reasons: Counter[str] = Counter()
    skipped = 0
    provenance_posts = 0
    prefilter_posts = 0
    full_scan_posts = 0

    for record in eligible:
        if candidate_mode == CandidateMode.FULL_SCAN:
            generated = [_pair(record, market, "full-scan") for market in markets]
            full_scan_posts += 1
        elif (
            candidate_mode in (CandidateMode.AUTO, CandidateMode.PROVENANCE)
            and record.market_id_candidate
        ):
            market = market_by_id.get(record.market_id_candidate)
            generated = [_pair(record, market, "provenance")] if market is not None else []
            provenance_posts += 1
        elif candidate_mode == CandidateMode.PROVENANCE:
            generated = []
        else:
            post = _record_to_post(record)
            post_tokens = _lower_tokens(post.text_normalized or post.text or "")
            meaningful_post = _meaningful(post_tokens)
            query_terms = _meaningful(_lower_tokens(record.query or ""))
            ranked = [
                match
                for prefilter in market_prefilters
                if (
                    match := _prefilter_match_pre(
                        prefilter, post_tokens, meaningful_post, query_terms
                    )
                )
                is not None
            ]
            ranked.sort(key=lambda match: (-match.score, match.market.market_id))
            generated = [
                _pair(record, match.market, match.reason)
                for match in ranked[:max_candidate_markets_per_post]
            ]
            prefilter_posts += 1

        if not generated:
            skipped += 1
            continue
        pairs.extend(generated)
        reasons.update(pair.inclusion_reason or "unknown" for pair in generated)

    modes_used: list[str] = []
    if provenance_posts:
        modes_used.append("provenance")
    if prefilter_posts:
        modes_used.append("prefilter")
    if full_scan_posts:
        modes_used.append("full-scan")
    diagnostics = CandidateGenerationDiagnostics(
        candidate_mode=candidate_mode,
        posts=len(eligible),
        markets=len(markets),
        candidate_rows_generated=len(pairs),
        average_candidate_markets_per_post=(len(pairs) / len(eligible) if eligible else 0.0),
        skipped_posts_no_candidate_anchors=skipped,
        provenance_posts=provenance_posts,
        prefilter_posts=prefilter_posts,
        full_scan_posts=full_scan_posts,
        fallback_mode_used="+".join(modes_used) or "none",
        inclusion_reasons=dict(reasons),
    )
    return CandidateGenerationResult(pairs=pairs, diagnostics=diagnostics)


def build_pairs_from_records(
    records: Iterable[SocialPostRecord],
    markets: Sequence[Market],
    *,
    include_duplicates: bool = False,
    candidate_mode: CandidateMode = CandidateMode.AUTO,
    max_candidate_markets_per_post: int = 10,
) -> list[CandidatePair]:
    """Compatibility wrapper returning only candidate pairs."""
    return build_pairs_from_records_with_diagnostics(
        records,
        markets,
        include_duplicates=include_duplicates,
        candidate_mode=candidate_mode,
        max_candidate_markets_per_post=max_candidate_markets_per_post,
    ).pairs


def build_pairs_cross_product(
    posts: Sequence[SocialPost],
    markets: Sequence[Market],
) -> list[CandidatePair]:
    """Pair every post with every market for fixtures and explicit evaluation."""
    return [
        CandidatePair(market=market, post=post, inclusion_reason="full-scan")
        for market in markets
        for post in posts
    ]


__all__ = [
    "CandidateGenerationDiagnostics",
    "CandidateGenerationResult",
    "CandidateMode",
    "CandidatePair",
    "build_pairs_cross_product",
    "build_pairs_from_records",
    "build_pairs_from_records_with_diagnostics",
]

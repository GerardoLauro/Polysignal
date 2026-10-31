"""Pydantic v2 domain models shared across PolySignal modules.

These types are the single source of truth for the shape of the data
that flows through the system. They are intentionally permissive on
optional fields (many social platforms expose different subsets of
metadata) and strict on identifiers, timestamps, and probability
ranges.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from polysignal.domain.enums import (
    ConfidenceBand,
    ConstraintMatch,
    EntityType,
    ForecastMethod,
    LinkMethod,
    MarketDomain,
    Platform,
    RelevanceLabel,
    ResolutionResult,
    SentimentLabel,
    StanceLabel,
    StanceMethod,
)


class _Base(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


# ---------------------------------------------------------------------------
# Markets
#
# Polymarket identifier mapping (verified against the live API in the
# next milestone — see ``polysignal.ingestion.polymarket.client`` and
# ``docs/DATA_MODEL.md``):
#
#   * ``Market.market_id`` is intended to hold the Polymarket
#     ``condition_id`` (the on-chain condition id) once verified —
#     it is the most stable cross-source identifier for a market.
#   * ``Market.slug`` mirrors the human-readable URL slug exposed by
#     the Gamma API.
#   * ``Outcome.outcome_id`` is intended to hold the per-outcome CLOB
#     token id (variously called ``token_id``, ``clob_token_id``, or
#     ``asset_id`` in different Polymarket payloads). It is also the
#     natural join key for the CLOB ``prices-history`` endpoint.
#   * ``PricePoint.market_id`` + ``PricePoint.outcome_id`` therefore
#     pair the ``condition_id`` with the per-outcome token id.
#
# If a future verification shows that more than one identifier per
# entity is genuinely needed (e.g. both a ``condition_id`` and a
# separate Polymarket numeric id), prefer adding named optional
# fields here rather than overloading the existing ``*_id`` fields —
# and update the docs in the same change.
# ---------------------------------------------------------------------------


class Outcome(_Base):
    market_id: str | None = None
    outcome_id: str  # Polymarket CLOB token id (see module header).
    label: str
    index: int = Field(ge=0)
    # Optional latest market-implied price for this outcome (when the
    # ingestor sees it inline on the Gamma payload). Probabilities live
    # in [0, 1].
    current_price: float | None = Field(default=None, ge=0.0, le=1.0)


class Market(_Base):
    market_id: str  # Polymarket ``condition_id`` (see module header).
    slug: str
    question: str
    description: str | None = None
    domain: MarketDomain = MarketDomain.OTHER
    # Coarse Traccia_2 thematic bucket (politics / finance / sports) derived
    # from ``domain``. ``None`` for science/other markets, which are not one
    # of the three target domains. Materialized so the dashboard and coverage
    # checks can group by the assignment's three categories directly.
    thematic_category: str | None = None
    created_at: datetime
    close_time: datetime | None = None
    resolved_at: datetime | None = None
    is_resolved: bool = False
    is_closed: bool = False
    resolution: ResolutionResult | None = None
    outcomes: list[Outcome] = Field(min_length=2)
    tags: list[str] = Field(default_factory=list)
    source_url: str | None = None
    # Optional Polymarket-specific identifiers and metrics. Each field
    # is permissive on purpose — the live Gamma payload only sometimes
    # exposes them, and we never want a missing field to drop a market.
    event_id: str | None = None
    condition_id: str | None = None  # mirrors ``market_id`` when known.
    volume: float | None = Field(default=None, ge=0.0)
    liquidity: float | None = Field(default=None, ge=0.0)

    @field_validator("outcomes")
    @classmethod
    def _unique_outcome_ids(cls, outcomes: list[Outcome]) -> list[Outcome]:
        ids = [o.outcome_id for o in outcomes]
        if len(ids) != len(set(ids)):
            raise ValueError("outcome_id values must be unique within a market")
        return outcomes


class PricePoint(_Base):
    market_id: str  # Polymarket ``condition_id`` (see module header).
    outcome_id: str  # Polymarket CLOB token id (see module header).
    timestamp: datetime
    price: float = Field(ge=0.0, le=1.0)
    volume: float | None = Field(default=None, ge=0.0)


# ---------------------------------------------------------------------------
# Social
# ---------------------------------------------------------------------------


class Engagement(_Base):
    likes: int | None = Field(default=None, ge=0)
    reposts: int | None = Field(default=None, ge=0)
    replies: int | None = Field(default=None, ge=0)
    views: int | None = Field(default=None, ge=0)
    reactions: int | None = Field(default=None, ge=0)
    score: int | None = None  # platform-specific score (e.g. Reddit upvotes - downvotes)


class Entity(_Base):
    text: str
    label: EntityType
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @field_validator("end")
    @classmethod
    def _end_after_start(cls, end: int, info: Any) -> int:
        start = info.data.get("start")
        if start is not None and end < start:
            raise ValueError("entity end offset must be >= start offset")
        return end


class SocialPost(_Base):
    post_id: str
    platform: Platform
    author_id: str | None = None
    author_handle: str | None = None
    created_at: datetime | None = None
    text: str
    text_normalized: str | None = None
    language: str | None = None
    url: str | None = None
    parent_post_id: str | None = None
    engagement: Engagement = Field(default_factory=Engagement)
    entities: list[Entity] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Branch 04 — social ingestion records
#
# ``SocialPostRecord`` is the denormalized, parquet-friendly view of a
# collected social post. It pairs the platform-native ``SocialPost``
# data with the *retrieval* context (the query that surfaced it and the
# candidate market_id that generated that query). One record represents
# one (post, query, market_candidate) tuple — so if multiple queries from
# multiple markets surface the same post, several records exist; rows
# after the first occurrence of ``(platform, post_id)`` are marked with
# ``is_duplicate=True`` to make filtering trivial.
#
# ``market_id_candidate`` strictly means "this post was retrieved using a
# query generated from this market". It is NOT a claim of semantic
# relevance — true relevance is decided in branch 05.
# ---------------------------------------------------------------------------


class MarketQuery(_Base):
    """One generated search query against a candidate Polymarket market."""

    market_id: str
    query: str
    query_type: str
    rank: int = Field(ge=0)
    source_fields: list[str] = Field(default_factory=list)
    generated_at: datetime


class SocialPostRecord(_Base):
    """A collected social post, denormalized for branch 04 parquet output."""

    post_id: str
    platform: Platform
    source_id: str
    url: str | None = None
    author_id: str | None = None
    author_name: str | None = None
    created_at: datetime | None = None
    text: str
    title: str | None = None
    subreddit_or_channel: str | None = None
    query: str | None = None
    query_type: str | None = None
    market_id_candidate: str | None = None
    retrieval_run_id: str | None = None
    retrieval_work_id: str | None = None
    retrieval_window_start: datetime | None = None
    retrieval_window_end: datetime | None = None
    retrieval_lookback_days: int | None = Field(default=None, ge=0)
    retrieval_window_label: str | None = None
    retrieval_mode: str | None = None
    temporal_valid_candidate: bool | None = None
    created_at_parse_status: str = "parsed"
    market_id_hint: str | None = None
    market_question_hint: str | None = None
    market_window_start_hint: datetime | None = None
    market_close_time_hint: datetime | None = None
    query_provenance: str | None = None
    temporal_relation_to_market: str | None = None
    leakage_eligible: bool | None = None
    exclusion_reason: str | None = None
    engagement_score: float = 0.0
    upvotes: int | None = None
    comments_count: int | None = None
    shares_count: int | None = None
    views_count: int | None = None
    reactions_count: int | None = None
    # --- Traccia_2 §2 unified social schema (additive, nullable) -------------
    # These columns make every platform emit the same Section-2 fields. They
    # default to ``None`` so existing parquet files (which predate them) read
    # back cleanly via Pydantic defaults — no rewrite of the large corpus.
    #
    # ``market_id`` / ``category`` are the *resolved* linked contract and its
    # ``thematic_category``. They are populated by a join against
    # ``linked_posts`` at dataset-export time (linking runs after ingestion),
    # so the normalizer leaves them ``None``.
    market_id: str | None = None
    category: str | None = None
    # Author provenance (nullable per platform — e.g. Telegram channels expose
    # neither a follower count nor a stable per-author URL).
    author_url: str | None = None
    author_followers: int | None = Field(default=None, ge=0)
    # Section-2 normalized engagement names. The platform-native columns above
    # (upvotes/comments_count/shares_count/views_count/reactions_count) are
    # retained; these are the unified cross-platform names.
    like_count: int | None = Field(default=None, ge=0)
    reply_count: int | None = Field(default=None, ge=0)
    repost_count: int | None = Field(default=None, ge=0)
    view_count: int | None = Field(default=None, ge=0)
    # Number of sampled comments for this post in ``post_comments.parquet``
    # (0 when none collected). The comment rows themselves live in the sibling
    # table, not inline, to keep this row flat.
    comment_sample_count: int = Field(default=0, ge=0)
    language: str | None = None
    raw_path: str | None = None
    collected_at: datetime
    is_duplicate: bool = False
    text_length: int = Field(default=0, ge=0)


class PostComment(_Base):
    """One sampled comment/reply attached to a social post (Traccia_2 §2).

    Stored in a sibling ``post_comments.parquet`` keyed by ``(platform,
    post_id)`` rather than inline on :class:`SocialPostRecord`, so the flat
    post row stays flat and the large post corpus is never rewritten. Reddit
    populates this from comment JSON; Telegram from discussion-group replies.
    NER / language detection run over ``text`` exactly as for top-level posts.
    """

    post_id: str
    platform: Platform
    comment_id: str
    parent_comment_id: str | None = None
    author_id: str | None = None
    author_url: str | None = None
    text: str = ""
    published_at: datetime | None = None
    like_count: int | None = Field(default=None, ge=0)
    reply_count: int | None = Field(default=None, ge=0)
    language: str | None = None
    market_id_candidate: str | None = None
    raw_path: str | None = None
    collected_at: datetime
    text_length: int = Field(default=0, ge=0)


class PostEnrichment(_Base):
    """Branch 13 — derived NLP enrichment for one social post (Traccia_2 §2).

    Materializes the post-level fields the assignment requires but that the
    raw collection schema does not carry: ``hashtags``, ``mentioned_entities``
    (NER over persons/orgs/locations), detected ``language``, plus the
    sentiment-analysis output (Task 2/3). It is written to a *separate*
    ``post_enrichment.parquet`` keyed by ``post_id`` and joined back to
    ``social_posts.parquet`` — the multi-hundred-MB raw corpus is never
    rewritten (the project's data-safety convention), mirroring how
    ``linked_posts`` and ``stance_labels`` are derived tables.
    """

    post_id: str
    platform: Platform
    language: str | None = None
    hashtags: list[str] = Field(default_factory=list)
    mentions: list[str] = Field(default_factory=list)
    mentioned_entities: list[dict[str, str]] = Field(default_factory=list)
    sentiment_label: SentimentLabel | None = None
    sentiment_compound: float | None = Field(default=None, ge=-1.0, le=1.0)
    sentiment_pos: float | None = Field(default=None, ge=0.0, le=1.0)
    sentiment_neu: float | None = Field(default=None, ge=0.0, le=1.0)
    sentiment_neg: float | None = Field(default=None, ge=0.0, le=1.0)
    sentiment_method: str | None = None
    entity_method: str | None = None
    enriched_at: datetime


class SocialPostSegment(_Base):
    """Branch 05G — one semantic segment derived from a parent social post.

    Original ``social_posts.parquet`` is never modified; this model is the
    schema for the derived ``social_post_segments.parquet`` dataset.
    """

    segment_id: str
    parent_post_id: str
    platform: Platform
    source: str | None = None
    segment_index: int = Field(ge=0)
    segment_text: str
    created_at: datetime
    url: str | None = None
    raw_path: str | None = None
    text_length: int = Field(default=0, ge=0)
    segmentation_method: str = "single"


# ---------------------------------------------------------------------------
# Linking and stance
# ---------------------------------------------------------------------------


class Link(_Base):
    post_id: str
    platform: Platform
    market_id: str
    method: LinkMethod
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: datetime
    evidence: list[str] = Field(default_factory=list)


class LinkedPost(_Base):
    """Branch 05 — one (market, post) pair with semantic-relevance metadata.

    This is the parquet row written to ``data/normalized/social/linked_posts.parquet``.
    Unlike :class:`Link` (Branch 04's lightweight per-post link record),
    ``LinkedPost`` carries the *decision* — score, label, constraint match,
    matched entities/terms, and a short explanation — that downstream
    branches (stance, dashboard, forecasting) need to filter on.

    ``relevance_score`` is a deterministic baseline score in ``[0, 1]``.
    ``confidence`` is a separate ``[0, 1]`` band that reflects how *sure*
    the scorer is, independent of the score itself (a near-empty post has
    a low score *and* low confidence; an exact constraint+entity match
    has a high score *and* high confidence).
    """

    link_id: str
    market_id: str
    post_id: str
    platform: Platform
    query: str | None = None
    query_type: str | None = None
    relevance_score: float = Field(ge=0.0, le=1.0)
    relevance_label: RelevanceLabel
    confidence: float = Field(ge=0.0, le=1.0)
    method: LinkMethod = LinkMethod.DETERMINISTIC_BASELINE
    explanation: str = ""
    matched_entities: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)
    constraint_match: ConstraintMatch = ConstraintMatch.UNKNOWN
    created_at: datetime
    post_created_at: datetime | None = None
    market_question: str = ""
    post_text_preview: str = ""
    raw_post_path: str | None = None
    # Branch 05A diagnostic field — count of contract-specific category
    # matches (event/action/product/threshold/temporal/constraint).
    signal_count: int = 0
    # Branch 05E additive audit fields. Constraints can only promote a
    # label after enough contract-specific anchor evidence exists.
    anchor_score: float = Field(default=0.0, ge=0.0, le=1.0)
    constraint_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    domain_mismatch: bool = False
    rejection_reason: str | None = None
    # Branch 05B optional fields. Defaults keep parquet output
    # compatible with 05A consumers: ``embedding_enabled=False`` and
    # ``semantic_similarity=None`` means "no overlay was applied".
    semantic_similarity: float | None = None
    embedding_model: str | None = None
    embedding_enabled: bool = False
    # Branch 05C optional fields. Baseline fields above remain unchanged
    # for auditability and for consumers that do not enable the LLM judge.
    llm_selected: bool = False
    llm_judged: bool = False
    llm_cache_hit: bool = False
    llm_invalid_output: bool = False
    llm_applied: bool = False
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_relevance_label: RelevanceLabel | None = None
    llm_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    llm_constraint_match: ConstraintMatch | None = None
    llm_reason: str | None = None
    final_relevance_label: RelevanceLabel | None = None
    final_relevance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    final_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    final_constraint_match: ConstraintMatch | None = None
    final_method: LinkMethod | None = None
    # Branch 05G optional segment audit fields. ``post_id`` continues to
    # carry the row identity used for scoring (which is the segment id
    # when segmentation is enabled); ``parent_post_id`` and ``segment_id``
    # let the dashboard trace each linked row back to the original post.
    parent_post_id: str | None = None
    segment_id: str | None = None
    segment_index: int | None = Field(default=None, ge=0)
    segment_text_preview: str | None = None


class Stance(_Base):
    post_id: str
    platform: Platform
    market_id: str
    outcome_id: str | None = None
    label: StanceLabel
    confidence: float = Field(ge=0.0, le=1.0)
    method: StanceMethod
    rationale: str | None = None


# ---------------------------------------------------------------------------
# Features and forecasts
# ---------------------------------------------------------------------------


class Driver(_Base):
    name: str
    contribution: float
    weight: float


class SocialPressureRecord(_Base):
    market_id: str
    outcome_id: str
    window_start: datetime
    window_end: datetime
    index_value: float = Field(ge=0.0, le=1.0)
    n_posts: int = Field(ge=0)
    drivers: list[Driver] = Field(default_factory=list)

    @field_validator("window_end")
    @classmethod
    def _window_end_after_start(cls, end: datetime, info: Any) -> datetime:
        start = info.data.get("window_start")
        if start is not None and end < start:
            raise ValueError("window_end must be >= window_start")
        return end


class ProbabilityEstimate(_Base):
    market_id: str
    outcome_id: str
    timestamp: datetime
    market_probability: float = Field(ge=0.0, le=1.0)
    social_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    adjusted_probability: float = Field(ge=0.0, le=1.0)
    confidence: ConfidenceBand = ConfidenceBand.LOW
    delta: float
    explanation: list[str] = Field(default_factory=list)
    method: ForecastMethod = ForecastMethod.BASELINE_BLEND

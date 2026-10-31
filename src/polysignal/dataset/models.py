"""Pydantic models for the high-signal Telegram dataset.

These models are persistence-friendly (extra="forbid", JSON-roundtrip-safe)
and live alongside ``polysignal.domain.models`` instead of inside it
because they are specific to the "high-signal dataset" workflow and
should not pollute the core domain layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from polysignal.domain.enums import ResolutionResult


class HighSignalDomain(str, Enum):
    """High-level domain bucket used for balanced selection.

    These are coarser than :class:`polysignal.domain.MarketDomain`. They
    intentionally collapse crypto/economy into a single
    ``finance_macro_crypto`` bucket and sports/culture into
    ``sports_entertainment`` because the user-facing brief asks for that
    grouping.
    """

    POLITICS = "politics"
    FINANCE_MACRO_CRYPTO = "finance_macro_crypto"
    SPORTS_ENTERTAINMENT = "sports_entertainment"
    OTHER = "other"


@dataclass(frozen=True)
class DatasetTargets:
    """Configurable dataset-size targets.

    Defaults match the brief; the CLI lets the operator override them. Do
    not bake these constants into other modules — go through this object
    so the rest of the pipeline stays configuration-driven.
    """

    selected_market_target: int = 50
    raw_telegram_messages_target_min: int = 10_000
    raw_telegram_messages_target_max: int = 30_000
    candidate_segments_target_min: int = 5_000
    candidate_segments_target_max: int = 15_000
    linked_segments_target_min: int = 1_000
    linked_segments_target_max: int = 3_000
    minimum_viable_linked_segments: int = 500
    domain_split: dict[HighSignalDomain, tuple[int, int]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.domain_split is None:
            object.__setattr__(
                self,
                "domain_split",
                {
                    HighSignalDomain.POLITICS: (15, 20),
                    HighSignalDomain.FINANCE_MACRO_CRYPTO: (15, 20),
                    HighSignalDomain.SPORTS_ENTERTAINMENT: (10, 20),
                },
            )


DATASET_TARGETS = DatasetTargets()


class _Base(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class HighSignalMarket(_Base):
    """One selected Polymarket market enriched for Telegram search.

    The field set is intentionally a superset of what
    :class:`polysignal.domain.Market` exposes — the analytical layer
    needs extra metadata (scores, search window, canonical keywords)
    that does not belong on the raw ingestion model.
    """

    market_id: str
    slug: str
    question: str
    domain: HighSignalDomain
    polymarket_domain: str
    outcomes: list[str] = Field(default_factory=list)
    is_resolved: bool = False
    is_closed: bool = False
    resolution: ResolutionResult | None = None
    close_time: datetime | None = None
    resolved_at: datetime | None = None
    temporal_anchor: datetime | None = None
    temporal_anchor_kind: str | None = None
    created_at: datetime
    volume: float | None = None
    liquidity: float | None = None
    has_price_history: bool = False
    social_searchability_score: float = Field(ge=0.0, le=1.0)
    telegram_searchability_score: float = Field(ge=0.0, le=1.0)
    market_priority_score: float = Field(ge=0.0, le=1.0)
    selection_reason: str
    search_window_start: datetime
    search_window_end: datetime
    canonical_keywords: list[str] = Field(default_factory=list)
    entity_candidates: list[str] = Field(default_factory=list)
    outcome_aliases: list[str] = Field(default_factory=list)
    suggested_telegram_queries: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TelegramSearchPlanEntry(_Base):
    """One concrete Telegram search plan row.

    Each row is the unit of work for ingestion: "for this market, ask
    *this channel* for messages matching *this query* between *these
    timestamps*". Plans intentionally include a row with no channel id
    (``channel_id=None``) when the operator has not configured a channel
    list — that row documents the intended query without faking
    a target.
    """

    market_id: str
    work_id: str = ""
    channel_id: str | None = None
    channel_name: str | None = None
    channel_domain: str | None = None
    query: str
    query_type: str
    priority: int = Field(ge=0)
    temporal_anchor: datetime | None = None
    temporal_anchor_kind: str | None = None
    lookback_days: int = Field(default=90, ge=1)
    window_label: str = "0-90d"
    search_window_start: datetime
    search_window_end: datetime
    expected_relevance: str = "candidate"
    notes: str | None = None


class TelegramCoverageEntry(_Base):
    """Per-market Telegram coverage summary used in the coverage report."""

    market_id: str
    question: str
    domain: HighSignalDomain
    plan_entries: int = Field(ge=0)
    raw_messages: int = Field(ge=0)
    normalized_messages: int = Field(ge=0)
    segments: int = Field(ge=0)
    matched_keywords: list[str] = Field(default_factory=list)
    insufficient_coverage: bool = False
    notes: str | None = None


class TelegramCoverageReport(_Base):
    """Top-level coverage report written to JSON."""

    generated_at: datetime
    selected_markets_total: int
    selected_markets_by_domain: dict[str, int] = Field(default_factory=dict)
    telegram_channels_configured: int = Field(ge=0)
    telegram_channels_enabled: list[str] = Field(default_factory=list)
    raw_messages_collected: int = Field(ge=0)
    normalized_messages: int = Field(ge=0)
    segments_generated: int = Field(ge=0)
    plan_entries_total: int = Field(ge=0)
    per_market: list[TelegramCoverageEntry] = Field(default_factory=list)
    top_markets_by_coverage: list[str] = Field(default_factory=list)
    markets_with_insufficient_coverage: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    targets: dict[str, int] = Field(default_factory=dict)


__all__ = [
    "DATASET_TARGETS",
    "DatasetTargets",
    "HighSignalDomain",
    "HighSignalMarket",
    "TelegramCoverageEntry",
    "TelegramCoverageReport",
    "TelegramSearchPlanEntry",
]

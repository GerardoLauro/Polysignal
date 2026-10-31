"""Pydantic model for Branch 08 Social Pressure Index (SPI) features.

A :class:`SocialPressureFeatures` row is the parquet-friendly, per-market
output of the SPI pipeline. It is intentionally *flat* and *interpretable*:
every column is either a raw count, a deterministic ratio/entropy, or a
signed pressure value in a documented range, so the row can be explained
without re-running the pipeline.

Scope and policy
----------------
* The row is computed only from **linked evidence, stance labels,
  confidence, timestamps, and source/channel/author metadata**, plus
  graph-compatible counts derived from the existing Branch 07 graph-load
  policy. Resolved market labels (the YES/NO outcome) are **never** used
  as features — only the close/resolution *timestamp* is consumed, as a
  temporal anchor.
* ``dataset_mode`` records whether the row came from the real
  (``"real"``) or semi-synthetic (``"synthetic"``) dataset, so synthetic
  rows can never be mistaken for real empirical evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from polysignal.storage.parquet_store import read_parquet, write_parquet

# pressure_direction values
PRESSURE_YES = "YES"
PRESSURE_NO = "NO"
PRESSURE_MIXED = "MIXED"
PRESSURE_NEUTRAL = "NEUTRAL"
PRESSURE_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
PRESSURE_DIRECTIONS = (
    PRESSURE_YES,
    PRESSURE_NO,
    PRESSURE_MIXED,
    PRESSURE_NEUTRAL,
    PRESSURE_INSUFFICIENT,
)


class _Base(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class SocialPressureFeatures(_Base):
    """Per-market Social Pressure Index feature row (Branch 08)."""

    # --- market identifiers -------------------------------------------------
    market_id: str
    condition_id: str | None = None
    question: str = ""
    domain: str | None = None
    close_time: datetime | None = None
    dataset_mode: str = "real"

    # --- evidence counts ----------------------------------------------------
    accepted_link_count: int = Field(default=0, ge=0)
    linked_post_count: int = Field(default=0, ge=0)
    unique_post_count: int = Field(default=0, ge=0)
    unique_author_count: int = Field(default=0, ge=0)
    unique_channel_count: int = Field(default=0, ge=0)
    relevant_count: int = Field(default=0, ge=0)
    weakly_relevant_count: int = Field(default=0, ge=0)
    ambiguous_count: int = Field(default=0, ge=0)
    skipped_count: int = Field(default=0, ge=0)

    # --- stance counts (raw, from stance_labels.parquet) --------------------
    supports_yes_count: int = Field(default=0, ge=0)
    supports_no_count: int = Field(default=0, ge=0)
    neutral_stance_count: int = Field(default=0, ge=0)
    ambiguous_stance_count: int = Field(default=0, ge=0)
    directional_stance_count: int = Field(default=0, ge=0)

    # --- pressure features --------------------------------------------------
    yes_pressure_raw: float = 0.0
    no_pressure_raw: float = 0.0
    net_pressure_raw: float = 0.0
    yes_pressure_weighted: float = 0.0
    no_pressure_weighted: float = 0.0
    net_pressure_weighted: float = 0.0
    pressure_direction: str = PRESSURE_INSUFFICIENT
    pressure_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    stance_balance: float = Field(default=0.0, ge=-1.0, le=1.0)
    confidence_mean: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_weighted_score: float = Field(default=0.5, ge=0.0, le=1.0)
    # Laplace/Beta-shrunk P(YES) = (yes_w + a) / (yes_w + no_w + 2a). Unlike the
    # raw ``confidence_weighted_score``, this pulls thin-evidence markets toward
    # 0.5 so a market with 1-2 directional posts is not reported as a 0/1
    # certainty; it converges to the raw score as directional evidence grows.
    spi_score_shrunk: float = Field(default=0.5, ge=0.0, le=1.0)

    # --- temporal features (each net balance in [-1, 1]) -------------------
    pressure_0_7d: float = Field(default=0.0, ge=-1.0, le=1.0)
    pressure_8_30d: float = Field(default=0.0, ge=-1.0, le=1.0)
    pressure_31_90d: float = Field(default=0.0, ge=-1.0, le=1.0)
    pressure_91_180d: float = Field(default=0.0, ge=-1.0, le=1.0)
    temporal_decay_pressure: float = Field(default=0.0, ge=-1.0, le=1.0)
    early_pressure: float = Field(default=0.0, ge=-1.0, le=1.0)
    late_pressure: float = Field(default=0.0, ge=-1.0, le=1.0)
    pressure_acceleration: float = Field(default=0.0, ge=-2.0, le=2.0)

    # --- quality features ---------------------------------------------------
    evidence_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_diversity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    channel_entropy: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    accepted_evidence_ratio: float = Field(default=0.0, ge=0.0, le=1.0)

    # --- graph-derived features (Branch 07 graph-load policy) ---------------
    graph_linked_post_count: int = Field(default=0, ge=0)
    graph_supports_count: int = Field(default=0, ge=0)
    graph_opposes_count: int = Field(default=0, ge=0)
    graph_author_count: int = Field(default=0, ge=0)
    graph_channel_count: int = Field(default=0, ge=0)
    graph_neighborhood_size: int = Field(default=0, ge=0)

    # --- engagement features (SECONDARY; log-scaled, never replace the SPI) --
    # Aggregated over the accepted directional stances' parent posts. Raw
    # engagement counts are heavy-tailed, so the weighted variants use
    # ``log1p`` to keep a single viral post from dominating the signal. The
    # primary, confidence-weighted SPI above is unaffected by these columns.
    engagement_total: float = Field(default=0.0, ge=0.0)
    engagement_mean: float = Field(default=0.0, ge=0.0)
    engagement_max: float = Field(default=0.0, ge=0.0)
    engagement_log_total: float = Field(default=0.0, ge=0.0)
    yes_pressure_engagement_weighted: float = Field(default=0.0, ge=0.0)
    no_pressure_engagement_weighted: float = Field(default=0.0, ge=0.0)
    net_pressure_engagement_weighted: float = 0.0


def write_social_pressure_features(
    rows: Sequence[SocialPressureFeatures], path: Path
) -> Path:
    """Persist SPI feature rows to a parquet file (overwrite-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(list(rows), path)
    return path


def read_social_pressure_features(path: Path) -> list[SocialPressureFeatures]:
    """Read an SPI feature parquet file back into model objects."""
    return read_parquet(path, SocialPressureFeatures)


__all__ = [
    "PRESSURE_DIRECTIONS",
    "PRESSURE_INSUFFICIENT",
    "PRESSURE_MIXED",
    "PRESSURE_NEUTRAL",
    "PRESSURE_NO",
    "PRESSURE_YES",
    "SocialPressureFeatures",
    "read_social_pressure_features",
    "write_social_pressure_features",
]

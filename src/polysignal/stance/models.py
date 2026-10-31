"""Pydantic models for outcome-aware stance records (Branches 06A-06E).

A :class:`StanceRecord` is the parquet-friendly row produced by the
stance pipeline. It deliberately tracks far more provenance than the
older :class:`polysignal.domain.Stance` (which is the lightweight
in-memory model used by the scaffold heuristic): downstream consumers
(SPI, forecaster, dashboard) need to be able to trace each stance back
to its segment, parent post, and the relevance verdict that admitted
it.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from polysignal.domain.enums import (
    ConstraintMatch,
    Platform,
    RelevanceLabel,
    StanceLabel,
    StanceMethod,
)
from polysignal.storage.parquet_store import read_parquet, write_parquet


class HumanStanceLabel(str, Enum):
    """Optional human-applied stance label used by the evaluation step."""

    SUPPORTS_YES = "supports_yes"
    SUPPORTS_NO = "supports_no"
    NEUTRAL = "neutral"
    AMBIGUOUS = "ambiguous"
    UNRELATED = "unrelated"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class _Base(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class StanceRecord(_Base):
    """One stance verdict for a (market, post/segment) pair.

    ``stance_score`` is a signed value in [-1, 1] where +1 means strong
    YES support and -1 means strong NO support. For Branch 06D
    multi-outcome rows, YES / NO are relative to ``target_outcome_name``.
    Neutral / ambiguous / insufficient_context all encode as
    ``stance_score = 0.0``.
    """

    stance_id: str
    market_id: str
    post_id: str
    segment_id: str | None = None
    parent_post_id: str | None = None
    outcome_id: str | None = None
    target_outcome: str = "yes"
    target_outcome_index: int | None = Field(default=None, ge=0)
    target_outcome_name: str | None = None
    target_aliases: list[str] = Field(default_factory=list)
    is_multi_outcome: bool = False
    stance_label: StanceLabel
    stance_score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    method: StanceMethod = StanceMethod.HEURISTIC
    explanation: str = ""
    matched_yes_cues: list[str] = Field(default_factory=list)
    matched_no_cues: list[str] = Field(default_factory=list)
    matched_uncertainty_cues: list[str] = Field(default_factory=list)
    negated_cues: list[str] = Field(default_factory=list)
    created_at: datetime
    post_created_at: datetime | None = None
    market_question: str = ""
    outcome_names: list[str] = Field(default_factory=list)
    segment_text_preview: str = ""
    relevance_label: RelevanceLabel | None = None
    relevance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    link_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    constraint_match: ConstraintMatch | None = None
    source_platform: Platform | None = None
    source_id: str | None = None
    source_channel: str | None = None
    # Branch 06B audit fields — populated only when the optional LLM
    # stance judge ran for this row. Default ``used_llm=False`` keeps
    # the parquet schema backward-compatible with 06A consumers.
    used_llm: bool = False
    llm_label: StanceLabel | None = None
    llm_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    llm_rationale: str | None = None
    llm_cache_key: str | None = None
    llm_cache_hit: bool = False
    llm_invalid_output: bool = False
    llm_applied: bool = False
    llm_provider: str | None = None
    llm_model: str | None = None


def write_stance_records(rows: Sequence[StanceRecord], path: Path) -> Path:
    """Persist stance records to a parquet file (overwrite-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(list(rows), path)
    return path


def read_stance_records(path: Path) -> list[StanceRecord]:
    """Read a stance parquet file back into model objects."""
    return read_parquet(path, StanceRecord)


__all__ = [
    "HumanStanceLabel",
    "StanceRecord",
    "read_stance_records",
    "write_stance_records",
]

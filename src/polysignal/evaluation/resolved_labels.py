"""Resolved-market evaluation labels (Branch 07 upstream-data milestone).

This module derives a canonical *resolved-only* view from the normalized
Polymarket datasets. The project currently focuses on already-resolved
markets for evaluation; active/open markets are deferred until after the
full pipeline is complete.

Scientific constraint
---------------------
A :class:`ResolvedMarketLabel` is an **evaluation target only**. The
winning outcome must never leak into social retrieval, query generation,
linking, stance classification, the Social Pressure Index, feature
engineering, or any prediction input. It exists so a later evaluation
step can score forecasts against ground truth.

The builder is a pure function over :class:`~polysignal.domain.Market`
and :class:`~polysignal.domain.Outcome` models, so it is fully offline,
deterministic, and unit-testable. It never contacts an API and never
fabricates a result: a market is only emitted when the upstream
normalizer already marked it resolved (``is_resolved``) or carried an
explicit ``resolution`` enum.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from polysignal.domain import Market, Outcome
from polysignal.domain.enums import ResolutionResult
from polysignal.storage.parquet_store import dataframe_to_models


class ResolvedMarketLabel(BaseModel):
    """One resolved market's ground-truth label (evaluation target only)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    market_id: str
    condition_id: str | None = None
    slug: str = ""
    question: str = ""
    domain: str = "other"
    created_at: datetime | None = None
    close_time: datetime | None = None
    resolved_at: datetime | None = None
    is_resolved: bool = False
    is_closed: bool = False
    resolution: str | None = None  # yes / no / invalid / multi_outcome
    winning_outcome_label: str | None = None
    winning_outcome_id: str | None = None
    winning_outcome_index: int | None = Field(default=None, ge=0)
    binary_yes_no: str | None = None  # "YES" / "NO" for binary YES/NO markets
    n_outcomes: int = Field(default=0, ge=0)
    # Which upstream fields determined the recorded result — provenance so
    # a reviewer can see *why* this market is considered resolved.
    source_fields: list[str] = Field(default_factory=list)


def _is_resolved(market: Market) -> bool:
    """A market counts as resolved when the normalizer said so.

    We accept either the explicit ``is_resolved`` flag or a populated
    ``resolution`` enum; both are set by
    :func:`polysignal.ingestion.polymarket.normalizer.normalize_market`
    only when the upstream Gamma payload reports a settled market.
    """
    return bool(market.is_resolved) or market.resolution is not None


def _winning_outcome(
    market: Market,
    outcomes: Sequence[Outcome],
) -> tuple[Outcome | None, str | None]:
    """Resolve the winning outcome and YES/NO mapping for a binary market.

    Returns ``(outcome, "YES"|"NO"|None)``. For non-binary or invalid
    resolutions the winner is ``None`` — we never guess.
    """
    if market.resolution is ResolutionResult.YES:
        wanted, binary = "yes", "YES"
    elif market.resolution is ResolutionResult.NO:
        wanted, binary = "no", "NO"
    else:
        return None, None
    match = next(
        (o for o in outcomes if (o.label or "").strip().lower() == wanted),
        None,
    )
    return match, binary


def build_resolved_labels(
    markets: Sequence[Market],
    outcomes: Sequence[Outcome] | None = None,
) -> list[ResolvedMarketLabel]:
    """Build resolved-only evaluation labels from normalized markets.

    ``outcomes`` may be supplied separately (the normalized
    ``outcomes.parquet``) or omitted, in which case each market's inline
    ``outcomes`` list is used. Only markets the upstream normalizer
    marked resolved are emitted.
    """
    by_market: dict[str, list[Outcome]] = {}
    for outcome in outcomes or []:
        if outcome.market_id:
            by_market.setdefault(outcome.market_id, []).append(outcome)

    labels: list[ResolvedMarketLabel] = []
    seen: set[str] = set()
    for market in markets:
        if not _is_resolved(market):
            continue
        # Gamma pagination can return the same market twice; the resolved
        # evaluation set must carry one row per market_id.
        if market.market_id in seen:
            continue
        seen.add(market.market_id)
        market_outcomes = by_market.get(market.market_id) or list(market.outcomes)
        market_outcomes = sorted(market_outcomes, key=lambda o: o.index)
        winner, binary = _winning_outcome(market, market_outcomes)

        source_fields: list[str] = []
        if market.is_closed:
            source_fields.append("is_closed")
        if market.is_resolved:
            source_fields.append("is_resolved")
        if market.resolution is not None:
            source_fields.append("resolution")
        if market.resolved_at is not None:
            source_fields.append("resolved_at")

        labels.append(
            ResolvedMarketLabel(
                market_id=market.market_id,
                condition_id=market.condition_id or market.market_id,
                slug=market.slug,
                question=market.question,
                domain=market.domain.value,
                created_at=market.created_at,
                close_time=market.close_time,
                resolved_at=market.resolved_at,
                is_resolved=market.is_resolved,
                is_closed=market.is_closed,
                resolution=market.resolution.value if market.resolution else None,
                winning_outcome_label=winner.label if winner else None,
                winning_outcome_id=winner.outcome_id if winner else None,
                winning_outcome_index=winner.index if winner else None,
                binary_yes_no=binary,
                n_outcomes=len(market_outcomes),
                source_fields=source_fields,
            )
        )
    return labels


def write_resolved_labels(labels: Sequence[ResolvedMarketLabel], path: Path) -> Path:
    """Write resolved labels to Parquet, preserving a stable schema when empty.

    When there are no resolved markets we still write a column-stable
    empty Parquet so downstream readers and the quality gate can detect
    the file's presence and its (zero) row count without a schema error.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if labels:
        frame = pd.DataFrame([label.model_dump(mode="json") for label in labels])
    else:
        frame = pd.DataFrame({name: pd.Series(dtype="object") for name in ResolvedMarketLabel.model_fields})
    frame.to_parquet(path, index=False)
    return path


def read_resolved_labels(path: Path) -> list[ResolvedMarketLabel]:
    """Read resolved labels back into models (empty list when absent/empty)."""
    if not path.exists():
        return []
    frame = pd.read_parquet(path)
    if frame.empty:
        return []
    return dataframe_to_models(frame, ResolvedMarketLabel)


__all__ = [
    "ResolvedMarketLabel",
    "build_resolved_labels",
    "read_resolved_labels",
    "write_resolved_labels",
]

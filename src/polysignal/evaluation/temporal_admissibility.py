"""Shared temporal-admissibility rules for historical social evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from polysignal.domain import Market


@dataclass(frozen=True)
class TemporalAdmissibility:
    """Decision envelope for one post relative to one market."""

    admissible: bool
    temporal_valid: bool
    reason: str | None
    anchor: datetime | None
    anchor_kind: str | None


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def market_temporal_anchor(market: Market) -> tuple[datetime | None, str | None]:
    """Return the conservative historical cutoff for a market."""
    if market.close_time is not None:
        return _utc(market.close_time), "close_time"
    if market.resolved_at is not None:
        return _utc(market.resolved_at), "resolved_at"
    return None, None


def is_historical_market(market: Market) -> bool:
    """Whether historical evidence requires an explicit temporal anchor."""
    return bool(market.is_closed or market.is_resolved or market.resolution is not None)


def assess_post_admissibility(
    market: Market,
    post_created_at: datetime | None,
    *,
    window_start: datetime | None = None,
) -> TemporalAdmissibility:
    """Accept active-market evidence or require pre-close historical evidence."""
    anchor, anchor_kind = market_temporal_anchor(market)
    if post_created_at is None:
        return TemporalAdmissibility(
            False,
            False,
            "post_timestamp_missing",
            anchor,
            anchor_kind,
        )
    if window_start is not None and _utc(post_created_at) < _utc(window_start):
        return TemporalAdmissibility(False, False, "outside_window", anchor, anchor_kind)
    if not is_historical_market(market):
        return TemporalAdmissibility(True, True, None, anchor, anchor_kind)
    if anchor is None:
        return TemporalAdmissibility(False, False, "temporal_anchor_missing", None, None)
    if _utc(post_created_at) > anchor:
        return TemporalAdmissibility(False, False, "post_close", anchor, anchor_kind)
    return TemporalAdmissibility(True, True, None, anchor, anchor_kind)


__all__ = [
    "TemporalAdmissibility",
    "assess_post_admissibility",
    "is_historical_market",
    "market_temporal_anchor",
]

"""Temporal aggregation helpers for the Social Pressure Index (Branch 08).

These functions are pure and deterministic. They operate on a flat list of
``TimedPressure`` items — one per accepted directional stance — where each
item carries:

* ``days_before``: days between the post timestamp and the market's temporal
  anchor (close/resolution time). Clamped to ``>= 0``; a post created after
  the anchor is treated as ``0`` days before (most recent).
* ``is_yes``: whether the stance supports YES (``True``) or NO (``False``).
* ``weight``: a non-negative weight (typically the stance confidence).

Every aggregation returns a *net balance* in ``[-1, 1]``:

    net = (yes_weight - no_weight) / (yes_weight + no_weight)

so ``+1`` means unanimous YES pressure, ``-1`` unanimous NO pressure, and
``0`` either a perfectly balanced window or an empty one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from statistics import median


@dataclass(frozen=True)
class TimedPressure:
    """One accepted directional stance positioned in time."""

    days_before: float
    is_yes: bool
    weight: float


def days_before(anchor: datetime, created_at: datetime) -> float:
    """Days between ``created_at`` and a later ``anchor`` (clamped at 0)."""
    delta = (anchor - created_at).total_seconds() / 86400.0
    return delta if delta > 0 else 0.0


def net_balance(items: list[TimedPressure]) -> float:
    """Net YES/NO balance of ``items`` in ``[-1, 1]`` (0 when empty)."""
    yes = sum(it.weight for it in items if it.is_yes)
    no = sum(it.weight for it in items if not it.is_yes)
    total = yes + no
    if total <= 0.0:
        return 0.0
    return (yes - no) / total


def window_pressure(items: list[TimedPressure], lo: float, hi: float) -> float:
    """Net balance over items with ``lo < days_before <= hi``.

    The first bucket should pass a negative ``lo`` so that posts exactly at
    the anchor (``days_before == 0``) are included.
    """
    in_window = [it for it in items if lo < it.days_before <= hi]
    return round(net_balance(in_window), 6)


def decay_pressure(items: list[TimedPressure], half_life_days: float) -> float:
    """Recency-weighted net balance using an exponential decay envelope.

    Each item's weight is multiplied by ``0.5 ** (days_before / half_life)``
    so evidence far from the anchor contributes near zero. A non-positive
    half-life collapses to the most-recent evidence only.
    """
    if not items:
        return 0.0
    decayed: list[TimedPressure] = []
    for it in items:
        if half_life_days <= 0.0:
            factor = 1.0 if it.days_before <= 0.0 else 0.0
        else:
            factor = 0.5 ** (it.days_before / half_life_days)
        decayed.append(TimedPressure(it.days_before, it.is_yes, it.weight * factor))
    return round(net_balance(decayed), 6)


def early_late_pressure(items: list[TimedPressure]) -> tuple[float, float]:
    """Return ``(early_pressure, late_pressure)`` split at the median time.

    ``late`` covers the half closest to the anchor (small ``days_before``);
    ``early`` covers the older half. With fewer than two items both equal the
    overall balance so ``pressure_acceleration`` is zero.
    """
    if len(items) < 2:
        overall = net_balance(items)
        return overall, overall
    cut = median(it.days_before for it in items)
    late = [it for it in items if it.days_before <= cut]
    early = [it for it in items if it.days_before > cut]
    early_val = net_balance(early) if early else 0.0
    late_val = net_balance(late) if late else 0.0
    return round(early_val, 6), round(late_val, 6)


def pressure_acceleration(early: float, late: float) -> float:
    """Change in directional lean from early to late evidence (``[-2, 2]``)."""
    return round(late - early, 6)


def standard_windows(window_days: tuple[int, ...], horizon_days: int) -> list[tuple[float, float]]:
    """Build inclusive-upper time buckets from cut points.

    ``window_days=(7, 30, 90)`` with ``horizon_days=180`` yields buckets
    ``[0-7], (7-30], (30-90], (90-180]``. The first ``lo`` is negative so the
    anchor itself is captured.
    """
    cuts = sorted(window_days)
    edges = [-math.inf, *cuts, horizon_days]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


__all__ = [
    "TimedPressure",
    "days_before",
    "decay_pressure",
    "early_late_pressure",
    "net_balance",
    "pressure_acceleration",
    "standard_windows",
    "window_pressure",
]

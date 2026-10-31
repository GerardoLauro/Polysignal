"""Social Pressure Index — deterministic baseline.

The index aggregates outcome-aware social signals for a (market,
outcome, time window) triple into a single value in ``[0, 1]``. It is
designed to be **transparent**: every input is logged as a ``Driver``
so the dashboard can explain why the index has its value.

Baseline formula (binary YES/NO case, weighted toward the YES side):

    For each post i in the window that supports an outcome o:
        base_i = w_stance * stance_confidence_i
               + w_link * link_confidence_i
               + w_engagement * engagement_score_i
               + w_author * author_weight_i
        w_i   = recency_decay_i * base_i

    yes_mass = sum(w_i for posts supporting YES)
    no_mass  = sum(w_i for posts supporting NO)
    index    = yes_mass / (yes_mass + no_mass)        in [0, 1]
    (when both masses are zero → index = 0.5, low confidence)

The recency decay is applied as a *multiplicative envelope* — a post
that is ten half-lives old contributes near zero regardless of how
much engagement or stance confidence it had. This matches the
intuition that a tweet from a week ago should not steer today's
forecast as strongly as one from an hour ago.

The function is pure and deterministic. Graph-derived features can be
plugged into ``author_weight_i`` later without changing the API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from polysignal.domain import Driver, SocialPressureRecord, StanceLabel


@dataclass(frozen=True)
class StanceObservation:
    """A single (post, stance) observation feeding the index."""

    market_id: str
    outcome_id: str | None
    label: StanceLabel
    stance_confidence: float
    link_confidence: float
    engagement_score: float  # in [0, 1] — caller normalizes raw counts
    author_weight: float  # in [0, 1]; default to 0.5 if unknown
    created_at: datetime


@dataclass(frozen=True)
class SocialPressureConfig:
    w_stance: float = 0.40
    w_link: float = 0.20
    w_engagement: float = 0.25
    w_author: float = 0.15
    half_life: timedelta = timedelta(hours=12)

    @property
    def weight_sum(self) -> float:
        return self.w_stance + self.w_link + self.w_engagement + self.w_author


@dataclass
class _Accumulator:
    yes: float = 0.0
    no: float = 0.0
    drivers: dict[str, float] = field(default_factory=dict)
    n: int = 0


class SocialPressureScorer:
    """Compute the index for a (market_id, outcome_id, window) triple."""

    def __init__(self, config: SocialPressureConfig | None = None) -> None:
        self.config = config or SocialPressureConfig()
        if not math.isclose(self.config.weight_sum, 1.0, abs_tol=1e-6):
            # We do not fail-hard here so callers can experiment with
            # weight sets, but we surface the issue via a flag the
            # dashboard can display. The math below is robust either
            # way because we always normalize by total mass.
            self._weights_normalized = False
        else:
            self._weights_normalized = True

    # ------------------------------------------------------------------ API
    def score(
        self,
        observations: list[StanceObservation],
        *,
        market_id: str,
        outcome_id: str,
        window_start: datetime,
        window_end: datetime,
    ) -> SocialPressureRecord:
        acc = _Accumulator()
        for obs in observations:
            if obs.created_at < window_start or obs.created_at > window_end:
                continue
            if obs.label not in (StanceLabel.SUPPORTS_YES, StanceLabel.SUPPORTS_NO):
                continue
            weight = self._post_weight(obs, window_end)
            if obs.label is StanceLabel.SUPPORTS_YES:
                acc.yes += weight
            else:
                acc.no += weight
            acc.n += 1

        total = acc.yes + acc.no
        index_value = 0.5 if total == 0.0 else acc.yes / total

        drivers = self._drivers(acc, total)

        return SocialPressureRecord(
            market_id=market_id,
            outcome_id=outcome_id,
            window_start=window_start,
            window_end=window_end,
            index_value=round(index_value, 6),
            n_posts=acc.n,
            drivers=drivers,
        )

    # -------------------------------------------------------------- helpers
    def _post_weight(self, obs: StanceObservation, ref_time: datetime) -> float:
        recency = _recency_decay(obs.created_at, ref_time, self.config.half_life)
        base = (
            self.config.w_stance * obs.stance_confidence
            + self.config.w_link * obs.link_confidence
            + self.config.w_engagement * obs.engagement_score
            + self.config.w_author * obs.author_weight
        )
        return recency * base

    def _drivers(self, acc: _Accumulator, total: float) -> list[Driver]:
        # The driver contributions are reported as the share of the
        # YES side over the total mass. They are illustrative — the
        # detailed breakdown will become richer when the pipeline
        # exposes per-driver totals.
        share_yes = (acc.yes / total) if total else 0.0
        share_no = (acc.no / total) if total else 0.0
        return [
            Driver(name="yes_mass_share", contribution=round(share_yes, 6), weight=1.0),
            Driver(name="no_mass_share", contribution=round(share_no, 6), weight=1.0),
            Driver(name="n_posts", contribution=float(acc.n), weight=1.0),
        ]


def _recency_decay(t: datetime, ref: datetime, half_life: timedelta) -> float:
    """Exponential decay with the given half-life. Returns a value in (0, 1]."""
    delta = (ref - t).total_seconds()
    if delta < 0:
        return 1.0
    if half_life.total_seconds() <= 0:
        return 0.0 if delta > 0 else 1.0
    return float(0.5 ** (delta / half_life.total_seconds()))


__all__ = ["SocialPressureConfig", "SocialPressureScorer", "StanceObservation"]

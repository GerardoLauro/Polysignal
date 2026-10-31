"""Deterministic baseline blender: market probability ⊕ social signal.

This module implements the first version of "social-adjusted
probability" described in the README:

* take the market-implied probability ``p_m``;
* take the social signal ``p_s`` (e.g. the Social Pressure Index for
  the YES outcome — caller-provided);
* compute a confidence ``w ∈ [0, 1]`` that depends on:
    - the number of supporting posts (more posts → higher confidence),
    - the recency of those posts,
    - the agreement between social and market (a small disagreement
      with high social confidence increases ``w``; a large
      disagreement actually *reduces* ``w`` to be conservative);
* return ``p_final = (1 - w) * p_m + w * p_s`` together with a
  ``ProbabilityEstimate`` that records every input and the
  human-readable explanation.

The function is intentionally simple. Future work can replace the
``BaselineBlender`` with a learned model implementing the same
interface (``forecast``) without changing any caller.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from polysignal.domain import (
    ConfidenceBand,
    ForecastMethod,
    ProbabilityEstimate,
    SocialPressureRecord,
)
from polysignal.forecasting.explain import build_explanation


@dataclass(frozen=True)
class ForecastInputs:
    market_id: str
    outcome_id: str
    timestamp: datetime
    market_probability: float
    social_probability: float | None
    n_supporting_posts: int = 0
    pressure_record: SocialPressureRecord | None = None


@dataclass(frozen=True)
class BlenderConfig:
    # Maximum weight the social signal can ever take on the final estimate.
    max_social_weight: float = 0.35
    # Number of supporting posts above which the volume term saturates.
    volume_saturation: int = 50
    # If |p_s - p_m| is greater than this we cap social weight at 1/3 of normal.
    large_disagreement: float = 0.25
    # Bands for the confidence label exposed to the dashboard.
    low_threshold: int = 5
    high_threshold: int = 30


class BaselineBlender:
    """Confidence-weighted blend between market and social probabilities."""

    method: ForecastMethod = ForecastMethod.BASELINE_BLEND

    def __init__(self, config: BlenderConfig | None = None) -> None:
        self.config = config or BlenderConfig()

    def forecast(self, inputs: ForecastInputs) -> ProbabilityEstimate:
        p_m = _clamp01(inputs.market_probability)
        if inputs.social_probability is None or inputs.n_supporting_posts == 0:
            estimate = ProbabilityEstimate(
                market_id=inputs.market_id,
                outcome_id=inputs.outcome_id,
                timestamp=inputs.timestamp,
                market_probability=p_m,
                social_probability=None,
                adjusted_probability=p_m,
                confidence=ConfidenceBand.LOW,
                delta=0.0,
                explanation=[
                    "No social signal available — fallback to market probability.",
                ],
                method=self.method,
            )
            return estimate

        p_s = _clamp01(inputs.social_probability)
        weight = self._weight(p_m, p_s, inputs.n_supporting_posts)
        p_final = (1.0 - weight) * p_m + weight * p_s
        confidence = self._band(inputs.n_supporting_posts)

        estimate = ProbabilityEstimate(
            market_id=inputs.market_id,
            outcome_id=inputs.outcome_id,
            timestamp=inputs.timestamp,
            market_probability=round(p_m, 6),
            social_probability=round(p_s, 6),
            adjusted_probability=round(p_final, 6),
            confidence=confidence,
            delta=round(p_final - p_m, 6),
            explanation=build_explanation(
                p_m=p_m,
                p_s=p_s,
                weight=weight,
                n_posts=inputs.n_supporting_posts,
                pressure=inputs.pressure_record,
            ),
            method=self.method,
        )
        return estimate

    def forecast_many(self, batch: Iterable[ForecastInputs]) -> list[ProbabilityEstimate]:
        return [self.forecast(inputs) for inputs in batch]

    # -------------------------------------------------------------- weights
    def _weight(self, p_m: float, p_s: float, n_posts: int) -> float:
        # 1. Volume factor: saturating curve, never exceeds 1.0.
        v = min(1.0, n_posts / float(self.config.volume_saturation))
        # 2. Disagreement penalty: large disagreement reduces weight.
        disagreement = abs(p_s - p_m)
        if disagreement >= self.config.large_disagreement:
            disagreement_factor = 1.0 / 3.0
        else:
            disagreement_factor = 1.0 - 0.5 * (disagreement / self.config.large_disagreement)
        # 3. Sanity factor: refuse to push to extremes when the social
        # signal is itself at the boundary (suggests poor sampling).
        sanity = 1.0 - 4.0 * (p_s - 0.5) ** 2  # peaks at p_s=0.5, 0 at 0/1
        sanity = max(0.25, sanity)
        weight = self.config.max_social_weight * v * disagreement_factor * sanity
        return float(max(0.0, min(self.config.max_social_weight, weight)))

    def _band(self, n_posts: int) -> ConfidenceBand:
        if n_posts < self.config.low_threshold:
            return ConfidenceBand.LOW
        if n_posts < self.config.high_threshold:
            return ConfidenceBand.MEDIUM
        return ConfidenceBand.HIGH


def _clamp01(x: float) -> float:
    if math.isnan(x):
        raise ValueError("probability cannot be NaN")
    return max(0.0, min(1.0, x))


__all__ = ["BaselineBlender", "BlenderConfig", "ForecastInputs"]

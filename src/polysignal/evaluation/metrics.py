"""Deterministic probabilistic-forecasting metrics.

All functions accept the same two-argument signature:

    metric(probabilities: Sequence[float], outcomes: Sequence[int]) -> float

* ``probabilities`` is the predicted probability of the positive class
  for each example, in ``[0, 1]``;
* ``outcomes`` is the realized binary outcome, ``0`` or ``1``.

The functions are intentionally pure-Python so that the test suite can
verify them against hand-computed values without importing
scikit-learn. The numbers match scikit-learn's definitions within
floating-point tolerance.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def _validate(probs: Sequence[float], outcomes: Sequence[int]) -> None:
    if len(probs) != len(outcomes):
        raise ValueError("probabilities and outcomes must have the same length")
    if len(probs) == 0:
        raise ValueError("at least one observation is required")
    for p in probs:
        if math.isnan(p) or not (0.0 <= p <= 1.0):
            raise ValueError(f"probability out of range: {p}")
    for y in outcomes:
        if y not in (0, 1):
            raise ValueError(f"outcome must be 0 or 1, got {y}")


def brier_score(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean squared error between predicted probabilities and outcomes."""
    _validate(probs, outcomes)
    return sum((p - y) ** 2 for p, y in zip(probs, outcomes, strict=True)) / len(probs)


def log_loss(
    probs: Sequence[float],
    outcomes: Sequence[int],
    *,
    eps: float = 1e-15,
) -> float:
    """Negative log-likelihood. Clipped by ``eps`` to avoid ``log(0)``."""
    _validate(probs, outcomes)
    total = 0.0
    for p, y in zip(probs, outcomes, strict=True):
        p_clipped = min(max(p, eps), 1 - eps)
        total -= y * math.log(p_clipped) + (1 - y) * math.log(1 - p_clipped)
    return total / len(probs)


def accuracy(
    probs: Sequence[float],
    outcomes: Sequence[int],
    *,
    threshold: float = 0.5,
) -> float:
    """Fraction of correct predictions at the given decision threshold."""
    _validate(probs, outcomes)
    correct = sum(
        1 for p, y in zip(probs, outcomes, strict=True) if (1 if p >= threshold else 0) == y
    )
    return correct / len(probs)


def _confusion_at_threshold(
    probs: Sequence[float],
    outcomes: Sequence[int],
    threshold: float,
) -> tuple[int, int, int, int]:
    """Return ``(tp, fp, fn, tn)`` for the positive class at ``threshold``."""
    tp = fp = fn = tn = 0
    for p, y in zip(probs, outcomes, strict=True):
        pred = 1 if p >= threshold else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 1:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    """F1 from per-class counts; defined as 0.0 when precision+recall is 0."""
    denominator = 2 * tp + fp + fn
    return (2 * tp) / denominator if denominator else 0.0


def f1_per_class(
    probs: Sequence[float],
    outcomes: Sequence[int],
    *,
    threshold: float = 0.5,
) -> dict[int, float]:
    """Per-class F1 for the binary problem at ``threshold`` (keys ``0`` and ``1``).

    Class ``1`` is the positive class; class ``0`` is scored symmetrically by
    swapping the roles of false positives and false negatives.
    """
    _validate(probs, outcomes)
    tp, fp, fn, tn = _confusion_at_threshold(probs, outcomes, threshold)
    return {
        1: _f1_from_counts(tp, fp, fn),
        0: _f1_from_counts(tn, fn, fp),
    }


def macro_f1(
    probs: Sequence[float],
    outcomes: Sequence[int],
    *,
    threshold: float = 0.5,
) -> float:
    """Unweighted mean of the per-class F1 scores (macro-averaged F1)."""
    per_class = f1_per_class(probs, outcomes, threshold=threshold)
    return (per_class[0] + per_class[1]) / 2.0


def expected_calibration_error(
    probs: Sequence[float],
    outcomes: Sequence[int],
    *,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error with equal-width bins.

    Returns the weighted absolute gap between mean predicted
    probability and mean observed frequency, averaged over bins.
    """
    _validate(probs, outcomes)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in zip(probs, outcomes, strict=True):
        idx = min(n_bins - 1, int(p * n_bins))
        bins[idx].append((p, y))
    total = float(len(probs))
    ece = 0.0
    for bin_items in bins:
        if not bin_items:
            continue
        mean_p = sum(p for p, _ in bin_items) / len(bin_items)
        mean_y = sum(y for _, y in bin_items) / len(bin_items)
        ece += (len(bin_items) / total) * abs(mean_p - mean_y)
    return ece


def calibration_bins(
    probs: Sequence[float],
    outcomes: Sequence[int],
    *,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Return populated equal-width calibration bins for inspection/export."""
    _validate(probs, outcomes)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for probability, outcome in zip(probs, outcomes, strict=True):
        index = min(n_bins - 1, int(probability * n_bins))
        bins[index].append((probability, outcome))
    rows: list[dict[str, Any]] = []
    for index, items in enumerate(bins):
        if not items:
            continue
        mean_probability = sum(probability for probability, _ in items) / len(items)
        observed_rate = sum(outcome for _, outcome in items) / len(items)
        rows.append(
            {
                "bin_index": index,
                "bin_lower": index / n_bins,
                "bin_upper": (index + 1) / n_bins,
                "count": len(items),
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
                "absolute_error": abs(mean_probability - observed_rate),
            }
        )
    return rows


__all__ = [
    "accuracy",
    "brier_score",
    "calibration_bins",
    "expected_calibration_error",
    "f1_per_class",
    "log_loss",
    "macro_f1",
]

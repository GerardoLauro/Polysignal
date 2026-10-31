"""Evaluation of stance predictions against optional human labels.

The evaluator is intentionally tolerant: when no human labels are
supplied, it produces a coverage-only report with a warning rather
than raising. That keeps the CLI runnable end-to-end on a fresh
checkout where no review has happened yet.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polysignal.domain import StanceLabel
from polysignal.stance.models import HumanStanceLabel, StanceRecord


@dataclass(frozen=True)
class _PerLabelMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class StanceEvaluation:
    """Top-level evaluation result."""

    total_predictions: int
    total_with_labels: int
    accuracy: float | None
    macro_f1: float | None
    coverage: float
    abstention_rate: float
    per_label_counts: dict[str, int] = field(default_factory=dict)
    per_label_metrics: dict[str, _PerLabelMetrics] = field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_predictions": self.total_predictions,
            "total_with_labels": self.total_with_labels,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "coverage": self.coverage,
            "abstention_rate": self.abstention_rate,
            "per_label_counts": dict(self.per_label_counts),
            "per_label_metrics": {
                label: {
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "support": metrics.support,
                }
                for label, metrics in self.per_label_metrics.items()
            },
            "confusion_matrix": {
                gold: dict(predicted) for gold, predicted in self.confusion_matrix.items()
            },
            "warnings": list(self.warnings),
        }


def _per_label_counts(records: Sequence[StanceRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = record.stance_label.value
        counts[key] = counts.get(key, 0) + 1
    return counts


_ABSTAIN_LABELS = {
    StanceLabel.AMBIGUOUS,
    StanceLabel.NEUTRAL,
    StanceLabel.INSUFFICIENT_CONTEXT,
    StanceLabel.UNRELATED,
}


def _abstention_rate(records: Sequence[StanceRecord]) -> float:
    if not records:
        return 0.0
    abstained = sum(1 for r in records if r.stance_label in _ABSTAIN_LABELS)
    return abstained / len(records)


def _coverage(records: Sequence[StanceRecord]) -> float:
    if not records:
        return 0.0
    directional = sum(
        1 for r in records
        if r.stance_label in {StanceLabel.SUPPORTS_YES, StanceLabel.SUPPORTS_NO}
    )
    return directional / len(records)


def _macro_f1(
    per_label: dict[str, _PerLabelMetrics],
) -> float | None:
    if not per_label:
        return None
    values = [metrics.f1 for metrics in per_label.values()]
    if not values:
        return None
    return sum(values) / len(values)


def _confusion_matrix(
    records: Sequence[StanceRecord],
    human_labels: Mapping[str, HumanStanceLabel],
) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for record in records:
        gold = human_labels.get(record.stance_id)
        if gold is None:
            continue
        row = matrix.setdefault(gold.value, {})
        key = record.stance_label.value
        row[key] = row.get(key, 0) + 1
    return matrix


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_stance(
    records: Sequence[StanceRecord],
    human_labels: Mapping[str, HumanStanceLabel] | None = None,
) -> StanceEvaluation:
    """Score ``records`` against an optional human-label dict.

    ``human_labels`` is keyed by ``stance_id``. Missing entries are
    treated as "no label available" — they contribute to the totals but
    not to the per-label metrics.
    """
    counts = _per_label_counts(records)
    coverage = _coverage(records)
    abstention = _abstention_rate(records)
    warnings: list[str] = []

    if not records:
        warnings.append("no stance records to evaluate")
    if not human_labels:
        warnings.append(
            "no human labels supplied — accuracy / macro-F1 reported as null"
        )
        return StanceEvaluation(
            total_predictions=len(records),
            total_with_labels=0,
            accuracy=None,
            macro_f1=None,
            coverage=coverage,
            abstention_rate=abstention,
            per_label_counts=counts,
            warnings=warnings,
        )

    correct = 0
    total_labelled = 0
    per_label_tp: dict[str, int] = {}
    per_label_fp: dict[str, int] = {}
    per_label_fn: dict[str, int] = {}
    per_label_support: dict[str, int] = {}

    label_universe = {label.value for label in StanceLabel}
    for record in records:
        gold = human_labels.get(record.stance_id)
        if gold is None:
            continue
        total_labelled += 1
        gold_value = gold.value
        predicted_value = record.stance_label.value
        per_label_support[gold_value] = per_label_support.get(gold_value, 0) + 1
        if gold_value == predicted_value:
            correct += 1
            per_label_tp[gold_value] = per_label_tp.get(gold_value, 0) + 1
        else:
            per_label_fp[predicted_value] = per_label_fp.get(predicted_value, 0) + 1
            per_label_fn[gold_value] = per_label_fn.get(gold_value, 0) + 1

    if total_labelled == 0:
        warnings.append(
            "human labels supplied but none matched any stance_id — check the join key"
        )
        return StanceEvaluation(
            total_predictions=len(records),
            total_with_labels=0,
            accuracy=None,
            macro_f1=None,
            coverage=coverage,
            abstention_rate=abstention,
            per_label_counts=counts,
            warnings=warnings,
        )

    accuracy = correct / total_labelled if total_labelled else 0.0

    per_label_metrics: dict[str, _PerLabelMetrics] = {}
    for label in sorted(label_universe):
        tp = per_label_tp.get(label, 0)
        fp = per_label_fp.get(label, 0)
        fn = per_label_fn.get(label, 0)
        support = per_label_support.get(label, 0)
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_label_metrics[label] = _PerLabelMetrics(
            precision=precision,
            recall=recall,
            f1=_f1(precision, recall),
            support=support,
        )

    return StanceEvaluation(
        total_predictions=len(records),
        total_with_labels=total_labelled,
        accuracy=accuracy,
        macro_f1=_macro_f1(per_label_metrics),
        coverage=coverage,
        abstention_rate=abstention,
        per_label_counts=counts,
        per_label_metrics=per_label_metrics,
        confusion_matrix=_confusion_matrix(records, human_labels),
        warnings=warnings,
    )


def write_evaluation_report(report: StanceEvaluation, path: Path) -> Path:
    """Serialise the evaluation report to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


__all__ = [
    "StanceEvaluation",
    "evaluate_stance",
    "write_evaluation_report",
]

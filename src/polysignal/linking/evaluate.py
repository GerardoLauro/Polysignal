"""Offline evaluation for the Branch 05 deterministic linker.

The harness compares a sequence of :class:`RelevanceJudgement`
predictions against the gold :class:`LabeledPair` set and returns
classification metrics and diagnostic breakdowns:

* accuracy, macro-F1;
* per-label precision / recall / F1 / support;
* confusion matrix (rows = gold, cols = predicted);
* constraint accuracy;
* per-label score distribution (mean / min / max);
* false positives and false negatives (top-N each);
* top errors by score magnitude.

The report can be exported to JSON via :func:`write_evaluation_report`
so a reviewer can diff before/after runs.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from polysignal.domain import ConstraintMatch, LinkedPost, RelevanceLabel
from polysignal.linking.fixtures import LabeledPair
from polysignal.linking.relevance import RelevanceJudgement


@dataclass(frozen=True)
class LabelMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ScoreDistribution:
    n: int
    mean: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class ErrorRow:
    market_id: str
    post_id: str
    gold: RelevanceLabel
    predicted: RelevanceLabel
    relevance_score: float
    constraint_match: str
    explanation: str


@dataclass(frozen=True)
class EvaluationReport:
    accuracy: float
    macro_f1: float
    per_label: dict[RelevanceLabel, LabelMetrics]
    confusion_matrix: dict[RelevanceLabel, dict[RelevanceLabel, int]]
    n_pairs: int
    matched_pairs: int
    constraint_accuracy: float
    score_distribution: dict[RelevanceLabel, ScoreDistribution]
    false_positives: list[ErrorRow] = field(default_factory=list)
    false_negatives: list[ErrorRow] = field(default_factory=list)
    top_errors: list[ErrorRow] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    llm_rows_selected: int = 0
    llm_rows_sent: int = 0
    llm_rows_judged: int = 0
    llm_rows_percent: float = 0.0
    llm_cached_judgments: int = 0
    llm_invalid_outputs: int = 0
    llm_label_changes: int = 0


def _index_predictions(
    predictions: Sequence[RelevanceJudgement | LinkedPost],
) -> dict[tuple[str, str], RelevanceJudgement | LinkedPost]:
    return {(p.market_id, p.post_id): p for p in predictions}


def _effective_label(prediction: RelevanceJudgement | LinkedPost) -> RelevanceLabel:
    return getattr(prediction, "final_relevance_label", None) or prediction.relevance_label


def _effective_constraint(prediction: RelevanceJudgement | LinkedPost) -> ConstraintMatch:
    return getattr(prediction, "final_constraint_match", None) or prediction.constraint_match


def _label_metrics(
    label: RelevanceLabel,
    y_true: Sequence[RelevanceLabel],
    y_pred: Sequence[RelevanceLabel],
) -> LabelMetrics:
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p == label)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != label and p == label)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p != label)
    support = sum(1 for t in y_true if t == label)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return LabelMetrics(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        support=support,
    )


def _err(gold: LabeledPair, pred: RelevanceJudgement | LinkedPost) -> ErrorRow:
    return ErrorRow(
        market_id=gold.market_id,
        post_id=gold.post_id,
        gold=gold.expected_label,
        predicted=_effective_label(pred),
        relevance_score=pred.relevance_score,
        constraint_match=_effective_constraint(pred).value,
        explanation=pred.explanation,
    )


def evaluate_predictions(
    predictions: Sequence[RelevanceJudgement | LinkedPost],
    labels: Sequence[LabeledPair],
    *,
    top_n: int = 10,
) -> EvaluationReport:
    """Compare predictions to gold labels and return a full metrics report."""
    pred_index = _index_predictions(predictions)
    y_true: list[RelevanceLabel] = []
    y_pred: list[RelevanceLabel] = []
    constraint_hits = 0
    notes: list[str] = []
    paired: list[tuple[LabeledPair, RelevanceJudgement | LinkedPost]] = []

    for gold in labels:
        prediction = pred_index.get((gold.market_id, gold.post_id))
        if prediction is None:
            notes.append(f"no prediction for ({gold.market_id}, {gold.post_id})")
            continue
        y_true.append(gold.expected_label)
        y_pred.append(_effective_label(prediction))
        paired.append((gold, prediction))
        if _effective_constraint(prediction) == gold.expected_constraint_match:
            constraint_hits += 1

    matched = len(y_true)
    if matched == 0:
        empty = {label: ScoreDistribution(0, 0.0, 0.0, 0.0) for label in RelevanceLabel}
        return EvaluationReport(
            accuracy=0.0,
            macro_f1=0.0,
            per_label={},
            confusion_matrix={},
            n_pairs=len(labels),
            matched_pairs=0,
            constraint_accuracy=0.0,
            score_distribution=empty,
            notes=notes or ["no overlap between predictions and labels"],
        )

    accuracy = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p) / matched
    per_label = {label: _label_metrics(label, y_true, y_pred) for label in RelevanceLabel}
    macro_f1 = sum(m.f1 for m in per_label.values()) / len(RelevanceLabel)
    confusion: dict[RelevanceLabel, dict[RelevanceLabel, int]] = {
        t: dict.fromkeys(RelevanceLabel, 0) for t in RelevanceLabel
    }
    for t, p in zip(y_true, y_pred, strict=True):
        confusion[t][p] += 1

    # Score distribution per predicted label.
    by_label: dict[RelevanceLabel, list[float]] = {label: [] for label in RelevanceLabel}
    for _, pred in paired:
        by_label[_effective_label(pred)].append(pred.relevance_score)
    distribution: dict[RelevanceLabel, ScoreDistribution] = {}
    for label, scores in by_label.items():
        if scores:
            distribution[label] = ScoreDistribution(
                n=len(scores),
                mean=round(statistics.mean(scores), 4),
                minimum=round(min(scores), 4),
                maximum=round(max(scores), 4),
            )
        else:
            distribution[label] = ScoreDistribution(0, 0.0, 0.0, 0.0)

    positives = {RelevanceLabel.RELEVANT, RelevanceLabel.WEAKLY_RELEVANT}
    false_positives = [
        _err(gold, pred)
        for gold, pred in paired
        if _effective_label(pred) in positives and gold.expected_label not in positives
    ]
    false_negatives = [
        _err(gold, pred)
        for gold, pred in paired
        if gold.expected_label in positives and _effective_label(pred) not in positives
    ]
    all_errors = [
        _err(gold, pred) for gold, pred in paired if gold.expected_label != _effective_label(pred)
    ]
    top_errors = sorted(all_errors, key=lambda r: r.relevance_score, reverse=True)[:top_n]
    llm_rows_selected = sum(bool(getattr(pred, "llm_selected", False)) for pred in predictions)
    llm_rows_judged = sum(bool(getattr(pred, "llm_judged", False)) for pred in predictions)
    llm_cached_judgments = sum(bool(getattr(pred, "llm_cache_hit", False)) for pred in predictions)
    llm_invalid_outputs = sum(
        bool(getattr(pred, "llm_invalid_output", False)) for pred in predictions
    )
    llm_rows_sent = sum(
        (
            bool(getattr(pred, "llm_judged", False))
            or bool(getattr(pred, "llm_invalid_output", False))
        )
        and not bool(getattr(pred, "llm_cache_hit", False))
        for pred in predictions
    )
    llm_label_changes = sum(_effective_label(pred) != pred.relevance_label for pred in predictions)

    return EvaluationReport(
        accuracy=round(accuracy, 4),
        macro_f1=round(macro_f1, 4),
        per_label=per_label,
        confusion_matrix=confusion,
        n_pairs=len(labels),
        matched_pairs=matched,
        constraint_accuracy=round(constraint_hits / matched, 4),
        score_distribution=distribution,
        false_positives=false_positives[:top_n],
        false_negatives=false_negatives[:top_n],
        top_errors=top_errors,
        notes=notes,
        llm_rows_selected=llm_rows_selected,
        llm_rows_sent=llm_rows_sent,
        llm_rows_judged=llm_rows_judged,
        llm_rows_percent=round(llm_rows_sent / len(predictions) * 100, 2) if predictions else 0.0,
        llm_cached_judgments=llm_cached_judgments,
        llm_invalid_outputs=llm_invalid_outputs,
        llm_label_changes=llm_label_changes,
    )


def _report_to_dict(report: EvaluationReport) -> dict:
    def _err_list(rows: Sequence[ErrorRow]) -> list[dict]:
        return [
            {
                "market_id": r.market_id,
                "post_id": r.post_id,
                "gold": r.gold.value,
                "predicted": r.predicted.value,
                "relevance_score": r.relevance_score,
                "constraint_match": r.constraint_match,
                "explanation": r.explanation,
            }
            for r in rows
        ]

    return {
        "accuracy": report.accuracy,
        "macro_f1": report.macro_f1,
        "constraint_accuracy": report.constraint_accuracy,
        "n_pairs": report.n_pairs,
        "matched_pairs": report.matched_pairs,
        "llm_metrics": {
            "rows_selected": report.llm_rows_selected,
            "rows_sent": report.llm_rows_sent,
            "rows_judged": report.llm_rows_judged,
            "rows_percent": report.llm_rows_percent,
            "cached_judgments": report.llm_cached_judgments,
            "invalid_outputs": report.llm_invalid_outputs,
            "label_changes": report.llm_label_changes,
        },
        "per_label": {
            label.value: {
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "support": m.support,
            }
            for label, m in report.per_label.items()
        },
        "confusion_matrix": {
            row_label.value: {
                col_label.value: report.confusion_matrix[row_label][col_label]
                for col_label in RelevanceLabel
            }
            for row_label in report.confusion_matrix
        },
        "score_distribution": {
            label.value: {
                "n": dist.n,
                "mean": dist.mean,
                "min": dist.minimum,
                "max": dist.maximum,
            }
            for label, dist in report.score_distribution.items()
        },
        "false_positives": _err_list(report.false_positives),
        "false_negatives": _err_list(report.false_negatives),
        "top_errors": _err_list(report.top_errors),
        "notes": list(report.notes),
    }


def write_evaluation_report(report: EvaluationReport, path: Path) -> Path:
    """Write the evaluation report as JSON for diffing across runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_report_to_dict(report), indent=2), encoding="utf-8")
    return path


__all__ = [
    "ErrorRow",
    "EvaluationReport",
    "LabelMetrics",
    "ScoreDistribution",
    "evaluate_predictions",
    "write_evaluation_report",
]

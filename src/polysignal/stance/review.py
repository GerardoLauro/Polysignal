"""Manual-review CSV writer for stance verdicts.

The review CSV is exported alongside ``stance_labels.parquet`` so a
human reviewer can label a curated subset before the LLM judge or
fine-tuning milestone consumes them.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from polysignal.domain import StanceLabel
from polysignal.stance.models import StanceRecord

# Stance labels that benefit most from human spot-checking.
_REVIEW_PRIORITY = (
    StanceLabel.SUPPORTS_YES,
    StanceLabel.SUPPORTS_NO,
    StanceLabel.AMBIGUOUS,
    StanceLabel.INSUFFICIENT_CONTEXT,
)


_COLUMNS = (
    "stance_id",
    "market_id",
    "market_question",
    "outcome_names",
    "target_outcome",
    "target_outcome_index",
    "target_outcome_name",
    "target_aliases",
    "is_multi_outcome",
    "post_id",
    "segment_id",
    "parent_post_id",
    "platform",
    "segment_text_preview",
    "predicted_stance_label",
    "stance_score",
    "confidence",
    "method",
    "explanation",
    "matched_yes_cues",
    "matched_no_cues",
    "matched_uncertainty_cues",
    "negated_cues",
    "relevance_label",
    "relevance_score",
    "link_confidence",
    "human_stance_label",
    "human_notes",
    "review_status",
)


def _select_rows(
    records: Sequence[StanceRecord],
    *,
    top_per_label: int,
    top_neutral: int,
) -> list[StanceRecord]:
    """Pick a curated subset: top-K per directional label + a few neutrals."""
    by_label: dict[StanceLabel, list[StanceRecord]] = {}
    for record in records:
        by_label.setdefault(record.stance_label, []).append(record)
    for bucket in by_label.values():
        bucket.sort(key=lambda r: r.confidence, reverse=True)

    chosen: list[StanceRecord] = []
    for label in _REVIEW_PRIORITY:
        chosen.extend(by_label.get(label, [])[:top_per_label])
    chosen.extend(by_label.get(StanceLabel.NEUTRAL, [])[:top_neutral])
    seen: set[str] = set()
    deduped: list[StanceRecord] = []
    for record in chosen:
        if record.stance_id in seen:
            continue
        seen.add(record.stance_id)
        deduped.append(record)
    return deduped


def _csv_safe(value: list[str] | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def write_manual_stance_review(
    records: Sequence[StanceRecord],
    path: Path,
    *,
    top_per_label: int = 50,
    top_neutral: int = 20,
) -> tuple[Path, int]:
    """Write the curated review CSV. Returns ``(path, rows_written)``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _select_rows(records, top_per_label=top_per_label, top_neutral=top_neutral)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_COLUMNS))
        writer.writeheader()
        for record in rows:
            writer.writerow(
                {
                    "stance_id": record.stance_id,
                    "market_id": record.market_id,
                    "market_question": record.market_question,
                    "outcome_names": _csv_safe(record.outcome_names),
                    "target_outcome": record.target_outcome,
                    "target_outcome_index": (
                        record.target_outcome_index
                        if record.target_outcome_index is not None
                        else ""
                    ),
                    "target_outcome_name": record.target_outcome_name or "",
                    "target_aliases": _csv_safe(record.target_aliases),
                    "is_multi_outcome": record.is_multi_outcome,
                    "post_id": record.post_id,
                    "segment_id": record.segment_id or "",
                    "parent_post_id": record.parent_post_id or "",
                    "platform": (
                        record.source_platform.value if record.source_platform else ""
                    ),
                    "segment_text_preview": record.segment_text_preview,
                    "predicted_stance_label": record.stance_label.value,
                    "stance_score": record.stance_score,
                    "confidence": record.confidence,
                    "method": record.method.value,
                    "explanation": record.explanation,
                    "matched_yes_cues": _csv_safe(record.matched_yes_cues),
                    "matched_no_cues": _csv_safe(record.matched_no_cues),
                    "matched_uncertainty_cues": _csv_safe(record.matched_uncertainty_cues),
                    "negated_cues": _csv_safe(record.negated_cues),
                    "relevance_label": (
                        record.relevance_label.value if record.relevance_label else ""
                    ),
                    "relevance_score": (
                        record.relevance_score if record.relevance_score is not None else ""
                    ),
                    "link_confidence": (
                        record.link_confidence if record.link_confidence is not None else ""
                    ),
                    "human_stance_label": "",
                    "human_notes": "",
                    "review_status": "pending",
                }
            )
    return path, len(rows)


__all__ = ["write_manual_stance_review"]

"""Stance pipeline: linked_posts.parquet → stance_labels.parquet.

The pipeline is intentionally framework-light:

1. Read ``linked_posts.parquet`` (or accept an in-memory list).
2. Filter rows that are eligible to be scored (relevance label,
   confidence floor, optional ambiguous inclusion).
3. Look up the segment text (when ``social_post_segments.parquet`` is
   available) so the scorer sees the full segment instead of the
   240-char preview the linking stage stored.
4. Join ordered outcome labels from ``outcomes.parquet`` when available.
5. Generate conservative aliases for each multi-outcome target.
6. Run :func:`polysignal.stance.scorer.score_stance` against every
   eligible row.
7. Build :class:`StanceRecord` rows with full provenance, expanding
   non-binary markets into one row per target outcome, and persist them.

Honest fallbacks
----------------

* Missing input parquet ⇒ raises ``FileNotFoundError`` with a clear
  message; the CLI wraps that into a non-zero exit code.
* Empty input ⇒ returns a result with ``records=[]`` and an
  ``empty_input`` reason; the parquet write is skipped unless ``force``
  is set, in which case an empty parquet is written so downstream
  pipelines don't trip on missing files.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from polysignal.domain import (
    LinkedPost,
    Market,
    Outcome,
    SocialPostSegment,
    StanceLabel,
    StanceMethod,
)
from polysignal.domain.enums import RelevanceLabel
from polysignal.evaluation import assess_post_admissibility
from polysignal.linking.linking_pipeline import read_linked_posts
from polysignal.stance.llm_judge import LLMStanceJudge, LLMStanceJudgeResult
from polysignal.stance.models import StanceRecord, write_stance_records
from polysignal.stance.scorer import ScorerConfig, StanceVerdict, score_stance
from polysignal.stance.target_aliases import generate_target_aliases
from polysignal.storage.parquet_store import read_parquet
from polysignal.storage.paths import DataPaths, get_data_paths

logger = logging.getLogger(__name__)

_ELIGIBLE_LABELS_DEFAULT = {RelevanceLabel.RELEVANT, RelevanceLabel.WEAKLY_RELEVANT}
_ELIGIBLE_LABELS_WITH_AMBIGUOUS = _ELIGIBLE_LABELS_DEFAULT | {RelevanceLabel.AMBIGUOUS}
DEFAULT_MIN_LINK_CONFIDENCE = 0.6


@dataclass
class StancePipelineResult:
    """Diagnostic envelope returned by :func:`run_stance_pipeline`."""

    records: list[StanceRecord] = field(default_factory=list)
    rows_read: int = 0
    rows_eligible: int = 0
    rows_scored: int = 0
    rows_skipped_label: int = 0
    rows_skipped_confidence: int = 0
    rows_skipped_text: int = 0
    rows_skipped_post_close: int = 0
    rows_skipped_temporal_anchor_missing: int = 0
    rows_skipped_post_timestamp_missing: int = 0
    by_label: dict[str, int] = field(default_factory=dict)
    output_path: Path | None = None
    # Branch 06B — counters populated when an LLM judge is supplied.
    llm_considered: int = 0
    llm_selected: int = 0
    llm_calls: int = 0
    llm_cache_hits: int = 0
    llm_applied: int = 0
    llm_invalid_outputs: int = 0
    llm_label_changes: int = 0


def _effective_label(row: LinkedPost) -> RelevanceLabel:
    return row.final_relevance_label or row.relevance_label


def _effective_confidence(row: LinkedPost) -> float:
    return row.final_confidence if row.final_confidence is not None else row.confidence


def _effective_relevance_score(row: LinkedPost) -> float:
    return (
        row.final_relevance_score
        if row.final_relevance_score is not None
        else row.relevance_score
    )


def _stance_id(
    market_id: str,
    post_id: str,
    target_outcome_index: int | None = None,
) -> str:
    seed_text = f"{market_id}|{post_id}"
    if target_outcome_index is not None:
        seed_text += f"|{target_outcome_index}"
    seed = seed_text.encode()
    return hashlib.sha1(seed).hexdigest()[:24]


def _build_segment_index(
    segments: Sequence[SocialPostSegment],
) -> dict[str, SocialPostSegment]:
    """Index segments by ``segment_id`` for fast lookup."""
    return {seg.segment_id: seg for seg in segments}


def _build_outcomes_index(outcomes: Sequence[Outcome]) -> dict[str, list[Outcome]]:
    """Index ordered outcomes by ``market_id``."""
    grouped: dict[str, list[Outcome]] = {}
    for outcome in outcomes:
        if outcome.market_id is not None:
            grouped.setdefault(outcome.market_id, []).append(outcome)
    return {
        market_id: sorted(rows, key=lambda row: row.index)
        for market_id, rows in grouped.items()
    }


def _resolve_text(
    row: LinkedPost,
    segment_index: dict[str, SocialPostSegment],
) -> tuple[str, str | None, str | None]:
    """Resolve the text to score and the (segment_id, parent_post_id) provenance.

    Branch 05G already wired ``LinkedPost.segment_id`` and
    ``LinkedPost.segment_text_preview``. We prefer the full segment
    text when an external index is available so the scorer can see
    cues that fell off the 240-char preview window.
    """
    segment_id = row.segment_id
    parent_post_id = row.parent_post_id
    if segment_id and segment_id in segment_index:
        seg = segment_index[segment_id]
        return seg.segment_text, segment_id, parent_post_id or seg.parent_post_id
    if row.segment_text_preview:
        return row.segment_text_preview, segment_id, parent_post_id
    return row.post_text_preview, segment_id, parent_post_id


def _is_eligible(
    row: LinkedPost,
    *,
    eligible_labels: set[RelevanceLabel],
    min_confidence: float,
) -> tuple[bool, str | None]:
    label = _effective_label(row)
    if label not in eligible_labels:
        return False, "label"
    if _effective_confidence(row) < min_confidence:
        return False, "confidence"
    return True, None


def _build_record(
    *,
    row: LinkedPost,
    text: str,
    segment_id: str | None,
    parent_post_id: str | None,
    now: datetime,
    config: ScorerConfig,
    outcome_names: list[str],
    target_outcome: Outcome | None = None,
    is_multi_outcome: bool = False,
    llm_judge: LLMStanceJudge | None = None,
) -> tuple[StanceRecord, LLMStanceJudgeResult | None]:
    canonical_keywords = row.matched_terms or row.matched_entities
    target_outcome_name = target_outcome.label if target_outcome is not None else None
    target_aliases = generate_target_aliases(target_outcome_name)
    baseline = score_stance(
        text=text,
        market_question=row.market_question,
        outcome_names=outcome_names,
        target_outcome_name=target_outcome_name,
        target_aliases=target_aliases,
        canonical_keywords=canonical_keywords,
        config=config,
    )
    judge_result: LLMStanceJudgeResult | None = None
    final_verdict: StanceVerdict = baseline
    if llm_judge is not None:
        judge_result = llm_judge.judge(
            text=text,
            market_question=row.market_question,
            outcome_names=outcome_names,
            target_outcome_name=target_outcome_name,
            target_aliases=target_aliases,
            canonical_keywords=canonical_keywords,
            baseline=baseline,
        )
        final_verdict = judge_result.merged

    preview = (text or "").replace("\n", " ").strip()[:240]
    method = (
        StanceMethod.HYBRID
        if judge_result is not None and judge_result.applied
        else StanceMethod.HEURISTIC
    )
    llm_verdict = judge_result.verdict if judge_result is not None else None
    record = StanceRecord(
        stance_id=_stance_id(
            row.market_id,
            row.post_id,
            target_outcome.index if target_outcome is not None else None,
        ),
        market_id=row.market_id,
        post_id=row.post_id,
        segment_id=segment_id,
        parent_post_id=parent_post_id,
        outcome_id=target_outcome.outcome_id if target_outcome is not None else None,
        target_outcome=target_outcome.label if target_outcome is not None else "yes",
        target_outcome_index=target_outcome.index if target_outcome is not None else None,
        target_outcome_name=target_outcome_name,
        target_aliases=target_aliases,
        is_multi_outcome=is_multi_outcome,
        stance_label=final_verdict.label,
        stance_score=final_verdict.stance_score,
        confidence=final_verdict.confidence,
        method=method,
        explanation=final_verdict.explanation,
        matched_yes_cues=final_verdict.matched_yes_cues,
        matched_no_cues=final_verdict.matched_no_cues,
        matched_uncertainty_cues=final_verdict.matched_uncertainty_cues,
        negated_cues=final_verdict.negated_cues,
        created_at=now,
        post_created_at=row.post_created_at,
        market_question=row.market_question,
        outcome_names=outcome_names,
        segment_text_preview=preview,
        relevance_label=_effective_label(row),
        relevance_score=_effective_relevance_score(row),
        link_confidence=_effective_confidence(row),
        constraint_match=row.final_constraint_match or row.constraint_match,
        source_platform=row.platform,
        source_id=row.post_id,
        source_channel=None,
        used_llm=judge_result is not None and judge_result.judged,
        llm_label=llm_verdict.stance_label if llm_verdict is not None else None,
        llm_confidence=llm_verdict.confidence if llm_verdict is not None else None,
        llm_rationale=llm_verdict.rationale if llm_verdict is not None else None,
        llm_cache_key=judge_result.cache_key if judge_result is not None else None,
        llm_cache_hit=bool(judge_result and judge_result.cache_hit),
        llm_invalid_output=bool(judge_result and judge_result.invalid_output),
        llm_applied=bool(judge_result and judge_result.applied),
        llm_provider=llm_judge.provider if llm_judge is not None else None,
        llm_model=llm_judge.model_id if llm_judge is not None else None,
    )
    return record, judge_result


def run_stance_pipeline(
    *,
    linked_posts: Iterable[LinkedPost] | None = None,
    linked_posts_path: Path | None = None,
    segments: Sequence[SocialPostSegment] | None = None,
    segments_path: Path | None = None,
    outcomes: Sequence[Outcome] | None = None,
    outcomes_path: Path | None = None,
    markets: Sequence[Market] | None = None,
    markets_path: Path | None = None,
    include_ambiguous: bool = False,
    min_link_confidence: float = DEFAULT_MIN_LINK_CONFIDENCE,
    limit: int | None = None,
    paths: DataPaths | None = None,
    output_path: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
    config: ScorerConfig | None = None,
    now: datetime | None = None,
    llm_judge: LLMStanceJudge | None = None,
) -> StancePipelineResult:
    """Run the deterministic stance pipeline."""
    paths = paths or get_data_paths()
    moment = (now or datetime.now(tz=timezone.utc)).astimezone(timezone.utc)
    cfg = config or ScorerConfig()

    if linked_posts is None:
        source = linked_posts_path or paths.linked_posts_parquet
        if not source.exists():
            raise FileNotFoundError(
                f"linked_posts parquet not found at {source}. "
                "Run `polysignal linking score-candidates` first."
            )
        rows = read_linked_posts(source)
    else:
        rows = list(linked_posts)

    seg_models: list[SocialPostSegment]
    if segments is not None:
        seg_models = list(segments)
    else:
        seg_source = segments_path or paths.social_post_segments_parquet
        seg_models = (
            read_parquet(seg_source, SocialPostSegment) if seg_source.exists() else []
        )
    segment_index = _build_segment_index(seg_models)

    outcome_models: list[Outcome]
    if outcomes is not None:
        outcome_models = list(outcomes)
    else:
        outcome_source = outcomes_path or paths.polymarket_outcomes_parquet
        outcome_models = read_parquet(outcome_source, Outcome) if outcome_source.exists() else []
    outcomes_index = _build_outcomes_index(outcome_models)
    market_models: list[Market]
    if markets is not None:
        market_models = list(markets)
    else:
        market_source = markets_path or paths.polymarket_markets_parquet
        market_models = read_parquet(market_source, Market) if market_source.exists() else []
    market_index = {market.market_id: market for market in market_models}

    eligible_labels = (
        _ELIGIBLE_LABELS_WITH_AMBIGUOUS if include_ambiguous else _ELIGIBLE_LABELS_DEFAULT
    )

    result = StancePipelineResult()
    result.rows_read = len(rows)

    records: list[StanceRecord] = []
    for row in rows:
        if limit is not None and result.rows_scored >= limit:
            break
        ok, reason = _is_eligible(
            row,
            eligible_labels=eligible_labels,
            min_confidence=min_link_confidence,
        )
        if not ok:
            if reason == "label":
                result.rows_skipped_label += 1
            elif reason == "confidence":
                result.rows_skipped_confidence += 1
            continue
        market = market_index.get(row.market_id)
        if market is not None:
            admissibility = assess_post_admissibility(market, row.post_created_at)
            if not admissibility.admissible:
                if admissibility.reason == "post_close":
                    result.rows_skipped_post_close += 1
                elif admissibility.reason == "temporal_anchor_missing":
                    result.rows_skipped_temporal_anchor_missing += 1
                elif admissibility.reason == "post_timestamp_missing":
                    result.rows_skipped_post_timestamp_missing += 1
                continue
        result.rows_eligible += 1
        text, segment_id, parent_post_id = _resolve_text(row, segment_index)
        if not (text or "").strip():
            result.rows_skipped_text += 1
            continue
        market_outcomes = outcomes_index.get(row.market_id, [])
        outcome_names = [outcome.label for outcome in market_outcomes]
        is_multi_outcome = len(market_outcomes) > 2
        targets: list[Outcome | None] = market_outcomes if is_multi_outcome else [None]
        for target_outcome in targets:
            record, _judge_result = _build_record(
                row=row,
                text=text,
                segment_id=segment_id,
                parent_post_id=parent_post_id,
                now=moment,
                config=cfg,
                outcome_names=outcome_names,
                target_outcome=target_outcome,
                is_multi_outcome=is_multi_outcome,
                llm_judge=llm_judge,
            )
            records.append(record)
            result.rows_scored += 1
            key = record.stance_label.value
            result.by_label[key] = result.by_label.get(key, 0) + 1

    if llm_judge is not None:
        result.llm_considered = llm_judge.stats.considered
        result.llm_selected = llm_judge.stats.selected
        result.llm_calls = llm_judge.stats.calls
        result.llm_cache_hits = llm_judge.stats.cache_hits
        result.llm_applied = llm_judge.stats.applied
        result.llm_invalid_outputs = llm_judge.stats.invalid_outputs
        result.llm_label_changes = llm_judge.stats.label_changes

    result.records = records

    target = output_path or paths.stance_labels_parquet
    if dry_run:
        return result
    if records or force:
        write_stance_records(records, target)
        result.output_path = target
    return result


__all__ = [
    "DEFAULT_MIN_LINK_CONFIDENCE",
    "StancePipelineResult",
    "run_stance_pipeline",
]


# Keep an unused import alive — the ``StanceLabel`` re-export is used by
# downstream consumers that ``from polysignal.stance.pipeline import``.
_ = StanceLabel

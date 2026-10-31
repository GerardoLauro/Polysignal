"""Tests for the deterministic stance pipeline (Branch 06A)."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from polysignal.domain import (
    LinkedPost,
    Market,
    MarketDomain,
    Outcome,
    Platform,
    SocialPostSegment,
    StanceLabel,
)
from polysignal.domain.enums import (
    ConstraintMatch,
    LinkMethod,
    RelevanceLabel,
)
from polysignal.linking.linking_pipeline import write_linked_posts
from polysignal.stance import (
    HumanStanceLabel,
    evaluate_stance,
    read_stance_records,
    run_stance_pipeline,
    write_manual_stance_review,
    write_stance_records,
)
from polysignal.stance.models import StanceRecord
from polysignal.storage.parquet_store import write_parquet
from polysignal.storage.paths import get_data_paths


def _linked(
    *,
    link_id: str,
    market_id: str,
    post_id: str,
    market_question: str,
    relevance_label: RelevanceLabel = RelevanceLabel.RELEVANT,
    confidence: float = 0.8,
    final_relevance_label: RelevanceLabel | None = None,
    final_confidence: float | None = None,
    segment_id: str | None = None,
    parent_post_id: str | None = None,
    segment_text_preview: str = "",
    post_text_preview: str = "",
    matched_terms: list[str] | None = None,
    matched_entities: list[str] | None = None,
) -> LinkedPost:
    return LinkedPost(
        link_id=link_id,
        market_id=market_id,
        post_id=post_id,
        platform=Platform.TELEGRAM,
        relevance_score=0.7,
        relevance_label=relevance_label,
        confidence=confidence,
        method=LinkMethod.DETERMINISTIC_BASELINE,
        explanation="test",
        matched_entities=matched_entities or [],
        matched_terms=matched_terms or [],
        constraint_match=ConstraintMatch.UNKNOWN,
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        post_created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        market_question=market_question,
        post_text_preview=post_text_preview,
        signal_count=2,
        anchor_score=0.6,
        constraint_score=0.0,
        domain_mismatch=False,
        final_relevance_label=final_relevance_label,
        final_confidence=final_confidence,
        parent_post_id=parent_post_id,
        segment_id=segment_id,
        segment_text_preview=segment_text_preview,
    )


def _segment(segment_id: str, parent_post_id: str, text: str) -> SocialPostSegment:
    return SocialPostSegment(
        segment_id=segment_id,
        parent_post_id=parent_post_id,
        platform=Platform.TELEGRAM,
        segment_index=0,
        segment_text=text,
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )


def test_pipeline_skips_unrelated_rows() -> None:
    rows = [
        _linked(
            link_id="l1",
            market_id="m1",
            post_id="p1",
            market_question="Will Bitcoin hit $150k by December 2026?",
            relevance_label=RelevanceLabel.UNRELATED,
            post_text_preview="Bitcoin is set to reach $150k by December 2026.",
            matched_terms=["Bitcoin", "150k"],
        ),
        _linked(
            link_id="l2",
            market_id="m1",
            post_id="p2",
            market_question="Will Bitcoin hit $150k by December 2026?",
            relevance_label=RelevanceLabel.RELEVANT,
            post_text_preview="Bitcoin is set to reach $150k by December 2026.",
            matched_terms=["Bitcoin", "150k"],
        ),
    ]
    result = run_stance_pipeline(linked_posts=rows, dry_run=True)
    assert result.rows_eligible == 1
    assert result.rows_skipped_label == 1
    assert result.rows_scored == 1
    assert result.records[0].stance_label is StanceLabel.SUPPORTS_YES


def test_pipeline_respects_min_link_confidence() -> None:
    rows = [
        _linked(
            link_id="l1",
            market_id="m1",
            post_id="p1",
            market_question="Will Bitcoin hit $150k by December 2026?",
            confidence=0.3,
            post_text_preview="Bitcoin is set to reach $150k by 2026.",
            matched_terms=["Bitcoin", "150k"],
        ),
    ]
    result = run_stance_pipeline(
        linked_posts=rows,
        min_link_confidence=0.7,
        dry_run=True,
    )
    assert result.rows_eligible == 0
    assert result.rows_skipped_confidence == 1


def test_pipeline_skips_post_close_rows_for_historical_market() -> None:
    row = _linked(
        link_id="late",
        market_id="m1",
        post_id="p1",
        market_question="Will Bitcoin hit $150k by December 2026?",
        post_text_preview="Bitcoin is set to reach $150k by December 2026.",
        matched_terms=["Bitcoin", "150k"],
    )
    row.post_created_at = datetime(2026, 6, 1, tzinfo=timezone.utc)
    market = Market(
        market_id="m1",
        slug="m1",
        question=row.market_question,
        domain=MarketDomain.CRYPTO,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        close_time=datetime(2026, 5, 1, tzinfo=timezone.utc),
        is_closed=True,
        outcomes=[
            Outcome(outcome_id="yes", label="Yes", index=0),
            Outcome(outcome_id="no", label="No", index=1),
        ],
    )
    result = run_stance_pipeline(linked_posts=[row], markets=[market], dry_run=True)
    assert result.rows_scored == 0
    assert result.rows_skipped_post_close == 1


def test_pipeline_skips_missing_timestamp_rows() -> None:
    row = _linked(
        link_id="missing-time",
        market_id="m1",
        post_id="p1",
        market_question="Will Bitcoin hit $150k by December 2026?",
        post_text_preview="Bitcoin is set to reach $150k by December 2026.",
        matched_terms=["Bitcoin", "150k"],
    )
    row.post_created_at = None
    market = Market(
        market_id="m1",
        slug="m1",
        question=row.market_question,
        domain=MarketDomain.CRYPTO,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        close_time=datetime(2026, 5, 1, tzinfo=timezone.utc),
        outcomes=[
            Outcome(outcome_id="yes", label="Yes", index=0),
            Outcome(outcome_id="no", label="No", index=1),
        ],
    )
    result = run_stance_pipeline(linked_posts=[row], markets=[market], dry_run=True)
    assert result.rows_scored == 0
    assert result.rows_skipped_post_timestamp_missing == 1


def test_pipeline_uses_final_relevance_label_when_present() -> None:
    rows = [
        _linked(
            link_id="l1",
            market_id="m1",
            post_id="p1",
            market_question="Will Bitcoin hit $150k by December 2026?",
            relevance_label=RelevanceLabel.UNRELATED,
            final_relevance_label=RelevanceLabel.WEAKLY_RELEVANT,
            post_text_preview="Bitcoin is set to reach $150k by 2026.",
            matched_terms=["Bitcoin"],
        ),
    ]
    result = run_stance_pipeline(linked_posts=rows, dry_run=True)
    # The final label promoted to WEAKLY_RELEVANT, so it must be scored.
    assert result.rows_scored == 1


def test_pipeline_includes_ambiguous_when_flag_is_set() -> None:
    rows = [
        _linked(
            link_id="l1",
            market_id="m1",
            post_id="p1",
            market_question="Will Bitcoin hit $150k by December 2026?",
            relevance_label=RelevanceLabel.AMBIGUOUS,
            post_text_preview="Bitcoin is set to reach $150k by 2026.",
            matched_terms=["Bitcoin"],
        ),
    ]
    excluded = run_stance_pipeline(linked_posts=rows, dry_run=True)
    assert excluded.rows_scored == 0
    included = run_stance_pipeline(
        linked_posts=rows,
        include_ambiguous=True,
        dry_run=True,
    )
    assert included.rows_scored == 1


def test_pipeline_uses_segment_text_when_segments_provided() -> None:
    rows = [
        _linked(
            link_id="l1",
            market_id="m1",
            post_id="seg-1",
            market_question="Will Bitcoin hit $150k by December 2026?",
            segment_id="seg-1",
            parent_post_id="telegram:c1:42",
            segment_text_preview="Bitcoin price (truncated)",
            post_text_preview="ignored",
            matched_terms=["Bitcoin", "150k"],
        ),
    ]
    segments = [
        _segment(
            "seg-1",
            "telegram:c1:42",
            "Analysts expect Bitcoin to reach $150k before 2026 ends.",
        )
    ]
    result = run_stance_pipeline(
        linked_posts=rows,
        segments=segments,
        dry_run=True,
    )
    assert result.rows_scored == 1
    record = result.records[0]
    assert record.stance_label is StanceLabel.SUPPORTS_YES
    assert "Analysts expect" in record.segment_text_preview


def test_pipeline_joins_ordered_outcome_names() -> None:
    row = _linked(
        link_id="l1",
        market_id="m-election",
        post_id="p1",
        market_question="Who will win the election?",
        post_text_preview="Candidate A is set to win the election.",
        matched_terms=["Candidate A", "election"],
    )
    outcomes = [
        Outcome(market_id="m-election", outcome_id="other", label="Other", index=2),
        Outcome(market_id="m-election", outcome_id="a", label="Candidate A", index=0),
        Outcome(market_id="m-election", outcome_id="b", label="Candidate B", index=1),
    ]
    result = run_stance_pipeline(
        linked_posts=[row],
        outcomes=outcomes,
        dry_run=True,
    )
    assert len(result.records) == 3
    assert {record.target_outcome_name for record in result.records} == {
        "Candidate A",
        "Candidate B",
        "Other",
    }
    assert {record.target_outcome_index for record in result.records} == {0, 1, 2}
    assert {record.outcome_id for record in result.records} == {"a", "b", "other"}
    assert len({record.stance_id for record in result.records}) == 3
    assert all(record.is_multi_outcome for record in result.records)
    assert all(
        record.outcome_names == ["Candidate A", "Candidate B", "Other"]
        for record in result.records
    )
    by_target = {record.target_outcome_name: record for record in result.records}
    assert by_target["Candidate A"].stance_label is StanceLabel.SUPPORTS_YES
    assert by_target["Candidate A"].target_aliases == ["candidate a"]
    assert by_target["Candidate B"].stance_label is StanceLabel.NEUTRAL
    assert by_target["Other"].stance_label is StanceLabel.NEUTRAL


def test_pipeline_missing_outcome_metadata_falls_back_to_empty_list(tmp_path: Path) -> None:
    paths = get_data_paths(root=tmp_path)
    row = _linked(
        link_id="l1",
        market_id="m1",
        post_id="p1",
        market_question="Will Bitcoin hit $150k by December 2026?",
        post_text_preview="Bitcoin is set to reach $150k by December 2026.",
        matched_terms=["Bitcoin", "150k"],
    )
    result = run_stance_pipeline(linked_posts=[row], paths=paths, dry_run=True)
    assert len(result.records) == 1
    assert result.records[0].outcome_names == []
    assert result.records[0].target_outcome == "yes"
    assert result.records[0].is_multi_outcome is False


def test_pipeline_binary_outcomes_preserve_yes_no_scoring() -> None:
    row = _linked(
        link_id="l1",
        market_id="m1",
        post_id="p1",
        market_question="Will Bitcoin hit $150k by December 2026?",
        post_text_preview="Bitcoin is unlikely to reach $150k by December 2026.",
        matched_terms=["Bitcoin", "150k"],
    )
    outcomes = [
        Outcome(market_id="m1", outcome_id="yes", label="Yes", index=0),
        Outcome(market_id="m1", outcome_id="no", label="No", index=1),
    ]
    result = run_stance_pipeline(linked_posts=[row], outcomes=outcomes, dry_run=True)
    record = result.records[0]
    assert len(result.records) == 1
    assert record.outcome_names == ["Yes", "No"]
    assert record.target_outcome == "yes"
    assert record.target_outcome_name is None
    assert record.is_multi_outcome is False
    assert record.stance_label is StanceLabel.SUPPORTS_NO


def test_pipeline_empty_input_writes_nothing(tmp_path: Path) -> None:
    paths = get_data_paths(root=tmp_path)
    result = run_stance_pipeline(linked_posts=[], paths=paths)
    assert result.rows_scored == 0
    assert result.output_path is None
    assert not paths.stance_labels_parquet.exists()


def test_pipeline_force_writes_empty_parquet(tmp_path: Path) -> None:
    paths = get_data_paths(root=tmp_path)
    result = run_stance_pipeline(linked_posts=[], paths=paths, force=True)
    assert result.output_path is not None
    assert paths.stance_labels_parquet.exists()


def test_pipeline_missing_input_path_raises(tmp_path: Path) -> None:
    paths = get_data_paths(root=tmp_path)
    with pytest.raises(FileNotFoundError):
        run_stance_pipeline(paths=paths)


def test_pipeline_writes_stance_labels_parquet_with_provenance(tmp_path: Path) -> None:
    paths = get_data_paths(root=tmp_path)
    rows = [
        _linked(
            link_id="l1",
            market_id="m1",
            post_id="p1",
            market_question="Will Bitcoin hit $150k by December 2026?",
            post_text_preview="Bitcoin is set to reach $150k by 2026.",
            matched_terms=["Bitcoin", "150k"],
        ),
    ]
    write_linked_posts(rows, paths.linked_posts_parquet)
    result = run_stance_pipeline(paths=paths)
    assert paths.stance_labels_parquet.exists()
    persisted = read_stance_records(paths.stance_labels_parquet)
    assert len(persisted) == 1
    record = persisted[0]
    assert record.market_id == "m1"
    assert record.post_id == "p1"
    assert record.source_platform is Platform.TELEGRAM
    assert record.relevance_label is RelevanceLabel.RELEVANT
    assert result.by_label[StanceLabel.SUPPORTS_YES.value] == 1


def test_pipeline_limit_caps_scored_rows() -> None:
    rows = [
        _linked(
            link_id=f"l{i}",
            market_id="m1",
            post_id=f"p{i}",
            market_question="Will Bitcoin hit $150k by December 2026?",
            post_text_preview="Bitcoin is set to reach $150k by 2026.",
            matched_terms=["Bitcoin"],
        )
        for i in range(5)
    ]
    result = run_stance_pipeline(linked_posts=rows, limit=2, dry_run=True)
    assert result.rows_scored == 2


# ---------------------------------------------------------------------------
# Manual review export + evaluation
# ---------------------------------------------------------------------------


def _record(stance_id: str, *, label: StanceLabel, confidence: float) -> StanceRecord:
    return StanceRecord(
        stance_id=stance_id,
        market_id="m1",
        post_id="p" + stance_id,
        stance_label=label,
        stance_score=0.0 if label is StanceLabel.NEUTRAL else 0.5,
        confidence=confidence,
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        market_question="Will Bitcoin hit $150k by December 2026?",
        segment_text_preview="text",
    )


def test_manual_review_includes_directional_and_neutral_buckets(tmp_path: Path) -> None:
    records = [
        _record("a", label=StanceLabel.SUPPORTS_YES, confidence=0.6),
        _record("b", label=StanceLabel.SUPPORTS_NO, confidence=0.55),
        _record("c", label=StanceLabel.AMBIGUOUS, confidence=0.2),
        _record("d", label=StanceLabel.INSUFFICIENT_CONTEXT, confidence=0.05),
        _record("e", label=StanceLabel.NEUTRAL, confidence=0.1),
    ]
    target = tmp_path / "review.csv"
    _, written = write_manual_stance_review(
        records, target, top_per_label=10, top_neutral=10
    )
    assert written == 5
    rows = list(csv.DictReader(target.open("r", encoding="utf-8-sig")))
    labels = {row["predicted_stance_label"] for row in rows}
    assert labels == {
        StanceLabel.SUPPORTS_YES.value,
        StanceLabel.SUPPORTS_NO.value,
        StanceLabel.AMBIGUOUS.value,
        StanceLabel.INSUFFICIENT_CONTEXT.value,
        StanceLabel.NEUTRAL.value,
    }


def test_evaluate_with_no_human_labels_returns_warning() -> None:
    records = [_record("a", label=StanceLabel.SUPPORTS_YES, confidence=0.4)]
    report = evaluate_stance(records, human_labels=None)
    assert report.accuracy is None
    assert report.macro_f1 is None
    assert any("no human labels" in w for w in report.warnings)
    assert report.total_predictions == 1
    assert report.per_label_counts[StanceLabel.SUPPORTS_YES.value] == 1


def test_evaluate_with_human_labels_computes_accuracy(tmp_path: Path) -> None:
    records = [
        _record("a", label=StanceLabel.SUPPORTS_YES, confidence=0.6),
        _record("b", label=StanceLabel.SUPPORTS_NO, confidence=0.5),
        _record("c", label=StanceLabel.SUPPORTS_YES, confidence=0.4),
    ]
    human_labels = {
        "a": HumanStanceLabel.SUPPORTS_YES,
        "b": HumanStanceLabel.SUPPORTS_NO,
        "c": HumanStanceLabel.SUPPORTS_NO,  # wrong prediction
    }
    report = evaluate_stance(records, human_labels=human_labels)
    assert report.accuracy == pytest.approx(2.0 / 3.0)
    assert report.total_with_labels == 3
    persisted = tmp_path / "report.json"
    from polysignal.stance.evaluate import write_evaluation_report

    write_evaluation_report(report, persisted)
    payload = json.loads(persisted.read_text(encoding="utf-8"))
    assert payload["accuracy"] == pytest.approx(2.0 / 3.0)


def test_write_and_read_stance_records_roundtrip(tmp_path: Path) -> None:
    record = _record("a", label=StanceLabel.SUPPORTS_YES, confidence=0.5)
    record.outcome_names = ["Candidate A", "Candidate B", "Other"]
    record.outcome_id = "candidate-a"
    record.target_outcome = "Candidate A"
    record.target_outcome_index = 0
    record.target_outcome_name = "Candidate A"
    record.target_aliases = ["candidate a"]
    record.is_multi_outcome = True
    records = [record]
    target = tmp_path / "out.parquet"
    write_stance_records(records, target)
    loaded = read_stance_records(target)
    assert loaded[0].stance_id == "a"
    assert loaded[0].stance_label is StanceLabel.SUPPORTS_YES
    assert loaded[0].outcome_names == ["Candidate A", "Candidate B", "Other"]
    assert loaded[0].outcome_id == "candidate-a"
    assert loaded[0].target_outcome == "Candidate A"
    assert loaded[0].target_outcome_index == 0
    assert loaded[0].target_outcome_name == "Candidate A"
    assert loaded[0].target_aliases == ["candidate a"]
    assert loaded[0].is_multi_outcome is True


def test_pipeline_skips_rows_without_text() -> None:
    rows = [
        _linked(
            link_id="l1",
            market_id="m1",
            post_id="p1",
            market_question="Will Bitcoin hit $150k by December 2026?",
            post_text_preview="",
            segment_text_preview="",
            matched_terms=["Bitcoin"],
        ),
    ]
    result = run_stance_pipeline(linked_posts=rows, dry_run=True)
    assert result.rows_skipped_text == 1
    assert result.rows_scored == 0


def test_pipeline_reads_segments_parquet_when_path_passed(tmp_path: Path) -> None:
    paths = get_data_paths(root=tmp_path)
    rows = [
        _linked(
            link_id="l1",
            market_id="m1",
            post_id="seg-1",
            market_question="Will Bitcoin hit $150k by December 2026?",
            segment_id="seg-1",
            parent_post_id="telegram:c1:42",
            segment_text_preview="cut-off preview",
            post_text_preview="ignored",
            matched_terms=["Bitcoin"],
        ),
    ]
    write_linked_posts(rows, paths.linked_posts_parquet)
    seg_path = tmp_path / "segments.parquet"
    write_parquet(
        [
            SocialPostSegment(
                segment_id="seg-1",
                parent_post_id="telegram:c1:42",
                platform=Platform.TELEGRAM,
                segment_index=0,
                segment_text=(
                    "Analysts expect Bitcoin to reach $150k before 2026 ends."
                ),
                created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
            )
        ],
        seg_path,
    )
    result = run_stance_pipeline(paths=paths, segments_path=seg_path)
    assert result.rows_scored == 1
    assert result.records[0].stance_label is StanceLabel.SUPPORTS_YES

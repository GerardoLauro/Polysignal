"""Offline tests for deterministic Graph Loader v1 planning and execution."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from polysignal.domain import (
    LinkedPost,
    Market,
    MarketDomain,
    Outcome,
    Platform,
    RelevanceLabel,
    SocialPostRecord,
    SocialPostSegment,
    StanceLabel,
)
from polysignal.domain.enums import ResolutionResult
from polysignal.graph.loaders import build_load_plan, execute_load_plan
from polysignal.stance.models import StanceRecord, write_stance_records
from polysignal.storage.parquet_store import read_parquet, write_parquet
from polysignal.storage.paths import DataPaths, get_data_paths

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def _outcome(market_id: str, outcome_id: str, label: str, index: int) -> Outcome:
    return Outcome(market_id=market_id, outcome_id=outcome_id, label=label, index=index)


def _market(market_id: str, outcomes: list[Outcome]) -> Market:
    return Market(
        market_id=market_id,
        slug=market_id,
        question=f"Will {market_id} happen?",
        domain=MarketDomain.POLITICS,
        created_at=NOW,
        outcomes=outcomes,
    )


def _post(post_id: str = "post-1") -> SocialPostRecord:
    return SocialPostRecord(
        post_id=post_id,
        platform=Platform.TELEGRAM,
        source_id=post_id,
        author_id="channel-1",
        author_name="Channel One",
        created_at=NOW,
        text="A sufficiently long graph loader fixture post.",
        collected_at=NOW,
    )


def _link(
    post_id: str,
    *,
    parent_post_id: str | None = None,
    link_id: str = "link-1",
    market_id: str = "binary",
) -> LinkedPost:
    return LinkedPost(
        link_id=link_id,
        market_id=market_id,
        post_id=post_id,
        parent_post_id=parent_post_id,
        segment_id=post_id if "#seg" in post_id else None,
        platform=Platform.TELEGRAM,
        query="fixture query",
        relevance_score=0.8,
        relevance_label=RelevanceLabel.RELEVANT,
        confidence=0.9,
        created_at=NOW,
    )


def _stance(
    stance_id: str,
    label: StanceLabel,
    *,
    market_id: str = "binary",
    post_id: str = "post-1#seg000",
    outcome_id: str | None = None,
) -> StanceRecord:
    return StanceRecord(
        stance_id=stance_id,
        market_id=market_id,
        post_id=post_id,
        segment_id=post_id if "#seg" in post_id else None,
        outcome_id=outcome_id,
        stance_label=label,
        stance_score=1.0 if label is StanceLabel.SUPPORTS_YES else -1.0,
        confidence=0.8,
        explanation="fixture rationale",
        created_at=NOW,
        source_platform=Platform.TELEGRAM,
    )


def _seed(
    root: Path,
    *,
    segment_link: bool = True,
    include_segments: bool = True,
    include_stances: bool = False,
) -> DataPaths:
    paths = get_data_paths(root=root)
    binary_outcomes = [
        _outcome("binary", "binary-yes", "Yes", 0),
        _outcome("binary", "binary-no", "No", 1),
    ]
    election_outcomes = [
        _outcome("election", "trump", "Donald Trump", 0),
        _outcome("election", "harris", "Kamala Harris", 1),
        _outcome("election", "other", "Other", 2),
    ]
    write_parquet(
        [_market("binary", binary_outcomes), _market("election", election_outcomes)],
        paths.polymarket_markets_parquet,
    )
    write_parquet([*binary_outcomes, *election_outcomes], paths.polymarket_outcomes_parquet)
    write_parquet([_post()], paths.social_posts_parquet)
    segment_id = "post-1#seg000"
    if include_segments:
        write_parquet(
            [
                SocialPostSegment(
                    segment_id=segment_id,
                    parent_post_id="post-1",
                    platform=Platform.TELEGRAM,
                    segment_index=0,
                    segment_text="A sufficiently long graph loader segment.",
                    created_at=NOW,
                )
            ],
            paths.social_post_segments_parquet,
        )
    links = [_link(segment_id if segment_link else "post-1")]
    if include_stances:
        links.append(
            _link(
                segment_id if segment_link else "post-1",
                link_id="link-election",
                market_id="election",
            )
        )
    write_parquet(links, paths.linked_posts_parquet)
    if include_stances:
        rows = [
            _stance("stance-yes", StanceLabel.SUPPORTS_YES),
            _stance("stance-no", StanceLabel.SUPPORTS_NO),
            _stance("stance-neutral", StanceLabel.NEUTRAL),
            _stance("stance-ambiguous", StanceLabel.AMBIGUOUS),
            _stance("stance-context", StanceLabel.INSUFFICIENT_CONTEXT),
            _stance("stance-unrelated", StanceLabel.UNRELATED),
            _stance(
                "stance-multi-no",
                StanceLabel.SUPPORTS_NO,
                market_id="election",
                outcome_id="trump",
            ),
            _stance("stance-multi-unresolved", StanceLabel.SUPPORTS_YES, market_id="election"),
        ]
        write_stance_records(rows, paths.stance_labels_parquet)
    return paths


def test_plan_reports_required_and_missing_optional_datasets(tmp_path: Path):
    paths = _seed(tmp_path)
    plan = build_load_plan(paths=paths)
    assert plan.datasets["markets"].row_count == 2
    assert plan.datasets["stance_labels"].exists is False
    assert "stance_labels" in plan.missing_optional_datasets
    assert plan.planned_node_counts == {
        "Contract": 2,
        "Domain": 1,
        "Outcome": 5,
        "Platform": 1,
        "Post": 1,
        "Author": 1,
    }
    assert plan.planned_relationship_counts["LINKED_TO"] == 1
    assert plan.planned_relationship_counts["SUPPORTS"] == 0


def test_segment_link_resolves_to_parent_post_and_keeps_evidence(tmp_path: Path):
    plan = build_load_plan(paths=_seed(tmp_path))
    row = plan.operation("link_post_contracts").rows[0]
    assert row["parent_post_id"] == "post-1"
    assert row["segment_id"] == "post-1#seg000"
    assert row["source_linked_post_id"] == "post-1#seg000"
    assert row["retrieval_query"] == "fixture query"


def test_direct_link_works_without_segments_dataset(tmp_path: Path):
    plan = build_load_plan(
        paths=_seed(tmp_path, segment_link=False, include_segments=False)
    )
    row = plan.operation("link_post_contracts").rows[0]
    assert row["parent_post_id"] == "post-1"
    assert row["segment_id"] is None
    assert "social_post_segments" in plan.missing_optional_datasets


def test_stance_edges_are_directional_and_abstentions_are_skipped(tmp_path: Path):
    plan = build_load_plan(paths=_seed(tmp_path, include_stances=True))
    supports = plan.operation("link_post_supports_outcome").rows
    opposes = plan.operation("link_post_opposes_outcome").rows
    assert {(row["stance_id"], row["outcome_id"]) for row in supports} == {
        ("stance-yes", "binary-yes"),
        ("stance-no", "binary-no"),
    }
    assert {(row["stance_id"], row["outcome_id"]) for row in opposes} == {
        ("stance-multi-no", "trump"),
    }
    assert plan.skipped_rows["stance_non_directional"] == 4
    assert plan.skipped_rows["stance_unresolved_endpoint"] == 1


def test_required_dataset_missing_is_actionable(tmp_path: Path):
    paths = get_data_paths(root=tmp_path)
    with pytest.raises(FileNotFoundError, match="markets"):
        build_load_plan(paths=paths)


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    def run_write(self, statement: str, rows: list[dict[str, Any]]) -> None:
        self.calls.append((statement, rows))


def test_execute_load_plan_batches_without_live_neo4j(tmp_path: Path):
    plan = build_load_plan(paths=_seed(tmp_path))
    client = _FakeClient()
    result = execute_load_plan(client, plan, batch_size=1)
    assert result.batches_executed == len(client.calls)
    assert result.rows_processed == sum(len(rows) for _, rows in client.calls)
    assert result.rows_processed > 0
    assert all(len(rows) == 1 for _, rows in client.calls)


def test_loader_accepts_only_temporally_valid_evidence(tmp_path: Path):
    paths = get_data_paths(root=tmp_path)
    outcomes = [_outcome("closed", "yes", "Yes", 0), _outcome("closed", "no", "No", 1)]
    market = Market(
        market_id="closed",
        slug="closed",
        question="Will the closed market happen?",
        domain=MarketDomain.POLITICS,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        close_time=datetime(2026, 5, 1, tzinfo=timezone.utc),
        is_closed=True,
        is_resolved=True,
        resolution=ResolutionResult.YES,
        outcomes=outcomes,
    )
    post = _post()
    write_parquet([market], paths.polymarket_markets_parquet)
    write_parquet(outcomes, paths.polymarket_outcomes_parquet)
    write_parquet([post], paths.social_posts_parquet)

    def link(
        link_id: str,
        label: RelevanceLabel,
        confidence: float,
        created_at: datetime,
    ) -> LinkedPost:
        row = _link("post-1", link_id=link_id, market_id="closed")
        row.relevance_label = label
        row.confidence = confidence
        row.post_created_at = created_at
        return row

    pre_close = datetime(2026, 4, 1, tzinfo=timezone.utc)
    post_close = datetime(2026, 5, 2, tzinfo=timezone.utc)
    write_parquet(
        [
            link("accepted", RelevanceLabel.RELEVANT, 0.9, pre_close),
            link("unrelated", RelevanceLabel.UNRELATED, 0.9, pre_close),
            link("ambiguous", RelevanceLabel.AMBIGUOUS, 0.9, pre_close),
            link("weak-low", RelevanceLabel.WEAKLY_RELEVANT, 0.49, pre_close),
            link("late", RelevanceLabel.RELEVANT, 0.9, post_close),
        ],
        paths.linked_posts_parquet,
    )
    plan = build_load_plan(paths=paths)
    [accepted] = plan.operation("link_post_contracts").rows
    assert accepted["link_id"] == "accepted"
    assert accepted["accepted"] is True
    assert accepted["temporal_valid"] is True
    assert accepted["temporal_anchor_kind"] == "close_time"
    assert plan.skipped_rows["skipped_unrelated"] == 1
    assert plan.skipped_rows["skipped_ambiguous"] == 1
    assert plan.skipped_rows["skipped_low_confidence"] == 1
    assert plan.skipped_rows["skipped_post_close"] == 1


def test_loader_skips_historical_link_without_temporal_anchor(tmp_path: Path):
    paths = _seed(tmp_path, segment_link=False, include_segments=False)
    markets = read_parquet(paths.polymarket_markets_parquet, Market)
    markets[0].is_closed = True
    write_parquet(markets, paths.polymarket_markets_parquet)
    plan = build_load_plan(paths=paths)
    assert plan.planned_relationship_counts["LINKED_TO"] == 0
    assert plan.skipped_rows["skipped_temporal_anchor_missing"] == 1


def test_loader_keeps_missing_timestamp_post_but_skips_its_link(tmp_path: Path):
    paths = _seed(tmp_path, segment_link=False, include_segments=False)
    posts = read_parquet(paths.social_posts_parquet, SocialPostRecord)
    posts[0].created_at = None
    posts[0].created_at_parse_status = "missing"
    write_parquet(posts, paths.social_posts_parquet)
    plan = build_load_plan(paths=paths)
    [post_row] = plan.operation("upsert_posts").rows
    assert post_row["created_at"] is None
    assert plan.operation("link_post_contracts").rows == ()
    assert plan.skipped_rows["skipped_post_timestamp_missing"] == 1

"""Offline tests for the Telegram coverage report and bundle manifest."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from polysignal.dataset import (
    HighSignalDomain,
    HighSignalMarket,
    build_coverage_report,
    build_manifest,
    build_telegram_search_plan,
    write_coverage_report,
    write_manifest,
)
from polysignal.dataset.models import DATASET_TARGETS
from polysignal.domain import Platform, SocialPostRecord, SocialPostSegment


def _market(market_id: str = "m-c") -> HighSignalMarket:
    return HighSignalMarket(
        market_id=market_id,
        slug=market_id,
        question="Will Bitcoin pass $200K by end of 2026?",
        domain=HighSignalDomain.FINANCE_MACRO_CRYPTO,
        polymarket_domain="crypto",
        outcomes=["Yes", "No"],
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        close_time=datetime(2026, 12, 31, tzinfo=timezone.utc),
        social_searchability_score=0.8,
        telegram_searchability_score=0.8,
        market_priority_score=0.8,
        selection_reason="test",
        search_window_start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        search_window_end=datetime(2026, 12, 31, tzinfo=timezone.utc),
        canonical_keywords=["Bitcoin", "200K"],
        suggested_telegram_queries=["Bitcoin 200K 2026"],
    )


def _telegram_record(
    *,
    post_id: str,
    market_id: str,
    text: str,
    created_at: datetime,
) -> SocialPostRecord:
    return SocialPostRecord(
        post_id=post_id,
        platform=Platform.TELEGRAM,
        source_id=post_id,
        created_at=created_at,
        text=text,
        market_id_candidate=market_id,
        query="bitcoin",
        collected_at=created_at,
    )


def _segment(parent_post_id: str, idx: int) -> SocialPostSegment:
    return SocialPostSegment(
        segment_id=f"{parent_post_id}#seg{idx:03d}",
        parent_post_id=parent_post_id,
        platform=Platform.TELEGRAM,
        segment_index=idx,
        segment_text="Bitcoin price update",
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )


def test_coverage_report_zero_messages_marks_all_markets_insufficient() -> None:
    markets = [_market("m1"), _market("m2")]
    plan = build_telegram_search_plan(markets, channels=())
    report = build_coverage_report(selected=markets, plan=plan)
    assert report.raw_messages_collected == 0
    assert report.normalized_messages == 0
    assert report.segments_generated == 0
    assert set(report.markets_with_insufficient_coverage) == {"m1", "m2"}
    assert report.plan_entries_total == len(plan)


def test_coverage_report_counts_messages_and_segments_per_market() -> None:
    market = _market("m1")
    plan = build_telegram_search_plan([market], channels=())
    messages = [
        _telegram_record(
            post_id=f"telegram:1:{i}",
            market_id="m1",
            text="Bitcoin price rally toward 200K",
            created_at=datetime(2026, 6, i + 1, tzinfo=timezone.utc),
        )
        for i in range(15)
    ]
    segments = [_segment(messages[0].post_id, idx) for idx in range(4)]
    report = build_coverage_report(
        selected=[market],
        plan=plan,
        messages=messages,
        segments=segments,
    )
    assert report.raw_messages_collected == 15
    assert report.normalized_messages == 15
    assert report.segments_generated == 4
    assert report.per_market[0].raw_messages == 15
    assert report.per_market[0].segments == 4
    assert "Bitcoin" in report.per_market[0].matched_keywords or report.per_market[0].matched_keywords == []


def test_coverage_report_ignores_non_telegram_platforms() -> None:
    market = _market("m1")
    plan = build_telegram_search_plan([market], channels=())
    reddit = SocialPostRecord(
        post_id="reddit:1",
        platform=Platform.REDDIT,
        source_id="reddit:1",
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        text="off topic",
        market_id_candidate="m1",
        collected_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    report = build_coverage_report(selected=[market], plan=plan, messages=[reddit])
    assert report.raw_messages_collected == 0
    assert report.normalized_messages == 0


def test_coverage_report_writes_valid_json(tmp_path: Path) -> None:
    market = _market("m1")
    plan = build_telegram_search_plan([market], channels=())
    report = build_coverage_report(selected=[market], plan=plan, limitations=["no telegram creds"])
    target = tmp_path / "report.json"
    write_coverage_report(report, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["selected_markets_total"] == 1
    assert "no telegram creds" in payload["limitations"]
    assert payload["targets"]["selected_market_target"] == DATASET_TARGETS.selected_market_target


def test_manifest_lists_every_artefact_and_uses_optional_paths(tmp_path: Path) -> None:
    payload = build_manifest(
        selected_markets_path=tmp_path / "markets.parquet",
        telegram_plan_parquet_path=tmp_path / "plan.parquet",
        telegram_plan_json_path=tmp_path / "plan.json",
        coverage_report_path=None,
        raw_run_dirs=(tmp_path / "run1",),
        normalized_messages_path=None,
        segments_path=None,
        channels_config_path=None,
        targets=DATASET_TARGETS,
        notes=("no telegram credentials yet",),
    )
    assert payload["artefacts"]["selected_markets_parquet"].endswith("markets.parquet")
    assert payload["artefacts"]["telegram_coverage_report_json"] is None
    assert payload["artefacts"]["raw_run_dirs"]
    target = tmp_path / "manifest.json"
    write_manifest(payload, target)
    assert json.loads(target.read_text(encoding="utf-8"))["dataset_name"] == "high_signal_telegram"

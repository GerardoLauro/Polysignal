"""Offline tests for the Telegram search plan builder."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from polysignal.dataset import (
    HighSignalDomain,
    HighSignalMarket,
    build_telegram_search_plan,
    write_telegram_search_plan,
)
from polysignal.ingestion.social.telegram_normalizer import TelegramChannel


def _market(
    market_id: str,
    *,
    domain: HighSignalDomain = HighSignalDomain.FINANCE_MACRO_CRYPTO,
    queries: tuple[str, ...] = ("Bitcoin 200K 2026", "Bitcoin", "BTC price"),
) -> HighSignalMarket:
    return HighSignalMarket(
        market_id=market_id,
        slug=market_id,
        question="Will Bitcoin pass $200K by end of 2026?",
        domain=domain,
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
        canonical_keywords=["Bitcoin", "200K", "2026"],
        suggested_telegram_queries=list(queries),
    )


def test_plan_emits_one_row_per_market_when_no_channels_configured() -> None:
    plan = build_telegram_search_plan([_market("m1")], channels=(), queries_per_market=3)
    assert all(entry.channel_id is None for entry in plan)
    assert {entry.market_id for entry in plan} == {"m1"}
    assert len(plan) == 9  # queries_per_market * 3 non-overlapping windows


def test_plan_prefers_channels_matching_market_domain() -> None:
    channels = [
        TelegramChannel(username="cointelegraph", domain="crypto"),
        TelegramChannel(username="politicschan", domain="politics"),
    ]
    market = _market("m-crypto", domain=HighSignalDomain.FINANCE_MACRO_CRYPTO)
    plan = build_telegram_search_plan([market], channels=channels, queries_per_market=2)
    channel_ids = {entry.channel_id for entry in plan}
    assert "cointelegraph" in channel_ids
    assert "politicschan" not in channel_ids


def test_plan_falls_back_to_general_news_channel_when_none_match_domain() -> None:
    channels = [TelegramChannel(username="generic", domain_tags=("news",))]
    market = _market("m-crypto", domain=HighSignalDomain.FINANCE_MACRO_CRYPTO)
    plan = build_telegram_search_plan([market], channels=channels, queries_per_market=2)
    assert {entry.channel_id for entry in plan} == {"generic"}
    assert any(entry.expected_relevance == "best-effort" for entry in plan)


def test_plan_respects_max_queries_per_channel() -> None:
    channels = [TelegramChannel(username="cointelegraph", domain="crypto")]
    market = _market(
        "m-c",
        queries=("Q1", "Q2", "Q3", "Q4", "Q5"),
    )
    plan = build_telegram_search_plan(
        [market],
        channels=channels,
        queries_per_market=5,
        max_queries_per_channel=2,
    )
    assert len(plan) == 6
    assert {entry.query for entry in plan} == {"Q1", "Q2"}


def test_plan_emits_non_overlapping_pre_close_bands() -> None:
    market = _market("m-w")
    plan = build_telegram_search_plan([market], channels=(), queries_per_market=1)
    assert [entry.window_label for entry in plan] == ["0-7d", "8-30d", "31-90d"]
    assert plan[0].search_window_end == market.search_window_end
    assert plan[1].search_window_end < plan[0].search_window_start
    assert plan[2].search_window_end < plan[1].search_window_start


def test_plan_emits_at_least_one_row_even_with_no_suggested_queries() -> None:
    market = _market("m-empty", queries=())
    plan = build_telegram_search_plan([market], channels=())
    assert len(plan) == 3
    assert plan[0].query  # falls back to a truncated question


def test_plan_serialises_to_parquet_and_json(tmp_path: Path) -> None:
    market = _market("m-w")
    entries = build_telegram_search_plan([market], channels=(), queries_per_market=1)
    parquet_path = tmp_path / "plan.parquet"
    json_path = tmp_path / "plan.json"
    write_telegram_search_plan(entries, parquet_path, json_path)
    assert parquet_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "generated_at" in payload
    assert payload["entries"][0]["market_id"] == "m-w"


def test_plan_routes_using_all_channel_domain_tags() -> None:
    channels = [TelegramChannel(username="mixed", domain_tags=("news", "politics"))]
    market = _market("m-politics", domain=HighSignalDomain.POLITICS)
    plan = build_telegram_search_plan([market], channels=channels, queries_per_market=1)
    assert {entry.channel_id for entry in plan} == {"mixed"}
    assert {entry.expected_relevance for entry in plan} == {"domain-match"}


def test_plan_work_ids_are_stable_and_binary_outcome_aliases_are_ignored() -> None:
    market = _market("m-stable", queries=()).model_copy(
        update={"outcome_aliases": ["Yes", "No", "Candidate Smith"]}
    )
    first = build_telegram_search_plan([market], channels=(), queries_per_market=3)
    second = build_telegram_search_plan([market], channels=(), queries_per_market=3)
    assert [entry.work_id for entry in first] == [entry.work_id for entry in second]
    assert {entry.query for entry in first} == {"Candidate Smith"}

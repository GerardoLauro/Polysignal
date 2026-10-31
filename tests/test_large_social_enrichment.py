"""Offline tests for the resumable Branch 12B public-web collector."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from polysignal.domain import Market, MarketDomain, Outcome, Platform, SocialPostRecord
from polysignal.ingestion.social.enrichment import FetchOutcome
from polysignal.ingestion.social.enrichment_config import (
    EnrichmentConfig,
    RedditSectionConfig,
    RedditSource,
    TelegramSectionConfig,
    TelegramSource,
)
from polysignal.ingestion.social.large_enrichment import (
    LargeEnrichmentOptions,
    build_market_query_plan,
    classify_temporal_relation,
    collect_large_enriched,
    market_window,
    read_json,
    reddit_global_search_url,
    reddit_restricted_search_url,
    validate_telegram_public_url,
)
from polysignal.ingestion.social.normalizer import build_record
from polysignal.ingestion.social.reddit_scrapling_adapter import reddit_raw_to_social_post
from polysignal.ingestion.social.segmentation import segment_records
from polysignal.storage.parquet_store import read_parquet
from polysignal.storage.paths import get_data_paths

FIXTURES = Path(__file__).parent / "fixtures" / "social_enrichment"
REDDIT_HTML = (FIXTURES / "old_reddit_sample.html").read_text(encoding="utf-8")
TELEGRAM_HTML = (FIXTURES / "telegram_sample.html").read_text(encoding="utf-8")
NOW = datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc)


class FakeFetcher:
    def __init__(self, responses: dict[str, FetchOutcome]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get(self, url: str) -> FetchOutcome:
        self.calls.append(url)
        return self.responses.get(
            url,
            FetchOutcome(url=url, ok=False, status=404, error="not configured"),
        )


def _market(*, created_at: datetime | None = None) -> Market:
    return Market(
        market_id="m1",
        condition_id="condition-1",
        slug="bitcoin-price-market",
        question="Will Bitcoin rise above $100K by December 2026?",
        domain=MarketDomain.CRYPTO,
        created_at=created_at or NOW - timedelta(days=30),
        close_time=NOW + timedelta(days=30),
        outcomes=[
            Outcome(outcome_id="m1-yes", label="Yes", index=0),
            Outcome(outcome_id="m1-no", label="No", index=1),
        ],
        tags=["crypto", "bitcoin"],
    )


def _config(tmp_path: Path) -> EnrichmentConfig:
    return EnrichmentConfig(
        path=tmp_path / "sources.yaml",
        reddit=RedditSectionConfig(
            rate_limit_seconds=0.0,
            max_pages_per_source=1,
            max_items_per_source=10,
            sources=(
                RedditSource(
                    name="reddit_crypto",
                    url="https://old.reddit.com/r/CryptoCurrency/new/",
                    subreddit="CryptoCurrency",
                    domains=("crypto",),
                ),
            ),
        ),
        telegram=TelegramSectionConfig(
            rate_limit_seconds=0.0,
            max_pages_per_source=1,
            max_items_per_source=10,
            sources=(
                TelegramSource(
                    name="telegram_cointelegraph",
                    channel="cointelegraph",
                    url="https://t.me/s/cointelegraph",
                    domains=("crypto",),
                ),
            ),
        ),
    )


def _no_sleep(_: float) -> None:
    return None


def test_market_window_and_temporal_classification() -> None:
    market = _market(created_at=NOW - timedelta(days=400))
    start, close, source = market_window(market, fallback_window_days=180) or (None, None, "")
    assert start == market.close_time - timedelta(days=180)
    assert close == market.close_time
    assert source == "fallback_close_minus_180_days"
    assert classify_temporal_relation(
        close - timedelta(days=1), window_start=start, close_time=close
    ).relation == "inside_window"
    assert classify_temporal_relation(
        close + timedelta(seconds=1), window_start=start, close_time=close
    ).relation == "post_close"
    assert classify_temporal_relation(
        start - timedelta(seconds=1), window_start=start, close_time=close
    ).relation == "outside_window"
    assert classify_temporal_relation(
        close - timedelta(days=1), window_start=None, close_time=close
    ).relation == "pre_close"
    missing = classify_temporal_relation(None, window_start=start, close_time=close)
    assert missing.leakage_eligible is False
    assert missing.exclusion_reason == "post_timestamp_missing"


def test_public_url_helpers_are_bounded_and_encoded() -> None:
    assert validate_telegram_public_url("https://t.me/s/cointelegraph")
    assert not validate_telegram_public_url("https://t.me/cointelegraph")
    assert not validate_telegram_public_url("https://example.com/s/cointelegraph")
    assert reddit_global_search_url("BTC $100K") == (
        "https://old.reddit.com/search/?q=BTC+%24100K&sort=new&t=all"
    )
    assert reddit_restricted_search_url("CryptoCurrency", "ETH merge") == (
        "https://old.reddit.com/r/CryptoCurrency/search/"
        "?q=ETH+merge&restrict_sr=on&sort=new&t=all"
    )


def test_market_query_plan_is_deterministic_and_market_aware(tmp_path: Path) -> None:
    source = _config(tmp_path).reddit.sources[0]  # type: ignore[union-attr]
    first = build_market_query_plan([_market()], [source], now=NOW)
    second = build_market_query_plan([_market()], [source], now=NOW)
    assert [row.model_dump(mode="json") for row in first] == [
        row.model_dump(mode="json") for row in second
    ]
    assert first
    assert len(first) <= 10
    assert first[0].market_id == "m1"
    assert first[0].window_source == "market_created_at"
    assert first[0].selected_sources == ["global", "reddit_crypto"]


def test_missing_timestamp_record_is_retained_but_not_segmented() -> None:
    source = RedditSource(name="r", url="x", subreddit="x")
    post = reddit_raw_to_social_post(
        {"title": "timestamp audit row", "timestamp_ms": None, "fullname": "t3_missing"},
        source=source,
        collection_method="scrapling_public_web",
        source_url="x",
    )
    assert post is not None
    record = build_record(
        post=post,
        query=None,
        market_id_candidate=None,
        raw_path=None,
        collected_at=NOW,
    )
    assert record.created_at is None
    assert record.created_at_parse_status == "missing"
    assert segment_records([record]) == []


def test_collect_large_enriched_flushes_checkpoints_and_resumes(tmp_path: Path) -> None:
    reddit_url = "https://old.reddit.com/r/CryptoCurrency/new/"
    telegram_url = "https://t.me/s/cointelegraph"
    responses = {
        reddit_url: FetchOutcome(url=reddit_url, ok=True, status=200, html=REDDIT_HTML),
        telegram_url: FetchOutcome(url=telegram_url, ok=True, status=200, html=TELEGRAM_HTML),
    }
    reddit_paths = get_data_paths(root=tmp_path / "reddit")
    telegram_paths = get_data_paths(root=tmp_path / "telegram")
    output_paths = get_data_paths(root=tmp_path / "combined")
    result = collect_large_enriched(
        config=_config(tmp_path),
        markets=[_market()],
        reddit_paths=reddit_paths,
        telegram_paths=telegram_paths,
        output_paths=output_paths,
        options=LargeEnrichmentOptions(
            target_total_records=5,
            target_reddit_records=2,
            target_telegram_records=3,
            max_pages_per_source=1,
            max_markets=1,
            max_queries_per_market=1,
            rate_limit_seconds=0.0,
        ),
        reddit_fetcher=FakeFetcher(responses),
        telegram_fetcher=FakeFetcher(responses),
        now=NOW,
        sleep=_no_sleep,
    )
    assert result.summary["final_unique"] == {"reddit": 2, "telegram": 3, "total": 5}
    assert result.summary["targets"]["reached"] is True
    assert result.summary["objective_targets"]["total"] == 50_000
    assert result.summary["objective_targets"]["reached"] is False
    assert result.summary["completion"]["reason"] == "staged_target_reached"
    assert result.summary["persisted_retrieval_rows"] == 5
    assert result.summary["usefulness"]["timestamp_parsed"] == 5
    assert result.summary["collection_progress"]["fetched_urls"] == 2
    assert output_paths.social_collection_state_json.exists()
    assert output_paths.social_source_progress_json.exists()
    assert output_paths.social_query_progress_json.exists()
    assert reddit_paths.social_collection_state_json.exists()
    assert reddit_paths.social_source_progress_json.exists()
    assert reddit_paths.social_query_progress_json.exists()
    assert telegram_paths.social_collection_state_json.exists()
    assert telegram_paths.social_source_progress_json.exists()
    assert telegram_paths.social_query_progress_json.exists()
    assert output_paths.market_queries_parquet.exists()
    assert output_paths.market_queries_csv.exists()
    assert output_paths.social_collection_summary_json.exists()
    assert len(read_parquet(output_paths.social_posts_parquet, SocialPostRecord)) == 5

    reddit_fetcher = FakeFetcher({})
    telegram_fetcher = FakeFetcher({})
    resumed = collect_large_enriched(
        config=_config(tmp_path),
        markets=[_market()],
        reddit_paths=reddit_paths,
        telegram_paths=telegram_paths,
        output_paths=output_paths,
        options=LargeEnrichmentOptions(
            target_total_records=5,
            target_reddit_records=2,
            target_telegram_records=3,
            max_pages_per_source=1,
            max_markets=1,
            max_queries_per_market=1,
            rate_limit_seconds=0.0,
        ),
        reddit_fetcher=reddit_fetcher,
        telegram_fetcher=telegram_fetcher,
        now=NOW,
        sleep=_no_sleep,
    )
    assert resumed.summary["final_unique"]["total"] == 5
    assert reddit_fetcher.calls == []
    assert telegram_fetcher.calls == []


def test_resume_rejects_changed_source_config(tmp_path: Path) -> None:
    reddit_url = "https://old.reddit.com/r/CryptoCurrency/new/"
    config = _config(tmp_path)
    reddit_paths = get_data_paths(root=tmp_path / "reddit")
    telegram_paths = get_data_paths(root=tmp_path / "telegram")
    output_paths = get_data_paths(root=tmp_path / "combined")
    collect_large_enriched(
        config=config,
        markets=[_market()],
        reddit_paths=reddit_paths,
        telegram_paths=telegram_paths,
        output_paths=output_paths,
        options=LargeEnrichmentOptions(
            target_total_records=2,
            target_reddit_records=2,
            target_telegram_records=0,
            max_pages_per_source=1,
            max_queries_per_market=1,
            rate_limit_seconds=0.0,
        ),
        reddit_fetcher=FakeFetcher(
            {reddit_url: FetchOutcome(url=reddit_url, ok=True, status=200, html=REDDIT_HTML)}
        ),
        telegram_fetcher=FakeFetcher({}),
        now=NOW,
        sleep=_no_sleep,
    )
    assert config.reddit is not None
    changed = replace(
        config,
        reddit=replace(config.reddit, sources=(replace(config.reddit.sources[0], priority="low"),)),
    )
    with pytest.raises(ValueError, match="source config"):
        collect_large_enriched(
            config=changed,
            markets=[_market()],
            reddit_paths=reddit_paths,
            telegram_paths=telegram_paths,
            output_paths=output_paths,
            options=LargeEnrichmentOptions(
                target_total_records=2,
                target_reddit_records=2,
                target_telegram_records=0,
                max_pages_per_source=1,
                max_queries_per_market=1,
                rate_limit_seconds=0.0,
            ),
            reddit_fetcher=FakeFetcher({}),
            telegram_fetcher=FakeFetcher({}),
            now=NOW,
            sleep=_no_sleep,
        )


def test_collect_large_enriched_dry_run_has_no_writes(tmp_path: Path) -> None:
    output_paths = get_data_paths(root=tmp_path / "combined")
    result = collect_large_enriched(
        config=_config(tmp_path),
        markets=[_market()],
        reddit_paths=get_data_paths(root=tmp_path / "reddit"),
        telegram_paths=get_data_paths(root=tmp_path / "telegram"),
        output_paths=output_paths,
        options=LargeEnrichmentOptions(dry_run=True, max_markets=1),
        now=NOW,
    )
    assert result.dry_run is True
    assert result.plan
    assert not output_paths.root.exists()


def test_telegram_cursor_pagination_checkpoints_each_page(tmp_path: Path) -> None:
    telegram_url = "https://t.me/s/cointelegraph"
    before_url = f"{telegram_url}?before=70101"
    reddit_paths = get_data_paths(root=tmp_path / "reddit")
    telegram_paths = get_data_paths(root=tmp_path / "telegram")
    output_paths = get_data_paths(root=tmp_path / "combined")
    second_page = (
        TELEGRAM_HTML.replace("70101", "70001")
        .replace("70103", "70003")
        .replace("70104", "70004")
    )

    class CheckpointFetcher(FakeFetcher):
        def get(self, url: str) -> FetchOutcome:
            if self.calls:
                state = read_json(output_paths.social_collection_state_json)
                progress = read_json(output_paths.social_source_progress_json)
                work = next(iter(progress.values()))
                assert state["fetched_urls"] == [telegram_url]
                assert work["pages"] == 1
                assert work["cursor"] == "70101"
            return super().get(url)

    fetcher = CheckpointFetcher(
        {
            telegram_url: FetchOutcome(url=telegram_url, ok=True, status=200, html=TELEGRAM_HTML),
            before_url: FetchOutcome(url=before_url, ok=True, status=200, html=second_page),
        }
    )
    result = collect_large_enriched(
        config=_config(tmp_path),
        markets=[_market()],
        reddit_paths=reddit_paths,
        telegram_paths=telegram_paths,
        output_paths=output_paths,
        options=LargeEnrichmentOptions(
            target_total_records=6,
            target_reddit_records=0,
            target_telegram_records=6,
            max_pages_per_source=2,
            max_queries_per_market=1,
            rate_limit_seconds=0.0,
        ),
        reddit_fetcher=FakeFetcher({}),
        telegram_fetcher=fetcher,
        now=NOW,
        sleep=_no_sleep,
    )
    assert fetcher.calls == [telegram_url, before_url]
    assert result.summary["final_unique"] == {"reddit": 0, "telegram": 6, "total": 6}


def test_telegram_shortfall_transfers_to_reddit_target(tmp_path: Path) -> None:
    reddit_url = "https://old.reddit.com/r/CryptoCurrency/new/"
    telegram_url = "https://t.me/s/cointelegraph"
    result = collect_large_enriched(
        config=_config(tmp_path),
        markets=[_market()],
        reddit_paths=get_data_paths(root=tmp_path / "reddit"),
        telegram_paths=get_data_paths(root=tmp_path / "telegram"),
        output_paths=get_data_paths(root=tmp_path / "combined"),
        options=LargeEnrichmentOptions(
            target_total_records=5,
            target_reddit_records=1,
            target_telegram_records=4,
            max_pages_per_source=1,
            max_queries_per_market=1,
            rate_limit_seconds=0.0,
        ),
        reddit_fetcher=FakeFetcher(
            {reddit_url: FetchOutcome(url=reddit_url, ok=True, status=200, html=REDDIT_HTML)}
        ),
        telegram_fetcher=FakeFetcher(
            {telegram_url: FetchOutcome(url=telegram_url, ok=True, status=200, html=TELEGRAM_HTML)}
        ),
        now=NOW,
        sleep=_no_sleep,
    )
    assert result.summary["final_unique"] == {"reddit": 2, "telegram": 3, "total": 5}
    assert result.summary["targets"]["telegram_shortfall_shifted_to_reddit"] == 1
    assert result.summary["targets"]["effective_reddit_target"] == 2
    assert result.summary["targets"]["reached"] is True


def test_duplicate_telegram_pages_report_exact_drops(tmp_path: Path) -> None:
    telegram_url = "https://t.me/s/cointelegraph"
    before_url = f"{telegram_url}?before=70101"
    telegram_paths = get_data_paths(root=tmp_path / "telegram")
    result = collect_large_enriched(
        config=replace(_config(tmp_path), reddit=None),
        markets=[_market()],
        reddit_paths=get_data_paths(root=tmp_path / "reddit"),
        telegram_paths=telegram_paths,
        output_paths=get_data_paths(root=tmp_path / "combined"),
        options=LargeEnrichmentOptions(
            target_total_records=10,
            target_reddit_records=0,
            target_telegram_records=10,
            max_pages_per_source=2,
            max_queries_per_market=1,
            rate_limit_seconds=0.0,
        ),
        telegram_fetcher=FakeFetcher(
            {
                telegram_url: FetchOutcome(
                    url=telegram_url, ok=True, status=200, html=TELEGRAM_HTML
                ),
                before_url: FetchOutcome(
                    url=before_url, ok=True, status=200, html=TELEGRAM_HTML
                ),
            }
        ),
        now=NOW,
        sleep=_no_sleep,
    )
    assert result.summary["persisted_retrieval_rows"] == 3
    assert result.summary["deduplication"]["telegram"]["exact_duplicates_dropped"] == 3
    platform_summary = read_json(telegram_paths.social_collection_summary_json)
    assert platform_summary["exact_duplicates_dropped"] == 3


def test_resume_reopens_work_when_page_cap_increases(tmp_path: Path) -> None:
    reddit_url = "https://old.reddit.com/r/CryptoCurrency/new/"
    next_url = "https://old.reddit.com/r/politics/new/?count=25&after=t3_bbb222"
    reddit_paths = get_data_paths(root=tmp_path / "reddit")
    telegram_paths = get_data_paths(root=tmp_path / "telegram")
    output_paths = get_data_paths(root=tmp_path / "combined")
    common = {
        "config": _config(tmp_path),
        "markets": [_market()],
        "reddit_paths": reddit_paths,
        "telegram_paths": telegram_paths,
        "output_paths": output_paths,
        "telegram_fetcher": FakeFetcher({}),
        "now": NOW,
        "sleep": _no_sleep,
    }
    collect_large_enriched(
        **common,
        options=LargeEnrichmentOptions(
            target_total_records=100,
            target_reddit_records=100,
            target_telegram_records=0,
            max_pages_per_source=1,
            max_pages_per_query=1,
            max_markets=1,
            max_queries_per_market=1,
            rate_limit_seconds=0.0,
        ),
        reddit_fetcher=FakeFetcher(
            {reddit_url: FetchOutcome(url=reddit_url, ok=True, status=200, html=REDDIT_HTML)}
        ),
    )
    fetcher = FakeFetcher(
        {next_url: FetchOutcome(url=next_url, ok=True, status=200, html="<html></html>")}
    )
    collect_large_enriched(
        **common,
        options=LargeEnrichmentOptions(
            target_total_records=100,
            target_reddit_records=100,
            target_telegram_records=0,
            max_pages_per_source=2,
            max_pages_per_query=1,
            max_markets=1,
            max_queries_per_market=1,
            rate_limit_seconds=0.0,
        ),
        reddit_fetcher=fetcher,
    )
    assert next_url in fetcher.calls


def test_reddit_429_opens_resume_circuit_breaker(tmp_path: Path) -> None:
    reddit_url = "https://old.reddit.com/r/CryptoCurrency/new/"
    reddit_paths = get_data_paths(root=tmp_path / "reddit")
    telegram_paths = get_data_paths(root=tmp_path / "telegram")
    output_paths = get_data_paths(root=tmp_path / "combined")
    common = {
        "config": _config(tmp_path),
        "markets": [_market()],
        "reddit_paths": reddit_paths,
        "telegram_paths": telegram_paths,
        "output_paths": output_paths,
        "options": LargeEnrichmentOptions(
            target_total_records=100,
            target_reddit_records=100,
            target_telegram_records=0,
            max_pages_per_source=1,
            max_markets=1,
            max_queries_per_market=1,
            rate_limit_seconds=0.0,
        ),
        "telegram_fetcher": FakeFetcher({}),
        "now": NOW,
        "sleep": _no_sleep,
    }
    collect_large_enriched(
        **common,
        reddit_fetcher=FakeFetcher(
            {reddit_url: FetchOutcome(url=reddit_url, ok=False, status=429)}
        ),
    )
    fetcher = FakeFetcher(
        {reddit_url: FetchOutcome(url=reddit_url, ok=True, status=200, html=REDDIT_HTML)}
    )
    resumed = collect_large_enriched(**common, reddit_fetcher=fetcher)
    assert fetcher.calls == []
    assert resumed.summary["circuit_breakers"]["reddit"]["reason"] == "http_status=429"


def test_existing_record_constructor_remains_backward_compatible() -> None:
    record = SocialPostRecord(
        post_id="t3_existing",
        platform=Platform.REDDIT,
        source_id="t3_existing",
        created_at=NOW,
        text="existing row",
        collected_at=NOW,
    )
    assert record.created_at_parse_status == "parsed"

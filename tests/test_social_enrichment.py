"""Offline tests for Branch 12 — Telegram + Reddit public-web dataset enrichment.

All tests are offline: HTML comes from small fixtures under
``tests/fixtures/social_enrichment`` and the network is replaced by a
:class:`FakeFetcher`. No live scraping happens here (live collection is opt-in
via the CLI, not the test suite).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from polysignal.domain import Market, MarketDomain, Outcome, Platform, SocialPostRecord
from polysignal.evaluation.benchmark_profiles import get_benchmark_profile
from polysignal.ingestion.social.enrichment import (
    FetchOutcome,
    combine_records,
    estimate_market_token_overlap,
    stable_post_id,
)
from polysignal.ingestion.social.enrichment_config import (
    EnrichmentOverrides,
    RedditSectionConfig,
    RedditSource,
    TelegramSectionConfig,
    TelegramSource,
    load_enrichment_config,
)
from polysignal.ingestion.social.enrichment_pipeline import (
    assemble_artifact_index,
    merge_enriched_datasets,
    read_downstream_coverage,
    render_coverage_report,
)
from polysignal.ingestion.social.reddit_scrapling_adapter import (
    collect_reddit_scrapling,
    find_next_page_url,
    parse_old_reddit_listing,
    reddit_raw_to_social_post,
)
from polysignal.ingestion.social.telegram_public_web_adapter import (
    collect_telegram_public_web,
    find_before_cursor,
    parse_telegram_public_page,
    telegram_public_raw_to_social_post,
)
from polysignal.storage.parquet_store import read_parquet
from polysignal.storage.paths import get_data_paths

FIXTURES = Path(__file__).parent / "fixtures" / "social_enrichment"
REDDIT_HTML = (FIXTURES / "old_reddit_sample.html").read_text(encoding="utf-8")
TELEGRAM_HTML = (FIXTURES / "telegram_sample.html").read_text(encoding="utf-8")
FIXED_NOW = datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc)


class FakeFetcher:
    """Deterministic fetcher: maps URLs to canned outcomes; no network."""

    def __init__(self, responses: dict[str, FetchOutcome]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get(self, url: str) -> FetchOutcome:
        self.calls.append(url)
        return self.responses.get(
            url, FetchOutcome(url=url, ok=False, status=404, error="not configured")
        )


def _no_sleep(_: float) -> None:
    return None


# ---------------------------------------------------------------------------
# Pure parsing
# ---------------------------------------------------------------------------


def test_parse_old_reddit_listing_extracts_fields_and_skips_ads() -> None:
    posts = parse_old_reddit_listing(REDDIT_HTML, default_subreddit="politics")
    assert len(posts) == 2  # promoted ad dropped
    first = posts[0]
    assert first["fullname"] == "t3_aaa111"
    assert first["author"] == "alice"
    assert first["subreddit"] == "politics"
    assert first["score"] == 128
    assert first["num_comments"] == 42
    # Nested <b> inside the title is flattened into plain text.
    assert first["title"] == "Election poll shifts ahead of November vote"
    assert first["timestamp_ms"] == 1780416497000


def test_find_next_page_url_unescapes_amp() -> None:
    url = find_next_page_url(REDDIT_HTML)
    assert url == "https://old.reddit.com/r/politics/new/?count=25&after=t3_bbb222"


def test_parse_telegram_public_page_extracts_messages() -> None:
    messages = parse_telegram_public_page(TELEGRAM_HTML, channel="cointelegraph")
    assert len(messages) == 3
    first = messages[0]
    assert first["channel"] == "cointelegraph"
    assert first["message_id"] == "70101"
    assert first["views"] == 8310
    assert "Bitcoin breaks above $80K" in first["text"]
    # <br> becomes a newline.
    assert "\n" in first["text"]
    # Media-only message (no text) still parses with views.
    assert messages[2]["text"] == ""
    assert messages[2]["views"] == 7740


def test_find_before_cursor() -> None:
    assert find_before_cursor(TELEGRAM_HTML) == "70101"
    assert find_before_cursor("<html></html>") is None


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_reddit_raw_to_social_post_normalizes_into_domain_model() -> None:
    raw = parse_old_reddit_listing(REDDIT_HTML, default_subreddit="politics")[0]
    source = RedditSource(name="reddit_politics", url="x", subreddit="politics", domains=("politics",))
    post = reddit_raw_to_social_post(
        raw,
        source=source,
        collection_method="scrapling_public_web",
        source_url="https://old.reddit.com/r/politics/new/",
        html_snapshot_path="/snap/p01.html",
    )
    assert post is not None
    assert post.platform is Platform.REDDIT
    assert post.post_id == "t3_aaa111"
    assert post.created_at.tzinfo is not None
    assert post.engagement.score == 128
    assert post.extra["collection_method"] == "scrapling_public_web"
    assert post.extra["source_name"] == "reddit_politics"
    assert post.extra["raw_snapshot_path"] == "/snap/p01.html"
    assert post.extra["subreddit"] == "politics"


def test_telegram_media_only_message_normalizes_with_views_and_no_text() -> None:
    raw = parse_telegram_public_page(TELEGRAM_HTML, channel="cointelegraph")[2]
    source = TelegramSource(
        name="telegram_cointelegraph", channel="cointelegraph", url="x", domains=("crypto",)
    )
    post = telegram_public_raw_to_social_post(
        raw, source=source, collection_method="telegram_public_web", source_url="x"
    )
    assert post is not None
    assert post.platform is Platform.TELEGRAM
    assert post.post_id == "telegram:cointelegraph:70104"
    assert post.text == ""
    assert post.engagement.views == 7740


def test_normalization_never_invents_timestamp() -> None:
    source = RedditSource(name="r", url="x", subreddit="x")
    post = reddit_raw_to_social_post(
        {"title": "no time", "timestamp_ms": None, "fullname": "t3_z"},
        source=source,
        collection_method="scrapling_public_web",
        source_url="x",
    )
    assert post is not None
    assert post.created_at is None
    assert post.extra["created_at_parse_status"] == "missing"


# ---------------------------------------------------------------------------
# Stable IDs + dedup
# ---------------------------------------------------------------------------


def test_stable_post_id_is_deterministic_and_distinct() -> None:
    assert stable_post_id("reddit", "politics", "title") == stable_post_id(
        "reddit", "politics", "title"
    )
    assert stable_post_id("reddit", "politics", "a") != stable_post_id("reddit", "politics", "b")
    assert stable_post_id("a", None, "b") == stable_post_id("a", None, "b")


def _record(post_id: str, platform: Platform, text: str = "hello world") -> SocialPostRecord:
    return SocialPostRecord(
        post_id=post_id,
        platform=platform,
        source_id=post_id,
        created_at=FIXED_NOW,
        text=text,
        collected_at=FIXED_NOW,
    )


def test_combine_records_flags_cross_source_duplicates() -> None:
    reddit = [_record("t3_a", Platform.REDDIT), _record("t3_b", Platform.REDDIT)]
    telegram = [_record("telegram:c:1", Platform.TELEGRAM)]
    duplicate = [_record("t3_a", Platform.REDDIT)]  # same identity as reddit[0]
    merged, dup_flagged = combine_records(reddit, telegram, duplicate)
    assert len(merged) == 3  # exact-duplicate retrieval tuple dropped on merge
    assert dup_flagged == 0
    # Identity dedup is deterministic: the same post_id seen again is dropped.
    _, dup2 = combine_records(reddit, [_record("t3_a", Platform.REDDIT, text="other")])
    assert dup2 == 0


# ---------------------------------------------------------------------------
# Config loading + validation
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path) -> Path:
    content = """
reddit_scrapling:
  enabled: true
  collection_method: "scrapling_public_web"
  rate_limit_seconds: 5
  max_pages_per_source: 10
  max_items_per_source: 300
  sources:
    - name: reddit_politics
      url: "https://old.reddit.com/r/politics/new/"
      subreddit: "politics"
      domains: ["politics"]
      priority: "high"
    - name: reddit_disabled
      subreddit: "x"
      enabled: false
    - name: reddit_flagged
      subreddit: "crypto"
      status: "needs_validation"
    - subreddit: "no_name"   # invalid: missing name
telegram_public_web:
  enabled: true
  collection_method: "telegram_public_web"
  sources:
    - name: telegram_cointelegraph
      channel: "cointelegraph"
      url: "https://t.me/s/cointelegraph"
      domains: ["crypto"]
"""
    path = tmp_path / "sources.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_enrichment_config_validates_and_skips_invalid(tmp_path: Path) -> None:
    cfg = load_enrichment_config(_write_config(tmp_path))
    assert cfg.reddit is not None and cfg.telegram is not None
    names = [s.name for s in cfg.reddit.sources]
    # enabled politics + flagged (needs_validation included by default); disabled + unnamed dropped
    assert names == ["reddit_politics", "reddit_flagged"]
    assert any(s["reason"] == "missing name/url" for s in cfg.skipped_sources)
    assert cfg.reddit.sources[1].needs_validation is True
    assert cfg.telegram.sources[0].channel == "cointelegraph"
    # subreddit-only source synthesizes the canonical /new/ URL.
    assert cfg.reddit.sources[1].url.endswith("/r/crypto/new/")


def test_overrides_skip_needs_validation_and_apply_limits(tmp_path: Path) -> None:
    cfg = load_enrichment_config(
        _write_config(tmp_path),
        overrides=EnrichmentOverrides(
            max_pages=2, max_items=20, rate_limit_seconds=0.0, include_needs_validation=False
        ),
    )
    assert [s.name for s in cfg.reddit.sources] == ["reddit_politics"]
    assert cfg.reddit.max_pages_per_source == 2
    assert cfg.reddit.max_items_per_source == 20
    assert cfg.reddit.rate_limit_seconds == 0.0


def test_only_sources_filter(tmp_path: Path) -> None:
    cfg = load_enrichment_config(
        _write_config(tmp_path),
        overrides=EnrichmentOverrides(only_sources=("telegram_cointelegraph",)),
    )
    assert [s.name for s in cfg.reddit.sources] == []
    assert [s.name for s in cfg.telegram.sources] == ["telegram_cointelegraph"]


def test_missing_sections_raise(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("some_other_section: {}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_enrichment_config(bad)
    with pytest.raises(FileNotFoundError):
        load_enrichment_config(tmp_path / "nope.yaml")


# ---------------------------------------------------------------------------
# Collection orchestration (FakeFetcher, no network)
# ---------------------------------------------------------------------------


def _reddit_section(url: str) -> RedditSectionConfig:
    return RedditSectionConfig(
        rate_limit_seconds=0.0,
        max_pages_per_source=1,
        max_items_per_source=50,
        sources=(RedditSource(name="reddit_politics", url=url, subreddit="politics", domains=("politics",)),),
    )


def _telegram_section(url: str) -> TelegramSectionConfig:
    return TelegramSectionConfig(
        rate_limit_seconds=0.0,
        max_pages_per_source=1,
        max_items_per_source=50,
        sources=(TelegramSource(name="telegram_cointelegraph", channel="cointelegraph", url=url, domains=("crypto",)),),
    )


def test_collect_reddit_writes_dataset_snapshots_and_is_idempotent(tmp_path: Path) -> None:
    url = "https://old.reddit.com/r/politics/new/"
    paths = get_data_paths(root=tmp_path / "data_reddit_scrapling_eval")
    fetcher = FakeFetcher({url: FetchOutcome(url=url, ok=True, status=200, html=REDDIT_HTML)})
    result = collect_reddit_scrapling(
        _reddit_section(url), paths=paths, fetcher=fetcher, run_id="r1", now=FIXED_NOW, sleep=_no_sleep
    )
    assert result.outcomes[0].status == "succeeded"
    assert result.new_records_built == 2
    assert result.rows_added == 2
    assert result.raw_html_snapshots == 1
    assert paths.social_posts_parquet.exists()
    assert paths.social_posts_parquet.with_suffix(".csv").exists()

    records = read_parquet(paths.social_posts_parquet, SocialPostRecord)
    assert {r.platform for r in records} == {Platform.REDDIT}
    assert all(r.retrieval_mode == "scrapling_public_web" for r in records)

    # Raw HTML snapshot lives under the reddit_scrapling raw zone.
    snapshots = list(paths.raw_social("reddit_scrapling").glob("**/*.html"))
    assert len(snapshots) == 1

    # Re-running a second run is idempotent (no new rows added).
    result2 = collect_reddit_scrapling(
        _reddit_section(url), paths=paths, fetcher=fetcher, run_id="r2", now=FIXED_NOW, sleep=_no_sleep
    )
    assert result2.rows_added == 0
    assert result2.exact_duplicates_dropped == 2


def test_collect_telegram_writes_dataset(tmp_path: Path) -> None:
    url = "https://t.me/s/cointelegraph"
    paths = get_data_paths(root=tmp_path / "data_telegram_enriched_eval")
    fetcher = FakeFetcher({url: FetchOutcome(url=url, ok=True, status=200, html=TELEGRAM_HTML)})
    result = collect_telegram_public_web(
        _telegram_section(url), paths=paths, fetcher=fetcher, run_id="t1", now=FIXED_NOW, sleep=_no_sleep
    )
    assert result.outcomes[0].status == "succeeded"
    assert result.new_records_built == 3
    records = read_parquet(paths.social_posts_parquet, SocialPostRecord)
    assert {r.platform for r in records} == {Platform.TELEGRAM}
    assert any(r.views_count == 8310 for r in records)


def test_collect_reddit_graceful_failure_on_blocked_source(tmp_path: Path) -> None:
    url = "https://old.reddit.com/r/politics/new/"
    paths = get_data_paths(root=tmp_path / "data_reddit_scrapling_eval")
    fetcher = FakeFetcher({url: FetchOutcome(url=url, ok=False, status=429, error="rate limited")})
    result = collect_reddit_scrapling(
        _reddit_section(url), paths=paths, fetcher=fetcher, run_id="r1", now=FIXED_NOW, sleep=_no_sleep
    )
    assert result.outcomes[0].status == "failed"
    assert "rate limited" in (result.outcomes[0].reason or "")
    assert result.new_records_built == 0  # run completed without raising


def test_collect_does_not_overwrite_other_roots(tmp_path: Path) -> None:
    # A pre-existing artifact in a *different* root must be untouched by an
    # enrichment collection writing to its own root.
    other_root = get_data_paths(root=tmp_path / "data_other_eval")
    other_root.social_posts_parquet.parent.mkdir(parents=True, exist_ok=True)
    sentinel = other_root.social_posts_parquet
    sentinel.write_bytes(b"SENTINEL-DO-NOT-TOUCH")

    url = "https://old.reddit.com/r/politics/new/"
    paths = get_data_paths(root=tmp_path / "data_reddit_scrapling_eval")
    fetcher = FakeFetcher({url: FetchOutcome(url=url, ok=True, status=200, html=REDDIT_HTML)})
    collect_reddit_scrapling(
        _reddit_section(url), paths=paths, fetcher=fetcher, run_id="r1", now=FIXED_NOW, sleep=_no_sleep
    )
    assert sentinel.read_bytes() == b"SENTINEL-DO-NOT-TOUCH"


# ---------------------------------------------------------------------------
# Merge + report + artifact index + profile paths
# ---------------------------------------------------------------------------


def _collect_both(tmp_path: Path):
    reddit_url = "https://old.reddit.com/r/politics/new/"
    tg_url = "https://t.me/s/cointelegraph"
    reddit_paths = get_data_paths(root=tmp_path / "data_reddit_scrapling_eval")
    telegram_paths = get_data_paths(root=tmp_path / "data_telegram_enriched_eval")
    collect_reddit_scrapling(
        _reddit_section(reddit_url),
        paths=reddit_paths,
        fetcher=FakeFetcher({reddit_url: FetchOutcome(url=reddit_url, ok=True, status=200, html=REDDIT_HTML)}),
        run_id="r1",
        now=FIXED_NOW,
        sleep=_no_sleep,
    )
    collect_telegram_public_web(
        _telegram_section(tg_url),
        paths=telegram_paths,
        fetcher=FakeFetcher({tg_url: FetchOutcome(url=tg_url, ok=True, status=200, html=TELEGRAM_HTML)}),
        run_id="t1",
        now=FIXED_NOW,
        sleep=_no_sleep,
    )
    return reddit_paths, telegram_paths


def test_merge_enriched_combines_platforms(tmp_path: Path) -> None:
    reddit_paths, telegram_paths = _collect_both(tmp_path)
    output_paths = get_data_paths(root=tmp_path / "data_social_enriched_eval")
    merge = merge_enriched_datasets(
        reddit_paths=reddit_paths, telegram_paths=telegram_paths, output_paths=output_paths
    )
    assert merge.reddit_rows == 2
    assert merge.telegram_rows == 3
    assert merge.platform_counts == {"reddit": 2, "telegram": 3}
    assert output_paths.social_posts_parquet.exists()
    assert output_paths.social_posts_parquet.with_suffix(".csv").exists()


def test_assemble_artifact_index_and_report(tmp_path: Path) -> None:
    reddit_paths, telegram_paths = _collect_both(tmp_path)
    output_paths = get_data_paths(root=tmp_path / "data_social_enriched_eval")
    merge = merge_enriched_datasets(
        reddit_paths=reddit_paths, telegram_paths=telegram_paths, output_paths=output_paths
    )
    output_paths.social_collection_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_paths.social_collection_summary_json.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-02T12:00:00+00:00",
                "baseline_12a_unique": {"reddit": 2, "telegram": 3},
                "added_12b_unique": {"reddit": 0, "telegram": 0, "total": 0},
                "final_unique": {"reddit": 2, "telegram": 3, "total": 5},
                "persisted_retrieval_rows": 5,
                "deduplication": {"reddit": {"exact_duplicates_dropped": 1}},
                "targets": {"rate_limit_seconds": 4.0, "reached": False},
                "objective_targets": {
                    "total": 50000,
                    "reddit": 35000,
                    "telegram": 15000,
                    "remaining_total": 49995,
                    "progress_ratio": 0.0001,
                    "reached": False,
                },
                "collection_progress": {
                    "fetched_urls": 2,
                    "reddit": {
                        "source_work_items": 1,
                        "source_pages_fetched": 1,
                        "query_work_items": 1,
                        "query_pages_fetched": 0,
                        "failed_sources": [],
                        "exhausted_sources": [],
                    },
                    "telegram": {
                        "source_work_items": 1,
                        "source_pages_fetched": 1,
                        "query_work_items": 0,
                        "query_pages_fetched": 0,
                        "failed_sources": [],
                        "exhausted_sources": [],
                    },
                },
                "market_query_artifacts": {"rows": 1},
                "usefulness": {"timestamp_parsed": 5},
                "completion": {
                    "reason": "resume_available",
                    "next_steps": "Resume with higher page caps.",
                },
            }
        ),
        encoding="utf-8",
    )
    index = assemble_artifact_index(
        config_path=Path("configs/social_enrichment_sources.example.yaml"),
        profile="real_social_enriched_main",
        reddit_summary={"sources_succeeded": ["reddit_politics"], "sources_failed": [],
                        "raw_extracted": 2, "new_records_built": 2, "normalized_total": 2,
                        "posts_path": str(reddit_paths.social_posts_parquet)},
        telegram_summary={"sources_succeeded": ["telegram_cointelegraph"], "sources_failed": [],
                          "raw_extracted": 3, "new_records_built": 3, "normalized_total": 3,
                          "posts_path": str(telegram_paths.social_posts_parquet)},
        merge_summary=merge.summary(),
        downstream=read_downstream_coverage(output_paths, profile="real_social_enriched_main"),
        verdict="BRANCH 12 READY",
    )
    assert index["final_profile"] == "real_social_enriched_main"
    assert index["final_verdict"] == "BRANCH 12 READY"
    assert index["combined_platform_breakdown"] == {"reddit": 2, "telegram": 3}
    assert index["branch_12b"]["latest_collection"]["persisted_retrieval_rows"] == 5
    assert index["branch_12b"]["latest_collection"]["deduplication"]["reddit"] == {
        "exact_duplicates_dropped": 1
    }
    assert index["branch_12b"]["after_downstream"]["spi_feature_rows"] == 0
    # Round-trips as JSON.
    assert json.loads(json.dumps(index))["successful_sources"] == [
        "reddit_politics",
        "telegram_cointelegraph",
    ]
    report = render_coverage_report(index)
    assert "not** official Reddit API access" in report or "not** official Reddit API" in report
    assert "BRANCH 12 READY" in report
    assert "old.reddit.com" in report
    assert "Persisted retrieval rows: `5`" in report


def test_enriched_profile_paths_are_isolated() -> None:
    profile = get_benchmark_profile("real_social_enriched_main")
    assert profile.default_root == "data_social_enriched_eval"
    assert profile.supports_real_empirical_claims is True
    paths = get_data_paths(root=Path("data_social_enriched_eval"))
    backtest = paths.backtest_profile("real_social_enriched_main")
    assert backtest.root.name == "real_social_enriched_main"
    assert backtest.metrics_summary_json.name == "metrics_summary.json"


def test_market_token_overlap_is_deterministic_and_bounded() -> None:
    records = [_record("t3_a", Platform.REDDIT, text="Bitcoin price prediction surge")]
    market = Market(
        market_id="m1",
        slug="bitcoin-new-high",
        question="Will Bitcoin price reach a new high?",
        domain=MarketDomain.CRYPTO,
        created_at=FIXED_NOW,
        outcomes=[
            Outcome(outcome_id="o1", label="Yes", index=0),
            Outcome(outcome_id="o2", label="No", index=1),
        ],
    )
    overlap = estimate_market_token_overlap(records, [market])
    assert overlap["markets_total"] == 1
    assert overlap["markets_with_token_overlap"] == 1
    assert 0.0 <= overlap["coverage_ratio"] <= 1.0

"""Offline tests for the social normalizer and the dedup rule."""

from __future__ import annotations

from datetime import datetime, timezone

from polysignal.domain import Engagement, Platform, SocialPost, SocialPostRecord
from polysignal.ingestion.social.normalizer import (
    build_record,
    deduplicate_records,
    merge_with_existing,
)
from polysignal.storage.parquet_store import read_parquet, write_parquet


def _post(
    post_id: str = "abc",
    text: str = "Hello world",
    extra: dict | None = None,
    score: int = 10,
    replies: int = 3,
) -> SocialPost:
    return SocialPost(
        post_id=post_id,
        platform=Platform.REDDIT,
        author_id="someone",
        author_handle="someone",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        text=text,
        url=f"https://www.reddit.com/r/test/comments/{post_id}/",
        engagement=Engagement(score=score, replies=replies),
        extra=extra or {"subreddit": "test", "title": "T", "num_comments": replies, "score": score},
    )


def test_build_record_carries_query_and_market_traceability():
    post = _post()
    now = datetime(2026, 5, 31, tzinfo=timezone.utc)
    record = build_record(
        post=post,
        query="trump 25th amendment",
        market_id_candidate="cond-1",
        raw_path="/tmp/raw/abc.json",
        collected_at=now,
    )
    assert record.post_id == "abc"
    assert record.platform is Platform.REDDIT
    assert record.query == "trump 25th amendment"
    assert record.market_id_candidate == "cond-1"
    assert record.raw_path == "/tmp/raw/abc.json"
    assert record.collected_at == now
    assert record.is_duplicate is False
    assert record.text_length == len("Hello world")
    assert record.subreddit_or_channel == "test"
    assert record.upvotes == 10
    assert record.comments_count == 3


def test_build_record_engagement_score_combines_signals():
    post = _post(score=10, replies=4)
    record = build_record(
        post=post,
        query="x",
        market_id_candidate="m",
        raw_path=None,
        collected_at=datetime.now(tz=timezone.utc),
    )
    assert record.engagement_score == 14.0


def test_deduplicate_records_keeps_first_marks_rest():
    now = datetime(2026, 5, 31, tzinfo=timezone.utc)
    p = _post(post_id="abc")
    a = build_record(
        post=p,
        query="query-a",
        market_id_candidate="cond-1",
        raw_path=None,
        collected_at=now,
    )
    b = build_record(
        post=p,
        query="query-b",
        market_id_candidate="cond-2",
        raw_path=None,
        collected_at=now,
    )
    out = deduplicate_records([a, b])
    assert len(out) == 2
    assert out[0].is_duplicate is False
    assert out[1].is_duplicate is True
    # Traceability is preserved — the second row keeps its own query/market.
    assert out[1].query == "query-b"
    assert out[1].market_id_candidate == "cond-2"


def test_deduplicate_records_falls_back_to_content_hash_when_post_id_missing():
    now = datetime(2026, 5, 31, tzinfo=timezone.utc)
    # Use the same crafted text + url + created_at, but no post_id.
    p = _post(post_id="unknown", text="identical body")
    a = build_record(
        post=p,
        query="q1",
        market_id_candidate="cond-1",
        raw_path=None,
        collected_at=now,
    )
    b = build_record(
        post=p,
        query="q2",
        market_id_candidate="cond-2",
        raw_path=None,
        collected_at=now,
    )
    out = deduplicate_records([a, b])
    assert out[0].is_duplicate is False
    assert out[1].is_duplicate is True


def test_merge_with_existing_appends_and_marks_duplicates():
    now = datetime(2026, 5, 31, tzinfo=timezone.utc)
    p = _post(post_id="abc")
    first = build_record(
        post=p,
        query="q1",
        market_id_candidate="cond-1",
        raw_path=None,
        collected_at=now,
    )
    # Save first record as "existing" (mark it not duplicate).
    second = build_record(
        post=p,
        query="q2",
        market_id_candidate="cond-2",
        raw_path=None,
        collected_at=now,
    )
    merged = merge_with_existing([second], [first])
    assert len(merged) == 2
    assert merged[0].is_duplicate is False
    assert merged[1].is_duplicate is True


def test_merge_with_existing_keeps_new_record_unique_when_no_collision():
    now = datetime(2026, 5, 31, tzinfo=timezone.utc)
    p1 = _post(post_id="a")
    p2 = _post(post_id="b")
    first = build_record(
        post=p1, query="q", market_id_candidate="m", raw_path=None, collected_at=now
    )
    second = build_record(
        post=p2, query="q", market_id_candidate="m", raw_path=None, collected_at=now
    )
    merged = merge_with_existing([second], [first])
    assert [r.post_id for r in merged] == ["a", "b"]
    assert all(not r.is_duplicate for r in merged)


# ---------------------------------------------------------------------------
# Fix 5 — idempotent dedup across runs (the exact same retrieval tuple
# must NOT be appended again).
# ---------------------------------------------------------------------------


def test_merge_with_existing_skips_exact_same_retrieval_tuple():
    """Same (platform, post_id, query, market_id_candidate) → skipped."""
    now = datetime(2026, 5, 31, tzinfo=timezone.utc)
    p = _post(post_id="abc")
    existing = build_record(
        post=p,
        query="trump president",
        market_id_candidate="cond-1",
        raw_path=None,
        collected_at=now,
    )
    new = build_record(
        post=p,
        query="trump president",  # exact same
        market_id_candidate="cond-1",  # exact same
        raw_path=None,
        collected_at=now,
    )
    merged = merge_with_existing([new], [existing])
    # Exact repeat must be skipped (otherwise re-runs grow unbounded).
    assert len(merged) == 1
    assert merged[0].is_duplicate is False


def test_merge_with_existing_appends_same_post_with_different_query():
    """Same post via a *different* query is fresh retrieval evidence."""
    now = datetime(2026, 5, 31, tzinfo=timezone.utc)
    p = _post(post_id="abc")
    existing = build_record(
        post=p,
        query="query-a",
        market_id_candidate="cond-1",
        raw_path=None,
        collected_at=now,
    )
    new = build_record(
        post=p,
        query="query-b",
        market_id_candidate="cond-1",
        raw_path=None,
        collected_at=now,
    )
    merged = merge_with_existing([new], [existing])
    assert len(merged) == 2
    assert merged[0].is_duplicate is False
    assert merged[1].is_duplicate is True
    # Both retrieval contexts are preserved.
    assert {r.query for r in merged} == {"query-a", "query-b"}


def test_merge_with_existing_appends_same_post_with_different_market():
    """Same post associated with a *different* candidate market is appended."""
    now = datetime(2026, 5, 31, tzinfo=timezone.utc)
    p = _post(post_id="abc")
    existing = build_record(
        post=p,
        query="q",
        market_id_candidate="cond-1",
        raw_path=None,
        collected_at=now,
    )
    new = build_record(
        post=p,
        query="q",
        market_id_candidate="cond-2",
        raw_path=None,
        collected_at=now,
    )
    merged = merge_with_existing([new], [existing])
    assert len(merged) == 2
    assert {r.market_id_candidate for r in merged} == {"cond-1", "cond-2"}


def test_merge_repeated_three_times_does_not_grow_parquet():
    """Idempotency stress test: 3 identical runs ⇒ 1 row, not 3."""
    now = datetime(2026, 5, 31, tzinfo=timezone.utc)
    p = _post(post_id="abc")

    def make() -> SocialPostRecord:
        return build_record(
            post=p,
            query="q",
            market_id_candidate="cond-1",
            raw_path=None,
            collected_at=now,
        )

    rows = []
    for _ in range(3):
        rows = merge_with_existing([make()], rows)
    assert len(rows) == 1


def test_merge_preserves_same_post_retrieved_in_distinct_windows():
    now = datetime(2026, 5, 31, tzinfo=timezone.utc)
    p = _post(post_id="abc")
    first = build_record(
        post=p,
        query="q",
        market_id_candidate="cond-1",
        raw_path=None,
        collected_at=now,
        retrieval_window_start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        retrieval_window_end=datetime(2026, 5, 8, tzinfo=timezone.utc),
    )
    second = build_record(
        post=p,
        query="q",
        market_id_candidate="cond-1",
        raw_path=None,
        collected_at=now,
        retrieval_window_start=datetime(2026, 5, 9, tzinfo=timezone.utc),
        retrieval_window_end=datetime(2026, 5, 31, tzinfo=timezone.utc),
    )
    assert len(merge_with_existing([second], [first])) == 2


def test_unified_parquet_roundtrip_keeps_reddit_and_telegram_rows(tmp_path):
    """Platform-specific nullable fields must survive one unified parquet file."""
    now = datetime(2026, 5, 31, tzinfo=timezone.utc)
    reddit = _post(post_id="shared-native-id")
    telegram = SocialPost(
        post_id="shared-native-id",
        platform=Platform.TELEGRAM,
        author_id="telegram-author",
        author_handle="telegram-author",
        created_at=now,
        text="Telegram message body",
        url="https://t.me/public_channel/42",
        engagement=Engagement(reposts=4, replies=3, views=1200),
        extra={"channel": "public_channel"},
    )
    records = [
        build_record(
            post=reddit,
            query="reddit query",
            market_id_candidate="market-reddit",
            raw_path=None,
            collected_at=now,
        ),
        build_record(
            post=telegram,
            query="telegram query",
            market_id_candidate="market-telegram",
            raw_path=None,
            collected_at=now,
        ),
    ]

    target = tmp_path / "social_posts.parquet"
    write_parquet(merge_with_existing(records, []), target)
    loaded = read_parquet(target, SocialPostRecord)

    assert [record.platform for record in loaded] == [Platform.REDDIT, Platform.TELEGRAM]
    assert all(record.is_duplicate is False for record in loaded)
    telegram_record = loaded[1]
    assert telegram_record.subreddit_or_channel == "public_channel"
    assert telegram_record.title is None
    assert telegram_record.upvotes is None
    assert telegram_record.comments_count == 3
    assert telegram_record.shares_count == 4
    assert telegram_record.engagement_score == 12.2

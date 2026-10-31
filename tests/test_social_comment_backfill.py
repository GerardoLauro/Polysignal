from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from polysignal.domain import Platform, PostComment
from polysignal.ingestion.social.comment_backfill import (
    combine_comment_tables,
    merge_comment_records,
    normalize_x_thread_rows,
    parse_human_count,
    reddit_thread_url,
    update_post_metadata,
)

NOW = datetime(2026, 6, 20, tzinfo=timezone.utc)


def test_reddit_thread_url_uses_rendered_old_reddit():
    url = reddit_thread_url(
        "https://www.reddit.com/r/test/comments/abc/title/?utm_source=x"
    )
    assert url == "https://old.reddit.com/r/test/comments/abc/title/?sort=top"


def test_parse_human_count():
    assert parse_human_count("1.2K Likes. Like") == 1200
    assert parse_human_count("45,001 Views") == 45001
    assert parse_human_count("") is None


def test_normalize_x_thread_rows_splits_original_and_replies():
    rows = [
        {
            "tweet_id": "100",
            "username": "origin",
            "text": "opening post",
            "like_label": "20 Likes",
            "reply_label": "4 Replies",
            "repost_label": "3 Reposts",
            "view_label": "1.5K Views",
        },
        {
            "tweet_id": "101",
            "username": "reply_user",
            "text": "reply",
            "created_at": "2026-06-20T10:00:00+00:00",
            "like_label": "5 Likes",
            "reply_label": "1 Reply",
        },
    ]
    replies, engagement = normalize_x_thread_rows(rows, original_tweet_id="100")
    assert len(replies) == 1
    assert replies[0]["tweet_id"] == "101"
    assert replies[0]["parent_tweet_id"] == "100"
    assert replies[0]["favorite_count"] == 5
    assert engagement["like_count"] == 20
    assert engagement["reply_count"] == 4
    assert engagement["view_count"] == 1500


def test_merge_comments_is_idempotent_and_updates_post_counts(tmp_path: Path):
    comments_path = tmp_path / "post_comments.parquet"
    posts_path = tmp_path / "social_posts.parquet"
    comment = PostComment(
        post_id="reddit:t3_1",
        platform=Platform.REDDIT,
        comment_id="c1",
        author_id="alice",
        text="A useful reply.",
        collected_at=NOW,
        text_length=15,
    )
    added, total = merge_comment_records([comment], comments_path)
    assert (added, total) == (1, 1)
    added, total = merge_comment_records([comment], comments_path)
    assert (added, total) == (0, 1)

    pd.DataFrame(
        [
            {
                "post_id": "reddit:t3_1",
                "platform": "reddit",
                "comment_sample_count": 0,
                "text": "post",
            }
        ]
    ).to_parquet(posts_path, index=False)
    updated = update_post_metadata(posts_path, comments_path)
    assert updated == 1
    posts = pd.read_parquet(posts_path)
    assert int(posts.iloc[0]["comment_sample_count"]) == 1


def test_combine_comment_tables_deduplicates(tmp_path: Path):
    left = tmp_path / "left.parquet"
    right = tmp_path / "right.parquet"
    output = tmp_path / "combined.parquet"
    row = {
        "platform": "reddit",
        "post_id": "p1",
        "comment_id": "c1",
        "text": "hello",
    }
    pd.DataFrame([row]).to_parquet(left, index=False)
    pd.DataFrame([row]).to_parquet(right, index=False)
    assert combine_comment_tables([left, right], output) == 1
    assert len(pd.read_parquet(output)) == 1

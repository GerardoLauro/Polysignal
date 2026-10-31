"""Offline tests for social comment/reply collection (Traccia_2 §2).

No network: Reddit comment JSON and Telegram reply objects are provided as
in-memory fixtures. Verifies the flattening, the schema, NER/language
enrichment, and the parquet writer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from polysignal.domain import Platform, PostComment
from polysignal.ingestion.social import (
    comment_counts_by_post,
    enrich_comment_text,
    parse_reddit_comment_html,
    parse_reddit_comment_json,
    praw_comments_to_records,
    telegram_replies_to_comments,
    write_post_comments,
)

FIXED_NOW = datetime(2026, 6, 19, tzinfo=timezone.utc)


def _reddit_payload() -> list:
    return [
        {"kind": "Listing", "data": {"children": [{"kind": "t3", "data": {"id": "post1"}}]}},
        {
            "kind": "Listing",
            "data": {
                "children": [
                    {
                        "kind": "t1",
                        "data": {
                            "id": "c1",
                            "author": "alice",
                            "body": "I think YES wins this $BTC market #crypto",
                            "score": 42,
                            "created_utc": 1_780_000_000,
                            "replies": {
                                "kind": "Listing",
                                "data": {
                                    "children": [
                                        {
                                            "kind": "t1",
                                            "data": {
                                                "id": "c2",
                                                "author": "bob",
                                                "body": "Disagree, NO is more likely.",
                                                "score": 7,
                                                "created_utc": 1_780_000_500,
                                            },
                                        }
                                    ]
                                },
                            },
                        },
                    },
                    {"kind": "more", "data": {"id": "_"}},
                ]
            },
        },
    ]


def test_parse_reddit_comment_json_flattens_forest():
    comments = parse_reddit_comment_json(
        _reddit_payload(),
        post_id="t3_post1",
        market_id_candidate="mkt-1",
        raw_path="raw/x.json",
        collected_at=FIXED_NOW,
    )
    assert [c.comment_id for c in comments] == ["c1", "c2"]
    top, nested = comments
    assert top.parent_comment_id is None
    assert nested.parent_comment_id == "c1"
    assert top.author_id == "alice"
    assert top.author_url == "https://www.reddit.com/user/alice"
    assert top.like_count == 42
    assert top.platform == Platform.REDDIT
    assert top.post_id == "t3_post1"
    assert top.market_id_candidate == "mkt-1"
    assert top.published_at == datetime.fromtimestamp(1_780_000_000, tz=timezone.utc)
    assert top.text_length == len(top.text)


def test_parse_reddit_comment_json_respects_max():
    comments = parse_reddit_comment_json(
        _reddit_payload(), post_id="t3_post1", max_comments=1
    )
    assert len(comments) == 1


def test_parse_reddit_comment_html_flattens_rendered_thread():
    rendered = """
    <html><body>
      <div class="thing comment" data-type="comment" data-fullname="t1_c1"
           data-author="alice" data-replies="1">
        <div class="entry unvoted">
          <p class="tagline">
            <a class="author" href="https://old.reddit.com/user/alice">alice</a>
            <span class="score unvoted" title="42">42 points</span>
            <time datetime="2026-06-19T10:00:00+00:00">now</time>
          </p>
          <form><div class="usertext"><div class="usertext-body">
            <div class="md"><p>I think YES wins.</p></div>
          </div></div></form>
        </div>
        <div class="child">
          <div class="thing comment" data-type="comment" data-fullname="t1_c2"
               data-author="bob" data-replies="0">
            <div class="entry unvoted">
              <p class="tagline">
                <a class="author" href="https://old.reddit.com/user/bob">bob</a>
                <span class="score unvoted" title="7">7 points</span>
                <time datetime="2026-06-19T10:05:00+00:00">now</time>
              </p>
              <form><div class="usertext"><div class="usertext-body">
                <div class="md"><p>No, I disagree.</p></div>
              </div></div></form>
            </div>
          </div>
        </div>
      </div>
    </body></html>
    """
    comments = parse_reddit_comment_html(
        rendered,
        post_id="reddit:t3_post1",
        market_id_candidate="mkt-1",
        raw_path="raw/thread.html",
        collected_at=FIXED_NOW,
    )
    assert [comment.comment_id for comment in comments] == ["c1", "c2"]
    assert comments[0].parent_comment_id is None
    assert comments[1].parent_comment_id == "c1"
    assert comments[0].author_id == "alice"
    assert comments[0].like_count == 42
    assert comments[0].reply_count == 1
    assert comments[0].published_at == datetime(
        2026, 6, 19, 10, 0, tzinfo=timezone.utc
    )
    assert comments[1].text == "No, I disagree."


def test_telegram_replies_to_comments():
    replies = [
        {"id": 10, "message": "great call!", "sender_username": "trader", "date": FIXED_NOW},
        {"id": 11, "message": "reply", "reply_to_msg_id": 10, "date": FIXED_NOW},
    ]
    comments = telegram_replies_to_comments(
        replies, post_id="telegram:chan:5", channel="chan", collected_at=FIXED_NOW
    )
    assert len(comments) == 2
    assert comments[0].comment_id == "telegram:chan:10"
    assert comments[0].author_url == "https://t.me/trader"
    assert comments[1].parent_comment_id == "telegram:chan:10"
    assert all(c.platform == Platform.TELEGRAM for c in comments)


def test_enrich_comment_text_adds_language_and_entities():
    comments = parse_reddit_comment_json(_reddit_payload(), post_id="t3_post1")
    rows = enrich_comment_text(comments)
    assert len(rows) == 2
    first = rows[0]
    assert "language" in first
    assert "#crypto" in first["hashtags"]
    assert isinstance(first["mentioned_entities"], list)


def test_write_post_comments_roundtrip(tmp_path: Path):
    comments = parse_reddit_comment_json(_reddit_payload(), post_id="t3_post1")
    out = tmp_path / "post_comments.parquet"
    write_post_comments(comments, out)
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 2
    # §2 comment schema columns present
    for col in (
        "post_id",
        "platform",
        "comment_id",
        "author_id",
        "text",
        "published_at",
        "like_count",
        "reply_count",
        "language",
    ):
        assert col in df.columns


def test_comment_counts_by_post():
    comments = [
        PostComment(post_id="p1", platform=Platform.REDDIT, comment_id="a", collected_at=FIXED_NOW),
        PostComment(post_id="p1", platform=Platform.REDDIT, comment_id="b", collected_at=FIXED_NOW),
        PostComment(post_id="p2", platform=Platform.REDDIT, comment_id="c", collected_at=FIXED_NOW),
    ]
    assert comment_counts_by_post(comments) == {"p1": 2, "p2": 1}


def test_praw_comments_to_records_skips_more_stubs():
    praw = [
        {
            "id": "x1",
            "author": "carol",
            "body": "first",
            "score": 5,
            "created_utc": 1_780_000_000,
            "parent_id": "t3_post1",
        },
        {  # nested reply
            "id": "x2",
            "author": "dan",
            "body": "second",
            "score": 1,
            "created_utc": 1_780_000_100,
            "parent_id": "t1_x1",
        },
        {"id": "more1"},  # MoreComments stub: no body -> skipped
    ]
    rows = praw_comments_to_records(praw, post_id="t3_post1", collected_at=FIXED_NOW)
    assert [r.comment_id for r in rows] == ["x1", "x2"]
    assert rows[0].parent_comment_id is None  # t3_ parent dropped
    assert rows[1].parent_comment_id == "x1"  # t1_ parent preserved
    assert rows[0].author_url == "https://www.reddit.com/user/carol"

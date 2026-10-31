"""Offline cross-platform schema-parity tests (Traccia_2 §2, Task D.3).

Every platform (Telegram, Reddit, Twitter/X) flows through the *same* unified
normalizer :func:`build_record`, so all three must emit an identical column set
and map their native engagement onto the §2 unified names
(``like_count``/``reply_count``/``repost_count``/``view_count``) consistently.

These tests are pure and offline — no network, no settings reads — and prove
the schema parity the assignment requires before any scrape runs.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from polysignal.domain import Engagement, Platform, SocialPost
from polysignal.ingestion.social.normalizer import build_record

_COLLECTED = datetime(2026, 6, 20, tzinfo=timezone.utc)


def _reddit_post() -> SocialPost:
    return SocialPost(
        post_id="t3_abc123",
        platform=Platform.REDDIT,
        author_id="u_politico",
        author_handle="politico",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        text="Will the bill pass before close?",
        url="https://old.reddit.com/r/politics/comments/abc123/x/",
        engagement=Engagement(score=42, replies=7),
        extra={"title": "Will the bill pass?", "subreddit": "politics",
               "score": 42, "num_comments": 7},
    )


def _telegram_post() -> SocialPost:
    return SocialPost(
        post_id="telegram:cryptosignals:5512",
        platform=Platform.TELEGRAM,
        author_id="cryptosignals",
        author_handle="cryptosignals",
        created_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
        text="Market is pricing this at 60%.",
        url="https://t.me/cryptosignals/5512",
        engagement=Engagement(reactions=15, views=12000, replies=3),
        extra={"channel": "cryptosignals", "subscribers": 48000},
    )


def _twitter_post() -> SocialPost:
    return SocialPost(
        post_id="twitter:Polymarket:1790000000000000001",
        platform=Platform.TWITTER,
        author_id="Polymarket",
        author_handle="Polymarket",
        created_at=datetime(2026, 5, 3, tzinfo=timezone.utc),
        text="New market live.",
        url="https://x.com/Polymarket/status/1790000000000000001",
        engagement=Engagement(likes=200, reposts=35, replies=12, views=45000),
        extra={"username": "Polymarket", "author_followers": 250000},
    )


def _record(post: SocialPost):
    return build_record(
        post=post,
        query="will the bill pass",
        market_id_candidate="0xMARKET",
        raw_path=None,
        collected_at=_COLLECTED,
        query_type="keyword",
    )


def test_all_platforms_emit_identical_columns():
    """The normalized record column set must be identical across platforms."""
    records = [_record(_reddit_post()), _record(_telegram_post()), _record(_twitter_post())]
    column_sets = [set(r.model_dump().keys()) for r in records]
    assert column_sets[0] == column_sets[1] == column_sets[2]
    # Spot-check that the §2 unified social fields are present in the schema.
    required = {
        "post_id", "platform", "market_id", "category", "url", "author_id",
        "author_url", "author_followers", "like_count", "reply_count",
        "repost_count", "view_count", "comment_sample_count", "language",
    }
    assert required <= column_sets[0]


def test_reddit_engagement_unified():
    record = _record(_reddit_post())
    assert record.platform == Platform.REDDIT
    assert record.like_count == 42  # upvotes -> like_count
    assert record.reply_count == 7  # num_comments -> reply_count
    assert record.author_url == "https://www.reddit.com/user/politico"


def test_telegram_engagement_unified():
    record = _record(_telegram_post())
    assert record.platform == Platform.TELEGRAM
    assert record.like_count == 15  # reactions -> like_count
    assert record.view_count == 12000  # views -> view_count
    assert record.reply_count == 3
    assert record.author_followers == 48000  # subscribers -> author_followers
    assert record.author_url == "https://t.me/cryptosignals"


def test_twitter_engagement_unified():
    record = _record(_twitter_post())
    assert record.platform == Platform.TWITTER
    assert record.like_count == 200  # likes -> like_count
    assert record.repost_count == 35  # reposts -> repost_count
    assert record.reply_count == 12
    assert record.view_count == 45000
    assert record.author_followers == 250000
    assert record.author_url == "https://x.com/Polymarket"


def test_author_url_none_when_handle_missing():
    """A Telegram channel post with no handle and no channel yields no URL."""
    post = SocialPost(
        post_id="telegram:anon:1",
        platform=Platform.TELEGRAM,
        author_id=None,
        author_handle=None,
        created_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
        text="anon",
        engagement=Engagement(),
        extra={},
    )
    assert _record(post).author_url is None


@pytest.mark.parametrize("factory", [_reddit_post, _telegram_post, _twitter_post])
def test_every_platform_record_roundtrips(factory):
    """Each platform record must serialize without raising (parquet-ready)."""
    record = _record(factory())
    dumped = record.model_dump(mode="json")
    assert dumped["post_id"] == factory().post_id
    assert dumped["collected_at"] is not None

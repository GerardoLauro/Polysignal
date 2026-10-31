"""Offline tests for the auth-gated X/Twitter adapter (Task D.1).

The adapter must (a) refuse a live collection without session cookies and never
fabricate posts, and (b) normalize authenticated-X JSON into the unified schema
including engagement, author followers, and a replies/comments sample. All
tests are offline (no network, no real cookies).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from polysignal.config import Settings, TwitterSettings
from polysignal.domain import Platform
from polysignal.ingestion.social.comments import x_replies_to_comments
from polysignal.ingestion.social.enrichment_config import TwitterSource
from polysignal.ingestion.social.twitter_scrapling_adapter import (
    XAuthError,
    parse_x_timeline_json,
    require_x_web_auth,
    twitter_raw_to_social_post,
    x_web_auth_available,
)


def _settings(auth: str = "", ct0: str = "") -> Settings:
    s = Settings()
    s.twitter = TwitterSettings(auth_token=auth, csrf_token=ct0)
    return s


# --- auth gating ------------------------------------------------------------


def test_require_auth_raises_without_cookies():
    with pytest.raises(XAuthError):
        require_x_web_auth(_settings())


def test_require_auth_returns_cookies_when_present():
    auth, ct0 = require_x_web_auth(_settings(auth="tok", ct0="csrf"))
    assert (auth, ct0) == ("tok", "csrf")


def test_auth_available_flag():
    assert x_web_auth_available(_settings(auth="tok", ct0="csrf")) is True
    assert x_web_auth_available(_settings(auth="tok")) is False
    assert x_web_auth_available(_settings()) is False


# --- GraphQL timeline JSON normalization ------------------------------------

_GRAPHQL_FIXTURE = {
    "data": {
        "user": {
            "result": {
                "timeline_v2": {
                    "timeline": {
                        "instructions": [
                            {
                                "type": "TimelineAddEntries",
                                "entries": [
                                    {
                                        "content": {
                                            "itemContent": {
                                                "tweet_results": {
                                                    "result": {
                                                        "rest_id": "1790000000000000001",
                                                        "core": {
                                                            "user_results": {
                                                                "result": {
                                                                    "legacy": {
                                                                        "screen_name": "Polymarket",
                                                                        "followers_count": 250000,
                                                                    }
                                                                }
                                                            }
                                                        },
                                                        "views": {"count": "45000"},
                                                        "legacy": {
                                                            "id_str": "1790000000000000001",
                                                            "full_text": "New market is live.",
                                                            "created_at": "Wed May 21 20:19:24 +0000 2025",
                                                            "favorite_count": 200,
                                                            "retweet_count": 35,
                                                            "reply_count": 12,
                                                        },
                                                    }
                                                }
                                            }
                                        }
                                    }
                                ],
                            }
                        ]
                    }
                }
            }
        }
    }
}


def test_parse_x_timeline_json_extracts_tweet():
    rows = parse_x_timeline_json(_GRAPHQL_FIXTURE, username="Polymarket")
    assert len(rows) == 1
    row = rows[0]
    assert row["tweet_id"] == "1790000000000000001"
    assert row["text"] == "New market is live."
    assert row["favorite_count"] == 200
    assert row["retweet_count"] == 35
    assert row["reply_count"] == 12
    assert row["view_count"] == 45000
    assert row["author_followers"] == 250000
    assert row["created_at"].startswith("2025-05-21T20:19:24")


def test_parse_x_timeline_json_empty_payload_no_fabrication():
    assert parse_x_timeline_json({"data": {}}) == []
    assert parse_x_timeline_json(None) == []


def test_normalized_x_post_carries_engagement():
    rows = parse_x_timeline_json(_GRAPHQL_FIXTURE)
    source = TwitterSource(
        name="twitter_polymarket",
        username="Polymarket",
        url="https://x.com/Polymarket",
        domains=("crypto",),
    )
    post = twitter_raw_to_social_post(
        rows[0],
        source=source,
        collection_method="twitter_authenticated_web",
        source_url=source.url,
    )
    assert post is not None
    assert post.platform == Platform.TWITTER
    assert post.engagement.likes == 200
    assert post.engagement.reposts == 35
    assert post.engagement.replies == 12
    assert post.engagement.views == 45000
    assert post.extra["author_followers"] == 250000


def test_public_web_post_has_no_fabricated_engagement():
    """A logged-out DOM row (no counts) must leave engagement empty, not zeroed."""
    source = TwitterSource(
        name="twitter_x", username="x", url="https://x.com/x", domains=("news",)
    )
    post = twitter_raw_to_social_post(
        {"username": "x", "text": "hello", "tweet_id": "1"},
        source=source,
        collection_method="twitter_public_web",
        source_url=source.url,
    )
    assert post is not None
    assert post.engagement.likes is None
    assert post.engagement.views is None
    assert "author_followers" not in post.extra


# --- replies / comments sample ----------------------------------------------


def test_x_replies_to_comments():
    replies = [
        {
            "tweet_id": "1790000000000000099",
            "username": "trader42",
            "text": "I think YES wins.",
            "favorite_count": 5,
            "reply_count": 1,
            "created_at": "Wed May 21 21:00:00 +0000 2025",
            "in_reply_to_status_id_str": "1790000000000000001",
        }
    ]
    comments = x_replies_to_comments(replies, post_id="twitter:Polymarket:1790000000000000001")
    assert len(comments) == 1
    c = comments[0]
    assert c.platform == Platform.TWITTER
    assert c.comment_id == "twitter:1790000000000000099"
    assert c.parent_comment_id == "twitter:1790000000000000001"
    assert c.author_url == "https://x.com/trader42"
    assert c.like_count == 5
    assert c.published_at == datetime(2025, 5, 21, 21, 0, tzinfo=timezone.utc)

"""Live Reddit smoke tests — opt-in only.

These tests exercise the real PRAW client against the public Reddit
search API. They are **skipped by default**: set the environment
variable ``POLYSIGNAL_RUN_LIVE_REDDIT_TESTS=1`` (and configure
``REDDIT_CLIENT_ID`` / ``REDDIT_CLIENT_SECRET`` / ``REDDIT_USER_AGENT``)
to enable them.

Convention
----------
* No assertions about the *content* of returned posts (Reddit changes).
* Only structural assertions: API contract holds, post fields are
  populated, no exceptions leak through.
"""

from __future__ import annotations

import os

import pytest

from polysignal.config import get_settings
from polysignal.ingestion.social.base import SearchQuery
from polysignal.ingestion.social.reddit import RedditIngestor

pytestmark = pytest.mark.skipif(
    os.environ.get("POLYSIGNAL_RUN_LIVE_REDDIT_TESTS") != "1",
    reason="set POLYSIGNAL_RUN_LIVE_REDDIT_TESTS=1 to run live Reddit tests",
)


@pytest.fixture(autouse=True)
def _refresh_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_live_reddit_search_returns_posts_for_obvious_query():
    """A simple, evergreen query should return at least one post."""
    ingestor = RedditIngestor(rate_limit_sleep_seconds=0.5)
    if not ingestor.is_available():
        pytest.skip("Reddit credentials not configured — set REDDIT_CLIENT_ID etc.")
    posts = list(ingestor.search(SearchQuery(text="Bitcoin", limit=3)))
    assert posts, "expected at least one result for 'Bitcoin'"
    for post in posts:
        assert post.post_id
        assert post.created_at.tzinfo is not None
        assert isinstance(post.text, str)


def test_live_reddit_search_raw_returns_paired_lists():
    """``search_raw`` must return two parallel lists of the same length."""
    ingestor = RedditIngestor(rate_limit_sleep_seconds=0.5)
    if not ingestor.is_available():
        pytest.skip("Reddit credentials not configured")
    posts, raws = ingestor.search_raw(SearchQuery(text="Bitcoin", limit=3))
    assert len(posts) == len(raws)
    for post, raw in zip(posts, raws, strict=True):
        assert raw["id"] == post.post_id

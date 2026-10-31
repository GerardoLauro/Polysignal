"""Offline tests for Reddit ingestor mapping and credential gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from polysignal.config import Settings
from polysignal.domain import Platform
from polysignal.ingestion.social.base import SearchQuery
from polysignal.ingestion.social.reddit import (
    RedditAPIError,
    RedditIngestor,
    submission_to_social_post,
)


@dataclass
class _FakeSubreddit:
    display_name: str


@dataclass
class _FakeAuthor:
    name: str


@dataclass
class _FakeSubmission:
    id: str = "abc123"
    title: str = "Trump 25th amendment debate intensifies"
    selftext: str = "Body of the post.\n\nMore details inside."
    subreddit: Any = None
    author: Any = None
    created_utc: float = 1716913200.0  # 2024-05-28T17:00:00Z
    score: int = 42
    num_comments: int = 7
    permalink: str = "/r/politics/comments/abc123/trump_25th_amendment_debate_intensifies/"
    url: str = "https://www.reddit.com/r/politics/comments/abc123/"
    over_18: bool = False
    ups: int = 50
    upvote_ratio: float = 0.83


def _fake_submission() -> _FakeSubmission:
    return _FakeSubmission(
        subreddit=_FakeSubreddit(display_name="politics"),
        author=_FakeAuthor(name="someone"),
    )


# ---------------------------------------------------------------------------
# Pure normalization
# ---------------------------------------------------------------------------


def test_submission_to_social_post_populates_core_fields():
    post = submission_to_social_post(_fake_submission())
    assert post.platform is Platform.REDDIT
    assert post.post_id == "abc123"
    assert post.author_handle == "someone"
    assert post.url and post.url.startswith("https://www.reddit.com/")
    assert "Trump 25th amendment" in post.text
    assert "Body of the post" in post.text
    assert post.engagement.score == 42
    assert post.engagement.replies == 7
    assert post.extra["subreddit"] == "politics"
    assert post.extra["title"].startswith("Trump 25th")
    assert post.extra["over_18"] is False


def test_submission_with_empty_selftext_uses_title_as_text():
    sub = _fake_submission()
    sub.selftext = ""
    post = submission_to_social_post(sub)
    assert post.text == sub.title
    assert post.extra["selftext"] == ""


def test_submission_with_none_author_normalizes_to_none():
    sub = _fake_submission()
    sub.author = None
    post = submission_to_social_post(sub)
    assert post.author_handle is None
    assert post.author_id is None


def test_submission_with_external_link_permalink_normalizes_url():
    sub = _fake_submission()
    sub.permalink = "https://external.example.com/post"
    post = submission_to_social_post(sub)
    # Already absolute → preserved verbatim.
    assert post.url == "https://external.example.com/post"


# ---------------------------------------------------------------------------
# Credential gating + error paths
# ---------------------------------------------------------------------------


def _settings_without_reddit() -> Settings:
    return Settings(
        reddit={"client_id": "", "client_secret": "", "user_agent": "x"},
    )


def _settings_with_reddit() -> Settings:
    return Settings(
        reddit={
            "client_id": "fake",
            "client_secret": "fake",
            "user_agent": "PolySignal-Test/0.1",
        },
    )


def test_reddit_is_unavailable_without_credentials():
    ingestor = RedditIngestor(settings=_settings_without_reddit())
    assert ingestor.is_available() is False


def test_reddit_search_without_credentials_raises_with_actionable_message():
    ingestor = RedditIngestor(settings=_settings_without_reddit())
    with pytest.raises(RuntimeError) as exc_info:
        list(ingestor.search(SearchQuery(text="anything")))
    assert "REDDIT_CLIENT_ID" in str(exc_info.value)


class _FakePraw:
    """Minimal stand-in for PRAW's ``Reddit`` exposing ``subreddit().search``."""

    def __init__(self, submissions, raise_on_search: Exception | None = None):
        self._submissions = submissions
        self._raise = raise_on_search
        self.read_only = False
        self.search_calls: list[tuple[str, dict[str, Any]]] = []

    def subreddit(self, name: str) -> _FakePraw:
        assert name == "all"
        return self

    def search(self, query: str, **kwargs):
        self.search_calls.append((query, kwargs))
        if self._raise is not None:
            raise self._raise
        return iter(self._submissions)


def test_reddit_search_returns_social_posts_via_injected_client():
    fake = _FakePraw([_fake_submission()])
    ingestor = RedditIngestor(
        settings=_settings_with_reddit(),
        reddit_client=fake,
        rate_limit_sleep_seconds=0.0,
    )
    posts = list(ingestor.search(SearchQuery(text="trump", limit=10)))
    assert len(posts) == 1
    assert posts[0].platform is Platform.REDDIT
    # Ensure the limit / sort were threaded through.
    assert fake.search_calls[0][1]["limit"] == 10
    assert fake.search_calls[0][1]["sort"] == "relevance"


def test_reddit_search_raw_returns_posts_and_raw_dicts():
    fake = _FakePraw([_fake_submission()])
    ingestor = RedditIngestor(
        settings=_settings_with_reddit(),
        reddit_client=fake,
        rate_limit_sleep_seconds=0.0,
    )
    posts, raws = ingestor.search_raw(SearchQuery(text="trump", limit=5))
    assert len(posts) == 1
    assert raws[0]["id"] == "abc123"
    assert raws[0]["subreddit"] == "politics"


def test_reddit_search_propagates_api_errors_as_redditapierror():
    fake = _FakePraw(
        [],
        raise_on_search=RuntimeError("simulated 429 from Reddit"),
    )
    ingestor = RedditIngestor(
        settings=_settings_with_reddit(),
        reddit_client=fake,
        rate_limit_sleep_seconds=0.0,
    )
    with pytest.raises(RedditAPIError) as exc_info:
        list(ingestor.search(SearchQuery(text="trump")))
    assert "Reddit" in str(exc_info.value)

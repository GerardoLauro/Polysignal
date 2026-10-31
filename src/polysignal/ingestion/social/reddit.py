"""Reddit ingestor backed by PRAW.

This module implements the ``SocialIngestor`` interface for Reddit. It
is deliberately conservative:

* PRAW is an *optional* dependency (``pip install polysignal[social]``).
  When the library is not installed, the ingestor still imports cleanly
  but ``search`` raises a clear ``RuntimeError`` instead of silently
  returning an empty list.
* Credentials are read from the environment via
  :class:`polysignal.config.settings.RedditSettings`. Missing
  credentials yield the same ``RuntimeError`` with an actionable
  message.
* HTTP / API errors from PRAW are caught and re-raised as
  :class:`RedditAPIError` so callers can distinguish "credentials are
  fine but Reddit refused" from "we never got far enough to try".
* No comments are fetched at this milestone. Posts (submissions) are
  the unit of interest. If a future milestone wants top comments, the
  hook is :meth:`RedditIngestor._submission_to_post` — extend it to
  include a few comment ``SocialPost`` objects on the same return.

The mapping from Reddit ``Submission`` to ``SocialPost`` lives in
:func:`submission_to_social_post`. It is a pure function over a thin
``RedditSubmission`` Protocol so tests can feed in fake objects.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from polysignal.config import Settings, get_settings
from polysignal.domain import Engagement, Platform, SocialPost
from polysignal.ingestion.social.base import SearchQuery, SocialIngestor

logger = logging.getLogger(__name__)


class RedditAPIError(RuntimeError):
    """Raised when PRAW reports an error talking to Reddit."""


@runtime_checkable
class RedditSubmission(Protocol):
    """Structural type matching the PRAW ``Submission`` attributes we use.

    Defining a Protocol here keeps the tests independent of PRAW: any
    object exposing these attributes can be normalized to a
    :class:`SocialPost`.
    """

    id: str
    title: str
    selftext: str
    subreddit: Any
    author: Any
    created_utc: float
    score: int
    num_comments: int
    permalink: str
    url: str
    over_18: bool


# ---------------------------------------------------------------------------
# Pure helpers (no PRAW import required)
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _subreddit_name(subreddit: Any) -> str | None:
    # PRAW's Subreddit exposes ``display_name``; a plain str works too.
    if subreddit is None:
        return None
    if isinstance(subreddit, str):
        return subreddit
    name = getattr(subreddit, "display_name", None)
    if isinstance(name, str):
        return name
    return None


def _author_name(author: Any) -> str | None:
    if author is None:
        return None
    if isinstance(author, str):
        return author
    name = getattr(author, "name", None)
    if isinstance(name, str):
        return name
    return None


def submission_to_social_post(submission: RedditSubmission) -> SocialPost:
    """Convert a PRAW-like ``Submission`` into a :class:`SocialPost`.

    The function is intentionally pure: pass any object exposing the
    ``RedditSubmission`` attributes (a real PRAW ``Submission`` or a
    fixture) and you get a domain model back. Branch 04 stores the
    Reddit-specific fields (``subreddit``, ``permalink``, ``num_comments``,
    ``score``, ``over_18``) under ``extra`` so the upstream normalizer
    can promote them into the flat :class:`SocialPostRecord` rows.
    """
    title = (submission.title or "").strip()
    selftext = (submission.selftext or "").strip()
    # We concatenate title + body so downstream text analysis (and
    # branch-05 semantic linking) sees the whole post. The original
    # title and body are still available in ``extra``.
    text = f"{title}\n\n{selftext}" if title and selftext else title or selftext

    created_at = datetime.fromtimestamp(float(submission.created_utc), tz=timezone.utc)

    subreddit_name = _subreddit_name(getattr(submission, "subreddit", None))
    author_name = _author_name(getattr(submission, "author", None))

    permalink = getattr(submission, "permalink", "") or ""
    if permalink and not permalink.startswith("http"):
        url = f"https://www.reddit.com{permalink}"
    else:
        url = permalink or getattr(submission, "url", None)

    engagement = Engagement(
        likes=None,
        reposts=None,
        replies=int(getattr(submission, "num_comments", 0) or 0),
        views=None,
        score=int(getattr(submission, "score", 0) or 0),
    )

    extra: dict[str, Any] = {
        "title": title,
        "selftext": selftext,
        "subreddit": subreddit_name,
        "permalink": permalink,
        "num_comments": int(getattr(submission, "num_comments", 0) or 0),
        "score": int(getattr(submission, "score", 0) or 0),
        "over_18": bool(getattr(submission, "over_18", False)),
        "external_url": getattr(submission, "url", None),
    }

    return SocialPost(
        post_id=str(submission.id),
        platform=Platform.REDDIT,
        author_id=author_name,  # Reddit usernames double as the id.
        author_handle=author_name,
        created_at=created_at,
        text=text,
        url=url,
        engagement=engagement,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Ingestor
# ---------------------------------------------------------------------------


def _import_praw() -> Any:
    """Import PRAW lazily so the rest of the package works without it."""
    try:
        import praw  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            'PRAW is required for Reddit ingestion. Install it with: pip install -e ".[social]"'
        ) from exc
    try:
        import prawcore  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover - prawcore ships with praw
        prawcore = None
    return praw, prawcore


def _prawcore_exception_type() -> type[BaseException]:
    """Return the most specific PRAW exception type available, or ``Exception``.

    Tests inject fake clients and may raise plain ``Exception`` subclasses;
    real PRAW raises ``prawcore.exceptions.PrawcoreException``. We prefer
    the specific type when prawcore is importable so production paths can
    still distinguish "Reddit said no" from "something blew up locally",
    but never *require* prawcore to be present.
    """
    try:
        import prawcore.exceptions  # type: ignore[import-untyped]
    except ImportError:
        return Exception
    return prawcore.exceptions.PrawcoreException


class RedditIngestor(SocialIngestor):
    """Real Reddit ingestor (PRAW-backed) with offline-friendly errors.

    Parameters
    ----------
    settings:
        Optional pre-built :class:`Settings`. Falls back to
        :func:`get_settings`.
    reddit_client:
        Optional pre-built PRAW ``Reddit`` instance. Tests inject a
        fake client so the suite never touches the network.
    rate_limit_sleep_seconds:
        Pause inserted between successive search calls to be polite to
        Reddit. Set to 0.0 in tests.
    """

    platform = Platform.REDDIT

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        reddit_client: Any | None = None,
        rate_limit_sleep_seconds: float = 1.0,
    ) -> None:
        self.settings = (settings or get_settings()).reddit
        self._client = reddit_client
        self._rate_limit_sleep_seconds = max(0.0, float(rate_limit_sleep_seconds))

    # ----------------------------------------------------------------- public
    def is_available(self) -> bool:
        return self.settings.configured

    def search(self, query: SearchQuery) -> Iterable[SocialPost]:
        if not self.is_available():
            raise RuntimeError(
                "Reddit credentials are not configured. "
                "Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and "
                "REDDIT_USER_AGENT in your .env file (see .env.example)."
            )
        client = self._ensure_client()
        return list(self._iter_search(client, query))

    def search_raw(self, query: SearchQuery) -> tuple[list[SocialPost], list[dict[str, Any]]]:
        """Like :meth:`search`, but also return the raw per-submission dicts.

        The raw dicts are what the pipeline persists under
        ``data/raw/social/reddit/<run>/``. Each dict is a
        JSON-serializable snapshot of the submission fields Branch 04
        keeps, plus a small handful of derived fields.
        """
        if not self.is_available():
            raise RuntimeError(
                "Reddit credentials are not configured. "
                "Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and "
                "REDDIT_USER_AGENT in your .env file (see .env.example)."
            )
        client = self._ensure_client()
        posts: list[SocialPost] = []
        raws: list[dict[str, Any]] = []
        for submission in self._iter_submissions(client, query):
            try:
                posts.append(submission_to_social_post(submission))
                raws.append(_submission_to_raw_dict(submission))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("reddit: skipping malformed submission %r: %s", submission, exc)
        return posts, raws

    # ---------------------------------------------------------------- private
    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        praw, _ = _import_praw()
        try:
            self._client = praw.Reddit(
                client_id=self.settings.client_id,
                client_secret=self.settings.client_secret,
                user_agent=self.settings.user_agent or "PolySignal-Research/0.1",
                username=self.settings.username or None,
                password=self.settings.password or None,
                check_for_async=False,
            )
            # PRAW reads in read-only mode by default unless username +
            # password are supplied; either way, set it explicitly.
            self._client.read_only = True
        except Exception as exc:  # pragma: no cover - praw import-level failure
            raise RedditAPIError(f"failed to construct PRAW client: {exc}") from exc
        return self._client

    def _iter_search(self, client: Any, query: SearchQuery) -> Iterator[SocialPost]:
        for submission in self._iter_submissions(client, query):
            try:
                yield submission_to_social_post(submission)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("reddit: skipping malformed submission %r: %s", submission, exc)

    def _iter_submissions(self, client: Any, query: SearchQuery) -> Iterator[Any]:
        praw_exc = _prawcore_exception_type()
        try:
            subreddit = client.subreddit("all")
            kwargs: dict[str, Any] = {"sort": "relevance", "syntax": "lucene"}
            if query.limit is not None:
                kwargs["limit"] = int(query.limit)
            results = subreddit.search(query.text, **kwargs)
        except praw_exc as exc:
            raise RedditAPIError(f"Reddit search failed for {query.text!r}: {exc}") from exc
        except Exception as exc:
            raise RedditAPIError(f"Unexpected Reddit error for {query.text!r}: {exc}") from exc

        try:
            for submission in results:
                yield submission
                if self._rate_limit_sleep_seconds > 0.0:
                    time.sleep(self._rate_limit_sleep_seconds)
        except praw_exc as exc:
            raise RedditAPIError(f"Reddit pagination failed for {query.text!r}: {exc}") from exc


def _submission_to_raw_dict(submission: RedditSubmission) -> dict[str, Any]:
    """JSON-serializable snapshot of a Reddit submission.

    PRAW objects are not directly JSON-serializable. We extract the
    fields branch 04 actually persists (the ones we either store in the
    normalized record or want available for debugging in
    ``data/raw/social/reddit/``).
    """
    return {
        "id": str(getattr(submission, "id", "")),
        "title": str(getattr(submission, "title", "") or ""),
        "selftext": str(getattr(submission, "selftext", "") or ""),
        "subreddit": _subreddit_name(getattr(submission, "subreddit", None)),
        "author": _author_name(getattr(submission, "author", None)),
        "created_utc": float(getattr(submission, "created_utc", 0.0) or 0.0),
        "score": int(getattr(submission, "score", 0) or 0),
        "num_comments": int(getattr(submission, "num_comments", 0) or 0),
        "permalink": str(getattr(submission, "permalink", "") or ""),
        "url": getattr(submission, "url", None),
        "over_18": bool(getattr(submission, "over_18", False)),
        "ups": int(getattr(submission, "ups", 0) or 0),
        "upvote_ratio": float(getattr(submission, "upvote_ratio", 0.0) or 0.0),
    }


__all__ = [
    "RedditAPIError",
    "RedditIngestor",
    "RedditSubmission",
    "submission_to_social_post",
]

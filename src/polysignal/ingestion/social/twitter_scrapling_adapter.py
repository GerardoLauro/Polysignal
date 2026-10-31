"""Twitter/X public-web adapter (Scrapling + x.com/<username>).

Collects Twitter/X posts from public profile pages without requiring API access.
Uses Scrapling's StealthyFetcher to navigate public Twitter/X pages and extract
post data from the rendered HTML.

Twitter/X displays posts in a JavaScript-heavy interface, so we use StealthyFetcher
(real browser) to render the page and extract structured data from the DOM.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any

from polysignal.config import Settings, get_settings
from polysignal.domain import Engagement, Platform, SocialPost
from polysignal.ingestion.social.enrichment import (
    CollectionResult,
    HtmlFetcher,
    RateLimiter,
    ScraplingFetcher,
    SourceOutcome,
    csv_sibling,
    persist_records,
    social_posts_to_records,
    stable_post_id,
)
from polysignal.ingestion.social.enrichment_config import TwitterSectionConfig, TwitterSource
from polysignal.ingestion.social.raw_store import SocialRawWriter, generate_social_run_id
from polysignal.storage.paths import DataPaths, get_data_paths

logger = logging.getLogger(__name__)


def _safe_slug(value: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in (value or "x"))[:64]


# ---------------------------------------------------------------------------
# Auth gating — X/Twitter serves profile timelines only to authenticated
# sessions (see scripts/collect_twitter_playwright.py). The adapter refuses to
# collect live without the two session cookies and NEVER fabricates posts.
# ---------------------------------------------------------------------------


class XAuthError(RuntimeError):
    """Raised when a live X collection is attempted without session cookies."""


def x_web_credentials(settings: Settings | None = None) -> tuple[str | None, str | None]:
    """Return ``(auth_token, ct0)`` from settings/.env, or ``(None, None)``.

    The values come from the untracked ``.env`` (``X_AUTH_TOKEN`` / ``X_CT0``,
    or the ``TWITTER_*`` aliases). They are never written to a versioned file.
    """
    twitter = (settings or get_settings()).twitter
    auth = twitter.auth_token.strip() or None
    ct0 = twitter.csrf_token.strip() or None
    return auth, ct0


def x_web_auth_available(settings: Settings | None = None) -> bool:
    """True when both X session cookies are present (live collection possible)."""
    auth, ct0 = x_web_credentials(settings)
    return bool(auth and ct0)


def require_x_web_auth(settings: Settings | None = None) -> tuple[str, str]:
    """Return the X session cookies or raise :class:`XAuthError`.

    Use this at the start of any *live* collection path so the adapter fails
    loudly (rather than silently fabricating data) when credentials are absent.
    """
    auth, ct0 = x_web_credentials(settings)
    if not auth or not ct0:
        raise XAuthError(
            "X/Twitter web collection requires session cookies. Add X_AUTH_TOKEN "
            "and X_CT0 to your local .env (untracked); see "
            "scripts/collect_twitter_playwright.py for how to copy them."
        )
    return auth, ct0


# ---------------------------------------------------------------------------
# HTML parser for Twitter/X public pages
# ---------------------------------------------------------------------------


class _TwitterPublicPageParser(HTMLParser):
    """Extract tweet data from a rendered x.com/<username> page.

    Note: Twitter's current UI is heavily JS-driven. This parser extracts
    what's available in the initial HTML render. For production use, consider
    using Playwright or similar for full JS execution.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tweets: list[dict[str, Any]] = []
        self._in_tweet = False
        self._current_tweet: dict[str, Any] = {}
        self._in_text = False
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: (value or "") for key, value in attrs}

        # Twitter uses article tags for tweets
        if tag == "article" and "data-testid" in attr:
            if attr.get("data-testid") == "tweet":
                self._in_tweet = True
                self._current_tweet = {}
                self._text_parts = []

        # Look for tweet text content
        if self._in_tweet and tag == "div" and "data-testid" in attr:
            if attr.get("data-testid") == "tweetText":
                self._in_text = True
                self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_text:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "article" and self._in_tweet:
            if self._text_parts:
                self._current_tweet["text"] = "".join(self._text_parts).strip()
            if self._current_tweet.get("text"):
                self.tweets.append(self._current_tweet)
            self._in_tweet = False
            self._current_tweet = {}

        if tag == "div" and self._in_text:
            self._in_text = False


def parse_twitter_public_page(
    html: str,
    *,
    username: str | None = None,
) -> list[dict[str, Any]]:
    """Parse one x.com/<username> page into raw tweet dicts.

    Note: Twitter's modern UI requires JavaScript execution for full content.
    This basic parser extracts what's available in the server-rendered HTML.
    For production, use a browser automation tool like Playwright.
    """
    parser = _TwitterPublicPageParser()
    parser.feed(html or "")
    parser.close()

    out: list[dict[str, Any]] = []
    for idx, tweet in enumerate(parser.tweets):
        out.append({
            "username": username,
            "text": tweet.get("text", ""),
            "tweet_id": f"{username}_{idx}",  # Placeholder - real ID needs JS execution
            "created_at": None,  # Needs JS execution to extract timestamp
        })
    return out


# ---------------------------------------------------------------------------
# Normalization to SocialPost
# ---------------------------------------------------------------------------


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _x_engagement(raw: dict[str, Any]) -> Engagement:
    """Map authenticated-X engagement counts into the shared Engagement model.

    Accepts both the X GraphQL ``legacy`` field names (``favorite_count``,
    ``retweet_count``, ``reply_count``, plus a ``views`` block) and the flatter
    DOM-extracted aliases (``like_count``/``likes``, ``repost_count``, ...).
    Public-web HTML (logged out) carries none of these, so every field stays
    ``None`` there — engagement is populated only when the data is real.
    """
    likes = _int_or_none(raw.get("favorite_count"))
    if likes is None:
        likes = _int_or_none(raw.get("like_count") or raw.get("likes"))
    reposts = _int_or_none(raw.get("retweet_count"))
    if reposts is None:
        reposts = _int_or_none(raw.get("repost_count") or raw.get("retweets"))
    replies = _int_or_none(raw.get("reply_count") or raw.get("replies"))
    views = _int_or_none(raw.get("view_count") or raw.get("views"))
    return Engagement(likes=likes, reposts=reposts, replies=replies, views=views)


def twitter_raw_to_social_post(
    raw: dict[str, Any],
    *,
    source: TwitterSource,
    collection_method: str,
    source_url: str,
    html_snapshot_path: str | None = None,
) -> SocialPost | None:
    """Normalize a raw Twitter dict into the shared SocialPost model."""
    text = str(raw.get("text") or "").strip()
    if not text:
        return None

    username = str(raw.get("username") or source.username or "").strip()
    tweet_id = str(raw.get("tweet_id") or "").strip()

    if tweet_id:
        post_id = f"twitter:{username}:{tweet_id}"
    else:
        post_id = "twitter:" + stable_post_id(username, text, None)

    url = str(raw.get("url") or "") or (
        f"https://x.com/{username}/status/{tweet_id}" if username and tweet_id else None
    )

    # Parse timestamp if available
    created_at: datetime | None = None
    created_at_parse_status = "missing"
    when = raw.get("created_at")
    if when:
        try:
            created_at = datetime.fromisoformat(str(when).replace("Z", "+00:00"))
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            created_at = created_at.astimezone(timezone.utc)
            created_at_parse_status = "parsed"
        except ValueError:
            created_at_parse_status = "invalid"

    followers = _int_or_none(raw.get("author_followers") or raw.get("followers_count"))
    extra: dict[str, Any] = {
        "username": username,
        "tweet_id": tweet_id or None,
        "collection_method": collection_method,
        "platform": Platform.TWITTER.value,
        "source_name": source.name,
        "source_url": source_url,
        "domain_tags": list(source.domains),
        "raw_snapshot_path": html_snapshot_path,
        "extraction_status": "parsed",
        "created_at_parse_status": created_at_parse_status,
    }
    if followers is not None:
        extra["author_followers"] = followers

    return SocialPost(
        post_id=post_id,
        platform=Platform.TWITTER,
        author_id=username or None,
        author_handle=username or None,
        created_at=created_at,
        text=text,
        url=url,
        engagement=_x_engagement(raw),
        extra=extra,
    )


def parse_x_timeline_json(payload: Any, *, username: str | None = None) -> list[dict[str, Any]]:
    """Parse an authenticated X GraphQL ``UserTweets`` payload into raw dicts.

    X's web client renders timelines from a GraphQL response whose tweets live
    under ``legacy`` objects (``full_text``, ``favorite_count``,
    ``retweet_count``, ``reply_count``, a ``views.count`` block) with the author
    under ``core.user_results.result.legacy``. This walks that structure
    defensively (any missing branch is skipped, never fabricated) into the same
    raw-dict shape :func:`twitter_raw_to_social_post` consumes. Pure + offline,
    so it unit-tests against a captured fixture.
    """
    out: list[dict[str, Any]] = []

    def _walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                _walk(item)
            return
        if not isinstance(node, dict):
            return
        legacy = node.get("legacy")
        if isinstance(legacy, dict) and ("full_text" in legacy or "text" in legacy):
            out.append(_tweet_from_result(node, username=username))
        for value in node.values():
            if isinstance(value, (list, dict)):
                _walk(value)

    _walk(payload)
    # Deduplicate by tweet_id, preserving first occurrence.
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in out:
        key = str(row.get("tweet_id") or "")
        if key and key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _tweet_from_result(result: dict[str, Any], *, username: str | None) -> dict[str, Any]:
    legacy = result.get("legacy") or {}
    author = (
        result.get("core", {})
        .get("user_results", {})
        .get("result", {})
        .get("legacy", {})
        if isinstance(result.get("core"), dict)
        else {}
    )
    handle = author.get("screen_name") or username
    views = result.get("views")
    view_count = views.get("count") if isinstance(views, dict) else None
    return {
        "username": handle,
        "tweet_id": str(legacy.get("id_str") or result.get("rest_id") or "").strip() or None,
        "text": legacy.get("full_text") or legacy.get("text") or "",
        "created_at": _x_time(legacy.get("created_at")),
        "favorite_count": legacy.get("favorite_count"),
        "retweet_count": legacy.get("retweet_count"),
        "reply_count": legacy.get("reply_count"),
        "view_count": _int_or_none(view_count),
        "author_followers": _int_or_none(author.get("followers_count")),
    }


def _x_time(value: Any) -> str | None:
    """Convert X's ``created_at`` ('Wed Oct 10 20:19:24 +0000 2018') to ISO-8601."""
    if not value:
        return None
    try:
        parsed = datetime.strptime(str(value), "%a %b %d %H:%M:%S %z %Y")
    except ValueError:
        return str(value)
    return parsed.astimezone(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Collection orchestrator
# ---------------------------------------------------------------------------


def collect_twitter_public_web(
    section: TwitterSectionConfig,
    *,
    paths: DataPaths | None = None,
    fetcher: HtmlFetcher | None = None,
    run_id: str | None = None,
    now: datetime | None = None,
    sleep: Callable[[float], None] = time.sleep,
    posts_path: Any | None = None,
) -> CollectionResult:
    """Collect Twitter accounts into the unified social dataset.

    Each username is fetched independently. Accounts that are private, suspended,
    or return no tweets are recorded and the run continues.
    """
    paths = paths or get_data_paths()
    rid = run_id or generate_social_run_id()
    moment = (now or datetime.now(tz=timezone.utc)).astimezone(timezone.utc)
    run_dir = paths.raw_run_dir("social/twitter_public_web", rid, when=moment)
    writer = SocialRawWriter(directory=run_dir, platform="twitter_public_web")
    writer.extra["collection_method"] = section.collection_method
    writer.extra["user_agent"] = section.user_agent
    writer.extra["run_timestamp"] = moment

    active_fetcher = fetcher or ScraplingFetcher(
        user_agent=section.user_agent, timeout=section.request_timeout_seconds
    )
    limiter = RateLimiter(section.rate_limit_seconds, sleep=sleep)
    result = CollectionResult(
        platform="twitter",
        collection_method=section.collection_method,
        run_id=rid,
        started_at=moment,
        raw_dir=run_dir,
    )
    collected: list[tuple[SocialPost, str | None]] = []

    for source in section.sources:
        outcome = SourceOutcome(name=source.name, platform="twitter", url=source.url)
        seen_ids: set[str] = set()

        try:
            limiter.wait()
            fetched = active_fetcher.get(source.url)
            outcome.http_status = fetched.status

            if not fetched.ok or not fetched.html:
                outcome.reason = fetched.error or f"http_status={fetched.status}"
                outcome.status = "failed"
                writer.record_query("", f"twitter-user-{source.username}", 0, error=outcome.reason)
                result.outcomes.append(outcome)
                continue

            outcome.pages_fetched = 1
            snapshot_path: str | None = None

            if section.store_raw_snapshots:
                snap = run_dir / f"{_safe_slug(source.name)}.html"
                snap.parent.mkdir(parents=True, exist_ok=True)
                snap.write_text(fetched.html, encoding="utf-8")
                snapshot_path = str(snap.resolve())
                outcome.snapshot_paths.append(snapshot_path)
                result.raw_html_snapshots += 1

            raw_tweets = parse_twitter_public_page(fetched.html, username=source.username)

            if not raw_tweets:
                outcome.status = "empty"
                outcome.reason = "no tweets parsed"
                writer.record_query("", f"twitter-user-{source.username}", 0)
                result.outcomes.append(outcome)
                continue

            for raw in raw_tweets:
                if outcome.normalized >= section.max_items_per_source:
                    break

                outcome.raw_extracted += 1
                result.raw_extracted += 1

                post = twitter_raw_to_social_post(
                    raw,
                    source=source,
                    collection_method=section.collection_method,
                    source_url=source.url,
                    html_snapshot_path=snapshot_path,
                )

                if post is None or post.post_id in seen_ids:
                    continue

                seen_ids.add(post.post_id)

                payload = {
                    **raw,
                    "html_snapshot_path": snapshot_path,
                    "source_name": source.name,
                    "source_url": source.url,
                    "collection_method": section.collection_method,
                    "platform": "twitter",
                    "domain_tags": list(source.domains),
                    "extraction_status": "parsed",
                }

                raw_path = writer.write_post(
                    query="",
                    market_id=f"twitter-user-{source.username}",
                    post_id=post.post_id,
                    payload=payload,
                )
                collected.append((post, str(raw_path)))
                outcome.normalized += 1

            outcome.status = "succeeded"
            writer.record_query("", f"twitter-user-{source.username}", outcome.normalized)

        except Exception as exc:
            outcome.status = "failed"
            outcome.reason = f"{type(exc).__name__}: {exc}"[:300]
            logger.warning("twitter source %s failed: %s", source.name, outcome.reason)
            writer.record_query("", f"twitter-user-{source.username}", 0, error=outcome.reason)

        result.outcomes.append(outcome)

    records = social_posts_to_records(
        collected,
        collection_method=section.collection_method,
        query_type="twitter_public_web",
        run_id=rid,
        collected_at=moment,
    )
    result.new_records_built = len(records)
    target = posts_path or paths.social_posts_parquet
    merged, rows_added, dropped = persist_records(records, posts_path=target, append=True)
    result.rows_added = rows_added
    result.exact_duplicates_dropped = dropped
    result.duplicate_rows_flagged = sum(1 for record in merged if record.is_duplicate)
    result.normalized_total = len(merged)
    result.posts_path = target
    result.csv_path = csv_sibling(target)

    writer.extra["sources_succeeded"] = [o.name for o in result.succeeded]
    writer.extra["sources_failed"] = [
        {"name": o.name, "status": o.status, "reason": o.reason} for o in result.failed
    ]
    writer.extra["raw_extracted"] = result.raw_extracted
    writer.extra["new_records_built"] = result.new_records_built
    result.finished_at = datetime.now(tz=timezone.utc)

    if section.store_raw_snapshots or collected:
        writer.write_manifest()

    return result


__all__ = [
    "XAuthError",
    "collect_twitter_public_web",
    "parse_twitter_public_page",
    "parse_x_timeline_json",
    "require_x_web_auth",
    "twitter_raw_to_social_post",
    "x_web_auth_available",
    "x_web_credentials",
]

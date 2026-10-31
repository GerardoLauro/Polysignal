"""Reddit public-web adapter (Scrapling + old.reddit.com).

Branch 12. Reddit API/PRAW approval was unavailable (Reddit never responded),
so — with the professor's approval — Reddit evidence is collected from public
``old.reddit.com`` listing pages using Scrapling's **stealth browser**
(``StealthyFetcher``), selected via the section's ``base_fetcher``. The plain
``Fetcher`` is hard-blocked by Reddit (HTTP 403 on every page and ``.json``
endpoint); the stealth browser renders the same public listing pages and is
served 200s. Still public + unauthenticated only: no login, no proxy rotation,
no CAPTCHA solving, and no deep comment scraping. Both the live ``/new/`` feed
and the historical ``/top/?t=year`` listing use this identical HTML markup.

The HTML parsing is a pure, deterministic, dependency-free state machine over
:class:`html.parser.HTMLParser`. Old Reddit renders every submission as a
``<div class="thing" data-fullname="t3_..." data-subreddit="..."
data-timestamp="..." data-permalink="..." ...>`` whose ``data-*`` attributes
carry the structured fields, with the title in a nested
``<a class="title ...">``. Parsing only static HTML keeps the parser fully unit
testable offline; the live ``Fetcher.get`` call is the sole network surface.
"""

from __future__ import annotations

import html as html_lib
import logging
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any

from polysignal.domain import Engagement, Platform, SocialPost
from polysignal.ingestion.social.enrichment import (
    CollectionResult,
    HtmlFetcher,
    RateLimiter,
    SourceOutcome,
    build_html_fetcher,
    csv_sibling,
    persist_records,
    social_posts_to_records,
    stable_post_id,
)
from polysignal.ingestion.social.enrichment_config import RedditSectionConfig, RedditSource
from polysignal.ingestion.social.raw_store import SocialRawWriter, generate_social_run_id
from polysignal.storage.paths import DataPaths, get_data_paths

logger = logging.getLogger(__name__)

_NEXT_BUTTON_RE = re.compile(r'class="next-button">\s*<a[^>]*\bhref="([^"]+)"', re.IGNORECASE)


def _int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_slug(value: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in (value or "x"))[:64]


# ---------------------------------------------------------------------------
# Pure HTML parser
# ---------------------------------------------------------------------------


class _OldRedditListingParser(HTMLParser):
    """Extract submission ``data-*`` attributes + title from an old.reddit page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.things: list[dict[str, Any]] = []
        self._div_depth = 0
        self._thing: dict[str, Any] | None = None
        self._thing_depth: int | None = None
        self._capture_title = False
        self._title_done = False
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: (value or "") for key, value in attrs}
        if tag == "div":
            self._div_depth += 1
            if self._thing is not None:
                return
            classes = attr.get("class", "").split()
            fullname = attr.get("data-fullname", "")
            if "thing" in classes and fullname.startswith("t3_"):
                self._thing = attr
                self._thing_depth = self._div_depth
                self._capture_title = False
                self._title_done = False
                self._title_parts = []
            return
        if (
            tag == "a"
            and self._thing is not None
            and not self._title_done
            and "title" in attr.get("class", "").split()
        ):
            self._capture_title = True

    def handle_data(self, data: str) -> None:
        if self._capture_title:
            self._title_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capture_title:
            self._capture_title = False
            self._title_done = True
        if tag == "div":
            if self._thing is not None and self._div_depth == self._thing_depth:
                self._thing["_title"] = "".join(self._title_parts).strip()
                self.things.append(self._thing)
                self._thing = None
                self._thing_depth = None
            self._div_depth = max(0, self._div_depth - 1)


def parse_old_reddit_listing(
    html: str,
    *,
    default_subreddit: str | None = None,
) -> list[dict[str, Any]]:
    """Parse one old.reddit ``/new/`` listing page into raw post dicts.

    Promoted/ad ``thing`` blocks are dropped. The returned dicts are
    JSON-serializable raw snapshots; normalization happens in
    :func:`reddit_raw_to_social_post`.
    """
    parser = _OldRedditListingParser()
    parser.feed(html or "")
    parser.close()
    out: list[dict[str, Any]] = []
    for thing in parser.things:
        if (thing.get("data-promoted") or "").lower() == "true":
            continue
        fullname = thing.get("data-fullname") or ""
        out.append(
            {
                "fullname": fullname,
                "id": fullname.removeprefix("t3_") if fullname else None,
                "subreddit": thing.get("data-subreddit") or default_subreddit,
                "author": thing.get("data-author") or None,
                "timestamp_ms": _int(thing.get("data-timestamp")),
                "outbound_url": thing.get("data-url") or None,
                "permalink": thing.get("data-permalink") or None,
                "domain": thing.get("data-domain") or None,
                "score": _int(thing.get("data-score")),
                "num_comments": _int(thing.get("data-comments-count")),
                "nsfw": (thing.get("data-nsfw") or "").lower() == "true",
                "rank": _int(thing.get("data-rank")),
                "title": thing.get("_title") or "",
            }
        )
    return out


def find_next_page_url(html: str) -> str | None:
    """Return the old.reddit ``next`` listing URL, if the simple link exists."""
    match = _NEXT_BUTTON_RE.search(html or "")
    if not match:
        return None
    return html_lib.unescape(match.group(1)).strip() or None


# ---------------------------------------------------------------------------
# Normalization to SocialPost
# ---------------------------------------------------------------------------


def reddit_raw_to_social_post(
    raw: dict[str, Any],
    *,
    source: RedditSource,
    collection_method: str,
    source_url: str,
    html_snapshot_path: str | None = None,
) -> SocialPost | None:
    """Normalize a raw old.reddit post dict into the shared ``SocialPost`` model.

    Returns ``None`` only when there is no usable title. Rows without a
    reliable timestamp are retained for audit and marked ineligible downstream.
    """
    title = str(raw.get("title") or "").strip()
    timestamp_ms = raw.get("timestamp_ms")
    if not title:
        return None
    created_at: datetime | None = None
    created_at_parse_status = "parsed"
    if timestamp_ms is None:
        created_at_parse_status = "missing"
    else:
        try:
            created_at = datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=timezone.utc)
        except (OverflowError, TypeError, ValueError):
            created_at_parse_status = "invalid"

    subreddit = str(raw.get("subreddit") or source.subreddit or "")
    fullname = str(raw.get("fullname") or "")
    post_id = fullname or "t3_" + stable_post_id(
        Platform.REDDIT.value, subreddit, title, raw.get("permalink")
    )
    permalink = str(raw.get("permalink") or "")
    if permalink and not permalink.startswith("http"):
        url = f"https://old.reddit.com{permalink}"
    else:
        url = permalink or raw.get("outbound_url")
    author = str(raw.get("author") or "") or None

    engagement = Engagement(
        score=raw.get("score"),
        replies=raw.get("num_comments"),
    )
    extra: dict[str, Any] = {
        "title": title,
        "subreddit": subreddit,
        "permalink": permalink or None,
        "comment_url": url,
        "num_comments": raw.get("num_comments"),
        "score": raw.get("score"),
        "over_18": bool(raw.get("nsfw")),
        "external_url": raw.get("outbound_url"),
        "domain": raw.get("domain"),
        "collection_method": collection_method,
        "platform": Platform.REDDIT.value,
        "source_name": source.name,
        "source_url": source_url,
        "domain_tags": list(source.domains),
        "raw_snapshot_path": html_snapshot_path,
        "extraction_status": "parsed",
        "created_at_parse_status": created_at_parse_status,
    }
    return SocialPost(
        post_id=post_id,
        platform=Platform.REDDIT,
        author_id=author,
        author_handle=author,
        created_at=created_at,
        text=title,
        url=url,
        engagement=engagement,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Collection orchestrator
# ---------------------------------------------------------------------------


def collect_reddit_scrapling(
    section: RedditSectionConfig,
    *,
    paths: DataPaths | None = None,
    fetcher: HtmlFetcher | None = None,
    run_id: str | None = None,
    now: datetime | None = None,
    sleep: Callable[[float], None] = time.sleep,
    posts_path: Any | None = None,
) -> CollectionResult:
    """Collect every enabled Reddit source into the unified social dataset.

    Each source is fetched independently. A source that blocks, errors, or
    returns no usable posts is recorded in the result and the run continues.
    Raw HTML snapshots and per-post JSON snapshots are persisted under
    ``<root>/raw/social/reddit_scrapling/<run-dir>/``.
    """
    paths = paths or get_data_paths()
    rid = run_id or generate_social_run_id()
    moment = (now or datetime.now(tz=timezone.utc)).astimezone(timezone.utc)
    run_dir = paths.raw_run_dir("social/reddit_scrapling", rid, when=moment)
    writer = SocialRawWriter(directory=run_dir, platform="reddit_scrapling")
    writer.extra["collection_method"] = section.collection_method
    writer.extra["user_agent"] = section.user_agent
    writer.extra["run_timestamp"] = moment

    active_fetcher = fetcher or build_html_fetcher(
        section.base_fetcher,
        user_agent=section.user_agent,
        timeout=section.request_timeout_seconds,
    )
    limiter = RateLimiter(section.rate_limit_seconds, sleep=sleep)
    result = CollectionResult(
        platform="reddit",
        collection_method=section.collection_method,
        run_id=rid,
        started_at=moment,
        raw_dir=run_dir,
    )
    collected: list[tuple[SocialPost, str | None]] = []

    for source in section.sources:
        outcome = SourceOutcome(name=source.name, platform="reddit", url=source.url)
        seen_ids: set[str] = set()
        url: str | None = source.url
        try:
            while (
                outcome.pages_fetched < section.max_pages_per_source
                and outcome.normalized < section.max_items_per_source
                and url
            ):
                limiter.wait()
                fetched = active_fetcher.get(url)
                outcome.http_status = fetched.status
                if not fetched.ok or not fetched.html:
                    outcome.reason = fetched.error or f"http_status={fetched.status}"
                    break
                page_no = outcome.pages_fetched + 1
                outcome.pages_fetched = page_no
                snapshot_path: str | None = None
                if section.store_raw_snapshots:
                    snap = run_dir / f"{_safe_slug(source.name)}__p{page_no:02d}.html"
                    snap.parent.mkdir(parents=True, exist_ok=True)
                    snap.write_text(fetched.html, encoding="utf-8")
                    snapshot_path = str(snap.resolve())
                    outcome.snapshot_paths.append(snapshot_path)
                    result.raw_html_snapshots += 1
                raw_posts = parse_old_reddit_listing(
                    fetched.html, default_subreddit=source.subreddit
                )
                for raw in raw_posts:
                    if outcome.normalized >= section.max_items_per_source:
                        break
                    outcome.raw_extracted += 1
                    result.raw_extracted += 1
                    post = reddit_raw_to_social_post(
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
                        "platform": "reddit",
                        "domain_tags": list(source.domains),
                        "extraction_status": "parsed",
                    }
                    raw_path = writer.write_post(
                        query="",
                        market_id=f"reddit-source-{source.name}",
                        post_id=post.post_id,
                        payload=payload,
                    )
                    collected.append((post, str(raw_path)))
                    outcome.normalized += 1
                url = find_next_page_url(fetched.html)
        except Exception as exc:
            outcome.status = "failed"
            outcome.reason = f"{type(exc).__name__}: {exc}"[:300]
            logger.warning("reddit source %s failed: %s", source.name, outcome.reason)
            writer.record_query("", f"reddit-source-{source.name}", 0, error=outcome.reason)
            result.outcomes.append(outcome)
            continue

        if outcome.normalized > 0:
            outcome.status = "succeeded"
        elif outcome.pages_fetched > 0:
            outcome.status = "empty"
            outcome.reason = outcome.reason or "no usable posts parsed"
        else:
            outcome.status = "failed"
            outcome.reason = outcome.reason or "no pages fetched"
        writer.record_query(
            "", f"reddit-source-{source.name}", outcome.normalized, error=outcome.reason
        )
        result.outcomes.append(outcome)

    records = social_posts_to_records(
        collected,
        collection_method=section.collection_method,
        query_type="reddit_public_web",
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
    "collect_reddit_scrapling",
    "find_next_page_url",
    "parse_old_reddit_listing",
    "reddit_raw_to_social_post",
]

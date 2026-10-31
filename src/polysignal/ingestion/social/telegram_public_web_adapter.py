"""Telegram public-web adapter (Scrapling + t.me/s/<channel>).

Branch 12. Telegram evidence is collected **only** from public channel preview
pages at ``https://t.me/s/<channel>`` when they are reachable without login.
Scrapling's plain ``Fetcher`` performs the fetch (no stealth, no proxy
rotation, no anti-bot circumvention); parsing is a pure, deterministic
``html.parser.HTMLParser`` state machine so it is fully unit testable offline.

Telegram renders each message as
``<div class="tgme_widget_message" data-post="channel/123"> ...
<div class="tgme_widget_message_text">text</div> ...
<a class="tgme_widget_message_date"><time datetime="..."></a> ...
<span class="tgme_widget_message_views">1.2K</span> ... </div>``.
"""

from __future__ import annotations

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
    ScraplingFetcher,
    SourceOutcome,
    csv_sibling,
    persist_records,
    social_posts_to_records,
    stable_post_id,
)
from polysignal.ingestion.social.enrichment_config import TelegramSectionConfig, TelegramSource
from polysignal.ingestion.social.raw_store import SocialRawWriter, generate_social_run_id
from polysignal.storage.paths import DataPaths, get_data_paths

logger = logging.getLogger(__name__)

_BEFORE_RE = re.compile(r'tme_messages_more[^>]*\bdata-before="(\d+)"', re.IGNORECASE)
_VIEWS_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*([KMB]?)$", re.IGNORECASE)


def _safe_slug(value: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in (value or "x"))[:64]


def _parse_views(text: str | None) -> int | None:
    if not text:
        return None
    match = _VIEWS_RE.match(text.strip())
    if not match:
        return None
    value = float(match.group(1))
    suffix = match.group(2).upper()
    multiplier = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[suffix]
    return int(value * multiplier)


# ---------------------------------------------------------------------------
# Pure HTML parser
# ---------------------------------------------------------------------------


class _TelegramPublicPageParser(HTMLParser):
    """Extract message blocks from a t.me/s/<channel> preview page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.messages: list[dict[str, Any]] = []
        self._div_depth = 0
        self._msg: dict[str, Any] | None = None
        self._msg_depth: int | None = None
        self._in_text = False
        self._text_depth: int | None = None
        self._text_parts: list[str] = []
        self._in_views = False
        self._views_parts: list[str] = []
        self._in_date_link = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: (value or "") for key, value in attrs}
        classes = attr.get("class", "").split()
        if tag == "div":
            self._div_depth += 1
            if self._msg is None and "tgme_widget_message" in classes and attr.get("data-post"):
                self._msg = {"data_post": attr.get("data-post", "")}
                self._msg_depth = self._div_depth
                self._in_text = False
                self._text_parts = []
            elif self._msg is not None and "tgme_widget_message_text" in classes and not self._in_text:
                self._in_text = True
                self._text_depth = self._div_depth
                self._text_parts = []
            return
        if self._msg is None:
            return
        if tag == "br" and self._in_text:
            self._text_parts.append("\n")
        elif tag == "time":
            when = attr.get("datetime")
            if when and (self._in_date_link or "datetime" not in self._msg):
                self._msg["datetime"] = when
        elif tag == "a" and "tgme_widget_message_date" in classes:
            self._in_date_link = True
            href = attr.get("href")
            if href:
                self._msg["url"] = href
        elif tag == "span" and "tgme_widget_message_views" in classes:
            self._in_views = True
            self._views_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_text:
            self._text_parts.append(data)
        elif self._in_views:
            self._views_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._msg is None:
            if tag == "div":
                self._div_depth = max(0, self._div_depth - 1)
            return
        if tag == "span" and self._in_views:
            self._msg.setdefault("views_text", "".join(self._views_parts).strip())
            self._in_views = False
        elif tag == "a" and self._in_date_link:
            self._in_date_link = False
        elif tag == "div":
            if self._in_text and self._div_depth == self._text_depth:
                self._msg["text"] = "".join(self._text_parts).strip()
                self._in_text = False
                self._text_depth = None
            elif self._msg_depth is not None and self._div_depth == self._msg_depth:
                self.messages.append(self._msg)
                self._msg = None
                self._msg_depth = None
        if tag == "div":
            self._div_depth = max(0, self._div_depth - 1)


def parse_telegram_public_page(
    html: str,
    *,
    channel: str | None = None,
) -> list[dict[str, Any]]:
    """Parse one t.me/s/<channel> page into raw message dicts."""
    parser = _TelegramPublicPageParser()
    parser.feed(html or "")
    parser.close()
    out: list[dict[str, Any]] = []
    for message in parser.messages:
        data_post = str(message.get("data_post") or "")
        msg_channel, _, message_id = data_post.partition("/")
        out.append(
            {
                "data_post": data_post,
                "channel": msg_channel or channel,
                "message_id": message_id or None,
                "text": message.get("text") or "",
                "datetime": message.get("datetime"),
                "url": message.get("url"),
                "views_text": message.get("views_text"),
                "views": _parse_views(message.get("views_text")),
            }
        )
    return out


def find_before_cursor(html: str) -> str | None:
    """Return the ``before`` pagination cursor for older messages, if present."""
    match = _BEFORE_RE.search(html or "")
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Normalization to SocialPost
# ---------------------------------------------------------------------------


def telegram_public_raw_to_social_post(
    raw: dict[str, Any],
    *,
    source: TelegramSource,
    collection_method: str,
    source_url: str,
    html_snapshot_path: str | None = None,
) -> SocialPost | None:
    """Normalize a raw t.me/s message dict into the shared ``SocialPost`` model.

    Returns ``None`` when there is neither usable text nor engagement metadata.
    Rows without a reliable timestamp are retained for audit and marked
    ineligible downstream.
    """
    text = str(raw.get("text") or "").strip()
    views = raw.get("views")
    if not text and views is None:
        return None
    when = raw.get("datetime")
    created_at: datetime | None = None
    created_at_parse_status = "missing"
    if when:
        try:
            created_at = datetime.fromisoformat(str(when).replace("Z", "+00:00"))
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            created_at = created_at.astimezone(timezone.utc)
            created_at_parse_status = "parsed"
        except ValueError:
            created_at_parse_status = "invalid"

    channel = str(raw.get("channel") or source.channel or "").strip()
    message_id = str(raw.get("message_id") or "").strip()
    if message_id:
        post_id = f"telegram:{channel}:{message_id}"
    else:
        post_id = "telegram:" + stable_post_id(channel, text, when)
    url = str(raw.get("url") or "") or (
        f"https://t.me/{channel}/{message_id}" if channel and message_id else None
    )

    extra: dict[str, Any] = {
        "channel": channel,
        "channel_title": channel,
        "message_id": message_id or None,
        "views_text": raw.get("views_text"),
        "collection_method": collection_method,
        "platform": Platform.TELEGRAM.value,
        "source_name": source.name,
        "source_url": source_url,
        "domain_tags": list(source.domains),
        "raw_snapshot_path": html_snapshot_path,
        "extraction_status": "parsed",
        "created_at_parse_status": created_at_parse_status,
    }
    return SocialPost(
        post_id=post_id,
        platform=Platform.TELEGRAM,
        author_id=channel or None,
        author_handle=channel or None,
        created_at=created_at,
        text=text,
        url=url,
        engagement=Engagement(views=views),
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Collection orchestrator
# ---------------------------------------------------------------------------


def collect_telegram_public_web(
    section: TelegramSectionConfig,
    *,
    paths: DataPaths | None = None,
    fetcher: HtmlFetcher | None = None,
    run_id: str | None = None,
    now: datetime | None = None,
    sleep: Callable[[float], None] = time.sleep,
    posts_path: Any | None = None,
) -> CollectionResult:
    """Collect every enabled Telegram channel into the unified social dataset.

    Each channel is fetched independently. A channel that is private, empty,
    blocked, or returns no public messages is recorded and the run continues.
    """
    paths = paths or get_data_paths()
    rid = run_id or generate_social_run_id()
    moment = (now or datetime.now(tz=timezone.utc)).astimezone(timezone.utc)
    run_dir = paths.raw_run_dir("social/telegram_public_web", rid, when=moment)
    writer = SocialRawWriter(directory=run_dir, platform="telegram_public_web")
    writer.extra["collection_method"] = section.collection_method
    writer.extra["user_agent"] = section.user_agent
    writer.extra["run_timestamp"] = moment

    active_fetcher = fetcher or ScraplingFetcher(
        user_agent=section.user_agent, timeout=section.request_timeout_seconds
    )
    limiter = RateLimiter(section.rate_limit_seconds, sleep=sleep)
    result = CollectionResult(
        platform="telegram",
        collection_method=section.collection_method,
        run_id=rid,
        started_at=moment,
        raw_dir=run_dir,
    )
    collected: list[tuple[SocialPost, str | None]] = []

    for source in section.sources:
        outcome = SourceOutcome(name=source.name, platform="telegram", url=source.url)
        seen_ids: set[str] = set()
        cursor: str | None = None
        try:
            while (
                outcome.pages_fetched < section.max_pages_per_source
                and outcome.normalized < section.max_items_per_source
            ):
                url = source.url if cursor is None else f"{source.url}?before={cursor}"
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
                raw_messages = parse_telegram_public_page(fetched.html, channel=source.channel)
                if not raw_messages:
                    break
                new_on_page = 0
                for raw in raw_messages:
                    if outcome.normalized >= section.max_items_per_source:
                        break
                    outcome.raw_extracted += 1
                    result.raw_extracted += 1
                    post = telegram_public_raw_to_social_post(
                        raw,
                        source=source,
                        collection_method=section.collection_method,
                        source_url=source.url,
                        html_snapshot_path=snapshot_path,
                    )
                    if post is None or post.post_id in seen_ids:
                        continue
                    seen_ids.add(post.post_id)
                    new_on_page += 1
                    payload = {
                        **raw,
                        "html_snapshot_path": snapshot_path,
                        "source_name": source.name,
                        "source_url": source.url,
                        "collection_method": section.collection_method,
                        "platform": "telegram",
                        "domain_tags": list(source.domains),
                        "extraction_status": "parsed",
                    }
                    raw_path = writer.write_post(
                        query="",
                        market_id=f"telegram-channel-{source.channel}",
                        post_id=post.post_id,
                        payload=payload,
                    )
                    collected.append((post, str(raw_path)))
                    outcome.normalized += 1
                next_cursor = find_before_cursor(fetched.html)
                if not next_cursor or next_cursor == cursor or new_on_page == 0:
                    break
                cursor = next_cursor
        except Exception as exc:
            outcome.status = "failed"
            outcome.reason = f"{type(exc).__name__}: {exc}"[:300]
            logger.warning("telegram source %s failed: %s", source.name, outcome.reason)
            writer.record_query("", f"telegram-channel-{source.channel}", 0, error=outcome.reason)
            result.outcomes.append(outcome)
            continue

        if outcome.normalized > 0:
            outcome.status = "succeeded"
        elif outcome.pages_fetched > 0:
            outcome.status = "empty"
            outcome.reason = outcome.reason or "no public messages parsed"
        else:
            outcome.status = "failed"
            outcome.reason = outcome.reason or "no pages fetched"
        writer.record_query(
            "", f"telegram-channel-{source.channel}", outcome.normalized, error=outcome.reason
        )
        result.outcomes.append(outcome)

    records = social_posts_to_records(
        collected,
        collection_method=section.collection_method,
        query_type="telegram_public_web",
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
    "collect_telegram_public_web",
    "find_before_cursor",
    "parse_telegram_public_page",
    "telegram_public_raw_to_social_post",
]

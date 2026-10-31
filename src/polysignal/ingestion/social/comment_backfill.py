"""Resumable Reddit and X comment/reply backfill.

The normal post collectors intentionally remain focused on discovery. This
module opens already-collected Reddit/X post threads, samples public replies,
stores raw snapshots, appends deduplicated ``PostComment`` rows, and updates
``comment_sample_count`` on the corresponding staging post table.
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

from polysignal.domain import PostComment
from polysignal.ingestion.social.comments import (
    enrich_comment_text,
    parse_reddit_comment_html,
    x_replies_to_comments,
)
from polysignal.ingestion.social.enrichment import RateLimiter, build_html_fetcher
from polysignal.ingestion.social.enrichment_config import load_enrichment_config
from polysignal.storage.parquet_store import (
    backup_parquet,
    read_dataframe,
    write_dataframe,
)
from polysignal.storage.run_lock import lock_is_active, run_lock

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/social_enrichment_sources.example.yaml")
DEFAULT_REDDIT_POSTS = Path("data/staging/reddit/normalized/social/social_posts.parquet")
DEFAULT_TWITTER_POSTS = Path("data/staging/twitter_web/social_posts.parquet")
DEFAULT_REDDIT_COMMENTS = Path(
    "data/staging/reddit/normalized/social/post_comments.parquet"
)
DEFAULT_TWITTER_COMMENTS = Path(
    "data/staging/twitter_web/post_comments.parquet"
)
DEFAULT_COMBINED_COMMENTS = Path(
    "data/staging/social_comments/normalized/social/post_comments.parquet"
)
DEFAULT_STATE = Path("data/staging/social_comments/state/comment_backfill.json")
DEFAULT_RAW_ROOT = Path("data/staging/social_comments/raw")
DEFAULT_SUMMARY = Path("data/staging/social_comments/reports/comment_backfill_summary.json")
SCRAPLING_LOCK = Path("data/staging/enriched/scrapling_web.lock")


@dataclass
class CommentBackfillOptions:
    """Runtime controls for a bounded, resumable comment collection pass."""

    platform: str = "all"
    config_path: Path = DEFAULT_CONFIG
    reddit_posts_path: Path = DEFAULT_REDDIT_POSTS
    twitter_posts_path: Path = DEFAULT_TWITTER_POSTS
    reddit_comments_path: Path = DEFAULT_REDDIT_COMMENTS
    twitter_comments_path: Path = DEFAULT_TWITTER_COMMENTS
    combined_comments_path: Path = DEFAULT_COMBINED_COMMENTS
    state_path: Path = DEFAULT_STATE
    raw_root: Path = DEFAULT_RAW_ROOT
    summary_path: Path = DEFAULT_SUMMARY
    reddit_max_posts: int = 100
    twitter_max_posts: int = 50
    comments_per_post: int = 10
    rate_limit_seconds: float = 3.0
    reset_progress: bool = False
    metadata_only: bool = False
    force_post_update: bool = False
    dry_run: bool = False


@dataclass
class PlatformBackfillResult:
    platform: str
    selected_posts: int = 0
    attempted_posts: int = 0
    successful_posts: int = 0
    failed_posts: int = 0
    sampled_comments: int = 0
    rows_added: int = 0
    total_comments: int = 0
    post_metadata_updated: bool = False
    engagement_updates: dict[str, dict[str, int | None]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload["engagement_updates"] = len(self.engagement_updates)
        return payload


@dataclass
class CommentBackfillResult:
    started_at: datetime
    finished_at: datetime | None = None
    platforms: list[PlatformBackfillResult] = field(default_factory=list)
    combined_comments: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "combined_comments": self.combined_comments,
            "platforms": [item.as_dict() for item in self.platforms],
        }


def _clean_scalar(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")[:100] or "unknown"


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _write_text_snapshot(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def load_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "reddit": {}, "twitter": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "reddit": {}, "twitter": {}}
    if not isinstance(payload, dict):
        return {"version": 1, "reddit": {}, "twitter": {}}
    payload.setdefault("version", 1)
    payload.setdefault("reddit", {})
    payload.setdefault("twitter", {})
    return payload


def save_progress(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
    _write_json_atomic(path, state)


def reddit_thread_url(url: str) -> str:
    """Normalize a Reddit permalink to rendered old-Reddit thread HTML."""
    parts = urlsplit(url.strip())
    path = parts.path
    if "/comments/" not in path:
        return ""
    host = "old.reddit.com"
    query = "sort=top"
    return urlunsplit(("https", host, path.rstrip("/") + "/", query, ""))


def _post_id_from_x_url(url: str) -> str | None:
    match = re.search(r"/status/(\d+)", url or "")
    return match.group(1) if match else None


def _candidate_posts(
    posts_path: Path,
    *,
    platform: str,
    completed: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    frame = read_dataframe(posts_path)
    if frame.empty or "post_id" not in frame.columns or "url" not in frame.columns:
        return []
    frame = frame[frame["url"].notna()].copy()
    frame = frame[~frame["post_id"].astype(str).isin(completed)]
    if platform == "reddit":
        if "comments_count" in frame.columns:
            frame["_priority"] = pd.to_numeric(
                frame["comments_count"], errors="coerce"
            ).fillna(0)
            frame = frame[frame["_priority"] > 0]
        else:
            frame["_priority"] = 0
    else:
        if "reply_count" in frame.columns:
            frame["_priority"] = pd.to_numeric(
                frame["reply_count"], errors="coerce"
            ).fillna(0)
        else:
            frame["_priority"] = 0
        frame["_created"] = pd.to_datetime(
            frame.get("created_at"), errors="coerce", utc=True
        )
    sort_columns = ["_priority"]
    ascending = [False]
    if "_created" in frame.columns:
        sort_columns.append("_created")
        ascending.append(False)
    frame = frame.sort_values(sort_columns, ascending=ascending)
    frame = frame.drop_duplicates(subset=["post_id"], keep="first")
    if limit > 0:
        frame = frame.head(limit)
    return [
        {key: _clean_scalar(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def merge_comment_records(
    comments: list[PostComment],
    path: Path,
) -> tuple[int, int]:
    """Enrich and idempotently merge comments into a staging parquet."""
    existing = read_dataframe(path)
    if comments:
        incoming = pd.DataFrame(enrich_comment_text(comments))
        merged = (
            pd.concat([existing, incoming], ignore_index=True, sort=False)
            if not existing.empty
            else incoming
        )
    else:
        merged = existing
    if merged.empty:
        return 0, 0
    before = len(existing)
    keys = ["platform", "post_id", "comment_id"]
    merged = merged.drop_duplicates(subset=keys, keep="last")
    merged = merged.sort_values(keys, kind="stable").reset_index(drop=True)
    backup_parquet(path)
    write_dataframe(merged, path)
    merged.to_csv(path.with_suffix(".csv"), index=False)
    return max(0, len(merged) - before), len(merged)


def combine_comment_tables(paths: list[Path], output_path: Path) -> int:
    frames = [read_dataframe(path) for path in paths if path.exists()]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return 0
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.drop_duplicates(
        subset=["platform", "post_id", "comment_id"], keep="last"
    )
    merged = merged.sort_values(
        ["platform", "post_id", "comment_id"], kind="stable"
    ).reset_index(drop=True)
    backup_parquet(output_path)
    write_dataframe(merged, output_path)
    merged.to_csv(output_path.with_suffix(".csv"), index=False)
    return len(merged)


def update_post_metadata(
    posts_path: Path,
    comments_path: Path,
    *,
    engagement_updates: dict[str, dict[str, int | None]] | None = None,
) -> int:
    """Update sampled-comment counts and optional X engagement atomically."""
    posts = read_dataframe(posts_path)
    if posts.empty:
        return 0
    comments = read_dataframe(comments_path)
    counts: dict[str, int] = {}
    if not comments.empty:
        counts = (
            comments.drop_duplicates(subset=["platform", "post_id", "comment_id"])
            .groupby("post_id")
            .size()
            .astype(int)
            .to_dict()
        )
    if "comment_sample_count" not in posts.columns:
        posts["comment_sample_count"] = 0
    mapped = posts["post_id"].astype(str).map(counts)
    posts["comment_sample_count"] = mapped.fillna(
        pd.to_numeric(posts["comment_sample_count"], errors="coerce").fillna(0)
    ).astype(int)

    for post_id, metrics in (engagement_updates or {}).items():
        mask = posts["post_id"].astype(str) == str(post_id)
        if not mask.any():
            continue
        for column, value in metrics.items():
            if value is None:
                continue
            if column not in posts.columns:
                posts[column] = None
            posts.loc[mask, column] = int(value)
    backup_parquet(posts_path)
    write_dataframe(posts, posts_path)
    return int((posts["comment_sample_count"] > 0).sum())


def _mark_progress(
    state: dict[str, Any],
    *,
    platform: str,
    post_id: str,
    status: str,
    comments: int = 0,
    error: str | None = None,
) -> None:
    current = state.setdefault(platform, {}).get(post_id, {})
    attempts = int(current.get("attempts", 0) or 0) + 1
    state[platform][post_id] = {
        "status": status,
        "comments": int(comments),
        "attempts": attempts,
        "error": error,
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def collect_reddit_comments(
    options: CommentBackfillOptions,
    state: dict[str, Any],
) -> PlatformBackfillResult:
    result = PlatformBackfillResult(platform="reddit")
    config = load_enrichment_config(options.config_path)
    section = config.reddit
    if section is None or not section.enabled or not section.parse_comments:
        logger.info("Reddit comments disabled by config")
        return result
    completed = {
        post_id
        for post_id, item in state.get("reddit", {}).items()
        if isinstance(item, dict) and item.get("status") == "success"
    }
    posts = _candidate_posts(
        options.reddit_posts_path,
        platform="reddit",
        completed=completed,
        limit=options.reddit_max_posts,
    )
    result.selected_posts = len(posts)
    logger.info("Reddit comment threads selected=%s", len(posts))
    if options.dry_run:
        return result

    fetcher = build_html_fetcher(
        section.base_fetcher,
        user_agent=section.user_agent,
        timeout=section.request_timeout_seconds,
    )
    limiter = RateLimiter(max(options.rate_limit_seconds, section.rate_limit_seconds))
    day = datetime.now(tz=timezone.utc).date().isoformat()
    for index, row in enumerate(posts, start=1):
        post_id = str(row["post_id"])
        url = reddit_thread_url(str(row["url"]))
        if not url:
            _mark_progress(
                state,
                platform="reddit",
                post_id=post_id,
                status="failed",
                error="invalid Reddit thread URL",
            )
            save_progress(options.state_path, state)
            result.failed_posts += 1
            continue
        result.attempted_posts += 1
        limiter.wait()
        fetched = fetcher.get(url)
        if not fetched.ok or not fetched.html:
            error = f"HTTP {fetched.status}: {fetched.error or 'thread unavailable'}"
            logger.warning("Reddit %s failed: %s", post_id, error)
            _mark_progress(
                state,
                platform="reddit",
                post_id=post_id,
                status="failed",
                error=error,
            )
            save_progress(options.state_path, state)
            result.failed_posts += 1
            continue

        raw_path = options.raw_root / "reddit" / day / f"{_safe_name(post_id)}.html"
        _write_text_snapshot(raw_path, fetched.html)
        comments = parse_reddit_comment_html(
            fetched.html,
            post_id=post_id,
            market_id_candidate=row.get("market_id_candidate"),
            raw_path=str(raw_path),
            max_comments=options.comments_per_post,
        )
        added, total = merge_comment_records(comments, options.reddit_comments_path)
        result.rows_added += added
        result.total_comments = total
        result.sampled_comments += len(comments)
        result.successful_posts += 1
        _mark_progress(
            state,
            platform="reddit",
            post_id=post_id,
            status="success",
            comments=len(comments),
        )
        save_progress(options.state_path, state)
        logger.info(
            "Reddit [%s/%s] %s comments=%s added=%s",
            index,
            len(posts),
            post_id,
            len(comments),
            added,
        )
    return result


_X_THREAD_JS = r"""
() => {
  const rows = [];
  function metric(article, testid) {
    const node = article.querySelector(`[data-testid="${testid}"]`);
    return node ? (node.getAttribute("aria-label") || node.innerText || "") : "";
  }
  for (const article of document.querySelectorAll('article[data-testid="tweet"]')) {
    const textNode = article.querySelector('div[data-testid="tweetText"]');
    const timeNode = article.querySelector("time");
    let tweetId = "", username = "", url = "";
    for (const a of article.querySelectorAll('a[href*="/status/"]')) {
      const href = a.getAttribute("href") || "";
      const match = href.match(/^\/([^/]+)\/status\/(\d+)/);
      if (match) {
        username = match[1];
        tweetId = match[2];
        url = "https://x.com" + href.split("?")[0];
        break;
      }
    }
    if (!tweetId || !textNode) continue;
    const viewsNode = article.querySelector('a[href$="/analytics"]');
    rows.push({
      tweet_id: tweetId,
      username,
      url,
      text: textNode.innerText || "",
      created_at: timeNode ? timeNode.getAttribute("datetime") : null,
      like_label: metric(article, "like"),
      reply_label: metric(article, "reply"),
      repost_label: metric(article, "retweet"),
      view_label: viewsNode ? (viewsNode.getAttribute("aria-label") || viewsNode.innerText || "") : ""
    });
  }
  return rows;
}
"""


def parse_human_count(value: str | None) -> int | None:
    """Parse the first X metric count from labels such as ``1.2K Likes``."""
    if not value:
        return None
    match = re.search(r"([\d,.]+)\s*([KMB]?)", str(value), flags=re.IGNORECASE)
    if not match:
        return None
    number = match.group(1).replace(",", "")
    try:
        numeric = float(number)
    except ValueError:
        return None
    multiplier = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    return max(0, int(numeric * multiplier[match.group(2).upper()]))


def normalize_x_thread_rows(
    rows: list[dict[str, Any]],
    *,
    original_tweet_id: str,
) -> tuple[list[dict[str, Any]], dict[str, int | None]]:
    """Split DOM rows into reply objects and original-post engagement."""
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        tweet_id = str(row.get("tweet_id") or "")
        if tweet_id:
            unique[tweet_id] = row
    original = unique.pop(original_tweet_id, {})
    engagement = {
        "like_count": parse_human_count(original.get("like_label")),
        "reply_count": parse_human_count(original.get("reply_label")),
        "comments_count": parse_human_count(original.get("reply_label")),
        "repost_count": parse_human_count(original.get("repost_label")),
        "shares_count": parse_human_count(original.get("repost_label")),
        "view_count": parse_human_count(original.get("view_label")),
        "views_count": parse_human_count(original.get("view_label")),
    }
    replies: list[dict[str, Any]] = []
    for row in unique.values():
        replies.append(
            {
                "tweet_id": row.get("tweet_id"),
                "username": row.get("username"),
                "text": row.get("text"),
                "created_at": row.get("created_at"),
                "favorite_count": parse_human_count(row.get("like_label")),
                "reply_count": parse_human_count(row.get("reply_label")),
                "parent_tweet_id": original_tweet_id,
            }
        )
    return replies, engagement


def _load_local_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _x_cookies() -> list[dict[str, Any]]:
    _load_local_env()
    auth = os.environ.get("X_AUTH_TOKEN") or os.environ.get("TWITTER_AUTH_TOKEN")
    ct0 = os.environ.get("X_CT0") or os.environ.get("TWITTER_CT0")
    if not auth or not ct0:
        raise RuntimeError("X_AUTH_TOKEN and X_CT0 are required for X reply collection")
    cookies: list[dict[str, Any]] = []
    for domain in (".x.com", ".twitter.com"):
        cookies.extend(
            [
                {
                    "name": "auth_token",
                    "value": auth,
                    "domain": domain,
                    "path": "/",
                    "httpOnly": True,
                    "secure": True,
                },
                {
                    "name": "ct0",
                    "value": ct0,
                    "domain": domain,
                    "path": "/",
                    "secure": True,
                },
            ]
        )
    return cookies


def collect_twitter_comments(
    options: CommentBackfillOptions,
    state: dict[str, Any],
) -> PlatformBackfillResult:
    result = PlatformBackfillResult(platform="twitter")
    config = load_enrichment_config(options.config_path)
    section = config.twitter
    if section is None or not section.enabled or not section.parse_comments:
        logger.info("X comments disabled by config")
        return result
    completed = {
        post_id
        for post_id, item in state.get("twitter", {}).items()
        if isinstance(item, dict) and item.get("status") == "success"
    }
    posts = _candidate_posts(
        options.twitter_posts_path,
        platform="twitter",
        completed=completed,
        limit=options.twitter_max_posts,
    )
    result.selected_posts = len(posts)
    logger.info("X reply threads selected=%s", len(posts))
    if options.dry_run:
        return result

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - local scraping environment has it
        raise RuntimeError("Playwright is required for authenticated X replies") from exc

    limiter = RateLimiter(max(options.rate_limit_seconds, section.rate_limit_seconds))
    day = datetime.now(tz=timezone.utc).date().isoformat()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 1800},
            locale="en-US",
        )
        context.add_cookies(_x_cookies())
        page = context.new_page()
        for index, row in enumerate(posts, start=1):
            post_id = str(row["post_id"])
            url = str(row["url"])
            original_tweet_id = _post_id_from_x_url(url)
            if not original_tweet_id:
                _mark_progress(
                    state,
                    platform="twitter",
                    post_id=post_id,
                    status="failed",
                    error="invalid X thread URL",
                )
                save_progress(options.state_path, state)
                result.failed_posts += 1
                continue
            result.attempted_posts += 1
            limiter.wait()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=45_000)
                with suppress(Exception):
                    page.wait_for_selector(
                        "article[data-testid='tweet']", timeout=20_000
                    )
                collected_rows: dict[str, dict[str, Any]] = {}
                stale = 0
                for _ in range(8):
                    for extracted in page.evaluate(_X_THREAD_JS):
                        tweet_id = str(extracted.get("tweet_id") or "")
                        if tweet_id:
                            collected_rows[tweet_id] = extracted
                    reply_count = len(
                        [key for key in collected_rows if key != original_tweet_id]
                    )
                    if reply_count >= options.comments_per_post:
                        break
                    before = len(collected_rows)
                    page.mouse.wheel(0, 1800)
                    page.wait_for_timeout(1200)
                    stale = stale + 1 if len(collected_rows) == before else 0
                    if stale >= 3:
                        break

                extracted_rows = list(collected_rows.values())
                replies, engagement = normalize_x_thread_rows(
                    extracted_rows,
                    original_tweet_id=original_tweet_id,
                )
                replies = replies[: options.comments_per_post]
                raw_base = options.raw_root / "twitter" / day / _safe_name(post_id)
                html_path = _write_text_snapshot(
                    raw_base.with_suffix(".html"), page.content()
                )
                json_path = raw_base.with_suffix(".json")
                _write_json_atomic(
                    json_path,
                    {
                        "post_id": post_id,
                        "url": url,
                        "original_tweet_id": original_tweet_id,
                        "rows": extracted_rows,
                    },
                )
                comments = x_replies_to_comments(
                    replies,
                    post_id=post_id,
                    market_id_candidate=row.get("market_id_candidate"),
                    raw_path=str(json_path),
                    max_comments=options.comments_per_post,
                )
                added, total = merge_comment_records(
                    comments, options.twitter_comments_path
                )
                result.rows_added += added
                result.total_comments = total
                result.sampled_comments += len(comments)
                result.successful_posts += 1
                result.engagement_updates[post_id] = engagement
                _mark_progress(
                    state,
                    platform="twitter",
                    post_id=post_id,
                    status="success",
                    comments=len(comments),
                )
                save_progress(options.state_path, state)
                logger.info(
                    "X [%s/%s] %s replies=%s added=%s raw=%s",
                    index,
                    len(posts),
                    post_id,
                    len(comments),
                    added,
                    html_path,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:240]}"
                logger.warning("X %s failed: %s", post_id, error)
                _mark_progress(
                    state,
                    platform="twitter",
                    post_id=post_id,
                    status="failed",
                    error=error,
                )
                save_progress(options.state_path, state)
                result.failed_posts += 1
        browser.close()
    return result


def run_comment_backfill(options: CommentBackfillOptions) -> CommentBackfillResult:
    """Run a bounded resumable backfill and persist a machine-readable summary."""
    result = CommentBackfillResult(started_at=datetime.now(tz=timezone.utc))
    if options.reset_progress and options.state_path.exists():
        options.state_path.unlink()
    state = load_progress(options.state_path)
    lock_path = options.state_path.parent / "comment_backfill.lock"
    with run_lock(lock_path, label="social-comment-backfill"):
        if not options.metadata_only:
            if options.platform in {"all", "reddit"}:
                result.platforms.append(collect_reddit_comments(options, state))
            if options.platform in {"all", "twitter"}:
                result.platforms.append(collect_twitter_comments(options, state))

        by_platform = {item.platform: item for item in result.platforms}
        if options.platform in {"all", "reddit"} and options.reddit_comments_path.exists():
            reddit_result = by_platform.setdefault(
                "reddit", PlatformBackfillResult(platform="reddit")
            )
            if lock_is_active(SCRAPLING_LOCK) and not options.force_post_update:
                logger.warning(
                    "Active Scrapling post collector detected; Reddit "
                    "comment_sample_count update deferred. Stop it and rerun "
                    "with --metadata-only."
                )
            elif not options.dry_run:
                update_post_metadata(
                    options.reddit_posts_path, options.reddit_comments_path
                )
                reddit_result.post_metadata_updated = True
        if options.platform in {"all", "twitter"} and options.twitter_comments_path.exists():
            twitter_result = by_platform.setdefault(
                "twitter", PlatformBackfillResult(platform="twitter")
            )
            if not options.dry_run:
                update_post_metadata(
                    options.twitter_posts_path,
                    options.twitter_comments_path,
                    engagement_updates=twitter_result.engagement_updates,
                )
                twitter_result.post_metadata_updated = True

        result.platforms = list(by_platform.values())
        if not options.dry_run:
            result.combined_comments = combine_comment_tables(
                [options.reddit_comments_path, options.twitter_comments_path],
                options.combined_comments_path,
            )
            save_progress(options.state_path, state)
        result.finished_at = datetime.now(tz=timezone.utc)
        if not options.dry_run:
            _write_json_atomic(options.summary_path, result.as_dict())
    return result


__all__ = [
    "CommentBackfillOptions",
    "CommentBackfillResult",
    "PlatformBackfillResult",
    "collect_reddit_comments",
    "collect_twitter_comments",
    "combine_comment_tables",
    "load_progress",
    "merge_comment_records",
    "normalize_x_thread_rows",
    "parse_human_count",
    "reddit_thread_url",
    "run_comment_backfill",
    "save_progress",
    "update_post_metadata",
]

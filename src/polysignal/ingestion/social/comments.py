"""Comment / reply collection for Traccia_2 §2 (``post_comments.parquet``).

Comments are stored in a sibling table keyed by ``(platform, post_id)`` rather
than inline on :class:`SocialPostRecord`, so the flat post corpus is never
rewritten. Two real platforms are wired:

* **Reddit** — old.reddit ``<permalink>.json`` returns a 2-element array
  ``[<post listing>, <comments listing>]``. :func:`parse_reddit_comment_json`
  walks the comment forest (including ``replies``) into flat
  :class:`~polysignal.domain.PostComment` rows.
* **Telegram** — discussion-group replies exposed by Telethon as message
  objects. :func:`telegram_replies_to_comments` normalizes those.

NER + language detection run over comment ``text`` exactly as for top-level
posts (:func:`enrich_comment_text`). The functions here are pure (no network,
no settings reads) so they unit-test against fixtures; the live fetch is the
caller's concern.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polysignal.domain import Platform, PostComment
from polysignal.nlp.extract import detect_language, extract_hashtags, get_entity_backend
from polysignal.storage.parquet_store import write_parquet


def _utc(value: Any) -> datetime | None:
    """Coerce an epoch-seconds number or ISO string into an aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            # Fall back to X/Twitter's native format ('Wed May 21 21:00:00 +0000 2025').
            try:
                parsed = datetime.strptime(value.strip(), "%a %b %d %H:%M:%S %z %Y")
            except ValueError:
                return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _author_url(platform: Platform, author: str | None) -> str | None:
    handle = (author or "").lstrip("@").strip()
    if not handle or handle in {"[deleted]", "[removed]"}:
        return None
    if platform == Platform.REDDIT:
        return f"https://www.reddit.com/user/{handle}"
    if platform == Platform.TELEGRAM:
        return f"https://t.me/{handle}"
    if platform == Platform.TWITTER:
        return f"https://x.com/{handle}"
    return None


# ---------------------------------------------------------------------------
# Reddit comment JSON
# ---------------------------------------------------------------------------


def parse_reddit_comment_json(
    payload: Any,
    *,
    post_id: str,
    market_id_candidate: str | None = None,
    raw_path: str | None = None,
    collected_at: datetime | None = None,
    max_comments: int | None = None,
) -> list[PostComment]:
    """Flatten an old.reddit ``<permalink>.json`` payload into PostComment rows.

    ``payload`` is the parsed JSON: ``[post_listing, comments_listing]``. Only
    the second listing is walked. ``more`` continuation stubs are skipped.
    """
    collected = collected_at or datetime.now(tz=timezone.utc)
    comments_listing: Any = None
    if isinstance(payload, list) and len(payload) >= 2:
        comments_listing = payload[1]
    elif isinstance(payload, dict):
        comments_listing = payload
    if not isinstance(comments_listing, dict):
        return []

    rows: list[PostComment] = []

    def _walk(children: Any, parent_id: str | None) -> None:
        if not isinstance(children, list):
            return
        for child in children:
            if max_comments is not None and len(rows) >= max_comments:
                return
            if not isinstance(child, dict) or child.get("kind") != "t1":
                continue
            data = child.get("data")
            if not isinstance(data, dict):
                continue
            body = str(data.get("body") or "").strip()
            comment_id = str(data.get("id") or data.get("name") or "").strip()
            if not comment_id:
                continue
            author = data.get("author")
            author = str(author) if author else None
            rows.append(
                PostComment(
                    post_id=post_id,
                    platform=Platform.REDDIT,
                    comment_id=comment_id,
                    parent_comment_id=parent_id,
                    author_id=author,
                    author_url=_author_url(Platform.REDDIT, author),
                    text=body,
                    published_at=_utc(data.get("created_utc")),
                    like_count=_int(data.get("score")),
                    reply_count=None,
                    market_id_candidate=market_id_candidate,
                    raw_path=raw_path,
                    collected_at=collected,
                    text_length=len(body),
                )
            )
            replies = data.get("replies")
            if isinstance(replies, dict):
                nested = replies.get("data", {}).get("children")
                _walk(nested, comment_id)

    _walk(comments_listing.get("data", {}).get("children"), None)
    return rows


def parse_reddit_comment_html(
    html_text: str,
    *,
    post_id: str,
    market_id_candidate: str | None = None,
    raw_path: str | None = None,
    collected_at: datetime | None = None,
    max_comments: int | None = None,
) -> list[PostComment]:
    """Parse rendered ``old.reddit.com`` thread HTML into comment rows.

    Reddit currently returns HTTP 403 for unauthenticated ``.json`` thread
    endpoints while the same public thread is available through the rendered
    old-Reddit HTML page. This parser is therefore the live public-web fallback
    used by the resumable comment backfill.
    """
    if not html_text.strip():
        return []
    try:
        from lxml import html
    except ImportError as exc:  # pragma: no cover - scraping extra supplies lxml
        raise RuntimeError(
            "lxml is required to parse rendered Reddit comments. "
            'Install the scraping extra with: pip install -e ".[scraping]"'
        ) from exc

    try:
        root = html.fromstring(html_text)
    except (TypeError, ValueError):
        return []

    collected = collected_at or datetime.now(tz=timezone.utc)
    rows: list[PostComment] = []
    comment_nodes = root.xpath(
        '//div[@data-type="comment" and starts-with(@data-fullname, "t1_")]'
    )
    for node in comment_nodes:
        if max_comments is not None and len(rows) >= max_comments:
            break
        fullname = str(node.get("data-fullname") or "")
        comment_id = fullname.removeprefix("t1_").strip()
        if not comment_id:
            continue

        body_nodes = node.xpath(
            './div[contains(concat(" ", normalize-space(@class), " "), " entry ")]'
            '//div[contains(concat(" ", normalize-space(@class), " "), " usertext-body ")]'
            '/div[contains(concat(" ", normalize-space(@class), " "), " md ")][1]'
        )
        if not body_nodes:
            continue
        body = re.sub(r"\s+", " ", body_nodes[0].text_content()).strip()
        if not body or body.lower() in {"[deleted]", "[removed]"}:
            continue

        author = str(node.get("data-author") or "").strip() or None
        author_nodes = node.xpath(
            './div[contains(concat(" ", normalize-space(@class), " "), " entry ")]'
            '//a[contains(concat(" ", normalize-space(@class), " "), " author ")][1]'
        )
        author_url = None
        if author_nodes:
            author = author or author_nodes[0].text_content().strip() or None
            author_url = str(author_nodes[0].get("href") or "").strip() or None

        time_nodes = node.xpath(
            './div[contains(concat(" ", normalize-space(@class), " "), " entry ")]'
            '//time[@datetime][1]'
        )
        published_at = _utc(time_nodes[0].get("datetime")) if time_nodes else None

        score_nodes = node.xpath(
            './div[contains(concat(" ", normalize-space(@class), " "), " entry ")]'
            '//span[contains(concat(" ", normalize-space(@class), " "), " score ")'
            ' and contains(concat(" ", normalize-space(@class), " "), " unvoted ")][1]'
        )
        if not score_nodes:
            score_nodes = node.xpath(
                './div[contains(concat(" ", normalize-space(@class), " "), " entry ")]'
                '//span[contains(concat(" ", normalize-space(@class), " "), " score ")][1]'
            )
        like_count = None
        if score_nodes:
            raw_score = str(score_nodes[0].get("title") or "").replace(",", "")
            if raw_score.lstrip("-").isdigit():
                like_count = max(0, int(raw_score))

        ancestors = node.xpath(
            'ancestor::div[@data-type="comment" and starts-with(@data-fullname, "t1_")][1]'
        )
        parent_comment_id = None
        if ancestors:
            parent_comment_id = str(ancestors[0].get("data-fullname") or "").removeprefix("t1_")
            parent_comment_id = parent_comment_id or None

        raw_replies = str(node.get("data-replies") or "")
        reply_count = int(raw_replies) if raw_replies.isdigit() else None
        rows.append(
            PostComment(
                post_id=post_id,
                platform=Platform.REDDIT,
                comment_id=comment_id,
                parent_comment_id=parent_comment_id,
                author_id=author,
                author_url=author_url or _author_url(Platform.REDDIT, author),
                text=body,
                published_at=published_at,
                like_count=like_count,
                reply_count=reply_count,
                market_id_candidate=market_id_candidate,
                raw_path=raw_path,
                collected_at=collected,
                text_length=len(body),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Telegram discussion-group replies
# ---------------------------------------------------------------------------


def telegram_replies_to_comments(
    replies: Iterable[Any],
    *,
    post_id: str,
    channel: str | None = None,
    market_id_candidate: str | None = None,
    raw_path: str | None = None,
    collected_at: datetime | None = None,
) -> list[PostComment]:
    """Normalize Telethon reply message objects/dicts into PostComment rows."""
    collected = collected_at or datetime.now(tz=timezone.utc)
    rows: list[PostComment] = []
    for reply in replies:
        get = reply.get if isinstance(reply, dict) else lambda k, _r=reply: getattr(_r, k, None)
        message_id = get("id") or get("message_id")
        if message_id is None:
            continue
        text = str(get("message") or get("text") or "").strip()
        sender = get("sender_username") or get("from_id") or get("author")
        sender = str(sender) if sender else None
        rows.append(
            PostComment(
                post_id=post_id,
                platform=Platform.TELEGRAM,
                comment_id=f"telegram:{channel or 'group'}:{message_id}",
                parent_comment_id=(
                    f"telegram:{channel or 'group'}:{get('reply_to_msg_id')}"
                    if get("reply_to_msg_id")
                    else None
                ),
                author_id=sender,
                author_url=_author_url(Platform.TELEGRAM, sender),
                text=text,
                published_at=_utc(get("date")),
                like_count=None,
                reply_count=None,
                market_id_candidate=market_id_candidate,
                raw_path=raw_path,
                collected_at=collected,
                text_length=len(text),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# X / Twitter replies (authenticated web path)
# ---------------------------------------------------------------------------


def x_replies_to_comments(
    replies: Iterable[Any],
    *,
    post_id: str,
    market_id_candidate: str | None = None,
    raw_path: str | None = None,
    collected_at: datetime | None = None,
    max_comments: int | None = None,
) -> list[PostComment]:
    """Normalize X reply objects/dicts into PostComment rows.

    Accepts either the flat DOM-extracted reply dict (``tweet_id``, ``text``,
    ``username``, ``like_count``, ...) or a GraphQL ``legacy`` reply object
    (``id_str``, ``full_text``, ``favorite_count``, ``reply_count``,
    ``in_reply_to_status_id_str``). Pure + offline.
    """
    collected = collected_at or datetime.now(tz=timezone.utc)
    rows: list[PostComment] = []
    for reply in replies:
        get = reply.get if isinstance(reply, dict) else lambda k, _r=reply: getattr(_r, k, None)
        comment_id = get("tweet_id") or get("id_str") or get("id")
        if comment_id is None:
            continue
        text = str(get("full_text") or get("text") or "").strip()
        author = get("username") or get("screen_name") or get("author")
        author = str(author) if author else None
        parent = get("in_reply_to_status_id_str") or get("parent_tweet_id")
        rows.append(
            PostComment(
                post_id=post_id,
                platform=Platform.TWITTER,
                comment_id=f"twitter:{comment_id}",
                parent_comment_id=f"twitter:{parent}" if parent else None,
                author_id=author,
                author_url=_author_url(Platform.TWITTER, author),
                text=text,
                published_at=_utc(get("created_at")),
                like_count=_int(get("favorite_count") or get("like_count")),
                reply_count=_int(get("reply_count")),
                market_id_candidate=market_id_candidate,
                raw_path=raw_path,
                collected_at=collected,
                text_length=len(text),
            )
        )
        if max_comments is not None and len(rows) >= max_comments:
            break
    return rows


# ---------------------------------------------------------------------------
# Enrichment + persistence
# ---------------------------------------------------------------------------


def enrich_comment_text(comments: Sequence[PostComment]) -> list[dict[str, Any]]:
    """Return comment rows augmented with language + NER (hashtags/entities).

    The base PostComment columns are preserved; the enrichment adds
    ``language`` (filled when missing), ``hashtags``, and ``mentioned_entities``
    so §2 content fields are present for comments too.
    """
    backend = get_entity_backend("auto")
    enriched: list[dict[str, Any]] = []
    for comment in comments:
        row = comment.model_dump(mode="json")
        text = comment.text or ""
        row["language"] = comment.language or detect_language(text)
        row["hashtags"] = extract_hashtags(text)
        row["mentioned_entities"] = [e.as_dict() for e in backend.entities(text)]
        enriched.append(row)
    return enriched


def write_post_comments(
    comments: Sequence[PostComment],
    path: Path,
    *,
    enrich: bool = True,
) -> Path:
    """Write PostComment rows to ``post_comments.parquet`` (NER/language by default)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if enrich:
        rows = enrich_comment_text(comments)
        import pandas as pd

        pd.DataFrame(rows).to_parquet(path, index=False)
    else:
        write_parquet(list(comments), path)
    return path


def comment_counts_by_post(comments: Sequence[PostComment]) -> dict[str, int]:
    """Count sampled comments per ``post_id`` (for ``comment_sample_count``)."""
    counts: dict[str, int] = {}
    for comment in comments:
        counts[comment.post_id] = counts.get(comment.post_id, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Reddit PRAW comment objects (live, credentialed path)
# ---------------------------------------------------------------------------


def praw_comments_to_records(
    praw_comments: Iterable[Any],
    *,
    post_id: str,
    market_id_candidate: str | None = None,
    raw_path: str | None = None,
    collected_at: datetime | None = None,
    max_comments: int | None = None,
) -> list[PostComment]:
    """Normalize PRAW ``Comment`` objects (or dicts) into PostComment rows.

    Accepts the flat output of ``submission.comments.list()``. ``MoreComments``
    placeholders (no ``body``) are skipped. PRAW ``parent_id`` carries a
    ``t1_``/``t3_`` prefix; the ``t1_`` (comment-parent) case is preserved.
    """
    collected = collected_at or datetime.now(tz=timezone.utc)
    rows: list[PostComment] = []
    for raw in praw_comments:
        if max_comments is not None and len(rows) >= max_comments:
            break
        get = raw.get if isinstance(raw, dict) else lambda k, _r=raw: getattr(_r, k, None)
        body = get("body")
        comment_id = get("id")
        if body is None or not comment_id:
            continue  # MoreComments stub or malformed
        author = get("author")
        author = str(author) if author else None
        parent = get("parent_id")
        parent_id = (
            str(parent)[3:] if isinstance(parent, str) and parent.startswith("t1_") else None
        )
        text = str(body).strip()
        rows.append(
            PostComment(
                post_id=post_id,
                platform=Platform.REDDIT,
                comment_id=str(comment_id),
                parent_comment_id=parent_id,
                author_id=author,
                author_url=_author_url(Platform.REDDIT, author),
                text=text,
                published_at=_utc(get("created_utc")),
                like_count=_int(get("score")),
                reply_count=None,
                market_id_candidate=market_id_candidate,
                raw_path=raw_path,
                collected_at=collected,
                text_length=len(text),
            )
        )
    return rows


__all__ = [
    "comment_counts_by_post",
    "enrich_comment_text",
    "parse_reddit_comment_html",
    "parse_reddit_comment_json",
    "praw_comments_to_records",
    "telegram_replies_to_comments",
    "write_post_comments",
    "x_replies_to_comments",
]

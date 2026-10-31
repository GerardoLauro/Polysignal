"""Normalize collected social posts into ``SocialPostRecord`` rows.

This module is the social-side equivalent of
:mod:`polysignal.ingestion.polymarket.normalizer`. The functions are
pure (no I/O, no settings reads) so they can be unit-tested against
fixture inputs.

Two operations live here:

1. :func:`build_record` — convert one ``(SocialPost, query, market_id,
   raw_path)`` tuple into a flat :class:`SocialPostRecord` suitable for
   Parquet writing.

2. :func:`deduplicate_records` — apply the project's deterministic
   dedup rule. The current design keeps **one row per
   (post, query, market_candidate) tuple** and sets
   ``is_duplicate=True`` on rows after the first occurrence of a given
   ``(platform, post_id)`` pair, ordered by ``collected_at``. Branch 05
   can either filter to ``is_duplicate == False`` or treat the
   duplicate rows as additional retrieval evidence.

The dedup key falls back to ``hash(platform + url + text +
created_at)`` when ``post_id`` is missing — this should be rare for
Reddit but is important for future platforms (notably Telegram, where
messages may not carry a stable id).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from polysignal.domain import Platform, SocialPost, SocialPostRecord
from polysignal.nlp.normalize import normalize_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _coerce_text_length(text: str | None) -> int:
    if not text:
        return 0
    return len(text)


def _engagement_score(post: SocialPost) -> float:
    """Combine likes / score / replies / views into a single weighted score.

    The score is intentionally simple — it is *not* a ranking signal,
    just a single number the dashboard can plot. Reddit posts surface
    via ``engagement.score`` (upvotes - downvotes) and
    ``engagement.replies`` (comment count); the formula below extends
    cleanly to platforms that expose likes / reposts / views.
    """
    eng = post.engagement
    score = 0.0
    if eng.score is not None:
        score += float(eng.score)
    if eng.likes is not None:
        score += float(eng.likes) * 0.5
    if eng.reposts is not None:
        score += float(eng.reposts) * 2.0
    if eng.replies is not None:
        score += float(eng.replies)
    if eng.views is not None:
        score += float(eng.views) * 0.001
    if eng.reactions is not None:
        score += float(eng.reactions) * 0.5
    return float(score)


def _reddit_title(post: SocialPost) -> str | None:
    value = post.extra.get("title")
    if isinstance(value, str) and value.strip():
        return value
    return None


def _channel_name(post: SocialPost) -> str | None:
    # Reddit stores ``extra.subreddit``; Telegram uses ``extra.channel``.
    for key in ("subreddit", "channel"):
        value = post.extra.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _upvotes(post: SocialPost) -> int | None:
    value = post.extra.get("score")
    if isinstance(value, int):
        return value
    if post.engagement.score is not None:
        return int(post.engagement.score)
    return None


def _comments_count(post: SocialPost) -> int | None:
    value = post.extra.get("num_comments")
    if isinstance(value, int):
        return value
    if post.engagement.replies is not None:
        return int(post.engagement.replies)
    return None


def _shares_count(post: SocialPost) -> int | None:
    if post.engagement.reposts is not None:
        return int(post.engagement.reposts)
    value = post.extra.get("shares")
    if isinstance(value, int):
        return value
    return None


def _views_count(post: SocialPost) -> int | None:
    if post.engagement.views is not None:
        return int(post.engagement.views)
    return None


def _reactions_count(post: SocialPost) -> int | None:
    if post.engagement.reactions is not None:
        return int(post.engagement.reactions)
    return None


def _author_url(post: SocialPost) -> str | None:
    """Best-effort per-author profile URL, unified across platforms.

    Prefer an explicit ``extra.author_url``. Otherwise synthesize the public
    profile URL from the handle: ``reddit.com/user/<h>``, ``t.me/<channel>``,
    ``x.com/<h>``. Returns ``None`` when no handle is available (common for
    some Telegram channels).
    """
    explicit = post.extra.get("author_url")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    handle = (post.author_handle or "").lstrip("@").strip()
    if not handle:
        # Telegram: fall back to the channel permalink when present.
        channel = _channel_name(post)
        if post.platform == Platform.TELEGRAM and channel:
            return f"https://t.me/{channel}"
        return None
    if post.platform == Platform.REDDIT:
        return f"https://www.reddit.com/user/{handle}"
    if post.platform == Platform.TELEGRAM:
        return f"https://t.me/{handle}"
    if post.platform == Platform.TWITTER:
        return f"https://x.com/{handle}"
    return None


def _author_followers(post: SocialPost) -> int | None:
    for key in ("author_followers", "followers", "subscribers", "subscriber_count"):
        value = post.extra.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _like_count(post: SocialPost) -> int | None:
    """Unified §2 ``like_count`` (Reddit upvotes, X likes, Telegram reactions)."""
    if post.engagement.likes is not None:
        return int(post.engagement.likes)
    if post.platform == Platform.REDDIT:
        return _upvotes(post)
    if post.platform == Platform.TELEGRAM and post.engagement.reactions is not None:
        return int(post.engagement.reactions)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_record(
    *,
    post: SocialPost,
    query: str | None,
    market_id_candidate: str | None,
    raw_path: str | None,
    collected_at: datetime,
    query_type: str | None = None,
    retrieval_run_id: str | None = None,
    retrieval_work_id: str | None = None,
    retrieval_window_start: datetime | None = None,
    retrieval_window_end: datetime | None = None,
    retrieval_lookback_days: int | None = None,
    retrieval_window_label: str | None = None,
    retrieval_mode: str | None = None,
    temporal_valid_candidate: bool | None = None,
    created_at_parse_status: str | None = None,
    market_question_hint: str | None = None,
    query_provenance: str | None = None,
    temporal_relation_to_market: str | None = None,
    leakage_eligible: bool | None = None,
    exclusion_reason: str | None = None,
) -> SocialPostRecord:
    """Build one :class:`SocialPostRecord` from a collected post.

    The returned record carries enough context for branch 05 to
    reconstruct *why* the post was collected (``query``,
    ``market_id_candidate``, ``raw_path``) without re-running ingestion.
    """
    text = post.text or ""
    return SocialPostRecord(
        post_id=post.post_id,
        platform=post.platform,
        source_id=post.post_id,
        url=post.url,
        author_id=post.author_id,
        author_name=post.author_handle,
        created_at=post.created_at,
        text=text,
        title=_reddit_title(post),
        subreddit_or_channel=_channel_name(post),
        query=query,
        query_type=query_type,
        market_id_candidate=market_id_candidate,
        retrieval_run_id=retrieval_run_id,
        retrieval_work_id=retrieval_work_id,
        retrieval_window_start=retrieval_window_start,
        retrieval_window_end=retrieval_window_end,
        retrieval_lookback_days=retrieval_lookback_days,
        retrieval_window_label=retrieval_window_label,
        retrieval_mode=retrieval_mode,
        temporal_valid_candidate=temporal_valid_candidate,
        created_at_parse_status=(
            created_at_parse_status
            or str(post.extra.get("created_at_parse_status") or "parsed")
        ),
        market_id_hint=market_id_candidate,
        market_question_hint=market_question_hint,
        market_window_start_hint=retrieval_window_start,
        market_close_time_hint=retrieval_window_end,
        query_provenance=query_provenance or query,
        temporal_relation_to_market=temporal_relation_to_market,
        leakage_eligible=leakage_eligible,
        exclusion_reason=exclusion_reason,
        engagement_score=_engagement_score(post),
        upvotes=_upvotes(post),
        comments_count=_comments_count(post),
        shares_count=_shares_count(post),
        views_count=_views_count(post),
        reactions_count=_reactions_count(post),
        author_url=_author_url(post),
        author_followers=_author_followers(post),
        like_count=_like_count(post),
        reply_count=_comments_count(post),
        repost_count=_shares_count(post),
        view_count=_views_count(post),
        language=post.language,
        raw_path=raw_path,
        collected_at=collected_at.astimezone(timezone.utc),
        is_duplicate=False,
        text_length=_coerce_text_length(text),
    )


def _post_identity_key(record: SocialPostRecord) -> str:
    """The "is this the same post" key.

    Prefer ``(platform, post_id)``; fall back to a content hash when
    ``post_id`` is missing or obviously synthesized.
    """
    if record.post_id and record.post_id.strip() and record.post_id != "unknown":
        return f"{record.platform.value}:{record.post_id}"
    parts = [
        record.platform.value,
        record.url or "",
        normalize_text(record.text or "")[:512],
        (
            record.created_at.astimezone(timezone.utc).isoformat()
            if record.created_at is not None
            else ""
        ),
    ]
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return f"{record.platform.value}:hash:{digest[:32]}"


def _retrieval_key(record: SocialPostRecord) -> str:
    """The "is this the same *retrieval event*" key.

    Two retrieval events are equivalent when their full
    ``(platform, post_id, query, market_id_candidate)`` tuple matches.
    This is the key we use to decide whether a re-run should *append*
    a new row or *skip* it entirely — without this, repeated runs grow
    the social-posts parquet without bound.
    """
    return (
        f"{_post_identity_key(record)}|q={record.query}|m={record.market_id_candidate}"
        f"|ws={record.retrieval_window_start}|we={record.retrieval_window_end}"
    )


def _apply_duplicate_flags(records: list[SocialPostRecord]) -> list[SocialPostRecord]:
    """Set ``is_duplicate`` deterministically based on post identity.

    The first row with a given ``_post_identity_key`` keeps
    ``is_duplicate=False``; subsequent rows are marked duplicates.
    Order is preserved (so existing rows on disk keep their place and
    new rows are appended at the end). Input records are not mutated —
    a fresh list is returned.
    """
    seen: set[str] = set()
    out: list[SocialPostRecord] = []
    for record in records:
        key = _post_identity_key(record)
        if key in seen:
            out.append(record.model_copy(update={"is_duplicate": True}))
        else:
            seen.add(key)
            out.append(record.model_copy(update={"is_duplicate": False}))
    return out


def deduplicate_records(records: Iterable[SocialPostRecord]) -> list[SocialPostRecord]:
    """Mark ``is_duplicate`` on second+ occurrences within a single batch.

    The input order is preserved. A new list is returned — input
    records are not mutated.
    """
    return _apply_duplicate_flags(list(records))


def merge_with_existing(
    new_records: Iterable[SocialPostRecord],
    existing_records: Iterable[SocialPostRecord],
) -> list[SocialPostRecord]:
    """Merge a new batch with whatever is already on disk.

    The merge is **idempotent**: a new record with the exact same
    retrieval tuple (post + query + market_candidate) as an existing
    record is dropped entirely. Records with a *new* retrieval tuple
    (same post, different query or different market_candidate) are
    appended, preserving traceability.

    The ``is_duplicate`` flag is recomputed deterministically over the
    merged list: it is ``True`` on any row whose post identity (the
    ``(platform, post_id)`` pair) is not seen for the first time.
    """
    existing_list = list(existing_records)
    existing_retrieval_keys = {_retrieval_key(r) for r in existing_list}
    merged: list[SocialPostRecord] = list(existing_list)
    for record in new_records:
        if _retrieval_key(record) in existing_retrieval_keys:
            continue
        merged.append(record)
        existing_retrieval_keys.add(_retrieval_key(record))
    return _apply_duplicate_flags(merged)


def to_record_dicts(records: Iterable[SocialPostRecord]) -> list[dict[str, Any]]:
    """Stable JSON-friendly dict representation used by tests / debug."""
    return [r.model_dump(mode="json") for r in records]


__all__ = [
    "build_record",
    "deduplicate_records",
    "merge_with_existing",
    "to_record_dicts",
]


# Defensive backwards-compat: a couple of older code paths in the dashboard
# expect ``Platform`` to be importable from this module. Keep the surface
# small but predictable.
_ = Platform

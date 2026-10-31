"""Telethon-backed ingestion for explicitly configured Telegram channels."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from polysignal.config import Settings, get_settings
from polysignal.domain import Engagement, Platform, SocialPost
from polysignal.ingestion.social.base import SearchQuery, SocialIngestor

logger = logging.getLogger(__name__)


class TelegramAPIError(RuntimeError):
    """Raised when Telegram refuses or cannot complete a channel fetch."""


def _import_telegram_client() -> Any:
    try:
        from telethon.sync import TelegramClient
    except ImportError as exc:
        raise RuntimeError(
            "Telethon is required for Telegram ingestion. Install it with: "
            'pip install -e ".[telegram]"'
        ) from exc
    return TelegramClient


def _utc(value: datetime | str) -> datetime:
    parsed = (
        datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _username(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip().removeprefix("@") or None


def _reply_count(message: Any) -> int | None:
    replies = getattr(message, "replies", None)
    value = getattr(replies, "replies", None)
    return int(value) if value is not None else None


def _reaction_count(message: Any) -> int | None:
    reactions = getattr(message, "reactions", None)
    results = getattr(reactions, "results", None)
    if not results:
        return None
    return sum(int(getattr(item, "count", 0) or 0) for item in results)


def _media_type(message: Any) -> str | None:
    media = getattr(message, "media", None)
    return type(media).__name__ if media is not None else None


def telegram_message_to_raw_dict(
    message: Any,
    channel: Any,
    *,
    requested_channel: str,
) -> dict[str, Any]:
    """Extract a JSON-safe Telegram message snapshot."""
    channel_username = _username(getattr(channel, "username", None)) or _username(requested_channel)
    channel_id = str(getattr(channel, "id", "") or requested_channel)
    title = str(getattr(channel, "title", "") or channel_username or requested_channel)
    message_id = int(message.id)
    return {
        "channel": {
            "id": channel_id,
            "username": channel_username,
            "title": title,
            "requested": requested_channel,
        },
        "message": {
            "id": message_id,
            "text": str(getattr(message, "message", "") or ""),
            "date": _utc(message.date).isoformat(),
            "views": getattr(message, "views", None),
            "forwards": getattr(message, "forwards", None),
            "replies": _reply_count(message),
            "reactions": _reaction_count(message),
            "media_type": _media_type(message),
            "post_author": getattr(message, "post_author", None),
            "grouped_id": getattr(message, "grouped_id", None),
            "edit_date": (
                _utc(message.edit_date).isoformat() if getattr(message, "edit_date", None) else None
            ),
        },
    }


def raw_message_to_social_post(payload: dict[str, Any]) -> SocialPost | None:
    """Normalize one safe raw Telegram snapshot into the shared model."""
    channel = payload.get("channel") or {}
    message = payload.get("message") or {}
    source_config = payload.get("source_config") or {}
    message_id = message.get("id")
    if message_id is None:
        return None
    text = str(message.get("text") or "").strip()
    useful_metadata = any(
        message.get(key) is not None
        for key in ("media_type", "views", "forwards", "replies", "reactions")
    )
    if not text and not useful_metadata:
        return None

    channel_username = _username(channel.get("username"))
    channel_id = str(channel.get("id") or channel_username or channel.get("requested") or "unknown")
    channel_title = str(channel.get("title") or channel_username or channel_id)
    url = f"https://t.me/{channel_username}/{message_id}" if channel_username else None
    return SocialPost(
        post_id=f"telegram:{channel_id}:{message_id}",
        platform=Platform.TELEGRAM,
        author_id=channel_id,
        author_handle=channel_title,
        created_at=_utc(str(message["date"])),
        text=text,
        language=str(source_config.get("language") or "") or None,
        url=url,
        engagement=Engagement(
            reposts=message.get("forwards"),
            replies=message.get("replies"),
            views=message.get("views"),
            reactions=message.get("reactions"),
        ),
        extra={
            "channel": channel_username or channel_title,
            "channel_id": channel_id,
            "channel_title": channel_title,
            "media_type": message.get("media_type"),
            "post_author": message.get("post_author"),
            "domain_tags": source_config.get("domain_tags") or [],
            "source_name": source_config.get("source_name"),
        },
    )


def telegram_message_to_social_post(
    message: Any,
    channel: Any,
    *,
    requested_channel: str,
) -> SocialPost | None:
    """Convert one Telethon-like message object into the shared model."""
    return raw_message_to_social_post(
        telegram_message_to_raw_dict(message, channel, requested_channel=requested_channel)
    )


class TelegramIngestor(SocialIngestor):
    """Fetch messages from explicit Telegram public-channel/group targets."""

    platform = Platform.TELEGRAM

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        channel: str | None = None,
        telegram_client: Any | None = None,
    ) -> None:
        self.settings = (settings or get_settings()).telegram
        self.channel = channel
        self._client = telegram_client
        self._started = False

    @property
    def session_path(self) -> Path:
        return self.settings.session_dir / self.settings.session_name

    def is_available(self) -> bool:
        return self.settings.configured

    def authenticate(self) -> None:
        """Create or refresh an interactive Telethon user session."""
        self._start_client()

    def close(self) -> None:
        if self._client is not None and hasattr(self._client, "disconnect"):
            self._client.disconnect()
        self._started = False

    def search(self, query: SearchQuery) -> Iterable[SocialPost]:
        posts, _ = self.search_raw(query)
        return posts

    def search_raw(self, query: SearchQuery) -> tuple[list[SocialPost], list[dict[str, Any]]]:
        if not self.channel:
            raise RuntimeError(
                "Telegram channel is required. Use `polysignal social telegram ingest "
                "--config config/telegram_channels.yml` or pass --channel."
            )
        return self.fetch_channel_raw(self.channel, query)

    def fetch_channel_raw(
        self,
        channel: str,
        query: SearchQuery,
    ) -> tuple[list[SocialPost], list[dict[str, Any]]]:
        """Fetch one explicitly named channel/group. Telethon handles rate limits."""
        client = self._start_client()
        try:
            entity = client.get_entity(channel)
            self._validate_channel_entity(entity, channel)
            kwargs: dict[str, Any] = {"limit": int(query.limit or 200)}
            if query.text.strip():
                kwargs["search"] = query.text.strip()
            if query.offset_id:
                # Resumable history walk: only return messages older than the
                # last id we already saved for this channel (newest-first
                # iteration), so repeated runs page backwards through history.
                kwargs["offset_id"] = int(query.offset_id)
            if query.until:
                # Telethon's offset_date is exclusive. Add one microsecond so
                # a message exactly at the pre-close boundary remains eligible.
                kwargs["offset_date"] = _utc(query.until) + timedelta(microseconds=1)
            messages = client.iter_messages(entity, **kwargs)
            posts: list[SocialPost] = []
            raws: list[dict[str, Any]] = []
            for message in messages:
                raw = telegram_message_to_raw_dict(message, entity, requested_channel=channel)
                post = raw_message_to_social_post(raw)
                if post is None:
                    continue
                if query.since and post.created_at < _utc(query.since):
                    break
                if query.until and post.created_at > _utc(query.until):
                    continue
                posts.append(post)
                raws.append(raw)
            return posts, raws
        except RuntimeError:
            raise
        except Exception as exc:
            raise TelegramAPIError(f"Telegram fetch failed for {channel!r}: {exc}") from exc

    def _start_client(self) -> Any:
        if not self.is_available():
            raise RuntimeError(
                "Telegram API credentials are not configured. Set "
                "POLYSIGNAL_TELEGRAM_API_ID and POLYSIGNAL_TELEGRAM_API_HASH in .env."
            )
        if self._client is None:
            try:
                api_id = int(self.settings.api_id)
            except ValueError as exc:
                raise RuntimeError("POLYSIGNAL_TELEGRAM_API_ID must be an integer.") from exc
            self.settings.session_dir.mkdir(parents=True, exist_ok=True)
            telegram_client = _import_telegram_client()
            self._client = telegram_client(str(self.session_path), api_id, self.settings.api_hash)
        if not self._started:
            self._client.start()
            self._started = True
        return self._client

    @staticmethod
    def _validate_channel_entity(entity: Any, requested_channel: str) -> None:
        if getattr(entity, "broadcast", False) or getattr(entity, "megagroup", False):
            return
        raise TelegramAPIError(
            f"Telegram target {requested_channel!r} is not a channel or public-style group. "
            "Private chats are not supported."
        )


__all__ = [
    "TelegramAPIError",
    "TelegramIngestor",
    "raw_message_to_social_post",
    "telegram_message_to_raw_dict",
    "telegram_message_to_social_post",
]

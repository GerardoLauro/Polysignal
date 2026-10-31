"""Validated loader for the Branch 12 social-enrichment source config.

The config file (``configs/social_enrichment_sources.example.yaml``) lists two
public-web sections — ``reddit_scrapling`` and ``telegram_public_web`` — each
carrying global crawl settings plus a list of sources. This module loads and
*validates* that file, skipping disabled sections, disabled sources, and
structurally invalid sources gracefully (a malformed source never aborts the
whole run). Runtime overrides (``--max-pages`` / ``--limit`` /
``--rate-limit`` / ``--only`` / dropping ``needs_validation`` sources) are
applied here so adapters receive already-resolved values.

The loader is pure (YAML in, dataclasses out) and contains no network or
filesystem side effects beyond reading the config file, so it is fully unit
testable against small fixtures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REDDIT_SECTION = "reddit_scrapling"
TELEGRAM_SECTION = "telegram_public_web"
TWITTER_SECTION = "twitter_public_web"

DEFAULT_REDDIT_COLLECTION_METHOD = "scrapling_public_web"
DEFAULT_TELEGRAM_COLLECTION_METHOD = "telegram_public_web"
DEFAULT_TWITTER_COLLECTION_METHOD = "twitter_public_web"


# ---------------------------------------------------------------------------
# Source + section dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RedditSource:
    """One public old.reddit.com listing source."""

    name: str
    url: str
    subreddit: str
    domains: tuple[str, ...] = ()
    priority: str = "medium"
    status: str | None = None

    @property
    def needs_validation(self) -> bool:
        return (self.status or "").strip().lower() == "needs_validation"


@dataclass(frozen=True)
class TelegramSource:
    """One public t.me/s/<channel> preview source."""

    name: str
    channel: str
    url: str
    domains: tuple[str, ...] = ()
    priority: str = "medium"
    status: str | None = None

    @property
    def needs_validation(self) -> bool:
        return (self.status or "").strip().lower() == "needs_validation"


@dataclass(frozen=True)
class TwitterSource:
    """One public x.com/<username> profile source."""

    name: str
    username: str
    url: str
    domains: tuple[str, ...] = ()
    priority: str = "medium"
    status: str | None = None

    @property
    def needs_validation(self) -> bool:
        return (self.status or "").strip().lower() == "needs_validation"


@dataclass(frozen=True)
class RedditSectionConfig:
    enabled: bool = True
    collection_method: str = DEFAULT_REDDIT_COLLECTION_METHOD
    base_fetcher: str = "Fetcher"
    user_agent: str = "PolySignalUniversityResearch/0.1"
    rate_limit_seconds: float = 5.0
    request_timeout_seconds: float = 30.0
    max_pages_per_source: int = 10
    max_items_per_source: int = 300
    store_raw_snapshots: bool = True
    parse_comments: bool = False
    notes: str = ""
    sources: tuple[RedditSource, ...] = ()

    @property
    def platform_slug(self) -> str:
        return "reddit_scrapling"


@dataclass(frozen=True)
class TelegramSectionConfig:
    enabled: bool = True
    collection_method: str = DEFAULT_TELEGRAM_COLLECTION_METHOD
    base_url_pattern: str = "https://t.me/s/{channel}"
    user_agent: str = "PolySignalUniversityResearch/0.1"
    rate_limit_seconds: float = 4.0
    request_timeout_seconds: float = 30.0
    max_pages_per_source: int = 10
    max_items_per_source: int = 500
    store_raw_snapshots: bool = True
    notes: str = ""
    sources: tuple[TelegramSource, ...] = ()

    @property
    def platform_slug(self) -> str:
        return "telegram_public_web"


@dataclass(frozen=True)
class TwitterSectionConfig:
    enabled: bool = True
    collection_method: str = DEFAULT_TWITTER_COLLECTION_METHOD
    base_url_pattern: str = "https://x.com/{username}"
    user_agent: str = "PolySignalUniversityResearch/0.1"
    rate_limit_seconds: float = 5.0
    request_timeout_seconds: float = 30.0
    max_items_per_source: int = 200
    store_raw_snapshots: bool = True
    parse_comments: bool = False
    notes: str = ""
    sources: tuple[TwitterSource, ...] = ()

    @property
    def platform_slug(self) -> str:
        return "twitter_public_web"


@dataclass(frozen=True)
class EnrichmentOverrides:
    """Runtime overrides applied at load time."""

    max_pages: int | None = None
    max_items: int | None = None
    rate_limit_seconds: float | None = None
    include_needs_validation: bool = True
    only_sources: tuple[str, ...] = ()
    dry_run: bool = False


@dataclass(frozen=True)
class EnrichmentConfig:
    """Parsed + validated enrichment config."""

    path: Path
    reddit: RedditSectionConfig | None = None
    telegram: TelegramSectionConfig | None = None
    twitter: TwitterSectionConfig | None = None
    skipped_sources: tuple[dict[str, str], ...] = field(default_factory=tuple)

    @property
    def reddit_sources(self) -> tuple[RedditSource, ...]:
        return self.reddit.sources if self.reddit and self.reddit.enabled else ()

    @property
    def telegram_sources(self) -> tuple[TelegramSource, ...]:
        return self.telegram.sources if self.telegram and self.telegram.enabled else ()

    @property
    def twitter_sources(self) -> tuple[TwitterSource, ...]:
        return self.twitter.sources if self.twitter and self.twitter.enabled else ()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_domains(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _matches_only(name: str, only: tuple[str, ...]) -> bool:
    return not only or name in only


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------


def _parse_reddit_section(
    raw: dict[str, Any],
    overrides: EnrichmentOverrides,
    skipped: list[dict[str, str]],
) -> RedditSectionConfig:
    enabled = _as_bool(raw.get("enabled"), True)
    rate_limit = (
        overrides.rate_limit_seconds
        if overrides.rate_limit_seconds is not None
        else _as_float(raw.get("rate_limit_seconds"), 5.0)
    )
    max_pages = (
        overrides.max_pages
        if overrides.max_pages is not None
        else _as_int(raw.get("max_pages_per_source"), 10)
    )
    max_items = (
        overrides.max_items
        if overrides.max_items is not None
        else _as_int(raw.get("max_items_per_source"), 300)
    )
    sources: list[RedditSource] = []
    for row in raw.get("sources") or []:
        parsed = _parse_reddit_source(row, overrides, skipped)
        if parsed is not None:
            sources.append(parsed)
    return RedditSectionConfig(
        enabled=enabled,
        collection_method=str(raw.get("collection_method") or DEFAULT_REDDIT_COLLECTION_METHOD),
        base_fetcher=str(raw.get("base_fetcher") or "Fetcher"),
        user_agent=str(raw.get("user_agent") or "PolySignalUniversityResearch/0.1"),
        rate_limit_seconds=max(0.0, rate_limit),
        request_timeout_seconds=_as_float(raw.get("request_timeout_seconds"), 30.0),
        max_pages_per_source=max(1, max_pages),
        max_items_per_source=max(1, max_items),
        store_raw_snapshots=_as_bool(raw.get("store_raw_snapshots"), True),
        parse_comments=_as_bool(raw.get("parse_comments"), False),
        notes=str(raw.get("notes") or "").strip(),
        sources=tuple(sources),
    )


def _parse_reddit_source(
    row: Any,
    overrides: EnrichmentOverrides,
    skipped: list[dict[str, str]],
) -> RedditSource | None:
    if not isinstance(row, dict):
        skipped.append({"section": REDDIT_SECTION, "name": str(row), "reason": "not a mapping"})
        return None
    name = str(row.get("name") or "").strip()
    url = str(row.get("url") or "").strip()
    subreddit = str(row.get("subreddit") or "").strip()
    if not name or not (url or subreddit):
        skipped.append(
            {"section": REDDIT_SECTION, "name": name or "<unnamed>", "reason": "missing name/url"}
        )
        return None
    if not _as_bool(row.get("enabled"), True):
        return None
    if not _matches_only(name, overrides.only_sources):
        return None
    status = str(row.get("status") or "").strip() or None
    if (
        not overrides.include_needs_validation
        and (status or "").lower() == "needs_validation"
    ):
        return None
    if not url and subreddit:
        url = f"https://old.reddit.com/r/{subreddit}/new/"
    return RedditSource(
        name=name,
        url=url,
        subreddit=subreddit or _subreddit_from_url(url),
        domains=_as_domains(row.get("domains")),
        priority=str(row.get("priority") or "medium"),
        status=status,
    )


def _parse_telegram_section(
    raw: dict[str, Any],
    overrides: EnrichmentOverrides,
    skipped: list[dict[str, str]],
) -> TelegramSectionConfig:
    enabled = _as_bool(raw.get("enabled"), True)
    rate_limit = (
        overrides.rate_limit_seconds
        if overrides.rate_limit_seconds is not None
        else _as_float(raw.get("rate_limit_seconds"), 4.0)
    )
    max_pages = (
        overrides.max_pages
        if overrides.max_pages is not None
        else _as_int(raw.get("max_pages_per_source"), 10)
    )
    max_items = (
        overrides.max_items
        if overrides.max_items is not None
        else _as_int(raw.get("max_items_per_source"), 500)
    )
    base_url_pattern = str(raw.get("base_url_pattern") or "https://t.me/s/{channel}")
    sources: list[TelegramSource] = []
    for row in raw.get("sources") or []:
        parsed = _parse_telegram_source(row, overrides, base_url_pattern, skipped)
        if parsed is not None:
            sources.append(parsed)
    return TelegramSectionConfig(
        enabled=enabled,
        collection_method=str(raw.get("collection_method") or DEFAULT_TELEGRAM_COLLECTION_METHOD),
        base_url_pattern=base_url_pattern,
        user_agent=str(raw.get("user_agent") or "PolySignalUniversityResearch/0.1"),
        rate_limit_seconds=max(0.0, rate_limit),
        request_timeout_seconds=_as_float(raw.get("request_timeout_seconds"), 30.0),
        max_pages_per_source=max(1, max_pages),
        max_items_per_source=max(1, max_items),
        store_raw_snapshots=_as_bool(raw.get("store_raw_snapshots"), True),
        notes=str(raw.get("notes") or "").strip(),
        sources=tuple(sources),
    )


def _parse_telegram_source(
    row: Any,
    overrides: EnrichmentOverrides,
    base_url_pattern: str,
    skipped: list[dict[str, str]],
) -> TelegramSource | None:
    if not isinstance(row, dict):
        skipped.append({"section": TELEGRAM_SECTION, "name": str(row), "reason": "not a mapping"})
        return None
    name = str(row.get("name") or "").strip()
    channel = str(row.get("channel") or "").strip().removeprefix("@")
    if not name or not channel:
        skipped.append(
            {
                "section": TELEGRAM_SECTION,
                "name": name or "<unnamed>",
                "reason": "missing name/channel",
            }
        )
        return None
    if not _as_bool(row.get("enabled"), True):
        return None
    if not _matches_only(name, overrides.only_sources):
        return None
    status = str(row.get("status") or "").strip() or None
    if (
        not overrides.include_needs_validation
        and (status or "").lower() == "needs_validation"
    ):
        return None
    url = str(row.get("url") or "").strip() or base_url_pattern.format(channel=channel)
    return TelegramSource(
        name=name,
        channel=channel,
        url=url,
        domains=_as_domains(row.get("domains")),
        priority=str(row.get("priority") or "medium"),
        status=status,
    )


def _subreddit_from_url(url: str) -> str:
    marker = "/r/"
    if marker in url:
        rest = url.split(marker, 1)[1]
        return rest.split("/", 1)[0]
    return ""


def _parse_twitter_section(
    raw: dict[str, Any],
    overrides: EnrichmentOverrides,
    skipped: list[dict[str, str]],
) -> TwitterSectionConfig:
    enabled = _as_bool(raw.get("enabled"), True)
    rate_limit = (
        overrides.rate_limit_seconds
        if overrides.rate_limit_seconds is not None
        else _as_float(raw.get("rate_limit_seconds"), 5.0)
    )
    max_items = (
        overrides.max_items
        if overrides.max_items is not None
        else _as_int(raw.get("max_items_per_source"), 200)
    )
    base_url_pattern = str(raw.get("base_url_pattern") or "https://x.com/{username}")
    sources: list[TwitterSource] = []
    for row in raw.get("sources") or []:
        parsed = _parse_twitter_source(row, overrides, base_url_pattern, skipped)
        if parsed is not None:
            sources.append(parsed)
    return TwitterSectionConfig(
        enabled=enabled,
        collection_method=str(raw.get("collection_method") or DEFAULT_TWITTER_COLLECTION_METHOD),
        base_url_pattern=base_url_pattern,
        user_agent=str(raw.get("user_agent") or "PolySignalUniversityResearch/0.1"),
        rate_limit_seconds=max(0.0, rate_limit),
        request_timeout_seconds=_as_float(raw.get("request_timeout_seconds"), 30.0),
        max_items_per_source=max(1, max_items),
        store_raw_snapshots=_as_bool(raw.get("store_raw_snapshots"), True),
        parse_comments=_as_bool(raw.get("parse_comments"), False),
        notes=str(raw.get("notes") or "").strip(),
        sources=tuple(sources),
    )


def _parse_twitter_source(
    row: Any,
    overrides: EnrichmentOverrides,
    base_url_pattern: str,
    skipped: list[dict[str, str]],
) -> TwitterSource | None:
    if not isinstance(row, dict):
        skipped.append({"section": TWITTER_SECTION, "name": str(row), "reason": "not a mapping"})
        return None
    name = str(row.get("name") or "").strip()
    username = str(row.get("username") or "").strip().removeprefix("@")
    if not name or not username:
        skipped.append(
            {
                "section": TWITTER_SECTION,
                "name": name or "<unnamed>",
                "reason": "missing name/username",
            }
        )
        return None
    if not _as_bool(row.get("enabled"), True):
        return None
    if not _matches_only(name, overrides.only_sources):
        return None
    status = str(row.get("status") or "").strip() or None
    if (
        not overrides.include_needs_validation
        and (status or "").lower() == "needs_validation"
    ):
        return None
    url = str(row.get("url") or "").strip() or base_url_pattern.format(username=username)
    return TwitterSource(
        name=name,
        username=username,
        url=url,
        domains=_as_domains(row.get("domains")),
        priority=str(row.get("priority") or "medium"),
        status=status,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_enrichment_config(
    path: Path,
    *,
    overrides: EnrichmentOverrides | None = None,
) -> EnrichmentConfig:
    """Load and validate the enrichment source config.

    Raises ``FileNotFoundError`` if the file is missing and ``ValueError`` if
    the top-level document is not a mapping or carries neither recognised
    section. Structurally invalid *sources* are skipped (recorded in
    ``skipped_sources``) rather than raising.
    """
    if not path.exists():
        raise FileNotFoundError(f"enrichment source config not found at {path}.")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("enrichment config must be a YAML mapping at the top level.")
    if REDDIT_SECTION not in payload and TELEGRAM_SECTION not in payload and TWITTER_SECTION not in payload:
        raise ValueError(
            f"enrichment config must contain at least one of "
            f"{REDDIT_SECTION!r} / {TELEGRAM_SECTION!r} / {TWITTER_SECTION!r}."
        )
    overrides = overrides or EnrichmentOverrides()
    skipped: list[dict[str, str]] = []

    reddit_section: RedditSectionConfig | None = None
    raw_reddit = payload.get(REDDIT_SECTION)
    if isinstance(raw_reddit, dict):
        reddit_section = _parse_reddit_section(raw_reddit, overrides, skipped)

    telegram_section: TelegramSectionConfig | None = None
    raw_telegram = payload.get(TELEGRAM_SECTION)
    if isinstance(raw_telegram, dict):
        telegram_section = _parse_telegram_section(raw_telegram, overrides, skipped)

    twitter_section: TwitterSectionConfig | None = None
    raw_twitter = payload.get(TWITTER_SECTION)
    if isinstance(raw_twitter, dict):
        twitter_section = _parse_twitter_section(raw_twitter, overrides, skipped)

    return EnrichmentConfig(
        path=path,
        reddit=reddit_section,
        telegram=telegram_section,
        twitter=twitter_section,
        skipped_sources=tuple(skipped),
    )


def with_section_limits(
    section: RedditSectionConfig | TelegramSectionConfig | TwitterSectionConfig,
    *,
    max_pages: int | None = None,
    max_items: int | None = None,
    rate_limit_seconds: float | None = None,
) -> RedditSectionConfig | TelegramSectionConfig | TwitterSectionConfig:
    """Return a copy of ``section`` with selected crawl limits overridden."""
    updates: dict[str, Any] = {}
    if max_pages is not None:
        updates["max_pages_per_source"] = max(1, max_pages)
    if max_items is not None:
        updates["max_items_per_source"] = max(1, max_items)
    if rate_limit_seconds is not None:
        updates["rate_limit_seconds"] = max(0.0, rate_limit_seconds)
    return replace(section, **updates)


__all__ = [
    "DEFAULT_REDDIT_COLLECTION_METHOD",
    "DEFAULT_TELEGRAM_COLLECTION_METHOD",
    "DEFAULT_TWITTER_COLLECTION_METHOD",
    "REDDIT_SECTION",
    "TELEGRAM_SECTION",
    "TWITTER_SECTION",
    "EnrichmentConfig",
    "EnrichmentOverrides",
    "RedditSectionConfig",
    "RedditSource",
    "TelegramSectionConfig",
    "TelegramSource",
    "TwitterSectionConfig",
    "TwitterSource",
    "load_enrichment_config",
    "with_section_limits",
]

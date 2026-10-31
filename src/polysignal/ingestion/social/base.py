"""Abstract base class shared by every social-platform ingestor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from polysignal.domain import Platform, SocialPost


@dataclass(frozen=True)
class SearchQuery:
    """A single search request issued to a social platform."""

    text: str
    market_id: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    limit: int | None = None
    # Resumable history walk: only messages strictly older than this id are
    # returned. Used by the staged Telegram collector to page backwards
    # through a channel's history across multiple runs without re-reading
    # what it already saved. ``None`` means "start from the newest message".
    offset_id: int | None = None


class SocialIngestor(ABC):
    """Common interface for Reddit / Twitter / Telegram ingestors.

    The contract is intentionally minimal: a way to test whether the
    backend is available, and a way to fetch posts for a query. The
    rest of the pipeline never instantiates concrete platforms by
    name — it depends on this interface.
    """

    platform: Platform

    @abstractmethod
    def is_available(self) -> bool:
        """Return True iff this ingestor has the credentials it needs."""

    @abstractmethod
    def search(self, query: SearchQuery) -> Iterable[SocialPost]:
        """Return posts matching ``query`` (most recent first)."""


__all__ = ["SearchQuery", "SocialIngestor"]

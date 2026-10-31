"""Build deterministic, market-window targeted Telegram search plans."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

from polysignal.dataset.models import (
    HighSignalDomain,
    HighSignalMarket,
    TelegramSearchPlanEntry,
)
from polysignal.ingestion.social.telegram_normalizer import TelegramChannel
from polysignal.storage.parquet_store import write_parquet

_DOMAIN_SYNONYMS: dict[str, HighSignalDomain] = {
    "politics": HighSignalDomain.POLITICS,
    "government": HighSignalDomain.POLITICS,
    "geopolitics": HighSignalDomain.POLITICS,
    "election": HighSignalDomain.POLITICS,
    "crypto": HighSignalDomain.FINANCE_MACRO_CRYPTO,
    "cryptocurrency": HighSignalDomain.FINANCE_MACRO_CRYPTO,
    "finance": HighSignalDomain.FINANCE_MACRO_CRYPTO,
    "macro": HighSignalDomain.FINANCE_MACRO_CRYPTO,
    "economy": HighSignalDomain.FINANCE_MACRO_CRYPTO,
    "markets": HighSignalDomain.FINANCE_MACRO_CRYPTO,
    "stocks": HighSignalDomain.FINANCE_MACRO_CRYPTO,
    "sports": HighSignalDomain.SPORTS_ENTERTAINMENT,
    "entertainment": HighSignalDomain.SPORTS_ENTERTAINMENT,
    "culture": HighSignalDomain.SPORTS_ENTERTAINMENT,
    "music": HighSignalDomain.SPORTS_ENTERTAINMENT,
    "movies": HighSignalDomain.SPORTS_ENTERTAINMENT,
    "tv": HighSignalDomain.SPORTS_ENTERTAINMENT,
}
_GENERAL_NEWS_TAGS = {"news", "general", "general_news"}
_GENERIC_BINARY_OUTCOMES = {"yes", "no"}


def _channel_domains(channel: TelegramChannel) -> set[HighSignalDomain]:
    tags = channel.domain_tags or ((channel.domain,) if channel.domain else ())
    return {
        domain
        for tag in tags
        if (domain := _DOMAIN_SYNONYMS.get(tag.strip().lower())) is not None
    }


def _is_general_news_channel(channel: TelegramChannel) -> bool:
    tags = channel.domain_tags or ((channel.domain,) if channel.domain else ())
    return bool({tag.strip().lower() for tag in tags} & _GENERAL_NEWS_TAGS)


def _channels_for_market(
    market: HighSignalMarket,
    channels: Sequence[TelegramChannel],
) -> list[TelegramChannel]:
    """Route to matching domain tags, then general-news fallback only."""
    matched = [channel for channel in channels if market.domain in _channel_domains(channel)]
    return matched or [channel for channel in channels if _is_general_news_channel(channel)]


def _expected_relevance(
    channel: TelegramChannel | None,
    market: HighSignalMarket,
) -> str:
    if channel is None:
        return "unassigned"
    if market.domain in _channel_domains(channel):
        return "domain-match"
    if _is_general_news_channel(channel):
        return "best-effort"
    return "cross-domain"


def _queries_for_market(market: HighSignalMarket, *, limit: int) -> list[str]:
    suggested = market.suggested_telegram_queries
    values = [*(suggested[:1]), *market.outcome_aliases, *(suggested[1:])]
    if not values:
        values = [market.question[:96]]
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in _GENERIC_BINARY_OUTCOMES:
            continue
        if cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        out.append(cleaned)
        if len(out) >= limit:
            break
    return out or [market.question[:96]]


def _window_bands(
    market: HighSignalMarket,
    lookback_days: Sequence[int],
) -> list[tuple[datetime, datetime, int, str]]:
    anchor = market.temporal_anchor or market.search_window_end
    floor = market.search_window_start
    bands: list[tuple[datetime, datetime, int, str]] = []
    previous = 0
    for lookback in sorted({int(value) for value in lookback_days if int(value) > 0}):
        start = max(floor, anchor - timedelta(days=lookback))
        end = anchor if previous == 0 else anchor - timedelta(days=previous, microseconds=1)
        if start <= end:
            start_day = 0 if previous == 0 else previous + 1
            bands.append((start, end, lookback, f"{start_day}-{lookback}d"))
        previous = lookback
    return bands


def _work_id(
    *,
    market_id: str,
    channel_id: str | None,
    query: str,
    start: datetime,
    end: datetime,
) -> str:
    payload = "|".join(
        (
            market_id,
            channel_id or "<unassigned>",
            query,
            start.astimezone(timezone.utc).isoformat(),
            end.astimezone(timezone.utc).isoformat(),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _query_type(priority: int) -> str:
    return {
        0: "core",
        1: "entity",
        2: "constraint",
        3: "anchor",
        4: "domain",
        5: "polymarket",
    }.get(priority, "extra")


def _plan_entry(
    *,
    market: HighSignalMarket,
    channel: TelegramChannel | None,
    query: str,
    priority: int,
    start: datetime,
    end: datetime,
    lookback: int,
    window_label: str,
    expected_relevance: str,
    notes: str | None,
) -> TelegramSearchPlanEntry:
    channel_id = channel.username if channel is not None else None
    return TelegramSearchPlanEntry(
        market_id=market.market_id,
        work_id=_work_id(
            market_id=market.market_id,
            channel_id=channel_id,
            query=query,
            start=start,
            end=end,
        ),
        channel_id=channel_id,
        channel_name=(channel.name or channel.username) if channel is not None else None,
        channel_domain=channel.domain if channel is not None else None,
        query=query,
        query_type=_query_type(priority),
        priority=priority,
        temporal_anchor=market.temporal_anchor or market.search_window_end,
        temporal_anchor_kind=market.temporal_anchor_kind or "search_window_end",
        lookback_days=lookback,
        window_label=window_label,
        search_window_start=start,
        search_window_end=end,
        expected_relevance=expected_relevance,
        notes=notes,
    )


def build_telegram_search_plan(
    markets: Sequence[HighSignalMarket],
    *,
    channels: Sequence[TelegramChannel] = (),
    queries_per_market: int = 6,
    max_queries_per_channel: int = 3,
    lookback_days: Sequence[int] = (7, 30, 90),
    extra_notes: str | None = None,
) -> list[TelegramSearchPlanEntry]:
    """Compose bounded rows for every market, channel, query, and time band."""
    entries: list[TelegramSearchPlanEntry] = []
    for market in markets:
        queries = _queries_for_market(market, limit=queries_per_market)
        bands = _window_bands(market, lookback_days)
        market_channels = _channels_for_market(market, channels)
        if not market_channels:
            market_channels = [None]  # type: ignore[list-item]
        for channel in market_channels:
            channel_queries = queries if channel is None else queries[:max_queries_per_channel]
            for priority, query in enumerate(channel_queries):
                for start, end, lookback, window_label in bands:
                    entries.append(
                        _plan_entry(
                            market=market,
                            channel=channel,
                            query=query,
                            priority=priority,
                            start=start,
                            end=end,
                            lookback=lookback,
                            window_label=window_label,
                            expected_relevance=_expected_relevance(channel, market),
                            notes=extra_notes
                            or (
                                "no matching Telegram channel configured - planned only"
                                if channel is None
                                else None
                            ),
                        )
                    )
    return entries


def write_telegram_search_plan(
    entries: Sequence[TelegramSearchPlanEntry],
    parquet_path: Path,
    json_path: Path,
    *,
    generated_at: datetime | None = None,
) -> tuple[Path, Path]:
    """Persist the plan to parquet and JSON."""
    moment = (generated_at or datetime.now(tz=timezone.utc)).astimezone(timezone.utc)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(list(entries), parquet_path)
    payload = {
        "generated_at": moment.isoformat(),
        "entries": [entry.model_dump(mode="json") for entry in entries],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return parquet_path, json_path


__all__ = [
    "build_telegram_search_plan",
    "write_telegram_search_plan",
]

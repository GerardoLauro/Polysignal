"""Normalize raw Polymarket payloads into the PolySignal domain model.

These functions are **pure** — they take a parsed JSON dict (or a list
of price-history entries) and return Pydantic domain models. They do
not perform any I/O and they do not raise on missing optional fields:
the rule is *accept what Polymarket actually returns, reject only when
the payload is meaningless*.

Scope of this milestone
-----------------------
Only binary YES/NO markets are normalized. Multi-outcome detection
lives in :mod:`polysignal.ingestion.polymarket.binary_filter` and runs
before the normalizer is invoked — see ``README.md`` for the rationale.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from dateutil import parser as dateparser

from polysignal.domain import Market, MarketDomain, Outcome, PricePoint, ResolutionResult
from polysignal.domain.enums import thematic_category
from polysignal.ingestion.polymarket.binary_filter import BinaryMarket, classify_market

logger = logging.getLogger(__name__)

# Game-style sports questions that carry no league keyword:
#   "Will <Team> win on 2026-06-14?"  /  "<Team A> vs. <Team B>"
_SPORTS_GAME_RE = re.compile(
    r"win on \d{4}-\d{2}-\d{2}|win on \w+ \d{1,2}|\bvs\.?\b|\bv\.\b", re.IGNORECASE
)


# Polymarket "category" / tag strings → our coarse MarketDomain enum.
# POLITICS deliberately includes geopolitics / war terms: prediction markets about
# Russia/Ukraine, Iran, ceasefires, troop movements etc. are geopolitical, and the
# social channels covering them (war/geopolitics news) must not be domain-rejected.
_DOMAIN_KEYWORDS: list[tuple[tuple[str, ...], MarketDomain]] = [
    (
        (
            "politic",
            "election",
            "president",
            "governor",
            "senate",
            "congress",
            "geopolit",
            "prime minister",
            "parliament",
            "war",
            "ceasefire",
            "military",
            "troop",
            "invasion",
            "nato",
            "sanction",
            "nuclear",
            "russia",
            "ukraine",
            "iran",
            "israel",
            "gaza",
            "putin",
            "zelensky",
            "trump",
            "biden",
        ),
        MarketDomain.POLITICS,
    ),
    (
        (
            "sport",
            "nfl",
            "nba",
            "mlb",
            "nhl",
            "soccer",
            "football",
            "tennis",
            "ufc",
            "boxing",
            "olympic",
            # competitions / leagues (high precision)
            "champions league",
            "premier league",
            "europa league",
            "la liga",
            "serie a",
            "bundesliga",
            "ligue 1",
            "world cup",
            "euro 2",
            "copa",
            "stanley cup",
            "nba finals",
            "super bowl",
            "world series",
            "grand prix",
            "formula 1",
            "wimbledon",
            "playoff",
            "golf",
            "pga",
            "cricket",
            "rugby",
            "fc",
            "fifa",
            "uefa",
        ),
        MarketDomain.SPORTS,
    ),
    (("crypto", "bitcoin", "ethereum", "btc", "eth", "blockchain", "token"), MarketDomain.CRYPTO),
    (("econom", "inflation", "fed", "gdp", "unemployment", "interest rate", "stock"), MarketDomain.ECONOMY),
    (
        (
            "culture",
            "music",
            "celebrity",
            "award",
            "movie",
            "film",
            "box office",
            "grammy",
            "oscar",
            "emmy",
            "billboard",
            "album",
            "netflix",
            "spotify",
            "concert",
            "gta",
            "rotten tomatoes",
            "entertain",
        ),
        MarketDomain.CULTURE,
    ),
    (
        ("science", "tech", "ai", "space", "weather", "climate", "covid", "health"),
        MarketDomain.SCIENCE,
    ),
]

# Short / ambiguous keywords that must match as a *whole word* (both
# boundaries) to avoid substring false positives. The classic bug this
# prevents is "russia" matching inside "Borussia" (a football club) and
# "war" matching "Warriors". Longer keywords use a left word-boundary only,
# so intended prefixes still match ("politic"→"political",
# "russia"→"russian", "iran"→"iranian").
# Keywords (longer than 3 chars) that still need whole-word matching because
# they are prefixes of unrelated words. Short keywords (<=3 chars: war, ai,
# tv, nfl, nba, btc, eth, fed, …) get whole-word matching automatically.
# "iran"/"russia"/"gaza" are intentionally NOT here, so they keep matching
# "iranian"/"russian"/"gazan" via left-boundary.
_WHOLE_WORD_KEYWORDS = frozenset({"nato", "copa", "golf"})


def _compile_keyword(kw: str) -> str:
    escaped = re.escape(kw)
    if len(kw) <= 3 or kw in _WHOLE_WORD_KEYWORDS:
        return rf"\b{escaped}\b"
    return rf"\b{escaped}"


# Precompiled per-domain alternation patterns (built once at import).
_DOMAIN_PATTERNS: list[tuple[re.Pattern[str], MarketDomain]] = [
    (re.compile("|".join(_compile_keyword(kw) for kw in keywords)), domain)
    for keywords, domain in _DOMAIN_KEYWORDS
]


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = dateparser.isoparse(text)
    except (ValueError, TypeError):
        try:
            parsed = dateparser.parse(text)
        except (ValueError, TypeError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_float(value: Any, *, minimum: float | None = None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if minimum is not None and out < minimum:
        return None
    return out


def _classify_domain(payload: dict[str, Any]) -> MarketDomain:
    candidates: list[str] = []
    # The question/slug/description carry the topic even when Polymarket's
    # category/tags are missing or wrong (e.g. a Russia-Ukraine ceasefire market
    # tagged "science"). Including them lets keyword classification recover the
    # real domain instead of defaulting to OTHER / a stray tag match.
    for key in ("category", "subcategory", "question", "slug", "description", "title"):
        value = payload.get(key)
        if isinstance(value, str):
            candidates.append(value)
    tags = _coerce_tag_list(payload.get("tags"))
    candidates.extend(tags)
    events = payload.get("events")
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict):
                for key in ("category", "subcategory", "title", "slug"):
                    value = event.get(key)
                    if isinstance(value, str):
                        candidates.append(value)
                candidates.extend(_coerce_tag_list(event.get("tags")))
    haystack = " ".join(c.lower() for c in candidates if c)
    if not haystack:
        return MarketDomain.OTHER
    for pattern, domain in _DOMAIN_PATTERNS:
        if pattern.search(haystack):
            return domain
    # Game-style sports markets ("Will <Team> win on YYYY-MM-DD?",
    # "<Team A> vs <Team B>") often carry no league keyword. Treat them as
    # sports only after the explicit keyword pass, so a political market like
    # "Will Trump win on …" is still caught as politics above.
    question = str(payload.get("question") or "")
    if _SPORTS_GAME_RE.search(question):
        return MarketDomain.SPORTS
    return MarketDomain.OTHER


def _coerce_tag_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                for key in ("label", "name", "slug", "tag"):
                    text = item.get(key)
                    if isinstance(text, str) and text:
                        out.append(text)
                        break
        return out
    if isinstance(value, str):
        return [value]
    return []


def _resolution_from_payload(
    payload: dict[str, Any],
    binary: BinaryMarket,
) -> ResolutionResult | None:
    """Best-effort mapping of Polymarket resolution state to our enum."""
    if not payload.get("closed"):
        return None
    yes_price = binary.yes_price
    no_price = binary.no_price
    if yes_price is not None and yes_price >= 0.99:
        return ResolutionResult.YES
    if no_price is not None and no_price >= 0.99:
        return ResolutionResult.NO
    status = payload.get("umaResolutionStatus") or payload.get("resolutionStatus")
    if isinstance(status, str):
        s = status.lower()
        if "invalid" in s or "disput" in s:
            return ResolutionResult.INVALID
    return None


def _event_id(payload: dict[str, Any]) -> str | None:
    events = payload.get("events")
    if isinstance(events, list) and events:
        first = events[0]
        if isinstance(first, dict):
            for key in ("id", "slug"):
                value = first.get(key)
                if value:
                    return str(value)
    event_id = payload.get("eventId") or payload.get("event_id")
    return str(event_id) if event_id else None


def _source_url(payload: dict[str, Any]) -> str | None:
    slug = payload.get("slug")
    if isinstance(slug, str) and slug:
        return f"https://polymarket.com/market/{slug}"
    explicit = payload.get("url")
    return str(explicit) if explicit else None


def normalize_market(
    payload: dict[str, Any],
    *,
    binary: BinaryMarket | None = None,
) -> Market:
    """Convert a raw Gamma market payload into a :class:`Market`.

    ``binary`` is the parsed binary-filter result. If omitted, the
    function re-classifies the payload itself; this keeps the
    normalizer usable in isolation (e.g. in tests).
    """
    if binary is None:
        result = classify_market(payload)
        if not result.is_binary or result.market is None:
            raise ValueError(
                f"normalize_market: payload is not a binary YES/NO market: {result.reason}"
            )
        binary = result.market

    condition_id = payload.get("conditionId") or payload.get("condition_id")
    raw_id = payload.get("id")
    market_id = str(condition_id) if condition_id else None
    if not market_id and raw_id is not None:
        market_id = str(raw_id)
    if not market_id:
        raise ValueError("normalize_market: payload has no condition_id / id")

    slug = payload.get("slug") or payload.get("marketSlug") or market_id
    question = payload.get("question") or payload.get("title") or slug
    description = payload.get("description")
    if description is not None and not isinstance(description, str):
        description = str(description)

    created_at = (
        _parse_datetime(payload.get("createdAt"))
        or _parse_datetime(payload.get("created_at"))
        or _parse_datetime(payload.get("startDate"))
    )
    if created_at is None:
        created_at = datetime.now(tz=timezone.utc)
    close_time = (
        _parse_datetime(payload.get("endDate"))
        or _parse_datetime(payload.get("end_date"))
        or _parse_datetime(payload.get("closedTime"))
        or _parse_datetime(payload.get("closed_time"))
    )
    resolved_at = (
        _parse_datetime(payload.get("umaResolutionTimestamp"))
        or _parse_datetime(payload.get("resolvedTime"))
        or _parse_datetime(payload.get("resolved_at"))
    )

    is_closed = bool(payload.get("closed"))
    is_resolved = bool(
        is_closed
        and (resolved_at is not None or _resolution_from_payload(payload, binary) is not None)
    )
    resolution = _resolution_from_payload(payload, binary)

    domain = _classify_domain(payload)
    category = thematic_category(domain)
    tags = _coerce_tag_list(payload.get("tags"))

    outcomes = build_outcomes(payload, binary)

    return Market(
        market_id=market_id,
        slug=str(slug),
        question=str(question),
        description=description,
        domain=domain,
        thematic_category=category.value if category else None,
        created_at=created_at,
        close_time=close_time,
        resolved_at=resolved_at,
        is_resolved=is_resolved,
        is_closed=is_closed,
        resolution=resolution,
        outcomes=outcomes,
        tags=tags,
        source_url=_source_url(payload),
        event_id=_event_id(payload),
        condition_id=str(condition_id) if condition_id else None,
        volume=_safe_float(payload.get("volumeNum") or payload.get("volume"), minimum=0.0),
        liquidity=_safe_float(payload.get("liquidityNum") or payload.get("liquidity"), minimum=0.0),
    )


def build_outcomes(
    payload: dict[str, Any],
    binary: BinaryMarket,
) -> list[Outcome]:
    """Build the two :class:`Outcome` rows for a binary market.

    The YES outcome is always emitted first (index 0), the NO outcome
    second. ``outcome_id`` falls back to a synthesized ``<id>-yes`` /
    ``-no`` when Polymarket does not expose the CLOB token id (rare,
    but defensible — it keeps the model valid and the downstream join
    explicit).
    """
    raw_id = payload.get("conditionId") or payload.get("id") or payload.get("slug") or "unknown"
    fallback = str(raw_id)
    yes_id = binary.yes_token_id or f"{fallback}-yes"
    no_id = binary.no_token_id or f"{fallback}-no"
    if yes_id == no_id:
        # Defensive: should never happen for a binary market, but
        # ``Outcome`` requires unique ids.
        no_id = f"{no_id}-2"
    return [
        Outcome(
            market_id=fallback,
            outcome_id=yes_id,
            label=binary.yes_label,
            index=0,
            current_price=_clamp_probability(binary.yes_price),
        ),
        Outcome(
            market_id=fallback,
            outcome_id=no_id,
            label=binary.no_label,
            index=1,
            current_price=_clamp_probability(binary.no_price),
        ),
    ]


def _clamp_probability(value: float | None) -> float | None:
    if value is None:
        return None
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def normalize_price_history(
    *,
    market_id: str,
    outcome_id: str,
    history: list[dict[str, Any]] | dict[str, Any],
) -> list[PricePoint]:
    """Convert a CLOB ``prices-history`` payload into ``PricePoint`` rows.

    Accepts either the raw envelope (``{"history": [...]}``) or the
    bare list. Each entry is ``{"t": <unix_seconds>, "p": <price>}``.
    Entries with unparsable timestamps or out-of-range prices are
    dropped with a debug log rather than failing the whole batch.
    """
    if isinstance(history, dict):
        entries = history.get("history") or history.get("data") or []
    else:
        entries = history
    if not isinstance(entries, list):
        return []

    points: list[PricePoint] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        ts_raw = entry.get("t")
        price_raw = entry.get("p")
        if ts_raw is None or price_raw is None:
            continue
        try:
            ts = float(ts_raw)
        except (TypeError, ValueError):
            continue
        try:
            price = float(price_raw)
        except (TypeError, ValueError):
            continue
        if not (0.0 <= price <= 1.0):
            logger.debug(
                "polymarket: dropping price point out of [0, 1] for %s/%s: %s",
                market_id,
                outcome_id,
                price,
            )
            continue
        timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
        points.append(
            PricePoint(
                market_id=market_id,
                outcome_id=outcome_id,
                timestamp=timestamp,
                price=price,
                volume=_safe_float(entry.get("v"), minimum=0.0),
            )
        )
    return points


__all__ = [
    "build_outcomes",
    "normalize_market",
    "normalize_price_history",
]

"""Select ~50 high-signal Polymarket markets, balanced across domains.

This module pairs the scoring functions in :mod:`polysignal.dataset.scoring`
with a small balanced-bucket selector. The algorithm is intentionally
simple:

1. Score every market.
2. Bucket markets by :class:`HighSignalDomain`.
3. Pull the top ``min`` markets from each bucket first (so no single
   bucket starves the others).
4. Fill any remaining slots by global priority score.
5. Stop at ``selected_market_target``.

If a domain has fewer high-quality markets than its minimum target, the
imbalance is recorded as a warning and returned — we never invent markets
to hit a target.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from polysignal.dataset.models import (
    DATASET_TARGETS,
    DatasetTargets,
    HighSignalDomain,
    HighSignalMarket,
)
from polysignal.dataset.scoring import (
    market_priority_score,
    social_searchability_score,
    telegram_searchability_score,
)
from polysignal.domain import Market, MarketDomain
from polysignal.ingestion.social.query_gen import (
    decompose_question,
    generate_queries_for_market,
)
from polysignal.storage.parquet_store import write_parquet

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


_DEFAULT_WINDOW_DAYS = 180
_FALLBACK_WINDOW_DAYS = 90
_GENERIC_BINARY_OUTCOMES = {"yes", "no"}
_MICRO_MARKET_PATTERNS = (
    re.compile(r"\b(?:temperature|weather|rain|snowfall|precipitation)\b", re.IGNORECASE),
    re.compile(r"\bexact score\b", re.IGNORECASE),
    re.compile(r"\bscore \d+[-:]\d+\b", re.IGNORECASE),
    re.compile(r"\b\d+[-\u2013]\d+\s+(?:tweets?|posts?)\b", re.IGNORECASE),
    re.compile(r"\b(?:tweets?|posts?)\s+from\b", re.IGNORECASE),
    re.compile(r"\b(?:beat|no official winner)\b", re.IGNORECASE),
    re.compile(r"\blead the epl in points at midway point\b", re.IGNORECASE),
    re.compile(r"\btop fantasy\b", re.IGNORECASE),
    re.compile(r"\bmost assists in the .* finals series\b", re.IGNORECASE),
    re.compile(r"\bdrop out in (?:january|february|march|april|may|june|july|august)\b", re.IGNORECASE),
    re.compile(r"\bbetween \d+(?:\.\d+)?[-\u2013]\d+(?:\.\d+)?\b", re.IGNORECASE),
    re.compile(r"\bbetween \$?\d+(?:\.\d+)?[^ ]* and \$?\d+(?:\.\d+)?", re.IGNORECASE),
    re.compile(r"\bgross\b.*\bopening weekend\b", re.IGNORECASE),
    re.compile(r"\bscore a touchdown\b", re.IGNORECASE),
    re.compile(r"\bwin gold in\b", re.IGNORECASE),
    re.compile(r"\bdraft another player\b", re.IGNORECASE),
    re.compile(r"\bnext gf\b", re.IGNORECASE),
)
_QUESTION_DOMAIN_PATTERNS: tuple[tuple[HighSignalDomain, re.Pattern[str]], ...] = (
    (
        HighSignalDomain.POLITICS,
        re.compile(
            r"\b(?:election|president|prime minister|parliament|senate|congress|"
            r"government|trump|biden|democrat|republican|speaker|nomination|"
            r"dropout|popular vote|primary|approval rating|replacement)\b",
            re.IGNORECASE,
        ),
    ),
    (
        HighSignalDomain.FINANCE_MACRO_CRYPTO,
        re.compile(
            r"\b(?:bitcoin|btc|ethereum|eth|crypto|fed|interest rate|inflation|"
            r"gdp|recession|stock|nasdaq|s&p|ipo|airdrop|token|hyperliquid|"
            r"net worth|net income|aramco|sbf)\b",
            re.IGNORECASE,
        ),
    ),
    (
        HighSignalDomain.SPORTS_ENTERTAINMENT,
        re.compile(
            r"\b(?:nba|nfl|nhl|mlb|motogp|formula 1|champions league|world cup|"
            r"finals?|grand prix|epl|paralympics|march madness|head coach|mvp|"
            r"playoffs?|scorer|constructor|japanese gp|tournament|retire|sign with|"
            r"lakers|draft|gold|freestyle|breaststroke|individual medley|stakes|"
            r"rbc heritage|terra wortmann|valero|cracker barrel|album|movie|film|"
            r"oscar|grammy|tony|gta vi)\b",
            re.IGNORECASE,
        ),
    ),
)


def classify_high_signal_domain(market: Market) -> HighSignalDomain:
    """Map ``MarketDomain`` to the coarser high-signal bucket.

    Crypto and Economy collapse into ``finance_macro_crypto``; Sports
    and Culture collapse into ``sports_entertainment``. Politics passes
    through verbatim. Anything else maps to ``other``.
    """
    question = market.question.strip()
    for domain, pattern in _QUESTION_DOMAIN_PATTERNS:
        if pattern.search(question):
            return domain
    if market.domain == MarketDomain.POLITICS:
        return HighSignalDomain.POLITICS
    if market.domain in {MarketDomain.CRYPTO, MarketDomain.ECONOMY}:
        return HighSignalDomain.FINANCE_MACRO_CRYPTO
    if market.domain in {MarketDomain.SPORTS, MarketDomain.CULTURE}:
        return HighSignalDomain.SPORTS_ENTERTAINMENT
    if market.domain == MarketDomain.SCIENCE:
        return HighSignalDomain.OTHER
    return HighSignalDomain.OTHER


def _canonical_keywords(market: Market) -> list[str]:
    decomposition = decompose_question(market.question)
    seen: set[str] = set()
    out: list[str] = []
    for value in (*decomposition.anchors, *decomposition.constraints):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        out.append(cleaned)
    # Also surface the highest-value entity tokens from the question.
    for token in re.findall(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\b", market.question):
        cleaned = token.strip()
        if not cleaned or cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        out.append(cleaned)
    return out[:8]


def _outcome_aliases(market: Market) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for outcome in market.outcomes:
        label = (outcome.label or "").strip()
        if not label:
            continue
        low = label.lower()
        if low in _GENERIC_BINARY_OUTCOMES:
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(label)
    return out


def _suggested_queries(market: Market) -> list[str]:
    queries = [query.query for query in generate_queries_for_market(market, max_queries=6)]
    values = [*(queries[:1]), *_outcome_aliases(market), *(queries[1:])]
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value.lower() in seen:
            continue
        seen.add(value.lower())
        out.append(value)
    return out


def _temporal_anchor(market: Market) -> tuple[datetime | None, str | None]:
    if market.close_time is not None:
        return market.close_time.astimezone(timezone.utc), "close_time"
    if market.resolved_at is not None:
        return market.resolved_at.astimezone(timezone.utc), "resolved_at"
    return None, None


def _is_resolved(market: Market) -> bool:
    return bool(market.is_resolved or market.resolution is not None)


def _is_obvious_micro_market(market: Market) -> bool:
    return any(pattern.search(market.question) for pattern in _MICRO_MARKET_PATTERNS)


def _deduplicate_markets(markets: Iterable[Market]) -> list[Market]:
    out: list[Market] = []
    seen: set[str] = set()
    for market in markets:
        if market.market_id in seen:
            continue
        seen.add(market.market_id)
        out.append(market)
    return out


def _search_window(
    market: Market,
    *,
    now: datetime,
    window_days: int,
) -> tuple[datetime, datetime]:
    end = market.close_time or market.resolved_at
    if end is None:
        raise ValueError(f"market {market.market_id!r} is missing a temporal anchor")
    end = end.astimezone(timezone.utc)
    start = market.created_at.astimezone(timezone.utc)
    if start >= end:
        # Pathological data — fall back to a small lookback window.
        start = end - timedelta(days=_FALLBACK_WINDOW_DAYS)
    earliest_start = end - timedelta(days=window_days)
    if start < earliest_start:
        start = earliest_start
    return start, end


def _selection_reason(
    market: Market,
    domain: HighSignalDomain,
    *,
    rank: int,
    soc: float,
    tg: float,
) -> str:
    parts = [f"domain={domain.value}", f"rank={rank}"]
    if soc >= 0.7:
        parts.append("clear question")
    if tg >= 0.7:
        parts.append("strong telegram signal")
    if (market.volume or 0) > 0:
        parts.append(f"vol={int(market.volume or 0):,}")
    if market.is_resolved:
        parts.append("resolved")
    return "; ".join(parts)


def _warnings_for(market: Market, *, has_price_history: bool) -> list[str]:
    warnings: list[str] = []
    if market.close_time is None:
        warnings.append("missing close_time")
    if not has_price_history:
        warnings.append("no price history available")
    if (market.volume or 0.0) == 0.0:
        warnings.append("zero volume")
    if len(market.question) < 12:
        warnings.append("very short question")
    return warnings


@dataclass
class HighSignalSelectionResult:
    """Diagnostic envelope returned by :func:`select_high_signal_markets`."""

    markets: list[HighSignalMarket] = field(default_factory=list)
    domain_counts: dict[HighSignalDomain, int] = field(default_factory=dict)
    imbalance_warnings: list[str] = field(default_factory=list)
    total_candidates: int = 0
    skipped_for_domain: int = 0
    duplicates_removed: int = 0
    skipped_unresolved: int = 0
    skipped_temporal_anchor_missing: int = 0
    skipped_micro_markets: int = 0
    skipped_other_markets: int = 0


def rank_markets(
    markets: Iterable[Market],
    *,
    has_price_history: Mapping[str, bool] | None = None,
) -> list[tuple[Market, HighSignalDomain, float, float, float]]:
    """Score every market and return rows sorted by priority desc.

    Each row is ``(market, domain_bucket, social_score, telegram_score,
    priority_score)``. The function is pure — callers pass in a mapping
    of ``market_id -> has_price_history`` if they have one.
    """
    has_history = has_price_history or {}
    scored: list[tuple[Market, HighSignalDomain, float, float, float]] = []
    for market in markets:
        domain = classify_high_signal_domain(market)
        soc = social_searchability_score(market)
        tg = telegram_searchability_score(market)
        priority = market_priority_score(
            market,
            has_price_history=bool(has_history.get(market.market_id, False)),
        )
        scored.append((market, domain, soc, tg, priority))
    scored.sort(key=lambda row: row[4], reverse=True)
    return scored


def select_high_signal_markets(
    markets: Sequence[Market],
    *,
    target: int | None = None,
    targets: DatasetTargets | None = None,
    has_price_history: Mapping[str, bool] | None = None,
    now: datetime | None = None,
    window_days: int = _DEFAULT_WINDOW_DAYS,
    resolved_only: bool = True,
    exclude_micro_markets: bool = True,
    include_other: bool = False,
) -> HighSignalSelectionResult:
    """Select up to ``target`` high-signal markets balanced by domain.

    The algorithm intentionally never invents data: if one domain bucket
    has fewer high-quality markets than its minimum target it is
    reported through ``imbalance_warnings`` and the remaining slots are
    filled from the global ranking instead.
    """
    cfg = targets or DATASET_TARGETS
    selected_target = target if target is not None else cfg.selected_market_target
    if selected_target <= 0:
        return HighSignalSelectionResult()
    moment = (now or datetime.now(tz=timezone.utc)).astimezone(timezone.utc)
    has_history = has_price_history or {}
    unique_markets = _deduplicate_markets(markets)
    candidates: list[Market] = []
    skipped_unresolved = 0
    skipped_anchor = 0
    skipped_micro = 0
    skipped_other = 0
    for market in unique_markets:
        if resolved_only and not _is_resolved(market):
            skipped_unresolved += 1
            continue
        anchor, _anchor_kind = _temporal_anchor(market)
        if anchor is None:
            skipped_anchor += 1
            continue
        if exclude_micro_markets and _is_obvious_micro_market(market):
            skipped_micro += 1
            continue
        if not include_other and classify_high_signal_domain(market) is HighSignalDomain.OTHER:
            skipped_other += 1
            continue
        candidates.append(market)
    scored = rank_markets(candidates, has_price_history=has_history)
    by_domain: dict[HighSignalDomain, list[tuple[Market, float, float, float]]] = defaultdict(list)
    for market, domain, soc, tg, prio in scored:
        by_domain[domain].append((market, soc, tg, prio))

    domain_split = cfg.domain_split
    chosen: list[tuple[Market, HighSignalDomain, float, float, float]] = []
    chosen_ids: set[str] = set()
    imbalance: list[str] = []

    # Phase 1 — pull the minimum from each balanced bucket. When the
    # requested target is smaller than the sum of configured minimums
    # (e.g. unit tests), scale every bucket down proportionally so no
    # single domain captures the whole budget.
    minima_sum = sum(min_count for min_count, _ in domain_split.values()) or 1
    scale = min(1.0, selected_target / minima_sum)
    for domain, (min_count, _max_count) in domain_split.items():
        bucket = by_domain.get(domain, [])
        if len(bucket) < min_count:
            imbalance.append(
                f"{domain.value}: only {len(bucket)} candidates "
                f"(minimum target {min_count})"
            )
        scaled_min = max(0, round(min_count * scale))
        for market, soc, tg, prio in bucket[:scaled_min]:
            if market.market_id in chosen_ids or len(chosen) >= selected_target:
                continue
            chosen.append((market, domain, soc, tg, prio))
            chosen_ids.add(market.market_id)

    # Phase 2 — fill any leftover slots by global priority, capped per
    # bucket at its max.
    domain_count: dict[HighSignalDomain, int] = defaultdict(int)
    for _market, domain, _soc, _tg, _prio in chosen:
        domain_count[domain] += 1

    if len(chosen) < selected_target:
        for market, domain, soc, tg, prio in scored:
            if len(chosen) >= selected_target:
                break
            if market.market_id in chosen_ids:
                continue
            max_count = domain_split.get(
                domain, (0, selected_target)
            )[1]
            # Buckets outside the configured split have no cap — they
            # fall under HighSignalDomain.OTHER and only get picked
            # after the configured domains are exhausted.
            if domain in domain_split and domain_count[domain] >= max_count:
                continue
            chosen.append((market, domain, soc, tg, prio))
            chosen_ids.add(market.market_id)
            domain_count[domain] += 1

    chosen.sort(key=lambda row: row[4], reverse=True)
    high_signal_markets: list[HighSignalMarket] = []
    for rank, (market, domain, soc, tg, prio) in enumerate(chosen):
        start, end = _search_window(market, now=moment, window_days=window_days)
        anchor, anchor_kind = _temporal_anchor(market)
        high_signal_markets.append(
            HighSignalMarket(
                market_id=market.market_id,
                slug=market.slug,
                question=market.question,
                domain=domain,
                polymarket_domain=market.domain.value,
                outcomes=[o.label for o in market.outcomes],
                is_resolved=market.is_resolved,
                is_closed=market.is_closed,
                resolution=market.resolution,
                close_time=market.close_time,
                resolved_at=market.resolved_at,
                temporal_anchor=anchor,
                temporal_anchor_kind=anchor_kind,
                created_at=market.created_at,
                volume=market.volume,
                liquidity=market.liquidity,
                has_price_history=bool(has_history.get(market.market_id, False)),
                social_searchability_score=soc,
                telegram_searchability_score=tg,
                market_priority_score=prio,
                selection_reason=_selection_reason(
                    market, domain, rank=rank, soc=soc, tg=tg
                ),
                search_window_start=start,
                search_window_end=end,
                canonical_keywords=_canonical_keywords(market),
                entity_candidates=_canonical_keywords(market),
                outcome_aliases=_outcome_aliases(market),
                suggested_telegram_queries=_suggested_queries(market),
                warnings=_warnings_for(
                    market,
                    has_price_history=bool(has_history.get(market.market_id, False)),
                ),
            )
        )

    domain_counts = {
        domain: sum(1 for m in high_signal_markets if m.domain == domain)
        for domain in HighSignalDomain
    }
    skipped = max(0, selected_target - len(high_signal_markets))
    return HighSignalSelectionResult(
        markets=high_signal_markets,
        domain_counts=domain_counts,
        imbalance_warnings=imbalance,
        total_candidates=len(scored),
        skipped_for_domain=skipped,
        duplicates_removed=len(markets) - len(unique_markets),
        skipped_unresolved=skipped_unresolved,
        skipped_temporal_anchor_missing=skipped_anchor,
        skipped_micro_markets=skipped_micro,
        skipped_other_markets=skipped_other,
    )


def write_high_signal_markets(
    markets: Sequence[HighSignalMarket],
    path: Path,
) -> Path:
    """Persist selected markets to a parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(list(markets), path)
    return path


__all__ = [
    "HighSignalSelectionResult",
    "classify_high_signal_domain",
    "rank_markets",
    "select_high_signal_markets",
    "write_high_signal_markets",
]

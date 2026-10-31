"""Coverage report for the high-signal Telegram dataset.

The report is computed from already-persisted artefacts:

* the selected ``high_signal_markets`` set;
* the persisted Telegram search plan;
* (optionally) ``social_posts.parquet`` and
  ``social_post_segments.parquet`` for actual collected data.

The function is pure-data — it never re-fetches Telegram. That keeps the
report reproducible offline.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from polysignal.dataset.models import (
    DATASET_TARGETS,
    DatasetTargets,
    HighSignalDomain,
    HighSignalMarket,
    TelegramCoverageEntry,
    TelegramCoverageReport,
    TelegramSearchPlanEntry,
)
from polysignal.domain import Platform, SocialPostRecord, SocialPostSegment


def _index_messages(
    messages: Sequence[SocialPostRecord],
) -> tuple[dict[str, list[SocialPostRecord]], int, int]:
    """Index messages by ``market_id_candidate`` (Telegram only).

    Returns the per-market mapping, the targeted raw count, and the
    targeted normalized count. Legacy channel-wide Telegram rows with
    no market provenance are intentionally excluded.
    """
    by_market: dict[str, list[SocialPostRecord]] = {}
    raw_total = 0
    normalized_total = 0
    for record in messages:
        if record.platform is not Platform.TELEGRAM:
            continue
        if not record.market_id_candidate:
            continue
        raw_total += 1
        normalized_total += 1
        key = record.market_id_candidate
        by_market.setdefault(key, []).append(record)
    return by_market, raw_total, normalized_total


def _index_segments(
    segments: Sequence[SocialPostSegment],
    parent_to_markets: dict[str, set[str]],
) -> tuple[dict[str, int], int]:
    """Count segments per market via the parent_post_id link."""
    counts: dict[str, int] = {}
    total = 0
    for segment in segments:
        if segment.platform is not Platform.TELEGRAM:
            continue
        market_ids = parent_to_markets.get(segment.parent_post_id)
        if not market_ids:
            continue
        total += 1
        for market_id in market_ids:
            counts[market_id] = counts.get(market_id, 0) + 1
    return counts, total


def _coverage_threshold(targets: DatasetTargets) -> int:
    """Per-market minimum considered "viable" coverage.

    Use the ratio between minimum_viable_linked_segments and
    selected_market_target so the threshold scales with whatever
    configuration the operator chose.
    """
    if targets.selected_market_target <= 0:
        return 0
    return max(1, targets.minimum_viable_linked_segments // targets.selected_market_target)


def build_coverage_report(
    *,
    selected: Sequence[HighSignalMarket],
    plan: Sequence[TelegramSearchPlanEntry],
    messages: Sequence[SocialPostRecord] = (),
    segments: Sequence[SocialPostSegment] = (),
    channels_configured: int = 0,
    enabled_channels: Sequence[str] = (),
    targets: DatasetTargets | None = None,
    now: datetime | None = None,
    limitations: Sequence[str] = (),
) -> TelegramCoverageReport:
    """Produce a :class:`TelegramCoverageReport` from already-persisted data."""
    cfg = targets or DATASET_TARGETS
    moment = (now or datetime.now(tz=timezone.utc)).astimezone(timezone.utc)
    plan_by_market: dict[str, int] = {}
    for entry in plan:
        plan_by_market[entry.market_id] = plan_by_market.get(entry.market_id, 0) + 1

    by_market, raw_total, normalized_total = _index_messages(messages)
    parent_to_markets: dict[str, set[str]] = {}
    for market_id, records in by_market.items():
        for record in records:
            parent_to_markets.setdefault(record.post_id, set()).add(market_id)
    seg_counts, seg_total = _index_segments(segments, parent_to_markets)

    threshold = _coverage_threshold(cfg)
    per_market: list[TelegramCoverageEntry] = []
    insufficient: list[str] = []
    for market in selected:
        records = by_market.get(market.market_id, [])
        keywords = sorted(
            {
                kw
                for kw in market.canonical_keywords
                if any(kw.lower() in (rec.text or "").lower() for rec in records)
            }
        )
        entry = TelegramCoverageEntry(
            market_id=market.market_id,
            question=market.question,
            domain=market.domain,
            plan_entries=plan_by_market.get(market.market_id, 0),
            raw_messages=len(records),
            normalized_messages=len(records),
            segments=seg_counts.get(market.market_id, 0),
            matched_keywords=keywords,
            insufficient_coverage=len(records) < threshold,
            notes=(
                None
                if records
                else "no Telegram messages collected for this market"
            ),
        )
        per_market.append(entry)
        if entry.insufficient_coverage:
            insufficient.append(market.market_id)

    top = sorted(
        per_market,
        key=lambda row: (row.raw_messages, row.segments),
        reverse=True,
    )
    top_ids = [row.market_id for row in top[:10]]

    by_domain_counts: dict[str, int] = {}
    for market in selected:
        key = market.domain.value
        by_domain_counts[key] = by_domain_counts.get(key, 0) + 1
    for domain in HighSignalDomain:
        by_domain_counts.setdefault(domain.value, 0)

    return TelegramCoverageReport(
        generated_at=moment,
        selected_markets_total=len(selected),
        selected_markets_by_domain=by_domain_counts,
        telegram_channels_configured=channels_configured,
        telegram_channels_enabled=list(enabled_channels),
        raw_messages_collected=raw_total,
        normalized_messages=normalized_total,
        segments_generated=seg_total,
        plan_entries_total=len(plan),
        per_market=per_market,
        top_markets_by_coverage=top_ids,
        markets_with_insufficient_coverage=insufficient,
        limitations=list(limitations),
        targets={
            "selected_market_target": cfg.selected_market_target,
            "raw_telegram_messages_target_min": cfg.raw_telegram_messages_target_min,
            "raw_telegram_messages_target_max": cfg.raw_telegram_messages_target_max,
            "candidate_segments_target_min": cfg.candidate_segments_target_min,
            "candidate_segments_target_max": cfg.candidate_segments_target_max,
            "linked_segments_target_min": cfg.linked_segments_target_min,
            "linked_segments_target_max": cfg.linked_segments_target_max,
            "minimum_viable_linked_segments": cfg.minimum_viable_linked_segments,
        },
    )


def write_coverage_report(report: TelegramCoverageReport, path: Path) -> Path:
    """Serialise the report to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


__all__ = [
    "build_coverage_report",
    "write_coverage_report",
]

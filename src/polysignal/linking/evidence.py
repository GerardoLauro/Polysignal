"""Accepted / rejected evidence export with per-row reasons and provenance.

The graph loader (:mod:`polysignal.graph.loaders`) applies the project's
acceptance policy when planning ``LINKED_TO`` relationships, but it only
reports *aggregate* skip counts. This module applies the **same policy**
row by row and persists two auditable tables plus a summary:

* ``data/analytical/evidence/accepted_evidence.parquet`` (+ ``.csv``)
* ``data/analytical/evidence/rejected_evidence.parquet`` (+ ``.csv``)
* ``data/analytical/evidence/evidence_summary.json``

Policy (identical to the loader — never looser):

1. the effective label is ``final_relevance_label`` (LLM-merged) falling
   back to the deterministic ``relevance_label``;
2. ``unrelated`` and ``ambiguous`` rows are rejected;
3. ``weakly_relevant`` rows need ``confidence >=``
   :data:`~polysignal.graph.loaders.ACCEPTED_WEAK_LINK_MIN_CONFIDENCE`;
4. the parent post and market must both exist (provenance resolvable);
5. the post must be temporally admissible
   (:func:`polysignal.evaluation.assess_post_admissibility` — pre-close
   for historical markets, timestamp required);
6. posts flagged ``is_duplicate`` are rejected (``duplicate_post``);
7. one row per ``(market_id, parent_post_id)`` — the highest
   ``final_confidence`` survives, the rest are rejected
   (``duplicate_evidence``).

Every output row carries provenance: retrieval query + query type, the
collection run / provenance string of the parent post, channel/author,
the raw snapshot path when recorded, and the scorer explanation.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from polysignal.domain import LinkedPost, Market, SocialPostRecord, SocialPostSegment
from polysignal.domain.enums import RelevanceLabel
from polysignal.evaluation import assess_post_admissibility
from polysignal.storage.parquet_store import read_parquet
from polysignal.storage.paths import DataPaths, get_data_paths


def _weak_link_floor() -> float:
    # Imported lazily: graph.loaders transitively imports the linking
    # package, so a module-level import here would be circular. The graph
    # loader stays the single owner of the acceptance threshold.
    from polysignal.graph.loaders import ACCEPTED_WEAK_LINK_MIN_CONFIDENCE

    return ACCEPTED_WEAK_LINK_MIN_CONFIDENCE

__all__ = ["EvidenceExportResult", "export_evidence"]


@dataclass
class EvidenceExportResult:
    """Summary of one evidence export run."""

    accepted_rows: int
    rejected_rows: int
    accepted_path: Path
    rejected_path: Path
    summary_path: Path
    summary: dict[str, Any]


def _segment_index(segments: list[SocialPostSegment]) -> dict[str, SocialPostSegment]:
    return {segment.segment_id: segment for segment in segments}


def _resolve_parent_post_id(
    link: LinkedPost,
    segments: dict[str, SocialPostSegment],
) -> str:
    segment = segments.get(link.post_id) or (
        segments.get(link.segment_id) if link.segment_id else None
    )
    if segment is not None:
        return segment.parent_post_id
    return link.parent_post_id or link.post_id


def _row(
    link: LinkedPost,
    *,
    parent_post_id: str,
    market: Market | None,
    post: SocialPostRecord | None,
    accepted: bool,
    rejection_stage: str | None,
    rejection_reason: str | None,
) -> dict[str, Any]:
    label = link.final_relevance_label or link.relevance_label
    confidence = link.final_confidence if link.final_confidence is not None else link.confidence
    post_created_at = link.post_created_at or (post.created_at if post is not None else None)
    return {
        "link_id": link.link_id,
        "market_id": link.market_id,
        "market_question": link.market_question or (market.question if market else None),
        "market_domain": (market.domain.value if market and market.domain else None),
        "market_close_time": (market.close_time.isoformat() if market and market.close_time else None),
        "platform": link.platform.value,
        "parent_post_id": parent_post_id,
        "segment_id": link.segment_id,
        "post_created_at": post_created_at.isoformat() if post_created_at else None,
        "accepted": accepted,
        "rejection_stage": rejection_stage,
        "rejection_reason": rejection_reason,
        "relevance_label": link.relevance_label.value,
        "final_relevance_label": label.value,
        "final_relevance_score": (
            link.final_relevance_score
            if link.final_relevance_score is not None
            else link.relevance_score
        ),
        "final_confidence": confidence,
        "constraint_match": (link.final_constraint_match or link.constraint_match).value,
        "final_method": (link.final_method or link.method).value,
        "scorer_rejection_reason": link.rejection_reason,
        "explanation": link.explanation,
        # provenance
        "retrieval_query": link.query,
        "query_type": link.query_type,
        "raw_post_path": link.raw_post_path,
        "post_url": post.url if post is not None else None,
        "post_author": post.author_name if post is not None else None,
        "post_channel": post.subreddit_or_channel if post is not None else None,
        "post_retrieval_run_id": post.retrieval_run_id if post is not None else None,
        "post_query_provenance": post.query_provenance if post is not None else None,
        "post_engagement_score": post.engagement_score if post is not None else None,
        "post_text_preview": link.post_text_preview,
        "llm_applied": link.llm_applied,
        "llm_model": link.llm_model,
    }


def _rejection_verdict(
    *,
    link: LinkedPost,
    label: RelevanceLabel,
    confidence: float,
    post: SocialPostRecord | None,
    market: Market | None,
    weak_link_floor: float,
    is_duplicate_post: bool,
    admissibility: Any,
) -> tuple[str | None, str | None]:
    """Return ``(stage, reason)`` if the link is rejected, else ``(None, None)``.

    Mirrors the graph-loader acceptance policy exactly. Pure function of
    its inputs, so it carries no loop-variable closure.
    """
    if post is None or market is None:
        reason = (
            "parent post missing from social_posts"
            if market is not None
            else "market missing from markets table"
        )
        return "endpoint", reason
    if label is RelevanceLabel.UNRELATED:
        return "relevance", link.rejection_reason or "label unrelated"
    if label is RelevanceLabel.AMBIGUOUS:
        return "relevance", link.rejection_reason or "label ambiguous"
    if label is RelevanceLabel.WEAKLY_RELEVANT and confidence < weak_link_floor:
        return (
            "confidence",
            f"weak link below confidence floor ({confidence:.2f} < {weak_link_floor})",
        )
    if is_duplicate_post:
        return "dedup", "duplicate_post (post identity already counted)"
    if admissibility is not None and not admissibility.admissible:
        return "temporal", admissibility.reason or "temporally inadmissible"
    return None, None


def export_evidence(
    *,
    paths: DataPaths | None = None,
    now: datetime | None = None,
) -> EvidenceExportResult:
    """Apply the acceptance policy row by row and persist both verdict sets."""
    paths = paths or get_data_paths()
    weak_link_floor = _weak_link_floor()
    links = read_parquet(paths.linked_posts_parquet, LinkedPost)
    posts = read_parquet(paths.social_posts_parquet, SocialPostRecord)
    segments = _segment_index(read_parquet(paths.social_post_segments_parquet, SocialPostSegment))
    markets = {m.market_id: m for m in read_parquet(paths.polymarket_markets_parquet, Market)}

    posts_by_key: dict[tuple[str, str], SocialPostRecord] = {}
    duplicate_post_keys: set[tuple[str, str]] = set()
    for post in posts:
        key = (post.platform.value, post.post_id)
        if key not in posts_by_key:
            posts_by_key[key] = post
        if post.is_duplicate:
            duplicate_post_keys.add(key)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    # (market_id, parent_post_id) -> index into accepted (for dedup keep-best)
    best_by_pair: dict[tuple[str, str], int] = {}

    for link in links:
        parent_post_id = _resolve_parent_post_id(link, segments)
        platform = link.platform.value
        post = posts_by_key.get((platform, parent_post_id))
        market = markets.get(link.market_id)
        label = link.final_relevance_label or link.relevance_label
        confidence = (
            link.final_confidence if link.final_confidence is not None else link.confidence
        )

        is_duplicate_post = (platform, parent_post_id) in duplicate_post_keys
        post_created_at = link.post_created_at or (post.created_at if post is not None else None)
        admissibility = (
            assess_post_admissibility(market, post_created_at) if market is not None else None
        )
        stage, reason = _rejection_verdict(
            link=link,
            label=label,
            confidence=confidence,
            post=post,
            market=market,
            weak_link_floor=weak_link_floor,
            is_duplicate_post=is_duplicate_post,
            admissibility=admissibility,
        )
        if stage is not None:
            rejected.append(
                _row(
                    link,
                    parent_post_id=parent_post_id,
                    market=market,
                    post=post,
                    accepted=False,
                    rejection_stage=stage,
                    rejection_reason=reason,
                )
            )
            continue

        pair = (link.market_id, parent_post_id)
        row = _row(
            link,
            parent_post_id=parent_post_id,
            market=market,
            post=post,
            accepted=True,
            rejection_stage=None,
            rejection_reason=None,
        )
        existing_index = best_by_pair.get(pair)
        if existing_index is None:
            best_by_pair[pair] = len(accepted)
            accepted.append(row)
            continue
        existing = accepted[existing_index]
        if (row["final_confidence"] or 0.0) > (existing["final_confidence"] or 0.0):
            # Demote the previous best to rejected, promote this row.
            existing["accepted"] = False
            existing["rejection_stage"] = "dedup"
            existing["rejection_reason"] = "duplicate_evidence (higher-confidence row kept)"
            rejected.append(existing)
            accepted[existing_index] = row
        else:
            row["accepted"] = False
            row["rejection_stage"] = "dedup"
            row["rejection_reason"] = "duplicate_evidence (higher-confidence row kept)"
            rejected.append(row)

    out_dir = paths.analytical / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = out_dir / "accepted_evidence.parquet"
    rejected_path = out_dir / "rejected_evidence.parquet"
    accepted_df = pd.DataFrame(accepted)
    rejected_df = pd.DataFrame(rejected)
    accepted_df.to_parquet(accepted_path, index=False)
    rejected_df.to_parquet(rejected_path, index=False)
    accepted_df.to_csv(out_dir / "accepted_evidence.csv", index=False)
    rejected_df.to_csv(out_dir / "rejected_evidence.csv", index=False)

    reason_counts = Counter(
        f"{row['rejection_stage']}: {row['rejection_reason']}" for row in rejected
    )
    per_market = Counter(row["market_id"] for row in accepted)
    per_platform = Counter(row["platform"] for row in accepted)
    summary = {
        "generated_at": (now or datetime.now(tz=timezone.utc)).isoformat(),
        "linked_rows_scored": len(links),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "weak_link_confidence_floor": weak_link_floor,
        "rejection_reasons": dict(reason_counts.most_common()),
        "accepted_by_market": dict(per_market.most_common()),
        "accepted_by_platform": dict(per_platform.most_common()),
        "markets_with_accepted_evidence": len(per_market),
    }
    summary_path = out_dir / "evidence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return EvidenceExportResult(
        accepted_rows=len(accepted),
        rejected_rows=len(rejected),
        accepted_path=accepted_path,
        rejected_path=rejected_path,
        summary_path=summary_path,
        summary=summary,
    )

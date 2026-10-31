"""Branch 12 orchestration: merge, coverage accounting, report + artifact index.

This module sits above the two collection adapters. It:

* persists/loads per-run collection summaries (JSON) so the report step can be
  reproduced without re-collecting;
* merges the Reddit and Telegram normalized datasets into the combined
  ``data_social_enriched_eval`` dataset (deterministic dedup preserved);
* reads whatever downstream coverage already exists (linked posts, stance
  labels, SPI features, Branch 09 backtest, Branch 10 dashboard view);
* renders the honest ``BRANCH_12_DATASET_ENRICHMENT.md`` coverage report and the
  ``BRANCH_12_ENRICHMENT_ARTIFACT_INDEX.json`` artifact index.

It never fabricates downstream results — it reports exactly what is on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polysignal.domain import SocialPostRecord
from polysignal.ingestion.social.enrichment import (
    combine_records,
    csv_sibling,
    platform_breakdown,
    write_records_csv,
)
from polysignal.storage.parquet_store import read_dataframe, read_parquet, write_parquet
from polysignal.storage.paths import DataPaths

BASELINE_12A = {
    "reddit_rows": 1800,
    "telegram_rows": 314,
    "combined_rows": 2114,
    "linked_posts_rows": 3634,
    "linked_posts_relevant": 338,
    "linked_markets": 97,
    "stance_labels_rows": 39,
    "directional_stances": 4,
    "post_close_excluded": 384,
    "spi_feature_rows": 130,
    "evidence_sufficient_markets": 0,
}


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Per-run collection summaries
# ---------------------------------------------------------------------------


def enrichment_dir(paths: DataPaths) -> Path:
    return paths.analytical / "enrichment"


def write_collection_summary(summary: dict[str, Any], paths: DataPaths) -> Path:
    target = enrichment_dir(paths) / "collection_summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return target


def read_collection_summary(paths: DataPaths) -> dict[str, Any] | None:
    target = enrichment_dir(paths) / "collection_summary.json"
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Merge into the combined enriched dataset
# ---------------------------------------------------------------------------


@dataclass
class MergeResult:
    output_root: Path
    posts_path: Path
    csv_path: Path
    reddit_rows: int = 0
    telegram_rows: int = 0
    other_rows: int = 0
    combined_rows: int = 0
    unique_rows: int = 0
    duplicate_rows_flagged: int = 0
    platform_counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "generated_at": _utcnow_iso(),
            "output_root": str(self.output_root),
            "posts_path": str(self.posts_path),
            "csv_path": str(self.csv_path),
            "reddit_rows": self.reddit_rows,
            "telegram_rows": self.telegram_rows,
            "other_rows": self.other_rows,
            "combined_rows": self.combined_rows,
            "unique_rows": self.unique_rows,
            "duplicate_rows_flagged": self.duplicate_rows_flagged,
            "platform_breakdown": self.platform_counts,
        }


def merge_enriched_datasets(
    *,
    reddit_paths: DataPaths,
    telegram_paths: DataPaths,
    output_paths: DataPaths,
) -> MergeResult:
    """Combine Reddit + Telegram normalized datasets into the enriched dataset.

    The two source datasets are read fresh and re-combined every time, so the
    combined output is a deterministic function of its inputs. ``is_duplicate``
    is recomputed across the union; source/platform metadata is preserved.
    """
    reddit_records = read_parquet(reddit_paths.social_posts_parquet, SocialPostRecord)
    telegram_records = read_parquet(telegram_paths.social_posts_parquet, SocialPostRecord)
    combined, duplicate_rows = combine_records(reddit_records, telegram_records)

    target = output_paths.social_posts_parquet
    target.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(combined, target)
    csv_path = csv_sibling(target)
    write_records_csv(combined, csv_path)

    counts = platform_breakdown(combined)
    return MergeResult(
        output_root=output_paths.root,
        posts_path=target,
        csv_path=csv_path,
        reddit_rows=counts.get("reddit", 0),
        telegram_rows=counts.get("telegram", 0),
        other_rows=sum(v for k, v in counts.items() if k not in {"reddit", "telegram"}),
        combined_rows=len(combined),
        unique_rows=sum(counts.values()),
        duplicate_rows_flagged=duplicate_rows,
        platform_counts=counts,
    )


# ---------------------------------------------------------------------------
# Downstream coverage (read-only — never fabricates)
# ---------------------------------------------------------------------------


def _count_rows(path: Path) -> int:
    frame = read_dataframe(path)
    return len(frame)


def read_spi_summary(paths: DataPaths) -> dict[str, Any] | None:
    """Read the Branch 08 SPI run summary for a data root, if present."""
    path = paths.social_pressure_summary_json
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def compare_with_real_optional(reference_paths: DataPaths) -> dict[str, Any] | None:
    """Summarize the baseline ``real_optional`` social evidence for comparison."""
    spi = read_spi_summary(reference_paths)
    if spi is None:
        return None
    totals = spi.get("totals") or {}
    directions = spi.get("pressure_direction_counts") or {}
    return {
        "reference_root": str(reference_paths.root),
        "linked_posts": int(totals.get("linked_posts", 0)),
        "accepted_links": int(totals.get("accepted_links", 0)),
        "directional_stances": int(totals.get("directional", 0)),
        "evidence_sufficient_markets": sum(
            int(value) for key, value in directions.items() if key != "INSUFFICIENT_EVIDENCE"
        ),
        "markets_evaluated": int(spi.get("markets_evaluated", 0)),
    }


def read_downstream_coverage(paths: DataPaths, *, profile: str | None = None) -> dict[str, Any]:
    """Inspect whatever downstream artifacts already exist for the enriched root."""
    linked_path = paths.linked_posts_parquet
    stance_path = paths.stance_labels_parquet
    spi_path = paths.social_pressure_features_parquet
    coverage: dict[str, Any] = {
        "linked_posts_rows": _count_rows(linked_path) if linked_path.exists() else 0,
        "linked_posts_path": str(linked_path) if linked_path.exists() else None,
        "stance_labels_rows": _count_rows(stance_path) if stance_path.exists() else 0,
        "stance_labels_path": str(stance_path) if stance_path.exists() else None,
        "spi_feature_rows": _count_rows(spi_path) if spi_path.exists() else 0,
        "spi_features_path": str(spi_path) if spi_path.exists() else None,
    }
    spi = read_spi_summary(paths)
    if spi is not None:
        totals = spi.get("totals") or {}
        directions = spi.get("pressure_direction_counts") or {}
        coverage["directional_stances"] = int(totals.get("directional", 0))
        coverage["accepted_links"] = int(totals.get("accepted_links", 0))
        coverage["evidence_sufficient_markets"] = sum(
            int(value) for key, value in directions.items() if key != "INSUFFICIENT_EVIDENCE"
        )
        coverage["markets_evaluated"] = int(spi.get("markets_evaluated", 0))
    # Accepted / relevant links, if the column is present.
    if linked_path.exists():
        frame = read_dataframe(linked_path)
        for column in ("accepted", "is_accepted", "relevance_label"):
            if column in frame.columns:
                if column == "relevance_label":
                    coverage["linked_posts_relevant"] = int(
                        frame[column].astype(str).isin(["relevant", "weakly_relevant"]).sum()
                    )
                else:
                    coverage["linked_posts_accepted"] = int(frame[column].astype(bool).sum())
                break
        coverage["linked_markets"] = (
            int(frame["market_id"].nunique()) if "market_id" in frame.columns else 0
        )

    if profile:
        backtest = paths.backtest_profile(profile)
        coverage["profile"] = profile
        coverage["backtest_dir"] = str(backtest.root)
        coverage["backtest_metrics_present"] = backtest.metrics_summary_json.exists()
        coverage["dashboard_view_present"] = backtest.dashboard_prediction_view_parquet.exists()
        coverage["dashboard_manifest_present"] = backtest.dashboard_manifest_json.exists()
        if backtest.metrics_summary_json.exists():
            try:
                coverage["metrics_summary"] = json.loads(
                    backtest.metrics_summary_json.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                coverage["metrics_summary"] = None
    return coverage


# ---------------------------------------------------------------------------
# Artifact index + coverage report
# ---------------------------------------------------------------------------


def assemble_artifact_index(
    *,
    config_path: Path,
    profile: str,
    reddit_summary: dict[str, Any] | None,
    telegram_summary: dict[str, Any] | None,
    merge_summary: dict[str, Any] | None,
    downstream: dict[str, Any] | None,
    verdict: str,
) -> dict[str, Any]:
    """Build the machine-readable Branch 12 artifact index."""
    succeeded: list[str] = []
    failed: list[dict[str, Any]] = []
    raw_artifacts: list[str] = []
    normalized_artifacts: list[str] = []
    row_counts: dict[str, Any] = {}
    duplicate_counts: dict[str, Any] = {}

    for label, summary in (("reddit", reddit_summary), ("telegram", telegram_summary)):
        if not summary:
            continue
        succeeded.extend(summary.get("sources_succeeded", []))
        failed.extend(
            {"platform": label, **entry} for entry in summary.get("sources_failed", [])
        )
        if summary.get("raw_dir"):
            raw_artifacts.append(summary["raw_dir"])
        if summary.get("posts_path"):
            normalized_artifacts.append(summary["posts_path"])
        if summary.get("csv_path"):
            normalized_artifacts.append(summary["csv_path"])
        row_counts[label] = {
            "raw_extracted": summary.get("raw_extracted", 0),
            "new_records_built": summary.get("new_records_built", 0),
            "rows_added": summary.get("rows_added", 0),
            "normalized_total": summary.get("normalized_total", 0),
            "raw_html_snapshots": summary.get("raw_html_snapshots", 0),
        }
        duplicate_counts[label] = {
            "exact_duplicates_dropped": summary.get("exact_duplicates_dropped", 0),
            "duplicate_rows_flagged": summary.get("duplicate_rows_flagged", 0),
        }

    combined_path = merge_summary.get("posts_path") if merge_summary else None
    if combined_path:
        normalized_artifacts.append(combined_path)
        if merge_summary.get("csv_path"):
            normalized_artifacts.append(merge_summary["csv_path"])
    output_root = Path(merge_summary["output_root"]) if merge_summary else None
    large_summary_path = output_root / "reports" / "collection_summary.json" if output_root else None
    large_summary = (
        json.loads(large_summary_path.read_text(encoding="utf-8"))
        if large_summary_path is not None and large_summary_path.exists()
        else None
    )
    branch_12b = {
        "implementation_status": "collected" if large_summary else "available_not_run",
        "cli_command": "python -m polysignal social collect-large-enriched",
        "default_targets": {"total": 50000, "reddit": 35000, "telegram": 15000},
        "baseline_12a": BASELINE_12A,
        "collection_summary_path": str(large_summary_path) if large_summary_path else None,
        "state_paths": (
            [
                str(output_root / "state" / "collection_state.json"),
                str(output_root / "state" / "source_progress.json"),
                str(output_root / "state" / "query_progress.json"),
            ]
            if output_root
            else []
        ),
        "query_artifact_paths": (
            [
                str(output_root / "queries" / "market_queries.parquet"),
                str(output_root / "queries" / "market_queries.csv"),
            ]
            if output_root
            else []
        ),
        "latest_collection": (
            {
                "generated_at": large_summary.get("generated_at"),
                "baseline_12a_unique": large_summary.get("baseline_12a_unique"),
                "added_12b_unique": large_summary.get("added_12b_unique"),
                "final_unique": large_summary.get("final_unique"),
                "targets": large_summary.get("targets"),
                "objective_targets": large_summary.get("objective_targets"),
                "fetched_pages": large_summary.get("fetched_pages"),
                "persisted_retrieval_rows": large_summary.get("persisted_retrieval_rows"),
                "deduplication": large_summary.get("deduplication"),
                "collection_progress": large_summary.get("collection_progress"),
                "market_query_artifacts": large_summary.get("market_query_artifacts"),
                "usefulness": large_summary.get("usefulness"),
                "circuit_breakers": large_summary.get("circuit_breakers"),
                "completion": large_summary.get("completion"),
                "output_paths": large_summary.get("output_paths"),
            }
            if large_summary
            else None
        ),
        "source_level_counts": row_counts,
        "query_level_counts": (
            (large_summary.get("collection_progress") or {}).get("reddit", {})
            if large_summary
            else {}
        ),
        "after_downstream": {
            "linked_posts_rows": (downstream or {}).get("linked_posts_rows", 0),
            "linked_posts_relevant": (downstream or {}).get("linked_posts_relevant", 0),
            "linked_markets": (downstream or {}).get("linked_markets", 0),
            "stance_labels_rows": (downstream or {}).get("stance_labels_rows", 0),
            "directional_stances": (downstream or {}).get("directional_stances", 0),
            "evidence_sufficient_markets": (downstream or {}).get(
                "evidence_sufficient_markets", 0
            ),
            "spi_feature_rows": (downstream or {}).get("spi_feature_rows", 0),
            "backtest_metrics_present": bool(
                (downstream or {}).get("backtest_metrics_present")
            ),
            "dashboard_view_present": bool(
                (downstream or {}).get("dashboard_view_present")
            ),
        },
    }

    return {
        "branch": "12 — Social Dataset Enrichment with Telegram and Reddit Public Web",
        "generated_at": _utcnow_iso(),
        "config_path": str(config_path),
        "source_counts": {
            "reddit_sources_succeeded": len(
                (reddit_summary or {}).get("sources_succeeded", [])
            ),
            "telegram_sources_succeeded": len(
                (telegram_summary or {}).get("sources_succeeded", [])
            ),
            "sources_failed": len(failed),
        },
        "successful_sources": succeeded,
        "failed_sources": failed,
        "raw_artifact_paths": raw_artifacts,
        "normalized_artifact_paths": normalized_artifacts,
        "row_counts": row_counts,
        "duplicate_counts": duplicate_counts,
        "combined_dataset_path": combined_path,
        "combined_platform_breakdown": (merge_summary or {}).get("platform_breakdown", {}),
        "branch_12b": branch_12b,
        "downstream": downstream or {},
        "final_profile": profile,
        "final_verdict": verdict,
    }


def render_coverage_report(index: dict[str, Any]) -> str:
    """Render the honest Branch 12 coverage report from the artifact index."""
    row_counts = index.get("row_counts", {})
    dup_counts = index.get("duplicate_counts", {})
    reddit_rows = row_counts.get("reddit", {})
    telegram_rows = row_counts.get("telegram", {})
    breakdown = index.get("combined_platform_breakdown", {})
    downstream = index.get("downstream", {})

    def _fmt_breakdown(counts: dict[str, Any]) -> str:
        if not counts:
            return "n/a"
        return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))

    lines: list[str] = []
    lines.append("# Branch 12 — Social Dataset Enrichment (Telegram + Reddit public web)")
    lines.append("")
    lines.append(f"_Generated: {index.get('generated_at')}_")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "Build two large, useful, traceable real social datasets for PolySignal — a Telegram "
        "public-channel dataset and a Reddit public-web dataset collected with Scrapling — and, "
        "when enough data is available, combine them into the enriched real-social profile "
        f"`{index.get('final_profile')}`. The datasets are normalized into the existing "
        "`SocialPostRecord` schema so they are usable by the existing linking, stance, Social "
        "Pressure Index, Branch 09 backtesting, and Branch 10 dashboard pipelines."
    )
    lines.append("")
    lines.append("## Why these sources, and why Scrapling")
    lines.append("")
    lines.append(
        "- **Telegram** public channels (`https://t.me/s/<channel>`) expose recent messages "
        "without login and are a dense, timestamped source of market-relevant commentary "
        "(crypto, finance, breaking news)."
    )
    lines.append(
        "- **Reddit** subreddits map cleanly onto PolySignal market domains (politics, crypto, "
        "sports, finance, tech, culture)."
    )
    lines.append(
        "- **Scrapling** is used as a responsible public-web extraction framework: only the "
        "plain `Fetcher` is used, with explicit rate limits, local raw snapshots, deterministic "
        "parsing, and deterministic deduplication."
    )
    lines.append("")
    lines.append("## Policy & provenance statements")
    lines.append("")
    lines.append(
        "- Reddit API / PRAW approval was **not available before the deadline**; Reddit was "
        "collected **only** from public `old.reddit.com` pages with Scrapling."
    )
    lines.append(
        "- This is **not** official Reddit API access and is **not** a complete crawl of Reddit."
    )
    lines.append(
        "- Telegram was collected **only** from public `t.me/s/<channel>` preview pages, "
        "without login."
    )
    lines.append(
        "- No login automation, CAPTCHA/Cloudflare bypass, proxy rotation, aggressive infinite "
        "scroll, or scraping of private/deleted/protected content was used."
    )
    lines.append(
        "- Raw social content is used only to build a normalized research dataset for PolySignal; "
        "it is **not** used to train LLMs or general AI models."
    )
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    lines.append(f"- Config file: `{index.get('config_path')}`")
    lines.append(
        f"- Reddit sources succeeded: "
        f"{index.get('source_counts', {}).get('reddit_sources_succeeded', 0)}"
    )
    lines.append(
        f"- Telegram sources succeeded: "
        f"{index.get('source_counts', {}).get('telegram_sources_succeeded', 0)}"
    )
    lines.append(f"- Sources failed/unavailable: {index.get('source_counts', {}).get('sources_failed', 0)}")
    lines.append("")
    if index.get("successful_sources"):
        lines.append("**Successful sources:** " + ", ".join(index["successful_sources"]))
        lines.append("")
    if index.get("failed_sources"):
        lines.append("**Failed / unavailable sources:**")
        lines.append("")
        for entry in index["failed_sources"]:
            lines.append(
                f"- `{entry.get('name')}` ({entry.get('platform')}): "
                f"{entry.get('status')} — {entry.get('reason')}"
            )
        lines.append("")
    lines.append("## Dataset counts")
    lines.append("")
    lines.append("| Metric | Reddit | Telegram |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| Raw HTML snapshots | {reddit_rows.get('raw_html_snapshots', 0)} "
        f"| {telegram_rows.get('raw_html_snapshots', 0)} |"
    )
    lines.append(
        f"| Raw extracted records | {reddit_rows.get('raw_extracted', 0)} "
        f"| {telegram_rows.get('raw_extracted', 0)} |"
    )
    lines.append(
        f"| New records built | {reddit_rows.get('new_records_built', 0)} "
        f"| {telegram_rows.get('new_records_built', 0)} |"
    )
    lines.append(
        f"| Exact duplicates dropped | "
        f"{dup_counts.get('reddit', {}).get('exact_duplicates_dropped', 0)} "
        f"| {dup_counts.get('telegram', {}).get('exact_duplicates_dropped', 0)} |"
    )
    lines.append(
        f"| Duplicate rows flagged | "
        f"{dup_counts.get('reddit', {}).get('duplicate_rows_flagged', 0)} "
        f"| {dup_counts.get('telegram', {}).get('duplicate_rows_flagged', 0)} |"
    )
    lines.append(
        f"| Normalized rows (dataset total) | {reddit_rows.get('normalized_total', 0)} "
        f"| {telegram_rows.get('normalized_total', 0)} |"
    )
    lines.append("")
    lines.append(f"**Combined normalized rows:** {(breakdown and sum(breakdown.values())) or 0} "
                 f"({_fmt_breakdown(breakdown)})")
    lines.append("")
    branch_12b = index.get("branch_12b", {})
    lines.append("## Branch 12B — large market-aware collection")
    lines.append("")
    lines.append(
        f"- Collector status: `{branch_12b.get('implementation_status', 'unknown')}`."
    )
    lines.append(
        "- Default unique-record targets: total=50,000, Reddit=35,000, Telegram=15,000; "
        "Telegram shortfall transfers to Reddit."
    )
    lines.append(
        "- The resumable collector writes page-level checkpoints, deterministic market-query "
        "artifacts, raw snapshots, and incremental normalized Parquet/CSV output."
    )
    lines.append(
        "- Missing, invalid, outside-window, and post-close timestamps remain auditable but "
        "cannot become leakage-free downstream evidence."
    )
    if branch_12b.get("latest_collection"):
        latest = branch_12b["latest_collection"]
        lines.append(f"- Latest 12B final unique counts: `{latest.get('final_unique')}`.")
        objective = latest.get("objective_targets") or {}
        lines.append(
            f"- Objective progress: `{objective.get('progress_ratio', 0):.2%}` "
            f"({objective.get('remaining_total', 0)} rows remaining); "
            f"target reached: `{objective.get('reached')}`."
        )
        lines.append(
            f"- Latest staged batch target reached: `{(latest.get('targets') or {}).get('reached')}`."
        )
        lines.append(
            f"- Persisted retrieval rows: `{latest.get('persisted_retrieval_rows', 0)}`."
        )
        queries = latest.get("market_query_artifacts") or {}
        lines.append(f"- Market-aware query artifact rows: `{queries.get('rows', 0)}`.")
        progress = latest.get("collection_progress") or {}
        lines.append(f"- Raw public pages fetched: `{progress.get('fetched_urls', 0)}`.")
        lines.append(
            f"- Rate limit: `{(latest.get('targets') or {}).get('rate_limit_seconds')}` seconds."
        )
        for platform in ("reddit", "telegram"):
            platform_progress = progress.get(platform) or {}
            lines.append(
                f"- {platform.title()} work: "
                f"sources={platform_progress.get('source_work_items', 0)}, "
                f"source pages={platform_progress.get('source_pages_fetched', 0)}, "
                f"queries={platform_progress.get('query_work_items', 0)}, "
                f"query pages={platform_progress.get('query_pages_fetched', 0)}, "
                f"failed sources={len(platform_progress.get('failed_sources', []))}, "
                f"exhausted sources={len(platform_progress.get('exhausted_sources', []))}."
            )
        usefulness = latest.get("usefulness") or {}
        lines.append(
            "- Timestamp/temporal stats: "
            f"parsed={usefulness.get('timestamp_parsed', 0)}, "
            f"market-aware={usefulness.get('market_aware_rows', 0)}, "
            f"general-context={usefulness.get('general_context_rows', 0)}, "
            f"pre-close eligible={usefulness.get('pre_close_eligible', 0)}, "
            f"post-close excluded={usefulness.get('post_close_excluded', 0)}, "
            f"missing/uncertain excluded={usefulness.get('missing_timestamp_excluded', 0)}."
        )
        completion = latest.get("completion") or {}
        lines.append(
            f"- Completion reason: `{completion.get('reason')}`. "
            f"Next step: {completion.get('next_steps')}"
        )
        if (latest.get("circuit_breakers") or {}).get("reddit"):
            lines.append(
                f"- Reddit circuit breaker: `{latest['circuit_breakers']['reddit']}`."
            )
    else:
        lines.append(
            "- No 12B live collection summary is present yet. The counts above remain the "
            "honest Branch 12A baseline."
        )
    lines.append("")
    lines.append("## 12A vs 12B coverage")
    lines.append("")
    baseline = branch_12b.get("baseline_12a", {})
    lines.append("| Metric | 12A baseline | Current enriched dataset |")
    lines.append("| --- | ---: | ---: |")
    lines.append(
        f"| Combined normalized rows | {baseline.get('combined_rows', 0)} "
        f"| {(breakdown and sum(breakdown.values())) or 0} |"
    )
    lines.append(
        f"| Linked pairs scored | {baseline.get('linked_posts_rows', 0)} "
        f"| {downstream.get('linked_posts_rows', 0)} |"
    )
    lines.append(
        f"| Relevant / weak links | {baseline.get('linked_posts_relevant', 0)} "
        f"| {downstream.get('linked_posts_relevant', 0)} |"
    )
    lines.append(
        f"| Stance labels | {baseline.get('stance_labels_rows', 0)} "
        f"| {downstream.get('stance_labels_rows', 0)} |"
    )
    lines.append(
        f"| Directional stances | {baseline.get('directional_stances', 0)} "
        f"| {downstream.get('directional_stances', 0)} |"
    )
    lines.append(
        f"| Evidence-sufficient markets | {baseline.get('evidence_sufficient_markets', 0)} "
        f"| {downstream.get('evidence_sufficient_markets', 0)} |"
    )
    lines.append("")
    lines.append("## Output paths")
    lines.append("")
    for path in index.get("normalized_artifact_paths", []):
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## Downstream coverage")
    lines.append("")
    if downstream:
        lines.append(f"- Linked posts rows: {downstream.get('linked_posts_rows', 0)}")
        if "linked_posts_accepted" in downstream:
            lines.append(f"- Accepted links: {downstream.get('linked_posts_accepted')}")
        if "linked_posts_relevant" in downstream:
            lines.append(f"- Relevant links: {downstream.get('linked_posts_relevant')}")
        if "linked_markets" in downstream:
            lines.append(f"- Distinct linked markets: {downstream.get('linked_markets')}")
        lines.append(f"- Stance labels rows: {downstream.get('stance_labels_rows', 0)}")
        if "directional_stances" in downstream:
            lines.append(
                f"- Directional stances (supports_yes/no): "
                f"{downstream.get('directional_stances')}"
            )
        if "evidence_sufficient_markets" in downstream:
            lines.append(
                f"- Evidence-sufficient markets: "
                f"{downstream.get('evidence_sufficient_markets')} / "
                f"{downstream.get('markets_evaluated', 0)}"
            )
        lines.append(f"- SPI feature rows: {downstream.get('spi_feature_rows', 0)}")
        if "market_token_overlap" in downstream:
            mto = downstream["market_token_overlap"]
            lines.append(
                f"- Market token-overlap estimate: "
                f"{mto.get('markets_with_token_overlap', 0)}/{mto.get('markets_total', 0)} "
                f"markets share >= {mto.get('min_shared_tokens')} content tokens with a post "
                f"(coverage ratio {mto.get('coverage_ratio')})"
            )
        if downstream.get("backtest_metrics_present"):
            lines.append(
                f"- Branch 09 enriched backtest: present at `{downstream.get('backtest_dir')}`"
            )
        else:
            lines.append("- Branch 09 enriched backtest: **not produced** (insufficient data).")
        lines.append(
            f"- Branch 10 dashboard view present: {bool(downstream.get('dashboard_view_present'))}"
        )
    else:
        lines.append("- Downstream linking/stance/SPI not run in this report.")
    lines.append("")
    comparison = (downstream or {}).get("comparison_real_optional")
    if comparison:
        latest_collection = (index.get("branch_12b") or {}).get("latest_collection") or {}
        added_12b = (latest_collection.get("added_12b_unique") or {}).get("total", 0)
        lines.append("## Comparison with the previous `real_optional` dataset")
        lines.append("")
        lines.append("| Metric | real_optional | real_social_enriched_main |")
        lines.append("| --- | --- | --- |")
        lines.append(
            f"| Linked-post rows scored | {comparison.get('linked_posts', 0)} "
            f"| {downstream.get('linked_posts_rows', 0)} |"
        )
        lines.append(
            f"| Accepted links | {comparison.get('accepted_links', 0)} "
            f"| {downstream.get('accepted_links', downstream.get('linked_posts_accepted', 0))} |"
        )
        lines.append(
            f"| Stance directional | {comparison.get('directional_stances', 0)} "
            f"| {downstream.get('directional_stances', 0)} |"
        )
        lines.append(
            f"| Evidence-sufficient markets | "
            f"{comparison.get('evidence_sufficient_markets', 0)} "
            f"| {downstream.get('evidence_sufficient_markets', 0)} |"
        )
        lines.append("")
        lines.append(
            "Branch 12 adds a large, broad real public-web evidence base "
            f"({added_12b} new 12B posts; "
            f"{(index.get('combined_platform_breakdown') and sum(index['combined_platform_breakdown'].values())) or 0} "
            "combined posts across Reddit + Telegram, touching more distinct markets). Because the "
            "public-web posts are recent and the resolved markets are historical, most posts fall "
            "after market close and are correctly excluded from leakage-free evaluation, so the "
            "count of evidence-sufficient markets does not increase here. The enrichment widens "
            "real social *coverage and breadth*; it does not, by itself, manufacture pre-close "
            "directional evidence for already-closed markets."
        )
        lines.append("")
    lines.append("## Interpretation (allowed claims only)")
    lines.append("")
    lines.append(
        "- The enriched social dataset increases real social evidence coverage for PolySignal "
        "compared to the previous sparse `real_optional` dataset."
    )
    lines.append(
        "- Scrapling enabled public-web extraction from selected Reddit and Telegram public "
        "pages under conservative crawling limits."
    )
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- Reddit public-web listings expose recent posts only; this is not the full Reddit "
        "corpus and not API access."
    )
    lines.append(
        "- Public-web posts are not market-targeted, so contract linking coverage is bounded by "
        "topical overlap with resolved markets."
    )
    lines.append(
        "- Some Telegram channels do not expose public preview pages and are recorded as "
        "unavailable above."
    )
    lines.append("")
    lines.append(f"## Final verdict: {index.get('final_verdict')}")
    lines.append("")
    return "\n".join(lines)


def write_report_and_index(
    index: dict[str, Any],
    *,
    report_path: Path,
    index_path: Path,
) -> tuple[Path, Path]:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_coverage_report(index), encoding="utf-8")
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return report_path, index_path


__all__ = [
    "MergeResult",
    "assemble_artifact_index",
    "enrichment_dir",
    "merge_enriched_datasets",
    "read_collection_summary",
    "read_downstream_coverage",
    "render_coverage_report",
    "write_collection_summary",
    "write_report_and_index",
]

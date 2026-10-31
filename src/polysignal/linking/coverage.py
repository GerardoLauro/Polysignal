"""Linking coverage statistics (Traccia_2 explicitly asks for these).

Computes, over all resolved contracts, the fraction with at least ``k`` linked
posts (``k = 1, 5, 10, 50``), broken down **per thematic domain** and **per
platform**, plus an overall roll-up. "Linked" means an *accepted* relevance
verdict (RELEVANT or WEAKLY_RELEVANT), using the effective label
(``final_relevance_label`` when the LLM judge ran, else ``relevance_label``).

Pure transforms over DataFrames; the CLI does the disk I/O.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_THRESHOLDS: tuple[int, ...] = (1, 5, 10, 50)
# RelevanceLabel enum values are lowercase on disk; match case-insensitively.
ACCEPTED_LABELS = {"relevant", "weakly_relevant"}


@dataclass
class CoverageStats:
    """Per-group coverage: contract counts and >=k fractions."""

    group: str
    total_contracts: int
    contracts_with_links: int
    counts_at_least_k: dict[str, int] = field(default_factory=dict)
    fraction_at_least_k: dict[str, float] = field(default_factory=dict)


@dataclass
class LinkingCoverageReport:
    generated_at: str
    thresholds: list[int]
    total_contracts: int
    total_linked_posts: int
    overall: CoverageStats
    by_domain: dict[str, CoverageStats] = field(default_factory=dict)
    by_platform: dict[str, CoverageStats] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "thresholds": self.thresholds,
            "total_contracts": self.total_contracts,
            "total_linked_posts": self.total_linked_posts,
            "overall": asdict(self.overall),
            "by_domain": {k: asdict(v) for k, v in self.by_domain.items()},
            "by_platform": {k: asdict(v) for k, v in self.by_platform.items()},
        }


def _effective_label_series(linked: pd.DataFrame) -> pd.Series:
    base = linked.get("relevance_label")
    if "final_relevance_label" in linked.columns:
        final = linked["final_relevance_label"]
        return final.where(final.notna() & (final.astype(str) != ""), base).astype(str)
    return base.astype(str)


def _stats_for(
    group_name: str,
    per_contract_counts: pd.Series,
    total_contracts: int,
    thresholds: tuple[int, ...],
) -> CoverageStats:
    counts: dict[str, int] = {}
    fractions: dict[str, float] = {}
    for k in thresholds:
        n = int((per_contract_counts >= k).sum())
        counts[str(k)] = n
        fractions[str(k)] = round(n / total_contracts, 6) if total_contracts else 0.0
    return CoverageStats(
        group=group_name,
        total_contracts=total_contracts,
        contracts_with_links=int((per_contract_counts >= 1).sum()),
        counts_at_least_k=counts,
        fraction_at_least_k=fractions,
    )


def build_linking_coverage_report(
    markets: pd.DataFrame,
    linked: pd.DataFrame,
    *,
    thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS,
) -> LinkingCoverageReport:
    """Compute per-domain / per-platform contract linking coverage."""
    market_ids = markets["market_id"].astype(str)
    total_contracts = int(market_ids.nunique())
    domain_by_market = dict(
        zip(market_ids, markets.get("thematic_category").astype("object"), strict=False)
    )

    accepted = pd.DataFrame(columns=["market_id", "platform"])
    if not linked.empty:
        work = linked.copy()
        work["_label"] = _effective_label_series(work).str.strip().str.lower()
        accepted = work[work["_label"].isin(ACCEPTED_LABELS)][["market_id", "platform"]].copy()
        accepted["market_id"] = accepted["market_id"].astype(str)
    total_linked_posts = len(accepted)

    # Overall: count accepted links per contract, zero-filled over all markets.
    per_contract = (
        accepted.groupby("market_id").size().reindex(market_ids.unique(), fill_value=0)
        if not accepted.empty
        else pd.Series(0, index=market_ids.unique())
    )
    overall = _stats_for("overall", per_contract, total_contracts, thresholds)

    # Per domain.
    by_domain: dict[str, CoverageStats] = {}
    domains = sorted({d for d in domain_by_market.values() if d is not None and pd.notna(d)})
    for domain in domains:
        dom_markets = [m for m, d in domain_by_market.items() if d == domain]
        dom_total = len(dom_markets)
        dom_counts = per_contract.reindex(dom_markets, fill_value=0)
        by_domain[domain] = _stats_for(domain, dom_counts, dom_total, thresholds)

    # Per platform: count accepted links per (platform, contract).
    by_platform: dict[str, CoverageStats] = {}
    if not accepted.empty:
        for platform in sorted(accepted["platform"].dropna().astype(str).unique()):
            plat = accepted[accepted["platform"].astype(str) == platform]
            plat_counts = (
                plat.groupby("market_id").size().reindex(market_ids.unique(), fill_value=0)
            )
            by_platform[platform] = _stats_for(
                platform, plat_counts, total_contracts, thresholds
            )

    return LinkingCoverageReport(
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        thresholds=list(thresholds),
        total_contracts=total_contracts,
        total_linked_posts=total_linked_posts,
        overall=overall,
        by_domain=by_domain,
        by_platform=by_platform,
    )


def write_linking_coverage_report(report: LinkingCoverageReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path


__all__ = [
    "ACCEPTED_LABELS",
    "DEFAULT_THRESHOLDS",
    "CoverageStats",
    "LinkingCoverageReport",
    "build_linking_coverage_report",
    "write_linking_coverage_report",
]

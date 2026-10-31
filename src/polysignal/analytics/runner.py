"""Analytics orchestrator (Traccia_2 Task 2).

Reads the canonical normalized datasets, runs the four required analyses, and
persists every artifact under ``data/analytical/analytics/``. The per-analysis
functions live in the sibling modules and stay pure; this module owns the
glue + I/O and returns a JSON-serialisable run summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from polysignal.analytics import correlation, cross_platform, descriptive, discourse, temporal
from polysignal.analytics._common import (
    accepted_links,
    domain_by_market,
    to_utc,
    yes_outcome_by_market,
)
from polysignal.storage.parquet_store import read_dataframe, write_dataframe
from polysignal.storage.paths import DataPaths, get_data_paths

MIN_POSTS_FOR_CORRELATION = 20


@dataclass
class AnalyticsResult:
    output_dir: Path
    written: dict[str, Path] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


def _daily_yes_series_by_market(
    prices: pd.DataFrame, yes_map: dict[str, str]
) -> dict[str, pd.Series]:
    """Build each market's daily YES price series once (avoids re-scans)."""
    if prices.empty:
        return {}
    yes_ids = set(yes_map.values())
    work = prices[prices["outcome_id"].isin(yes_ids)][
        ["market_id", "outcome_id", "timestamp", "price"]
    ].copy()
    work["timestamp"] = to_utc(work["timestamp"])
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work = work.dropna(subset=["timestamp", "price"])
    out: dict[str, pd.Series] = {}
    for market_id, yes_outcome in yes_map.items():
        slab = work[(work["market_id"] == market_id) & (work["outcome_id"] == yes_outcome)]
        if not slab.empty:
            out[market_id] = (
                slab.set_index("timestamp")["price"].sort_index().resample("1D").last().dropna()
            )
    return out


def run_correlation(
    linked: pd.DataFrame,
    enrichment: pd.DataFrame,
    markets: pd.DataFrame,
    outcomes: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    min_posts: int = MIN_POSTS_FOR_CORRELATION,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Per-market lead/lag + alignment + spike study, plus an aggregate roll-up."""
    per_cols = ["market_id", "domain", "n_posts", "best_lag", "best_corr", "relation",
                "price_direction", "sentiment_direction", "aligned", "n_spikes",
                "mean_spike_price_move"]
    acc = accepted_links(linked)
    yes_map = yes_outcome_by_market(outcomes)
    dom = domain_by_market(markets)
    if acc.empty or not yes_map or prices.empty:
        return pd.DataFrame(columns=per_cols), pd.DataFrame(columns=["date", "social_z", "price_move", "market_id"]), {}

    price_series = _daily_yes_series_by_market(prices, yes_map)
    enr = (
        enrichment[["post_id", "sentiment_compound"]].drop_duplicates("post_id")
        if not enrichment.empty
        else pd.DataFrame(columns=["post_id", "sentiment_compound"])
    )
    rows: list[dict] = []
    spike_rows: list[pd.DataFrame] = []
    for market_id, slab in acc.groupby("market_id"):
        price = price_series.get(market_id)
        if price is None or len(slab) < min_posts or len(price) < 4:
            continue
        posts = slab.merge(enr, on="post_id", how="left")
        social = temporal.social_activity_series(
            posts, time_col="post_created_at", sentiment_col="sentiment_compound", engagement_col=None
        )
        if social.empty:
            continue
        ll = correlation.lead_lag(price, social["n_posts"])
        bl = correlation.best_lag(ll)
        align = correlation.sentiment_price_alignment(price, social)
        spikes = correlation.spike_event_study(price, social["n_posts"])
        if not spikes.empty:
            spikes = spikes.assign(market_id=market_id)
            spike_rows.append(spikes)
        rows.append(
            {
                "market_id": market_id,
                "domain": dom.get(market_id),
                "n_posts": len(slab),
                "best_lag": bl["lag"],
                "best_corr": bl["corr"],
                "relation": bl["relation"],
                "price_direction": align["price_direction"],
                "sentiment_direction": align["sentiment_direction"],
                "aligned": align["aligned"],
                "n_spikes": len(spikes),
                "mean_spike_price_move": (
                    round(float(spikes["price_move"].abs().mean()), 4) if not spikes.empty else 0.0
                ),
            }
        )
    per_market = pd.DataFrame(rows, columns=per_cols)
    spike_study = (
        pd.concat(spike_rows, ignore_index=True)
        if spike_rows
        else pd.DataFrame(columns=["date", "social_z", "price_move", "market_id"])
    )
    summary = _correlation_summary(per_market)
    return per_market, spike_study, summary


def _correlation_summary(per_market: pd.DataFrame) -> dict[str, Any]:
    if per_market.empty:
        return {"n_markets": 0}
    aligned = per_market["aligned"].dropna()
    relation_counts = per_market["relation"].value_counts().to_dict()
    return {
        "n_markets": len(per_market),
        "relation_counts": {str(k): int(v) for k, v in relation_counts.items()},
        "alignment_rate": round(float(aligned.mean()), 4) if len(aligned) else None,
        "mean_best_corr": round(float(per_market["best_corr"].dropna().mean()), 4)
        if per_market["best_corr"].notna().any()
        else None,
        "total_spikes": int(per_market["n_spikes"].sum()),
    }


def run_all(*, paths: DataPaths | None = None, output_dir: Path | None = None) -> AnalyticsResult:
    """Run every Task-2 analysis and persist the artifacts."""
    paths = paths or get_data_paths()
    out_dir = output_dir or (paths.analytical / "analytics")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _read(path: Path) -> pd.DataFrame:
        return read_dataframe(path) if path.exists() else pd.DataFrame()

    markets = _read(paths.polymarket_markets_parquet)
    outcomes = _read(paths.polymarket_outcomes_parquet)
    prices = _read(paths.polymarket_price_points_parquet)
    enrichment = _read(paths.normalized_social / "post_enrichment.parquet")
    linked = _read(paths.linked_posts_parquet)
    topics = _read(paths.normalized_social / "topics.parquet")

    tables: dict[str, pd.DataFrame] = {}
    # 2.1 Descriptive
    tables["domain_distribution"] = descriptive.domain_distribution(markets)
    tables["contract_summary"] = descriptive.contract_summary(markets)
    volatility = descriptive.price_volatility(markets, outcomes, prices)
    tables["price_volatility"] = volatility
    tables["volatility_by_domain"] = descriptive.volatility_by_domain(volatility)
    tables["price_informativeness"] = descriptive.price_informativeness(markets, outcomes, prices)
    # 2.2 Discourse
    tables["discourse_volume"] = discourse.volume_by_platform_domain(linked, markets)
    tables["activity_timeline"] = discourse.activity_timeline(linked, markets)
    tables["sentiment_distribution"] = discourse.sentiment_distribution(linked, enrichment, markets)
    tables["entity_frequency"] = discourse.entity_frequency(linked, enrichment, markets)
    tables["topic_summary"] = discourse.topic_summary(topics)
    # 2.3 Correlation
    corr_per, spike_study, corr_summary = run_correlation(
        linked, enrichment, markets, outcomes, prices
    )
    tables["correlation_per_market"] = corr_per
    tables["spike_event_study"] = spike_study
    # 2.4 Cross-platform
    comparison = cross_platform.cross_platform_comparison(linked, markets)
    tables["cross_platform_comparison"] = comparison
    tables["cross_platform_summary"] = cross_platform.cross_platform_summary(comparison)

    result = AnalyticsResult(output_dir=out_dir)
    for name, df in tables.items():
        path = out_dir / f"{name}.parquet"
        write_dataframe(df, path)
        result.written[name] = path
        result.summary.setdefault("row_counts", {})[name] = len(df)

    result.summary["generated_at"] = datetime.now(tz=timezone.utc).isoformat()
    result.summary["correlation"] = corr_summary
    result.summary["cross_platform"] = cross_platform.overall_platform_state(comparison)
    if not tables["price_informativeness"].empty:
        result.summary["price_informativeness"] = tables["price_informativeness"].to_dict("records")

    import json

    summary_path = out_dir / "analytics_summary.json"
    summary_path.write_text(json.dumps(result.summary, indent=2, default=str), encoding="utf-8")
    result.written["analytics_summary"] = summary_path
    return result


__all__ = ["AnalyticsResult", "run_all", "run_correlation"]

"""Descriptive analysis of Polymarket contracts (Traccia_2 Task 2.1).

Characterizes the 300 resolved contracts along domain, duration, trading
volume, and price volatility, and measures how *informative* the market price
is at increasing temporal distance from resolution (T-30/14/7/3/1 days).

All functions are pure transforms over DataFrames; persistence lives in
:mod:`polysignal.analytics.runner`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from polysignal.analytics._common import (
    DOMAIN_COLUMN,
    final_yes_outcome,
    rank_auc,
    resolution_anchor,
    to_utc,
    yes_outcome_by_market,
)

DEFAULT_HORIZONS_DAYS: tuple[int, ...] = (30, 14, 7, 3, 1)


def domain_distribution(markets: pd.DataFrame) -> pd.DataFrame:
    """Contract counts per thematic domain, split by resolution outcome."""
    if markets.empty:
        return pd.DataFrame(columns=["domain", "n_contracts", "resolved_yes", "resolved_no"])
    col = DOMAIN_COLUMN if DOMAIN_COLUMN in markets else "domain"
    res = markets["resolution"].astype("string").str.lower()
    frame = pd.DataFrame({"domain": markets[col].astype("string"), "resolution": res})
    grouped = frame.groupby("domain")
    out = pd.DataFrame(
        {
            "n_contracts": grouped.size(),
            "resolved_yes": grouped["resolution"].apply(lambda s: int((s == "yes").sum())),
            "resolved_no": grouped["resolution"].apply(lambda s: int((s == "no").sum())),
        }
    ).reset_index()
    return out.sort_values("domain").reset_index(drop=True)


def contract_summary(markets: pd.DataFrame) -> pd.DataFrame:
    """Per-domain duration (days) and trading-volume summary statistics."""
    if markets.empty:
        return pd.DataFrame(
            columns=["domain", "n_contracts", "duration_days_mean", "duration_days_median",
                     "volume_mean", "volume_median", "volume_total"]
        )
    col = DOMAIN_COLUMN if DOMAIN_COLUMN in markets else "domain"
    created = to_utc(markets["created_at"])
    anchor = resolution_anchor(markets)
    duration_days = (anchor - created).dt.total_seconds() / 86400.0
    frame = pd.DataFrame(
        {
            "domain": markets[col].astype("string"),
            "duration_days": duration_days,
            "volume": pd.to_numeric(markets.get("volume"), errors="coerce"),
        }
    )
    grouped = frame.groupby("domain")
    out = pd.DataFrame(
        {
            "n_contracts": grouped.size(),
            "duration_days_mean": grouped["duration_days"].mean().round(2),
            "duration_days_median": grouped["duration_days"].median().round(2),
            "volume_mean": grouped["volume"].mean().round(2),
            "volume_median": grouped["volume"].median().round(2),
            "volume_total": grouped["volume"].sum().round(2),
        }
    ).reset_index()
    return out.sort_values("domain").reset_index(drop=True)


def price_volatility(
    markets: pd.DataFrame, outcomes: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    """Per-market YES-price volatility (std of daily price) + lifetime drift.

    Returns one row per market: ``market_id, domain, volatility, price_range,
    n_days``. Volatility is the standard deviation of the daily-resampled YES
    price over the contract's life.
    """
    empty_cols = ["market_id", "domain", "volatility", "price_range", "n_days"]
    if prices.empty or outcomes.empty:
        return pd.DataFrame(columns=empty_cols)
    yes_map = yes_outcome_by_market(outcomes)
    domain_col = DOMAIN_COLUMN if DOMAIN_COLUMN in markets else "domain"
    dom = dict(zip(markets["market_id"], markets[domain_col].astype("string"), strict=False))

    work = prices[["market_id", "outcome_id", "timestamp", "price"]].copy()
    work["timestamp"] = to_utc(work["timestamp"])
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    rows: list[dict] = []
    for market_id, yes_outcome in yes_map.items():
        slab = work[(work["market_id"] == market_id) & (work["outcome_id"] == yes_outcome)]
        slab = slab.dropna(subset=["timestamp", "price"])
        if slab.empty:
            continue
        daily = (
            slab.set_index("timestamp")["price"].sort_index().resample("1D").last().dropna()
        )
        if daily.empty:
            continue
        rows.append(
            {
                "market_id": market_id,
                "domain": dom.get(market_id),
                "volatility": float(daily.std(ddof=0)) if len(daily) > 1 else 0.0,
                "price_range": float(daily.max() - daily.min()),
                "n_days": len(daily),
            }
        )
    return pd.DataFrame(rows, columns=empty_cols)


def volatility_by_domain(volatility: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-market volatility to a per-domain mean/median."""
    if volatility.empty:
        return pd.DataFrame(columns=["domain", "n_contracts", "volatility_mean", "volatility_median"])
    grouped = volatility.groupby("domain")
    out = pd.DataFrame(
        {
            "n_contracts": grouped.size(),
            "volatility_mean": grouped["volatility"].mean().round(4),
            "volatility_median": grouped["volatility"].median().round(4),
        }
    ).reset_index()
    return out.sort_values("domain").reset_index(drop=True)


def _price_at(daily: pd.Series, target) -> float | None:
    """Last YES price at or before ``target`` (as-of lookup)."""
    eligible = daily[daily.index <= target]
    if eligible.empty:
        return None
    return float(eligible.iloc[-1])


def price_informativeness(
    markets: pd.DataFrame,
    outcomes: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS_DAYS,
) -> pd.DataFrame:
    """Brier / accuracy / AUC of the YES price at each horizon-before-resolution.

    For every binary market, the YES price at T-``h`` days before its resolution
    anchor is treated as a probabilistic forecast of the actual outcome (1=YES,
    0=NO). The curve quantifies how price informativeness improves as resolution
    approaches.
    """
    cols = ["horizon_days", "n_contracts", "brier", "accuracy", "auc", "mean_price"]
    yes_map = yes_outcome_by_market(outcomes)
    outcome_map = final_yes_outcome(markets)
    if not yes_map or not outcome_map or prices.empty:
        return pd.DataFrame(columns=cols)
    anchor_by_market = dict(zip(markets["market_id"], resolution_anchor(markets), strict=False))

    work = prices[["market_id", "outcome_id", "timestamp", "price"]].copy()
    work["timestamp"] = to_utc(work["timestamp"])
    work["price"] = pd.to_numeric(work["price"], errors="coerce")

    # Pre-build each market's daily YES series once.
    daily_by_market: dict[str, pd.Series] = {}
    for market_id, yes_outcome in yes_map.items():
        slab = work[(work["market_id"] == market_id) & (work["outcome_id"] == yes_outcome)]
        slab = slab.dropna(subset=["timestamp", "price"])
        if not slab.empty:
            daily_by_market[market_id] = (
                slab.set_index("timestamp")["price"].sort_index().resample("1D").last().dropna()
            )

    rows: list[dict] = []
    for horizon in horizons:
        scores: list[float] = []
        labels: list[float] = []
        for market_id, daily in daily_by_market.items():
            anchor = anchor_by_market.get(market_id)
            outcome = outcome_map.get(market_id)
            if anchor is None or pd.isna(anchor) or outcome is None:
                continue
            price = _price_at(daily, anchor - pd.Timedelta(days=horizon))
            if price is None:
                continue
            scores.append(price)
            labels.append(outcome)
        if not scores:
            rows.append({"horizon_days": horizon, "n_contracts": 0, "brier": np.nan,
                         "accuracy": np.nan, "auc": np.nan, "mean_price": np.nan})
            continue
        arr_s = np.asarray(scores)
        arr_y = np.asarray(labels)
        brier = float(np.mean((arr_s - arr_y) ** 2))
        accuracy = float(np.mean((arr_s >= 0.5).astype(float) == arr_y))
        auc = rank_auc(scores, labels)
        rows.append(
            {
                "horizon_days": horizon,
                "n_contracts": len(scores),
                "brier": round(brier, 4),
                "accuracy": round(accuracy, 4),
                "auc": round(auc, 4) if auc is not None else np.nan,
                "mean_price": round(float(arr_s.mean()), 4),
            }
        )
    return pd.DataFrame(rows, columns=cols).sort_values("horizon_days", ascending=False).reset_index(drop=True)

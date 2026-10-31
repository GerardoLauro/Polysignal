"""Temporal series builders for prices and social activity (Task 2.3 support).

These produce the aligned daily series the correlation analysis consumes:
a per-market YES price series and a per-market social-activity series (post
volume, mean sentiment, engagement). Pure transforms over DataFrames.
"""

from __future__ import annotations

import pandas as pd

from polysignal.analytics._common import to_utc


def price_series(
    prices: pd.DataFrame, market_id: str, outcome_id: str, *, freq: str = "1D"
) -> pd.Series:
    """Clean, time-sorted, resampled YES-price series for one market/outcome."""
    slab = prices[(prices["market_id"] == market_id) & (prices["outcome_id"] == outcome_id)]
    if slab.empty:
        return pd.Series(dtype="float64")
    work = slab[["timestamp", "price"]].copy()
    work["timestamp"] = to_utc(work["timestamp"])
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work = work.dropna(subset=["timestamp", "price"])
    if work.empty:
        return pd.Series(dtype="float64")
    return work.set_index("timestamp")["price"].sort_index().resample(freq).last().dropna()


def social_activity_series(
    posts: pd.DataFrame,
    *,
    freq: str = "1D",
    time_col: str = "post_created_at",
    sentiment_col: str | None = "sentiment_compound",
    engagement_col: str | None = "engagement_score",
) -> pd.DataFrame:
    """Aggregate posts into a per-bucket activity frame.

    Columns: ``n_posts`` (volume), and ``mean_sentiment`` / ``engagement`` when
    those input columns are present. Index is the resampled timestamp.
    """
    if posts.empty or time_col not in posts.columns:
        return pd.DataFrame(columns=["n_posts"])
    work = posts.copy()
    work[time_col] = to_utc(work[time_col])
    work = work.dropna(subset=[time_col]).set_index(time_col).sort_index()
    if work.empty:
        return pd.DataFrame(columns=["n_posts"])
    agg: dict[str, pd.Series] = {"n_posts": work.resample(freq).size()}
    if sentiment_col and sentiment_col in work.columns:
        agg["mean_sentiment"] = (
            pd.to_numeric(work[sentiment_col], errors="coerce").resample(freq).mean()
        )
    if engagement_col and engagement_col in work.columns:
        agg["engagement"] = (
            pd.to_numeric(work[engagement_col], errors="coerce").resample(freq).sum()
        )
    out = pd.DataFrame(agg)
    out["n_posts"] = out["n_posts"].fillna(0).astype(int)
    return out

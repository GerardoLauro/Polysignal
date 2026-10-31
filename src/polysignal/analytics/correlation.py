"""Signal-market correlation analysis (Traccia_2 Task 2.3).

Studies the temporal relationship between social signals (post volume,
aggregate sentiment, engagement) and Polymarket price movements:

* **lead/lag** cross-correlation — does social precede, coincide, or follow the
  price? (positive lag = social leads);
* **sentiment-direction vs. price-direction** alignment;
* **activity-spike event study** — detect posting-volume spikes and measure the
  price move in a window around them.

Pure transforms over aligned daily series (see :mod:`.temporal`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_MAX_LAG_DAYS = 7
SPIKE_Z_THRESHOLD = 2.0


def lead_lag(
    price: pd.Series, social: pd.Series, *, max_lag_days: int = DEFAULT_MAX_LAG_DAYS
) -> pd.DataFrame:
    """Cross-correlation of social signal vs daily |price change| across lags.

    For lag ``k``, correlates ``social(t)`` with ``|price_change|(t+k)``; a
    positive ``k`` with high correlation means social activity *leads* the price
    move by ``k`` days. Returns ``DataFrame[lag, corr]``.
    """
    cols = ["lag", "corr"]
    if price.empty or social.empty:
        return pd.DataFrame(columns=cols)
    price_move = price.diff().abs()
    idx = price_move.index.union(social.index)
    pm = price_move.reindex(idx)
    sv = social.reindex(idx).astype("float64")
    rows: list[dict] = []
    for lag in range(-max_lag_days, max_lag_days + 1):
        shifted = pm.shift(-lag)
        pair = pd.concat([sv, shifted], axis=1).dropna()
        if len(pair) < 3 or pair.iloc[:, 0].std(ddof=0) == 0 or pair.iloc[:, 1].std(ddof=0) == 0:
            corr = np.nan
        else:
            corr = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
        rows.append({"lag": lag, "corr": round(corr, 4) if pd.notna(corr) else np.nan})
    return pd.DataFrame(rows, columns=cols)


def best_lag(lead_lag_table: pd.DataFrame) -> dict:
    """Return the lag of maximum absolute correlation + a lead/coincide/follow tag."""
    if lead_lag_table.empty or lead_lag_table["corr"].abs().max() != lead_lag_table["corr"].abs().max():
        return {"lag": None, "corr": None, "relation": "insufficient_data"}
    valid = lead_lag_table.dropna(subset=["corr"])
    if valid.empty:
        return {"lag": None, "corr": None, "relation": "insufficient_data"}
    row = valid.loc[valid["corr"].abs().idxmax()]
    lag = int(row["lag"])
    relation = "social_leads" if lag > 0 else ("social_follows" if lag < 0 else "coincident")
    return {"lag": lag, "corr": round(float(row["corr"]), 4), "relation": relation}


def sentiment_price_alignment(price: pd.Series, social: pd.DataFrame) -> dict:
    """Compare the *direction* of mean sentiment change to the price change.

    Returns the sign of the net price move, the sign of the net sentiment move,
    and whether they agree over the contract's observed window.
    """
    out = {"price_direction": None, "sentiment_direction": None, "aligned": None}
    if price.empty or social.empty or "mean_sentiment" not in social.columns:
        return out
    price_delta = float(price.iloc[-1] - price.iloc[0])
    sent = pd.to_numeric(social["mean_sentiment"], errors="coerce").dropna()
    if sent.empty:
        return out
    sent_delta = float(sent.iloc[-1] - sent.iloc[0]) if len(sent) > 1 else float(sent.iloc[0])
    pdir = int(np.sign(price_delta))
    sdir = int(np.sign(sent_delta))
    out.update(
        {
            "price_direction": pdir,
            "sentiment_direction": sdir,
            "aligned": bool(pdir == sdir and pdir != 0),
        }
    )
    return out


def spike_event_study(
    price: pd.Series,
    social: pd.Series,
    *,
    z_threshold: float = SPIKE_Z_THRESHOLD,
    window_days: int = 3,
) -> pd.DataFrame:
    """Detect social-volume spikes and measure the subsequent price move.

    A spike is a day whose social volume exceeds ``mean + z_threshold*std``. For
    each spike, reports the absolute price change over the following
    ``window_days``. Returns ``DataFrame[date, social_z, price_move]``.
    """
    cols = ["date", "social_z", "price_move"]
    if price.empty or social.empty or len(social) < 4:
        return pd.DataFrame(columns=cols)
    sv = social.astype("float64")
    mean, std = sv.mean(), sv.std(ddof=0)
    if std == 0:
        return pd.DataFrame(columns=cols)
    z = (sv - mean) / std
    spikes = z[z >= z_threshold]
    rows: list[dict] = []
    for date, zval in spikes.items():
        after = price[(price.index >= date) & (price.index <= date + pd.Timedelta(days=window_days))]
        if len(after) < 2:
            continue
        rows.append(
            {
                "date": date.isoformat(),
                "social_z": round(float(zval), 3),
                "price_move": round(float(after.iloc[-1] - after.iloc[0]), 4),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def per_platform_share(linked_accepted: pd.DataFrame, market_id: str) -> pd.DataFrame:
    """Per-platform share of accepted-linked posts for one market."""
    cols = ["platform", "n_posts", "share"]
    if linked_accepted.empty:
        return pd.DataFrame(columns=cols)
    slab = linked_accepted[linked_accepted["market_id"] == market_id]
    if slab.empty:
        return pd.DataFrame(columns=cols)
    counts = slab.groupby("platform").size().rename("n_posts").reset_index()
    counts["share"] = (counts["n_posts"] / counts["n_posts"].sum()).round(4)
    return counts.sort_values("n_posts", ascending=False).reset_index(drop=True)

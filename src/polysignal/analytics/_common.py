"""Shared helpers for the Task-2 analytics layer.

Pure transforms over DataFrames — no disk I/O, no settings reads — so every
analysis unit-tests against small fixtures. The orchestrator
(:mod:`polysignal.analytics.runner`) and the CLI own the I/O.
"""

from __future__ import annotations

import pandas as pd

# A link is "accepted" when its effective relevance verdict is RELEVANT or
# WEAKLY_RELEVANT. Enum values are lowercase on disk; match case-insensitively.
ACCEPTED_LABELS = {"relevant", "weakly_relevant"}

# The three Traccia_2 thematic domains (clean 100/100/100 split lives in
# ``markets.thematic_category``; ``domain`` is the finer Polymarket tag).
DOMAIN_COLUMN = "thematic_category"


def to_utc(series: pd.Series) -> pd.Series:
    """Parse a column of ISO-8601 strings/datetimes to tz-aware UTC."""
    return pd.to_datetime(series, utc=True, errors="coerce")


def effective_label(linked: pd.DataFrame) -> pd.Series:
    """Effective relevance label: ``final_relevance_label`` else ``relevance_label``."""
    final = linked.get("final_relevance_label")
    base = linked.get("relevance_label")
    if final is None:
        return base.astype("string") if base is not None else pd.Series(dtype="string")
    out = final.astype("string")
    if base is not None:
        out = out.fillna(base.astype("string"))
    return out


def accepted_links(linked: pd.DataFrame) -> pd.DataFrame:
    """Filter ``linked_posts`` to accepted (RELEVANT/WEAKLY_RELEVANT) rows."""
    if linked.empty:
        return linked
    labels = effective_label(linked).str.lower()
    return linked[labels.isin(ACCEPTED_LABELS)].copy()


def resolution_anchor(markets: pd.DataFrame) -> pd.Series:
    """Per-market resolution timestamp: ``resolved_at`` when present, else ``close_time``.

    Resolved markets frequently leave ``resolved_at`` null in this corpus, so the
    close time is the reliable resolution anchor for horizon/lifetime math.
    """
    resolved = to_utc(markets["resolved_at"]) if "resolved_at" in markets else pd.Series(pd.NaT, index=markets.index)
    close = to_utc(markets["close_time"]) if "close_time" in markets else pd.Series(pd.NaT, index=markets.index)
    return resolved.fillna(close)


def yes_outcome_by_market(outcomes: pd.DataFrame) -> dict[str, str]:
    """Map ``market_id -> outcome_id`` of the YES leg (label 'Yes', else index 0)."""
    if outcomes.empty:
        return {}
    work = outcomes.copy()
    work["_label"] = work["label"].astype("string").str.strip().str.lower()
    yes = work[work["_label"] == "yes"]
    out = dict(zip(yes["market_id"], yes["outcome_id"], strict=False))
    # Fallback to index 0 for any market without an explicit 'Yes' label.
    if "index" in work.columns:
        idx0 = work[work["index"] == 0]
        for mid, oid in zip(idx0["market_id"], idx0["outcome_id"], strict=False):
            out.setdefault(mid, oid)
    return out


def final_yes_outcome(markets: pd.DataFrame) -> dict[str, float]:
    """Map ``market_id -> 1.0`` if the market resolved YES, ``0.0`` if NO."""
    if markets.empty or "resolution" not in markets:
        return {}
    res = markets["resolution"].astype("string").str.strip().str.lower()
    out: dict[str, float] = {}
    for mid, value in zip(markets["market_id"], res, strict=False):
        if value == "yes":
            out[mid] = 1.0
        elif value == "no":
            out[mid] = 0.0
    return out


def domain_by_market(markets: pd.DataFrame) -> dict[str, str]:
    """Map ``market_id -> thematic_category`` (politics/finance/sports)."""
    if markets.empty:
        return {}
    col = DOMAIN_COLUMN if DOMAIN_COLUMN in markets else "domain"
    return dict(zip(markets["market_id"], markets[col].astype("string"), strict=False))


def rank_auc(scores: list[float], labels: list[float]) -> float | None:
    """ROC-AUC via the Mann-Whitney U statistic (no sklearn dependency).

    Returns ``None`` when only one class is present (AUC undefined).
    """
    pos = [s for s, y in zip(scores, labels, strict=False) if y == 1.0]
    neg = [s for s, y in zip(scores, labels, strict=False) if y == 0.0]
    if not pos or not neg:
        return None
    paired = sorted(zip(scores, labels, strict=False), key=lambda t: t[0])
    rank_sum = 0.0
    i = 0
    n = len(paired)
    while i < n:
        j = i
        while j < n and paired[j][0] == paired[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-based average rank for ties
        for k in range(i, j):
            if paired[k][1] == 1.0:
                rank_sum += avg_rank
        i = j
    n_pos, n_neg = len(pos), len(neg)
    u = rank_sum - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)

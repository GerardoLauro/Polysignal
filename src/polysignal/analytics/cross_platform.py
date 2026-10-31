"""Cross-platform comparison (Traccia_2 Task 2.4).

Per contract, characterizes which platform carries the signal and whether the
platforms are complementary or redundant. The current corpus is ~99% Telegram,
so most contracts are honestly reported as **single-platform**; the comparison
machinery is fully implemented and auto-populates once X/Reddit have data — no
multi-platform numbers are fabricated.

Pure transforms over DataFrames.
"""

from __future__ import annotations

import pandas as pd

from polysignal.analytics._common import accepted_links, domain_by_market

# A second platform is "substantial" (→ genuinely complementary/redundant rather
# than incidental) when it holds at least this share of a contract's posts.
SUBSTANTIAL_SHARE = 0.2


def _effective_confidence(linked: pd.DataFrame) -> pd.Series:
    final = linked.get("final_confidence")
    base = linked.get("confidence")
    if final is None:
        return pd.to_numeric(base, errors="coerce") if base is not None else pd.Series(dtype="float64")
    out = pd.to_numeric(final, errors="coerce")
    if base is not None:
        out = out.fillna(pd.to_numeric(base, errors="coerce"))
    return out


def cross_platform_comparison(linked: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """One row per contract: platform mix, dominant platform, and state."""
    cols = ["market_id", "domain", "n_platforms", "platforms", "n_posts",
            "dominant_platform", "dominant_share", "state"]
    acc = accepted_links(linked)
    if acc.empty:
        return pd.DataFrame(columns=cols)
    dom = domain_by_market(markets)
    acc = acc.copy()
    acc["_conf"] = _effective_confidence(acc)
    rows: list[dict] = []
    for market_id, slab in acc.groupby("market_id"):
        counts = slab.groupby("platform").size().sort_values(ascending=False)
        total = int(counts.sum())
        platforms = list(counts.index)
        dominant = str(counts.index[0])
        dominant_share = round(float(counts.iloc[0] / total), 4)
        if len(platforms) == 1:
            state = "single_platform"
        elif (counts.iloc[1] / total) >= SUBSTANTIAL_SHARE:
            state = "multi_platform_complementary"
        else:
            state = "multi_platform_dominated"
        rows.append(
            {
                "market_id": market_id,
                "domain": dom.get(market_id),
                "n_platforms": len(platforms),
                "platforms": ",".join(platforms),
                "n_posts": total,
                "dominant_platform": dominant,
                "dominant_share": dominant_share,
                "state": state,
            }
        )
    return pd.DataFrame(rows, columns=cols).sort_values("n_posts", ascending=False).reset_index(drop=True)


def cross_platform_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-contract comparison to a per-domain state breakdown."""
    cols = ["domain", "n_contracts", "single_platform", "multi_platform",
            "multi_complementary", "pct_single_platform"]
    if comparison.empty:
        return pd.DataFrame(columns=cols)
    rows: list[dict] = []
    for domain, slab in comparison.groupby("domain", dropna=False):
        n = len(slab)
        single = int((slab["state"] == "single_platform").sum())
        complementary = int((slab["state"] == "multi_platform_complementary").sum())
        rows.append(
            {
                "domain": domain,
                "n_contracts": n,
                "single_platform": single,
                "multi_platform": n - single,
                "multi_complementary": complementary,
                "pct_single_platform": round(single / n, 4) if n else 0.0,
            }
        )
    out = pd.DataFrame(rows, columns=cols).sort_values("domain").reset_index(drop=True)
    return out


def overall_platform_state(comparison: pd.DataFrame) -> dict:
    """High-level honesty flag for the dashboard cross-platform view."""
    if comparison.empty:
        return {"state": "no_data", "n_contracts": 0, "multi_platform_contracts": 0,
                "platforms_present": []}
    multi = int((comparison["n_platforms"] > 1).sum())
    present = sorted({p for row in comparison["platforms"] for p in str(row).split(",") if p})
    state = "multi_platform" if multi > 0 else "single_platform_only"
    return {
        "state": state,
        "n_contracts": len(comparison),
        "multi_platform_contracts": multi,
        "platforms_present": present,
    }

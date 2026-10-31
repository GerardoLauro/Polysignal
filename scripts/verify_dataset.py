"""Verify the collected dataset against the Traccia_2 coverage requirements.

Reports: resolved-contract counts per thematic domain (target 100 each),
price-history coverage and span, and (if present) social enrichment /
linking coverage. Read-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from polysignal.domain.enums import MarketDomain, thematic_category  # noqa: E402

D = ROOT / "data" / "normalized"
A = ROOT / "data" / "analytical"


def _bar(n: int, target: int = 100) -> str:
    return "OK" if n >= target else ("near" if n >= target * 0.6 else "SHORT")


def main() -> None:
    m = pd.read_parquet(D / "polymarket" / "markets.parquet")
    # backfill thematic_category from domain if missing (older runs)
    if "thematic_category" not in m.columns or m["thematic_category"].isna().all():
        m["thematic_category"] = m["domain"].map(
            lambda d: (tc.value if (tc := thematic_category(MarketDomain(d))) else None)
        )

    print("=" * 64)
    print("MARKET COVERAGE")
    print("=" * 64)
    print(f"total markets         : {len(m)}")
    print(f"resolved (is_resolved): {int(m.get('is_resolved', pd.Series(dtype=bool)).sum())}")
    print("\nresolved contracts per THEMATIC domain (target >= 100):")
    cat = m["thematic_category"].fillna("(none)")
    for c in ["politics", "finance", "sports", "(none)"]:
        n = int((cat == c).sum())
        flag = _bar(n) if c != "(none)" else ""
        print(f"  {c:10s} {n:4d}   {flag}")
    print("\nfine domain distribution:")
    print(m["domain"].value_counts().to_string())
    if "resolution" in m.columns:
        print("\nresolution outcomes:", dict(m["resolution"].value_counts(dropna=False)))

    # prices
    p = pd.read_parquet(D / "polymarket" / "price_points.parquet")
    yes = p[~p["outcome_id"].astype(str).str.endswith("-no")]
    per = p.groupby("market_id").size()
    print("\n" + "=" * 64)
    print("PRICE TIME SERIES")
    print("=" * 64)
    print(f"price points (all outcomes): {len(p):,}")
    print(f"markets with prices        : {p['market_id'].nunique()} / {len(m)}")
    print(f"points per market (min/median/max): "
          f"{int(per.min())}/{int(per.median())}/{int(per.max())}")
    ts = pd.to_datetime(p["timestamp"], utc=True, errors="coerce")
    print(f"timestamp span             : {ts.min()} -> {ts.max()}")

    # social enrichment (optional)
    pe = D / "social" / "post_enrichment.parquet"
    if pe.exists():
        e = pd.read_parquet(pe, columns=["sentiment_label", "platform"])
        print("\n" + "=" * 64)
        print("SOCIAL ENRICHMENT (post_enrichment.parquet)")
        print("=" * 64)
        print(f"enriched posts: {len(e):,}")
        print("sentiment:", dict(e["sentiment_label"].value_counts(dropna=False)))
        print("platform :", dict(e["platform"].value_counts(dropna=False)))

    # linking (optional)
    lp = D / "social" / "linked_posts.parquet"
    if lp.exists():
        import pyarrow.parquet as pq
        n = pq.ParquetFile(lp).metadata.num_rows
        ldf = pd.read_parquet(lp, columns=["market_id", "final_relevance_label", "relevance_label"])
        lab = ldf["final_relevance_label"].fillna(ldf["relevance_label"])
        print("\n" + "=" * 64)
        print("LINKING (linked_posts.parquet)")
        print("=" * 64)
        print(f"link rows: {n:,} | distinct markets linked: {ldf['market_id'].nunique()}")
        print("labels:", dict(lab.value_counts(dropna=False)))

    print("\nDONE")


if __name__ == "__main__":
    main()

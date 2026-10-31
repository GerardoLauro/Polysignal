"""Designate the curated high-coverage subset for the deep social analyses.

Per the chosen strategy: all 300 markets satisfy the collection requirement
and carry the descriptive/price analyses; the markets that actually have
enough social evidence form the subset used for the social-discourse,
signal-correlation, sentiment, and prediction analyses. This keeps those
analyses honest (computed where evidence exists) without dropping the
balanced 300 collection.

Writes ``data/analytical/curated_analysis_markets.parquet`` + a summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
NORM = ROOT / "data" / "normalized"
ANALYTICAL = ROOT / "data" / "analytical"

MIN_DIRECTIONAL = 3   # directional stances for "sufficient evidence"
MIN_RELEVANT = 5      # relevant/weakly-relevant links as a softer inclusion


def main() -> None:
    m = pd.read_parquet(NORM / "polymarket/markets.parquet",
                        columns=["market_id", "question", "thematic_category", "domain", "resolution"])
    st = pd.read_parquet(NORM / "social/stance_labels.parquet", columns=["market_id", "stance_label"])
    lp = pd.read_parquet(NORM / "social/linked_posts.parquet",
                         columns=["market_id", "final_relevance_label", "relevance_label"])

    direc = st[st["stance_label"].isin(["supports_yes", "supports_no"])]
    direc_per = direc.groupby("market_id").size().rename("directional_stances")
    lab = lp["final_relevance_label"].fillna(lp["relevance_label"])
    rel = lp[lab.isin(["relevant", "weakly_relevant"])]
    rel_per = rel.groupby("market_id").size().rename("relevant_links")

    cov = m.set_index("market_id").join(direc_per).join(rel_per)
    cov["directional_stances"] = cov["directional_stances"].fillna(0).astype(int)
    cov["relevant_links"] = cov["relevant_links"].fillna(0).astype(int)
    cov["in_curated"] = (cov["directional_stances"] >= MIN_DIRECTIONAL) | (
        cov["relevant_links"] >= MIN_RELEVANT
    )

    curated = cov[cov["in_curated"]].reset_index()
    print("=" * 60)
    print("CURATED HIGH-COVERAGE SUBSET (deep social analyses)")
    print("=" * 60)
    print(f"curated markets: {len(curated)} / 300")
    print("\nby thematic domain:")
    print(curated["thematic_category"].value_counts().to_string())
    print("\nall-300 coverage (markets with >=1 relevant link), by domain:")
    any_cov = cov[cov["relevant_links"] > 0]
    print(any_cov["thematic_category"].value_counts().to_string())

    ANALYTICAL.mkdir(parents=True, exist_ok=True)
    curated.to_parquet(ANALYTICAL / "curated_analysis_markets.parquet", index=False)
    summary = {
        "min_directional_stances": MIN_DIRECTIONAL,
        "min_relevant_links": MIN_RELEVANT,
        "curated_market_count": int(len(curated)),
        "curated_by_domain": curated["thematic_category"].value_counts().to_dict(),
        "any_coverage_by_domain": any_cov["thematic_category"].value_counts().to_dict(),
        "total_markets": int(len(m)),
    }
    (ANALYTICAL / "curated_analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {ANALYTICAL / 'curated_analysis_markets.parquet'}")
    print(f"wrote {ANALYTICAL / 'curated_analysis_summary.json'}")


if __name__ == "__main__":
    main()

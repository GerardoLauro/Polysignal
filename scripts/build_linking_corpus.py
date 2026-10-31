"""Build the corpus for a correct full re-link against the 300 markets.

Scope rule: a post created *before* a market existed cannot be relevant
evidence for it, and the markets live in 2024-2026 — so the linking corpus
is the **post-2024 window** of the canonical corpus, UNION the freshly
collected domain-matched staging posts (sports / entertainment / finance).
This replaces the earlier 200k recency sample (which under-counted evidence)
and is non-destructive: the canonical `social_posts.parquet` is untouched.

Usage:
    python scripts/build_linking_corpus.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
SOCIAL = ROOT / "data" / "normalized" / "social"
CANON = SOCIAL / "social_posts.parquet"
OUT = SOCIAL / "_linking_corpus.parquet"
CUTOFF = "2024-01-01"

# Freshly collected domain-matched staging corpora to fold in.
EXTRA = [
    ROOT / "data/staging/telegram_web_dm/normalized/social/social_posts.parquet",
]


def main() -> None:
    canon = pd.read_parquet(CANON)
    ts = pd.to_datetime(canon["created_at"], utc=True, errors="coerce")
    canon = canon[ts >= pd.Timestamp(CUTOFF, tz="UTC")].copy()
    print(f"canonical post-{CUTOFF} rows: {len(canon):,}")

    frames = [canon]
    for p in EXTRA:
        if p.exists():
            extra = pd.read_parquet(p)
            # align columns to the canonical schema
            extra = extra.reindex(columns=canon.columns)
            frames.append(extra)
            print(f"+ staging {p.parent.parent.parent.parent.name}: {len(extra):,} rows")
        else:
            print(f"(skip, not present) {p}")

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["platform", "post_id"], keep="first")
    print(f"merged {before:,} -> deduped {len(merged):,}")
    merged.to_parquet(OUT, index=False)
    print(f"wrote {len(merged):,} rows -> {OUT}")
    print("platforms:", dict(merged["platform"].value_counts()))


if __name__ == "__main__":
    main()

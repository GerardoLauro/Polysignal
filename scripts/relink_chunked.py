"""Memory-safe full re-link: chunk the corpus so each linking call is bounded.

The full 721k-post corpus scores ~2.7M candidate pairs, and materializing
that many Pydantic LinkedPost models at once exhausts RAM. This splits the
corpus into chunks, runs the real `polysignal linking score-candidates`
command on each in its own subprocess (which releases memory on exit), keeps
only the non-unrelated rows from each, and concatenates them into the
canonical linked_posts.parquet. Pipeline code is unchanged.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SOCIAL = ROOT / "data" / "normalized" / "social"
CORPUS = SOCIAL / "_linking_corpus.parquet"
LINKED = SOCIAL / "linked_posts.parquet"
TMP = SOCIAL / "_relink_tmp"
CHUNK = 150_000
KEEP = {"relevant", "weakly_relevant", "ambiguous"}


def main() -> None:
    TMP.mkdir(exist_ok=True)
    df = pd.read_parquet(CORPUS)
    n = len(df)
    print(f"corpus rows: {n:,}  chunk size: {CHUNK:,}")
    n_chunks = (n + CHUNK - 1) // CHUNK

    kept_frames: list[pd.DataFrame] = []
    for i in range(n_chunks):
        chunk = df.iloc[i * CHUNK : (i + 1) * CHUNK]
        cpath = TMP / f"chunk_{i}.parquet"
        lpath = TMP / f"linked_{i}.parquet"
        chunk.to_parquet(cpath, index=False)
        print(f"\n[chunk {i + 1}/{n_chunks}] {len(chunk):,} posts -> linking ...", flush=True)
        r = subprocess.run(
            [str(PY), "-m", "polysignal", "linking", "score-candidates",
             "--posts-path", str(cpath), "--output-path", str(lpath), "--candidate-mode", "auto"],
            cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if r.returncode != 0 or not lpath.exists():
            print("  linking failed:\n", (r.stdout or "")[-1500:], (r.stderr or "")[-1500:])
            sys.exit(1)
        lk = pd.read_parquet(lpath)
        lab = lk["final_relevance_label"].fillna(lk["relevance_label"])
        keep = lk[lab.isin(KEEP)].copy()
        kept_frames.append(keep)
        print(f"  scored {len(lk):,} pairs -> kept {len(keep):,} non-unrelated")
        cpath.unlink(missing_ok=True)
        lpath.unlink(missing_ok=True)
        del lk, keep, chunk

    merged = pd.concat(kept_frames, ignore_index=True)
    # de-dup on the row identity used downstream (post/segment id + market)
    key = [c for c in ("market_id", "post_id") if c in merged.columns]
    merged = merged.drop_duplicates(subset=key, keep="first")
    merged.to_parquet(LINKED, index=False)
    lab = merged["final_relevance_label"].fillna(merged["relevance_label"])
    print(f"\nwrote {LINKED} : {len(merged):,} rows, {merged['market_id'].nunique()} markets")
    print("labels:", dict(lab.value_counts()))
    try:
        TMP.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()

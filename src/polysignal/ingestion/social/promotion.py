"""Promote staged social collections into the canonical dataset, safely.

Collection writes only to ``data/staging/<source>/`` (Telegram API, Telegram
web, Reddit web/API, enriched). Promotion is the deliberate, backed-up step
that merges those staged rows into the canonical
``data/normalized/social/social_posts.parquet`` the product reads.

Safety contract:

* the canonical file is **backed up** before it is touched;
* a present-but-corrupt canonical file is **quarantined**, never silently
  overwritten;
* the merge **de-duplicates** across sources on the retrieval tuple and
  recomputes ``is_duplicate`` on post identity, preserving provenance;
* **synthetic / UI-stress** rows (``retrieval_mode == 'ui_stress_public_social'``)
  are excluded by default so demo data never contaminates real evidence;
* the write goes through the shrink-guarded :func:`safe_write_parquet`.

The merge is done at the DataFrame level so a multi-hundred-thousand-row
Telegram dataset promotes in seconds without rehydrating Pydantic models.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from polysignal.storage.parquet_store import (
    CorruptParquetError,
    DatasetShrinkError,
    backup_parquet,
    parquet_is_readable,
    parquet_row_count,
    quarantine_corrupt_parquet,
    read_dataframe,
    write_dataframe,
)
from polysignal.storage.paths import DataPaths, get_data_paths

# Marker left by the UI-stress augmentation on synthetic rows.
SYNTHETIC_RETRIEVAL_MODE = "ui_stress_public_social"

DEFAULT_PROMOTE_SOURCES: tuple[str, ...] = (
    "telegram_api",
    "telegram_web",
    "reddit",
    "enriched",
)

# Retrieval-event identity (a re-collection of the same post for the same
# query/market is the same row). Post identity for the is_duplicate flag.
_RETRIEVAL_KEYS = ["platform", "post_id", "query", "market_id_candidate"]
_IDENTITY_KEYS = ["platform", "post_id"]


def _staging_social_parquet(data_root: DataPaths, source: str):
    return data_root.root / "staging" / source / "normalized" / "social" / "social_posts.parquet"


def _drop_synthetic(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty or "retrieval_mode" not in df.columns:
        return df, 0
    # NA retrieval_mode = a real collected row that was never tagged (e.g. the
    # Telegram API collector does not set retrieval_mode). It is NOT synthetic.
    # fillna(False) is essential: an all-NA nullable-boolean mask makes ``~mask``
    # all-NA, and ``df.loc[~mask]`` then silently drops EVERY row while
    # ``mask.sum()`` reports 0 removed -- a silent wipe of real evidence.
    mask = (df["retrieval_mode"].astype("string") == SYNTHETIC_RETRIEVAL_MODE).fillna(False)
    removed = int(mask.sum())
    return df.loc[~mask].copy(), removed


def promote_staging_social(
    *,
    canonical_paths: DataPaths | None = None,
    sources: tuple[str, ...] = DEFAULT_PROMOTE_SOURCES,
    exclude_synthetic: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Merge staged social rows into the canonical parquet (backed up, deduped).

    Returns a summary dict with per-source contributions, dedup counts, the
    canonical backup path, and the before/after row counts. With ``dry_run``
    nothing is written — useful for previewing what a promotion would do.
    """
    canonical_paths = canonical_paths or get_data_paths()
    output = canonical_paths.social_posts_parquet

    # Strict read of the canonical target: quarantine (never clobber) corruption.
    try:
        existing = read_dataframe(output) if parquet_is_readable(output) else pd.DataFrame()
        if output.exists() and not parquet_is_readable(output):
            raise CorruptParquetError(output, ValueError("unreadable"))
    except CorruptParquetError:
        quarantine_corrupt_parquet(output)
        existing = pd.DataFrame()

    before_rows = len(existing)
    frames: list[pd.DataFrame] = []
    per_source: dict[str, Any] = {}
    synthetic_removed_total = 0
    if not existing.empty:
        frames.append(existing)

    for source in sources:
        parquet = _staging_social_parquet(canonical_paths, source)
        rows = parquet_row_count(parquet)
        if not parquet.exists():
            per_source[source] = {"present": False, "rows": 0, "promoted": 0}
            continue
        if rows is None:
            per_source[source] = {"present": True, "rows": 0, "promoted": 0, "unreadable": True}
            continue
        df = read_dataframe(parquet)
        raw_rows = len(df)
        synthetic_removed = 0
        if exclude_synthetic:
            df, synthetic_removed = _drop_synthetic(df)
            synthetic_removed_total += synthetic_removed
        frames.append(df)
        per_source[source] = {
            "present": True,
            "rows": raw_rows,
            "promoted": len(df),
            "synthetic_excluded": synthetic_removed,
        }

    if not frames:
        return {
            "output": str(output),
            "before_rows": before_rows,
            "after_rows": before_rows,
            "per_source": per_source,
            "synthetic_excluded": synthetic_removed_total,
            "dry_run": dry_run,
            "note": "no staging sources present; nothing to promote",
        }

    combined = pd.concat(frames, ignore_index=True)
    # Dedup on the retrieval event, then recompute the post-identity duplicate flag.
    present_retrieval_keys = [k for k in _RETRIEVAL_KEYS if k in combined.columns]
    deduped = combined.drop_duplicates(subset=present_retrieval_keys, keep="first").reset_index(
        drop=True
    )
    retrieval_dropped = int(len(combined) - len(deduped))
    identity_keys = [k for k in _IDENTITY_KEYS if k in deduped.columns]
    if identity_keys and "is_duplicate" in deduped.columns:
        deduped["is_duplicate"] = deduped.duplicated(subset=identity_keys, keep="first")

    after_rows = len(deduped)
    backup = None
    if not dry_run:
        backup = backup_parquet(output)
        # Write the merged DataFrame atomically via the model-free fast path,
        # but still go through the shrink guard by checking row counts first.
        prev = parquet_row_count(output)
        if prev is not None and after_rows < prev * 0.5:
            raise DatasetShrinkError(
                f"refusing to promote: canonical has {prev} rows, merged result only "
                f"{after_rows}. Existing file left intact (backup: {backup})."
            )
        write_dataframe(deduped, output)

    return {
        "output": str(output),
        "before_rows": before_rows,
        "after_rows": after_rows,
        "rows_added": after_rows - before_rows,
        "retrieval_duplicates_dropped": retrieval_dropped,
        "synthetic_excluded": synthetic_removed_total,
        "per_source": per_source,
        "backup": str(backup) if backup else None,
        "dry_run": dry_run,
    }


__all__ = [
    "DEFAULT_PROMOTE_SOURCES",
    "SYNTHETIC_RETRIEVAL_MODE",
    "promote_staging_social",
]

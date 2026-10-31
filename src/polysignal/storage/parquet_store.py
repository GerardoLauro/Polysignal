"""Parquet read/write helpers tailored to PolySignal's domain types.

The store is intentionally simple: one Parquet file per normalized
entity type (``markets``, ``prices``, ``posts``, ...). Future scaling
work can introduce partitioned datasets without changing the call
sites because the public API only exposes ``read`` / ``write``.

Timestamp serialization
-----------------------
Models are dumped via ``BaseModel.model_dump(mode="json")``, which
serializes timezone-aware ``datetime`` objects as **ISO 8601 UTC
strings** (e.g. ``"2025-11-04T23:59:00+00:00"``). This keeps the
Parquet schema fully self-describing and round-trips through
:func:`dataframe_to_models` (Pydantic re-parses the string back to a
timezone-aware ``datetime``). The trade-off is that downstream
analytic queries which want to filter by date must parse the column
(``pd.to_datetime(df["timestamp"], utc=True)``) before comparing —
the column is *not* a pandas datetime64 dtype.

Migrating to a native ``timestamp[us, tz=UTC]`` Arrow type is
straightforward but is deferred until a downstream module actually
needs it; doing it now would change the Parquet schema and require
coordinated updates in the analytics modules
(``polysignal.analytics`` parses these ISO strings via ``to_utc``).
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
import pyarrow.lib as pa_lib
import pyarrow.parquet as pq
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Errors that mean "this parquet file is unreadable / truncated", as opposed
# to a programming error. A long resumable collection that is interrupted
# mid-write can leave a partial file; treat such files as empty rather than
# crashing the whole run on the next read.
_CORRUPT_PARQUET_ERRORS = (pa_lib.ArrowInvalid, pa_lib.ArrowIOError, OSError, ValueError)


class CorruptParquetError(RuntimeError):
    """An existing parquet file is present on disk but cannot be read.

    Distinct from "missing": a *missing* file legitimately means an empty
    dataset, but a *present-but-unreadable* file (e.g. the 0-byte
    ``social_posts.parquet`` that triggered a 600k-message data loss) must
    never be silently treated as empty on a read-modify-write path, because
    the subsequent overwrite would erase the real (recoverable) data. Callers
    on the merge/append path raise this so they can quarantine the bad file
    and recover instead of clobbering it.
    """

    def __init__(self, path: Path, original: BaseException) -> None:
        self.path = path
        self.original = original
        super().__init__(
            f"existing parquet {path} is present but unreadable "
            f"({type(original).__name__}: {original}); refusing to treat it as empty."
        )


class DatasetShrinkError(RuntimeError):
    """A guarded overwrite would drop far more rows than expected.

    The classic data-loss signature is overwriting a ~600k-row dataset with a
    ~3k-row one because an upstream read silently returned empty. The guarded
    writer refuses such a catastrophic shrink unless the caller explicitly
    opts in (``allow_shrink=True``)."""


def _timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parquet_is_readable(path: Path) -> bool:
    """Whether ``path`` exists and can be opened as a valid parquet file."""
    if not path.exists():
        return False
    try:
        pq.ParquetFile(path)
    except _CORRUPT_PARQUET_ERRORS:
        return False
    return True


def parquet_row_count(path: Path) -> int | None:
    """Cheap row count via parquet metadata; ``None`` if missing/unreadable.

    Reads only the footer, so it is safe to call on a multi-hundred-megabyte
    file before deciding whether an overwrite would shrink it.
    """
    if not path.exists():
        return None
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except _CORRUPT_PARQUET_ERRORS:
        return None


def quarantine_corrupt_parquet(path: Path) -> Path | None:
    """Move an unreadable parquet aside for forensics; return the new path.

    The file lands under ``<dir>/_corrupt/<name>.<timestamp>.corrupt`` so it is
    out of the dataset's way but preserved (a 0-byte file is still worth
    keeping as evidence of what happened). Returns ``None`` if the file is
    already gone.
    """
    if not path.exists():
        return None
    quarantine_dir = path.parent / "_corrupt"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    dest = quarantine_dir / f"{path.name}.{_timestamp()}.corrupt"
    with contextlib.suppress(OSError):
        os.replace(path, dest)
        logger.warning("parquet_store: quarantined unreadable parquet %s -> %s", path, dest)
        return dest
    return None


def backup_parquet(path: Path, *, keep: int = 8) -> Path | None:
    """Copy a readable parquet to ``<dir>/_backups/`` before it is overwritten.

    Keeps the most recent ``keep`` backups and prunes older ones. Returns the
    backup path, or ``None`` if there was nothing valid to back up.
    """
    if not parquet_is_readable(path):
        return None
    backup_dir = path.parent / "_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / f"{path.name}.{_timestamp()}.bak"
    try:
        shutil.copy2(path, dest)
    except OSError as exc:  # pragma: no cover - best effort
        logger.warning("parquet_store: backup of %s failed: %s", path, exc)
        return None
    backups = sorted(backup_dir.glob(f"{path.name}.*.bak"))
    for stale in backups[:-keep]:
        with contextlib.suppress(OSError):
            stale.unlink()
    return dest


def _atomic_replace(tmp: Path, path: Path, *, attempts: int = 5, base_delay: float = 0.2) -> None:
    """``os.replace`` with retries for transient Windows file locks.

    On Windows ``os.replace`` can raise ``PermissionError`` (WinError 5) or an
    ``OSError`` sharing violation (WinError 32) when an antivirus scanner or
    the search indexer is momentarily holding the freshly-written temp file or
    the destination. These are transient, so retry with a short backoff before
    giving up.
    """
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except (PermissionError, OSError) as exc:
            last_exc = exc
            time.sleep(base_delay * (2**attempt))
    raise OSError(f"atomic replace of {path} failed after {attempts} attempts: {last_exc}")


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to ``path`` atomically.

    Writes to a sibling ``*.tmp`` file and ``os.replace``s it into place, so a
    reader never observes a half-written (e.g. 4-byte ``PAR1``-only) file and an
    interrupted write cannot corrupt the existing artifact. The final swap
    retries through transient Windows file locks.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        df.to_parquet(tmp, index=False)
        _atomic_replace(tmp, path)
    finally:
        if tmp.exists():
            with contextlib.suppress(OSError):  # best-effort cleanup
                tmp.unlink()


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file, returning an empty frame if it is missing/corrupt."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except _CORRUPT_PARQUET_ERRORS as exc:
        logger.warning(
            "parquet_store: ignoring unreadable parquet %s (%s: %s); treating as empty.",
            path,
            type(exc).__name__,
            exc,
        )
        return pd.DataFrame()


def read_parquet_strict(path: Path, model_cls: type[T]) -> list[T]:
    """Read a parquet for a read-modify-write path, never hiding corruption.

    Unlike :func:`read_parquet` (which treats an unreadable file as empty,
    fine for read-only analytics), this raises :class:`CorruptParquetError`
    when the file exists but cannot be parsed. The merge/append path uses it
    so a corrupt ``social_posts.parquet`` is quarantined and recovered rather
    than silently overwritten with a near-empty dataset.
    """
    if not path.exists():
        return []
    try:
        df = pd.read_parquet(path)
    except _CORRUPT_PARQUET_ERRORS as exc:
        raise CorruptParquetError(path, exc) from exc
    return dataframe_to_models(df, model_cls)


def safe_write_parquet(
    models: Sequence[BaseModel],
    path: Path,
    *,
    allow_shrink: bool = False,
    min_keep_ratio: float = 0.5,
    backup: bool = True,
    label: str = "",
) -> Path:
    """Overwrite ``path`` with data-loss guards: backup + catastrophic-shrink refusal.

    Before replacing a readable existing file this snapshots it under
    ``_backups/`` and, unless ``allow_shrink`` is set, refuses to write a
    dataset that would drop below ``min_keep_ratio`` of the current row count
    (the 600k -> 3k overwrite that motivated this). A refused write keeps the
    existing file untouched and dumps the rejected payload next to it as
    ``<name>.rejected-<ts>.parquet`` for inspection.
    """
    new_count = len(models)
    prev_count = parquet_row_count(path)
    if (
        prev_count is not None
        and prev_count > 0
        and not allow_shrink
        and new_count < prev_count * min_keep_ratio
    ):
        reject = path.with_name(f"{path.name}.rejected-{_timestamp()}.parquet")
        with contextlib.suppress(Exception):
            _write_parquet_atomic(models_to_dataframe(models), reject)
        raise DatasetShrinkError(
            f"refusing to overwrite {label or path.name}: existing has {prev_count} rows, "
            f"new payload has only {new_count} (< {min_keep_ratio:.0%}). "
            f"Existing file left intact; rejected payload written to {reject}. "
            "Pass allow_shrink=True if this shrink is intentional."
        )
    if backup:
        backup_parquet(path)
    _write_parquet_atomic(models_to_dataframe(models), path)
    return path


def models_to_dataframe(models: Sequence[BaseModel]) -> pd.DataFrame:
    """Convert a sequence of Pydantic models to a flat dataframe.

    Nested models are stored as JSON-serializable dicts in their columns;
    consumers that need to query them should normalize further.
    """
    if not models:
        return pd.DataFrame()
    return pd.DataFrame([m.model_dump(mode="json") for m in models])


def dataframe_to_models(df: pd.DataFrame, model_cls: type[T]) -> list[T]:
    """Reverse of :func:`models_to_dataframe` for a single model class."""
    if df is None or df.empty:
        return []
    rows = [_normalize_missing_values(row) for row in df.to_dict(orient="records")]
    return [model_cls.model_validate(row) for row in rows]


def _normalize_missing_values(value: Any) -> Any:
    """Convert pandas scalar missing values back to JSON-compatible ``None``."""
    if isinstance(value, dict):
        return {key: _normalize_missing_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_missing_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_missing_values(item) for item in value)
    if pd.api.types.is_scalar(value):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value


def write_parquet(models: Sequence[BaseModel], path: Path) -> Path:
    """Write a sequence of models to ``path`` as Parquet (overwrites, atomic)."""
    df = models_to_dataframe(models)
    _write_parquet_atomic(df, path)
    return path


def write_dataframe(df: pd.DataFrame, path: Path) -> Path:
    """Atomically overwrite ``path`` with a DataFrame (model-free fast path)."""
    _write_parquet_atomic(df, path)
    return path


def append_parquet(models: Iterable[BaseModel], path: Path) -> Path:
    """Append models to ``path``, merging with whatever is already there.

    This is the simple deterministic version: load → concat → write. It
    is appropriate for the dataset sizes expected in this project. A
    future version can switch to an Arrow-native append.
    """
    new_df = models_to_dataframe(list(models))
    existing = _safe_read_parquet(path)
    merged = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    _write_parquet_atomic(merged, path)
    return path


def read_parquet(path: Path, model_cls: type[T]) -> list[T]:
    """Read a Parquet file and rehydrate the rows as Pydantic models."""
    df = _safe_read_parquet(path)
    return dataframe_to_models(df, model_cls)


def read_dataframe(path: Path) -> pd.DataFrame:
    """Read a Parquet file as a raw DataFrame (used by analytics)."""
    return _safe_read_parquet(path)


def upsert_parquet(
    models: Sequence[BaseModel],
    path: Path,
    key_columns: Sequence[str],
) -> Path:
    """Deduplicating append: keep the latest row per composite key (atomic)."""
    new_df = models_to_dataframe(models)
    if new_df.empty:
        return path
    existing = _safe_read_parquet(path)
    merged: pd.DataFrame = (
        pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    )
    merged = merged.drop_duplicates(subset=list(key_columns), keep="last")
    _write_parquet_atomic(merged, path)
    return path


__all__ = [
    "CorruptParquetError",
    "DatasetShrinkError",
    "append_parquet",
    "backup_parquet",
    "dataframe_to_models",
    "models_to_dataframe",
    "parquet_is_readable",
    "parquet_row_count",
    "quarantine_corrupt_parquet",
    "read_dataframe",
    "read_parquet",
    "read_parquet_strict",
    "safe_write_parquet",
    "upsert_parquet",
    "write_dataframe",
    "write_parquet",
]


# A small helper used in tests / scripts to assert that an iterable of
# objects is non-empty. Keeping it here means storage callers can
# import a single place for "shape checks" before persisting.
def require_nonempty(items: Sequence[Any], label: str) -> None:
    if not items:
        raise ValueError(f"refusing to persist empty payload for {label}")

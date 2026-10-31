"""Raw payload persistence for Polymarket ingestion.

Every ingestion run dumps its untouched API payloads under a
run-scoped subdirectory below ``data/raw/polymarket/``. The design
rules live in ``docs/DATA_MODEL.md`` (section "Raw payload
persistence"):

* raw before normalized;
* run-scoped directory named ``<YYYY-MM-DD>/<YYYYMMDDTHHMMSSZ>-<run_id>/``;
* append-only — files are never overwritten or mutated;
* a ``_manifest.json`` describes the run (endpoint, params, start /
  finish, page count).

Files are written as plain JSON (one file per page or per object). The
filenames already encode pagination, so we do not use JSONL.
"""

from __future__ import annotations

import json
import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def generate_run_id(now: datetime | None = None) -> str:
    """Return a short, sortable run id.

    The id combines a UTC timestamp with a short random suffix. It is
    passed verbatim to ``DataPaths.raw_run_dir``, which prepends the
    layout's own ``YYYY-MM-DD/HHMMSSZ`` prefix; the timestamp inside
    the id makes it safe to use anywhere (filenames, log lines, the
    ``_manifest.json``) on its own.
    """
    moment = (now or _utcnow()).astimezone(timezone.utc)
    return moment.strftime("%H%M%SZ") + "-" + secrets.token_hex(3)


@dataclass
class RawRunWriter:
    """Append-only writer for one run-scoped raw directory.

    Use one instance per logical run (one fetch-markets, one
    fetch-prices). The directory is created lazily — instantiating
    the writer does not touch the filesystem.
    """

    directory: Path
    endpoint: str
    params: dict[str, Any]
    user_agent: str | None = None
    started_at: datetime = field(default_factory=_utcnow)
    finished_at: datetime | None = None
    page_count: int = 0
    object_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def _ensure(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- pages
    def write_page(self, page_index: int, payload: Any) -> Path:
        """Persist one paginated chunk as ``page-NNNN.json``."""
        self._ensure()
        path = self.directory / f"page-{page_index:04d}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        self.page_count = max(self.page_count, page_index + 1)
        if isinstance(payload, list):
            self.object_count += len(payload)
        else:
            self.object_count += 1
        return path

    def write_named(self, name: str, payload: Any) -> Path:
        """Persist a single object under a custom filename.

        ``name`` should be filename-safe (no path separators). Used for
        price histories where the natural key is the token id rather
        than a page number.
        """
        if "/" in name or "\\" in name:
            raise ValueError(f"raw file name must not contain path separators: {name!r}")
        if not name.endswith(".json"):
            name = f"{name}.json"
        if len(name) > 80:
            original_name = name
            digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:20]
            name = f"{name[:24]}__{digest}.json"
            self.extra.setdefault("named_files", {})[name] = original_name
        self._ensure()
        path = self.directory / name
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        self.object_count += 1
        return path

    # -------------------------------------------------------------- manifest
    def write_manifest(self) -> Path:
        """Finalize the run by writing ``_manifest.json``."""
        self._ensure()
        self.finished_at = _utcnow()
        manifest = {
            "endpoint": self.endpoint,
            "params": _jsonable(self.params),
            "user_agent": self.user_agent,
            "started_at": self.started_at.astimezone(timezone.utc).isoformat(),
            "finished_at": self.finished_at.astimezone(timezone.utc).isoformat(),
            "page_count": self.page_count,
            "object_count": self.object_count,
            "extra": _jsonable(self.extra),
        }
        path = self.directory / "_manifest.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2, sort_keys=True)
        return path


def _jsonable(value: Any) -> Any:
    """Coerce a value into something ``json.dump`` accepts."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return str(value)


__all__ = ["RawRunWriter", "generate_run_id"]

"""Raw snapshot persistence for social ingestion.

This is the social-side equivalent of
:mod:`polysignal.ingestion.polymarket.raw_store`. Each ingestion run
dumps a JSON-serializable per-submission/per-message snapshot under a
run-scoped subdirectory below ``data/raw/social/<platform>/``. The
layout rules match the project convention captured in
``docs/DATA_MODEL.md``:

* raw before normalized;
* run-scoped directory named ``<YYYY-MM-DD>/<YYYYMMDDTHHMMSSZ>-<run_id>/``;
* append-only run directories — an exact repeated retrieval path within
  one run is an idempotent rewrite;
* a ``_manifest.json`` describes the run (platform, queries issued,
  start / finish, file count).

Each retrieval event is persisted to a deterministic filename built
from the market id, post id, and a short query hash. This preserves
lineage when several queries surface the same post.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def generate_social_run_id(now: datetime | None = None) -> str:
    """Return a short, sortable run id for a social ingestion run."""
    moment = (now or _utcnow()).astimezone(timezone.utc)
    return moment.strftime("%H%M%SZ") + "-" + secrets.token_hex(3)


def _safe_filename(value: str) -> str:
    """Turn an id into a filename-safe slug (no path separators)."""
    if not value:
        return "unknown"
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in value)[:96]


@dataclass
class SocialRawWriter:
    """Append-only writer for one run-scoped raw directory.

    Use one instance per platform per logical run (one Reddit
    ingestion, one Telegram crawl). The directory is created lazily —
    instantiating the writer does not touch the filesystem.
    """

    directory: Path
    platform: str
    started_at: datetime = field(default_factory=_utcnow)
    finished_at: datetime | None = None
    object_count: int = 0
    duplicate_writes: int = 0
    queries: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    _written_paths: set[Path] = field(default_factory=set, init=False, repr=False)

    def _ensure(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- writes
    def write_post(
        self,
        query: str,
        market_id: str,
        post_id: str,
        payload: Any,
        *,
        context: dict[str, Any] | None = None,
    ) -> Path:
        """Persist one raw post payload, with query lineage in the name.

        Each retrieval gets its own file: the filename embeds the
        market id, post id, **and** a short SHA-256 prefix of the query
        text. That way, when the same post is surfaced by two
        different queries for the same market, the two raw files do
        not collide and the query lineage is preserved on disk —
        not only inside the JSON wrapper.

        Within one run, re-issuing the exact same ``(query, market,
        post)`` triple overwrites the file with identical content,
        which is the desired idempotent behaviour.
        """
        self._ensure()
        context = context or {}
        hash_input = f"{query or ''}|{context.get('retrieval_work_id') or ''}"
        query_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:8]
        name = f"{_safe_filename(market_id)}__{_safe_filename(post_id)}__{query_hash}.json"
        path = self.directory / name
        wrapped = {
            "query": query,
            "query_hash": query_hash,
            "market_id": market_id,
            "post_id": post_id,
            "collected_at": _utcnow().isoformat(),
            "payload": payload,
        }
        wrapped.update(_jsonable(context))
        with path.open("w", encoding="utf-8") as fh:
            json.dump(wrapped, fh, ensure_ascii=False, indent=2, sort_keys=True)
        if path not in self._written_paths:
            self._written_paths.add(path)
            self.object_count += 1
        else:
            self.duplicate_writes += 1
        return path

    def record_query(
        self,
        query: str,
        market_id: str,
        post_count: int,
        error: str | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Add a per-query entry to the manifest."""
        row = {
            "query": query,
            "market_id": market_id,
            "post_count": int(post_count),
            "error": error,
        }
        if context:
            row.update(_jsonable(context))
        self.queries.append(row)

    # -------------------------------------------------------------- manifest
    def write_manifest(self) -> Path:
        """Finalize the run by writing ``_manifest.json``."""
        self._ensure()
        self.finished_at = _utcnow()
        manifest = {
            "platform": self.platform,
            "started_at": self.started_at.astimezone(timezone.utc).isoformat(),
            "finished_at": self.finished_at.astimezone(timezone.utc).isoformat(),
            "object_count": self.object_count,
            "duplicate_writes": self.duplicate_writes,
            "queries": self.queries,
            "extra": _jsonable(self.extra),
        }
        path = self.directory / "_manifest.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2, sort_keys=True)
        return path


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return str(value)


__all__ = ["SocialRawWriter", "generate_social_run_id"]

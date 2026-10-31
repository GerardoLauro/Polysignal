"""Simple deterministic on-disk caches.

The LLM cache and the feature cache both follow the same pattern:

    key  →  json blob

Keys are SHA-256 hashes derived from a stable input. The cache is the
backbone of reproducibility: given the same input, future runs reuse
the previous output instead of re-querying paid APIs or recomputing
expensive features.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_key(payload: Any) -> str:
    """Return a deterministic SHA-256 hex digest for an arbitrary payload.

    The payload is JSON-serialized with sorted keys so that equivalent
    Python objects produce the same hash regardless of dict ordering.
    """
    serialized = json.dumps(payload, sort_keys=True, default=_json_fallback)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _json_fallback(obj: Any) -> Any:
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


class JSONCache:
    """A trivial directory-backed JSON cache."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- core
    def _path_for(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def has(self, key: str) -> bool:
        return self._path_for(key).exists()

    def get(self, key: str) -> Any | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def set(self, key: str, value: Any) -> None:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(value, fh, ensure_ascii=False, indent=2, default=_json_fallback)

    def delete(self, key: str) -> None:
        path = self._path_for(key)
        if path.exists():
            path.unlink()

    # -------------------------------------------------------------- helpers
    def get_or_compute(self, key: str, compute):  # type: ignore[no-untyped-def]
        """Return cached value or compute, store, and return it.

        ``compute`` is a zero-argument callable. We avoid running it
        until we know the key is missing.
        """
        if self.has(key):
            return self.get(key)
        value = compute()
        self.set(key, value)
        return value


__all__ = ["JSONCache", "stable_key"]

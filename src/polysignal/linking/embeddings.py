"""Embedding-provider abstraction for the Branch 05B overlay.

Design goals
------------
1. **Optional.** The default test suite must not download a model or
   contact the network. The CLI defaults to ``NullEmbeddingProvider``
   (no-op).
2. **Pluggable.** The scorer only depends on the
   :class:`EmbeddingProvider` protocol — a ``model_id`` string, an
   ``is_available()`` check, and an ``embed(text)`` returning a vector.
3. **Cacheable.** Repeated calls on the same ``(model_id, text)`` pair
   hit a JSON-on-disk cache so reruns are cheap and reproducible.
4. **Deterministic fallback.** If ``sentence-transformers`` is not
   installed and ``--use-embeddings`` was set, the CLI emits a warning
   and falls back to the null provider rather than crashing.

Providers
---------
* :class:`NullEmbeddingProvider` — disabled by construction
  (``is_available() -> False``); ``embed`` returns ``[0.0]``.
* :class:`SentenceTransformerEmbeddingProvider` — lazy-loads the
  ``sentence-transformers`` package; raises a clear error if missing.
* :class:`StaticMockEmbeddingProvider` — used by tests; takes a
  ``{text: vector}`` map and returns the matching vector.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Anything that can turn a string into a fixed-size vector."""

    model_id: str

    def is_available(self) -> bool: ...

    def embed(self, text: str) -> list[float]: ...


class NullEmbeddingProvider:
    """No-op provider — default when embeddings are disabled."""

    model_id: str = "null"

    def is_available(self) -> bool:
        return False

    def embed(self, text: str) -> list[float]:
        return [0.0]


class StaticMockEmbeddingProvider:
    """Test-only provider returning vectors from a pre-supplied map.

    ``vectors`` maps the *raw text* (as the scorer will pass it) to the
    desired embedding. Unknown texts fall back to a deterministic
    bag-of-words vector so tests don't fail on a missed entry.
    """

    def __init__(self, vectors: Mapping[str, list[float]], *, model_id: str = "mock") -> None:
        self._vectors = dict(vectors)
        self.model_id = model_id

    def is_available(self) -> bool:
        return True

    def embed(self, text: str) -> list[float]:
        if text in self._vectors:
            return list(self._vectors[text])
        # Deterministic fallback: 16-dim hash-based unit vector.
        digest = hashlib.sha1(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:16]]


class SentenceTransformerEmbeddingProvider:
    """Optional sentence-transformers-backed provider.

    The dependency is imported lazily so test collection never fails
    when the package is missing. Use ``is_available()`` to check before
    calling ``embed``.
    """

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_id = model_id
        self._model = None
        self._import_error: str | None = None

    def is_available(self) -> bool:
        if self._model is not None:
            return True
        try:  # pragma: no cover - exercised only when the dep is installed
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

            self._model = SentenceTransformer(self.model_id)
            return True
        except Exception as exc:  # pragma: no cover - depends on env
            self._import_error = str(exc)
            return False

    def embed(self, text: str) -> list[float]:
        if not self.is_available():  # pragma: no cover - depends on env
            raise RuntimeError(
                "SentenceTransformerEmbeddingProvider not available: "
                f"{self._import_error}. Install polysignal[embeddings] or "
                "pass --no-embeddings."
            )
        vector = self._model.encode(text, normalize_embeddings=True)  # type: ignore[union-attr]
        return [float(x) for x in vector.tolist()]  # pragma: no cover - depends on env


class CachedEmbeddingProvider:
    """JSON-on-disk cache wrapper around any :class:`EmbeddingProvider`."""

    def __init__(self, inner: EmbeddingProvider, cache_dir: Path) -> None:
        self._inner = inner
        self.model_id = inner.model_id
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path = self._cache_dir / f"{_safe_filename(self.model_id)}.json"
        self._memory: dict[str, list[float]] = {}
        if self._cache_path.exists():
            try:
                self._memory = json.loads(self._cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._memory = {}

    def is_available(self) -> bool:
        return self._inner.is_available()

    def embed(self, text: str) -> list[float]:
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        cached = self._memory.get(key)
        if cached is not None:
            return cached
        vector = self._inner.embed(text)
        self._memory[key] = vector
        try:
            self._cache_path.write_text(json.dumps(self._memory), encoding="utf-8")
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("embedding cache write failed: %s", exc)
        return vector


def _safe_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in value)[:120]


def build_default_provider(
    *,
    enabled: bool,
    model_id: str,
    cache_dir: Path | None,
) -> EmbeddingProvider:
    """Build the provider implied by the CLI flags.

    ``enabled=False`` -> :class:`NullEmbeddingProvider`.
    ``enabled=True`` + sentence-transformers present -> wrapped + cached.
    ``enabled=True`` + sentence-transformers missing -> log a warning,
    return :class:`NullEmbeddingProvider`.
    """
    if not enabled:
        return NullEmbeddingProvider()
    candidate = SentenceTransformerEmbeddingProvider(model_id=model_id)
    if not candidate.is_available():
        logger.warning("sentence-transformers not available; embedding overlay disabled.")
        return NullEmbeddingProvider()
    if cache_dir is not None:
        return CachedEmbeddingProvider(candidate, cache_dir)
    return candidate


__all__ = [
    "CachedEmbeddingProvider",
    "EmbeddingProvider",
    "NullEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "StaticMockEmbeddingProvider",
    "build_default_provider",
]

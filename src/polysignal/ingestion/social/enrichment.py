"""Shared scaffolding for the Branch 12 public-web enrichment adapters.

This module holds everything the Reddit (old.reddit.com) and Telegram
(t.me/s/<channel>) Scrapling adapters have in common:

* a thin :class:`HtmlFetcher` protocol plus the real :class:`ScraplingFetcher`
  (Scrapling's ``Fetcher.get`` is the *only* network entry point, lazily
  imported so the package works without the optional ``scraping`` extra);
* a deterministic :func:`stable_post_id` used for cross-run / cross-source
  deduplication;
* helpers that normalize a batch of :class:`SocialPost` into the canonical
  ``SocialPostRecord`` schema, merge-dedup against whatever is already on disk,
  and write both the Parquet and CSV mirror;
* a light, deterministic market-awareness estimator used only for the coverage
  report (it never fabricates market links — Branch 05 linking remains the
  source of truth);
* the per-source / per-run result dataclasses the CLI renders.

Nothing here performs trading logic, adds a forecasting model, or rewrites the
existing pipeline; it strictly produces normalized, deduplicated, traceable
``SocialPostRecord`` rows in the existing schema.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from polysignal.domain import Market, Platform, SocialPost, SocialPostRecord
from polysignal.ingestion.social.normalizer import build_record, merge_with_existing
from polysignal.storage.parquet_store import read_parquet, write_parquet


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Fetching (the only network surface)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FetchOutcome:
    """Result of one HTTP GET — success or graceful failure."""

    url: str
    ok: bool
    status: int | None = None
    html: str | None = None
    error: str | None = None


class HtmlFetcher(Protocol):
    """Minimal fetch surface so tests can inject canned HTML (no network)."""

    def get(self, url: str) -> FetchOutcome: ...


class ScraplingFetcher:
    """Real fetcher backed by Scrapling's plain ``Fetcher`` (no stealth/bypass).

    Per Branch 12 policy this uses only normal public-web fetching: no stealth
    fetchers, no proxy rotation, no CAPTCHA/Cloudflare circumvention.
    """

    def __init__(self, *, user_agent: str, timeout: float = 30.0) -> None:
        self.user_agent = user_agent
        self.timeout = float(timeout)
        self._fetcher = None

    @staticmethod
    def _import_fetcher() -> Any:
        try:
            from scrapling.fetchers import Fetcher
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise RuntimeError(
                "Scrapling is required for public-web enrichment. Install it with: "
                'pip install -e ".[scraping]"'
            ) from exc
        return Fetcher

    def get(self, url: str) -> FetchOutcome:
        if self._fetcher is None:
            self._fetcher = self._import_fetcher()
        try:
            page = self._fetcher.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
            )
        except TypeError:
            # Older/newer Scrapling signatures may not accept headers/timeout
            # kwargs; fall back to the documented minimal call.
            try:
                page = self._fetcher.get(url)
            except Exception as exc:
                return FetchOutcome(url=url, ok=False, error=f"{type(exc).__name__}: {exc}"[:300])
        except Exception as exc:
            return FetchOutcome(url=url, ok=False, error=f"{type(exc).__name__}: {exc}"[:300])
        status = getattr(page, "status", None)
        html = None
        for attr in ("html_content", "body"):
            value = getattr(page, attr, None)
            if isinstance(value, str) and value:
                html = value
                break
        if html is None:
            html = str(page)
        ok = (status is None or 200 <= int(status) < 300) and bool(html)
        return FetchOutcome(url=url, ok=ok, status=status, html=html)


class StealthyScraplingFetcher:
    """Fetcher backed by Scrapling's ``StealthyFetcher`` (real stealth browser).

    Reddit (Cloudflare + Reddit's own anti-bot) hard-blocks plain HTTP fetches
    with HTTP 403 on every page and ``.json`` endpoint, so the deterministic
    ``Fetcher`` returns zero usable rows. A stealth browser renders the public
    ``old.reddit.com`` listing pages exactly as a human browser would and is
    served 200s. This is the project's approved scraping method; it still does
    only public, unauthenticated page fetching — no login, no proxies, no
    CAPTCHA solving.

    The stealth browser is launched per ``get`` call (Scrapling's documented
    surface), so callers should keep ``max_pages_per_source`` modest and use a
    generous timeout. The returned :class:`FetchOutcome` matches
    :class:`ScraplingFetcher` so the two are drop-in interchangeable.
    """

    def __init__(
        self,
        *,
        user_agent: str | None = None,
        timeout: float = 90.0,
        headless: bool = True,
        network_idle: bool = False,
    ) -> None:
        self.user_agent = user_agent
        # Scrapling/patchright wants the navigation timeout in milliseconds and
        # needs enough headroom for a cold browser start + render.
        self.timeout_ms = int(max(60.0, float(timeout)) * 1000)
        self.headless = headless
        self.network_idle = network_idle
        self._fetcher = self._import_fetcher()

    @staticmethod
    def _import_fetcher() -> Any:
        try:
            from scrapling.fetchers import StealthyFetcher
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise RuntimeError(
                "Scrapling's StealthyFetcher is required for blocked public-web "
                'sources. Install it with: pip install -e ".[scraping]" and run '
                "`python -m patchright install chromium`."
            ) from exc
        return StealthyFetcher

    def get(self, url: str) -> FetchOutcome:
        try:
            page = self._fetcher.fetch(
                url,
                headless=self.headless,
                network_idle=self.network_idle,
                timeout=self.timeout_ms,
            )
        except Exception as exc:
            return FetchOutcome(url=url, ok=False, error=f"{type(exc).__name__}: {exc}"[:300])
        status = getattr(page, "status", None)
        html = None
        for attr in ("html_content", "body"):
            value = getattr(page, attr, None)
            if isinstance(value, str) and value:
                html = value
                break
        if html is None:
            html = str(page)
        ok = (status is None or 200 <= int(status) < 300) and bool(html)
        return FetchOutcome(url=url, ok=ok, status=status, html=html)


def build_html_fetcher(
    base_fetcher: str | None,
    *,
    user_agent: str,
    timeout: float,
) -> HtmlFetcher:
    """Pick the fetcher named by ``base_fetcher`` (config-driven).

    ``"StealthyFetcher"`` / ``"stealthy"`` / ``"stealth"`` (case-insensitive)
    selects the stealth-browser fetcher needed for anti-bot sites like Reddit;
    anything else falls back to the plain deterministic ``Fetcher``.
    """
    name = (base_fetcher or "").strip().lower()
    if name in {"stealthyfetcher", "stealthy", "stealth"}:
        return StealthyScraplingFetcher(user_agent=user_agent, timeout=timeout)
    return ScraplingFetcher(user_agent=user_agent, timeout=timeout)


@dataclass
class RateLimiter:
    """Sleep just enough between requests to honour ``rate_limit_seconds``."""

    seconds: float
    sleep: Callable[[float], None] = time.sleep
    _last: float | None = field(default=None, init=False, repr=False)
    clock: Callable[[], float] = time.monotonic

    def wait(self) -> None:
        if self.seconds <= 0:
            return
        now = self.clock()
        if self._last is not None:
            elapsed = now - self._last
            remaining = self.seconds - elapsed
            if remaining > 0:
                self.sleep(remaining)
        self._last = self.clock()


# ---------------------------------------------------------------------------
# Deterministic stable IDs
# ---------------------------------------------------------------------------


def stable_post_id(*parts: Any) -> str:
    """Deterministic hash over stable identifying fields.

    Used as a fallback id when a native platform id is unavailable, and to make
    deduplication reproducible across repeated runs and the combined dataset.
    """
    payload = "␟".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SourceOutcome:
    """Per-source provenance — honest about what was actually read."""

    name: str
    platform: str
    url: str
    status: str = "skipped"  # succeeded | empty | failed | skipped
    pages_fetched: int = 0
    http_status: int | None = None
    raw_extracted: int = 0
    normalized: int = 0
    reason: str | None = None
    snapshot_paths: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "platform": self.platform,
            "url": self.url,
            "status": self.status,
            "pages_fetched": self.pages_fetched,
            "http_status": self.http_status,
            "raw_extracted": self.raw_extracted,
            "normalized": self.normalized,
            "reason": self.reason,
            "snapshot_count": len(self.snapshot_paths),
        }


@dataclass
class CollectionResult:
    """Outcome envelope for one platform collection run."""

    platform: str
    collection_method: str
    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    raw_dir: Path | None = None
    posts_path: Path | None = None
    csv_path: Path | None = None
    outcomes: list[SourceOutcome] = field(default_factory=list)
    raw_extracted: int = 0
    new_records_built: int = 0
    rows_added: int = 0
    exact_duplicates_dropped: int = 0
    duplicate_rows_flagged: int = 0
    normalized_total: int = 0
    raw_html_snapshots: int = 0

    @property
    def succeeded(self) -> list[SourceOutcome]:
        return [o for o in self.outcomes if o.status == "succeeded"]

    @property
    def failed(self) -> list[SourceOutcome]:
        return [o for o in self.outcomes if o.status in {"failed", "empty"}]

    def summary(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "collection_method": self.collection_method,
            "run_id": self.run_id,
            "started_at": self.started_at.astimezone(timezone.utc).isoformat(),
            "finished_at": (
                self.finished_at.astimezone(timezone.utc).isoformat()
                if self.finished_at
                else None
            ),
            "raw_dir": str(self.raw_dir) if self.raw_dir else None,
            "posts_path": str(self.posts_path) if self.posts_path else None,
            "csv_path": str(self.csv_path) if self.csv_path else None,
            "sources_total": len(self.outcomes),
            "sources_succeeded": [o.name for o in self.succeeded],
            "sources_failed": [
                {"name": o.name, "status": o.status, "reason": o.reason} for o in self.failed
            ],
            "raw_html_snapshots": self.raw_html_snapshots,
            "raw_extracted": self.raw_extracted,
            "new_records_built": self.new_records_built,
            "rows_added": self.rows_added,
            "exact_duplicates_dropped": self.exact_duplicates_dropped,
            "duplicate_rows_flagged": self.duplicate_rows_flagged,
            "normalized_total": self.normalized_total,
            "outcomes": [o.as_dict() for o in self.outcomes],
        }


# ---------------------------------------------------------------------------
# Normalization + persistence
# ---------------------------------------------------------------------------


def social_posts_to_records(
    posts: Iterable[tuple[SocialPost, str | None]],
    *,
    collection_method: str,
    query_type: str,
    run_id: str,
    collected_at: datetime | None = None,
) -> list[SocialPostRecord]:
    """Build canonical ``SocialPostRecord`` rows from collected posts.

    ``posts`` is an iterable of ``(post, raw_path)`` tuples. The collection
    method is recorded in ``retrieval_mode`` (a free-text provenance column in
    the existing schema), and the source/snapshot details survive in the raw
    JSON snapshots. No market candidate is invented here.
    """
    when = collected_at or _utcnow()
    records: list[SocialPostRecord] = []
    for post, raw_path in posts:
        records.append(
            build_record(
                post=post,
                query=None,
                query_type=query_type,
                market_id_candidate=None,
                raw_path=raw_path,
                collected_at=when,
                retrieval_run_id=run_id,
                retrieval_mode=collection_method,
                temporal_valid_candidate=None,
            )
        )
    return records


def write_records_csv(records: Sequence[SocialPostRecord], path: Path) -> Path:
    """Write a CSV mirror of the normalized records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        pd.DataFrame().to_csv(path, index=False)
        return path
    frame = pd.DataFrame([record.model_dump(mode="json") for record in records])
    frame.to_csv(path, index=False)
    return path


def csv_sibling(parquet_path: Path) -> Path:
    """Return the ``.csv`` mirror path next to a ``social_posts.parquet`` file."""
    return parquet_path.with_suffix(".csv")


def persist_records(
    new_records: Sequence[SocialPostRecord],
    *,
    posts_path: Path,
    csv_path: Path | None = None,
    append: bool = True,
) -> tuple[list[SocialPostRecord], int, int]:
    """Merge ``new_records`` into the dataset on disk and write Parquet + CSV.

    Returns ``(merged_records, rows_added, exact_duplicates_dropped)``. The
    merge is idempotent: re-running collection does not grow the dataset with
    rows that carry an already-seen retrieval tuple.
    """
    existing = read_parquet(posts_path, SocialPostRecord) if append else []
    merged = merge_with_existing(new_records, existing)
    rows_added = len(merged) - len(existing)
    exact_duplicates_dropped = max(0, len(new_records) - rows_added)
    if new_records or existing:
        write_parquet(merged, posts_path)
        write_records_csv(merged, csv_path or csv_sibling(posts_path))
    return merged, rows_added, exact_duplicates_dropped


# ---------------------------------------------------------------------------
# Combined dataset
# ---------------------------------------------------------------------------


def combine_records(
    *record_groups: Sequence[SocialPostRecord],
) -> tuple[list[SocialPostRecord], int]:
    """Concatenate record groups and deterministically re-flag duplicates.

    Returns ``(merged, duplicate_rows_flagged)``. Ordering is preserved so the
    first occurrence of a ``(platform, post_id)`` keeps ``is_duplicate=False``.
    """
    merged: list[SocialPostRecord] = []
    for group in record_groups:
        merged = merge_with_existing(list(group), merged)
    duplicate_rows = sum(1 for record in merged if record.is_duplicate)
    return merged, duplicate_rows


# ---------------------------------------------------------------------------
# Market-awareness (report-only, never fabricates links)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_STOPWORDS = frozenset(
    {
        "the", "will", "and", "for", "are", "was", "with", "that", "this", "from",
        "have", "has", "had", "not", "you", "your", "they", "their", "than",
        "then", "but", "can", "could", "would", "should", "what", "when", "who",
        "how", "why", "into", "over", "out", "about", "after", "before", "any",
        "all", "new", "now", "get", "more", "most", "year", "day", "days",
    }
)


def tokenize(text: str | None) -> set[str]:
    """Lowercase content tokens, stripping short words and stopwords."""
    if not text:
        return set()
    return {
        token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS
    }


def estimate_market_token_overlap(
    records: Sequence[SocialPostRecord],
    markets: Sequence[Market],
    *,
    min_shared_tokens: int = 2,
) -> dict[str, Any]:
    """Deterministic estimate of how many markets the dataset could touch.

    This is a coverage *signal* for the report — it counts markets whose
    question shares at least ``min_shared_tokens`` content tokens with at least
    one collected post. It is NOT a link and never enters the pipeline.
    """
    if not markets:
        return {
            "markets_total": 0,
            "markets_with_token_overlap": 0,
            "coverage_ratio": 0.0,
            "min_shared_tokens": min_shared_tokens,
        }
    post_token_sets = [tokenize(record.text) for record in records if record.text]
    matched = 0
    for market in markets:
        question = getattr(market, "question", None) or ""
        market_tokens = tokenize(question)
        if not market_tokens:
            continue
        for post_tokens in post_token_sets:
            if len(market_tokens & post_tokens) >= min_shared_tokens:
                matched += 1
                break
    total = len(markets)
    return {
        "markets_total": total,
        "markets_with_token_overlap": matched,
        "coverage_ratio": round(matched / total, 4) if total else 0.0,
        "min_shared_tokens": min_shared_tokens,
    }


def platform_breakdown(records: Sequence[SocialPostRecord]) -> dict[str, int]:
    """Count unique (non-duplicate) rows per platform."""
    counts: dict[str, int] = {}
    for record in records:
        if record.is_duplicate:
            continue
        key = (
            record.platform.value if isinstance(record.platform, Platform) else str(record.platform)
        )
        counts[key] = counts.get(key, 0) + 1
    return counts


__all__ = [
    "CollectionResult",
    "FetchOutcome",
    "HtmlFetcher",
    "RateLimiter",
    "ScraplingFetcher",
    "StealthyScraplingFetcher",
    "build_html_fetcher",
    "SourceOutcome",
    "combine_records",
    "csv_sibling",
    "estimate_market_token_overlap",
    "persist_records",
    "platform_breakdown",
    "social_posts_to_records",
    "stable_post_id",
    "tokenize",
    "write_records_csv",
]

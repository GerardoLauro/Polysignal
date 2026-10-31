"""End-to-end social ingestion orchestration.

Branch 04 ties together:

1. reading ``data/normalized/polymarket/markets.parquet`` for the
   candidate market set;
2. generating queries with
   :mod:`polysignal.ingestion.social.query_gen`;
3. dispatching each query to a platform ingestor (currently Reddit;
   direct Telegram channel collection has a dedicated CLI path);
4. persisting raw snapshots under
   ``data/raw/social/<platform>/<run-dir>/``;
5. normalizing into :class:`SocialPostRecord`;
6. deduplicating (idempotent across runs);
7. writing ``data/normalized/social/social_posts.parquet`` and
   ``data/normalized/social/social_queries.parquet``.

The pipeline is deliberately I/O-aware but framework-light: each step
is a small function that can be unit-tested in isolation. The orchestrator
returns a structured :class:`SocialIngestionResult` so the CLI can render
a faithful summary instead of "kept N" headlines.

Dry-run rule
------------
``dry_run=True`` short-circuits *before* any network call: the pipeline
generates queries, persists them to disk (so the planned workload is
auditable), and returns. It does **not** require platform credentials.
This matters because the operator must be able to preview "what would
my queries look like" without standing up a Reddit application first.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from polysignal.config import Settings, get_settings
from polysignal.domain import Market, MarketQuery, Platform, SocialPostRecord
from polysignal.ingestion.social.base import SearchQuery, SocialIngestor
from polysignal.ingestion.social.normalizer import (
    build_record,
    merge_with_existing,
)
from polysignal.ingestion.social.query_gen import generate_queries
from polysignal.ingestion.social.raw_store import SocialRawWriter, generate_social_run_id
from polysignal.storage.parquet_store import (
    read_dataframe,
    read_parquet,
    write_parquet,
)
from polysignal.storage.paths import DataPaths, get_data_paths

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------


@dataclass
class QueryAttempt:
    """One issued query against one platform, with its outcome."""

    market_id: str
    query: str
    query_type: str
    rank: int
    post_count: int = 0
    error: str | None = None


@dataclass
class SocialIngestionResult:
    """Summary returned by :func:`run_social_ingestion`."""

    platform: Platform
    run_id: str
    markets_seen: int = 0
    queries_generated: int = 0
    queries_executed: int = 0
    posts_collected: int = 0
    unique_posts: int = 0
    duplicates: int = 0
    attempts: list[QueryAttempt] = field(default_factory=list)
    queries: list[MarketQuery] = field(default_factory=list)
    records: list[SocialPostRecord] = field(default_factory=list)
    raw_dir: Path | None = None
    social_posts_parquet: Path | None = None
    queries_parquet: Path | None = None


# ---------------------------------------------------------------------------
# Market reading
# ---------------------------------------------------------------------------


def _coerce_market_value(value: Any) -> Any:
    """Pre-Pydantic cleanup for one parquet cell.

    Pandas hands back values in three problematic shapes depending on
    *who* wrote the parquet file and which dtype was inferred:

    * ``NaT`` / ``NaN`` for nullable datetimes and floats — Pydantic
      strict mode rejects these for ``datetime | None`` / ``float | None``.
    * ``numpy.ndarray`` for list-valued cells (``outcomes``, ``tags``) —
      Pydantic accepts a plain list but not all ndarray flavors.
    * ``numpy`` scalar types (``np.bool_``, ``np.int64``) — usually
      accepted by Pydantic v2 but worth normalizing for stable behaviour.

    This helper is *defensive*: it only normalizes the shapes above and
    passes everything else through unchanged.
    """
    if value is None:
        return None
    if value is pd.NaT:
        return None
    # pd.NaT shows up as a NaTType; pd.isna catches both NaT and NaN
    # for scalar inputs. We must *not* call pd.isna on arrays / lists
    # (it returns an elementwise mask that breaks downstream checks).
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        ts = value.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    # Coerce numpy arrays of dicts (outcomes) or strings (tags) to plain
    # Python lists so Pydantic's list validator accepts them. Must run
    # *before* the pd.isna check below — pd.isna on a numpy array
    # returns an element-wise mask, not a scalar.
    if hasattr(value, "tolist") and not isinstance(value, str | bytes | dict):
        try:
            converted = value.tolist()
        except Exception:  # pragma: no cover - defensive
            converted = value
        if converted is not value:
            return _coerce_market_value(converted)
    # Detect pandas / numpy NaT and NaN scalars on the remaining
    # (non-array) values without tripping on arrays or containers.
    if not isinstance(value, list | dict | tuple | set | str | bytes):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {key: _coerce_market_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_coerce_market_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_coerce_market_value(item) for item in value)
    return value


def _coerce_market_list(value: Any, *, field_name: str) -> Any:
    """Normalize a parquet list cell while keeping invalid outcomes visible.

    Depending on the parquet writer, list-valued columns may round-trip
    as arrays, tuples, lists, or JSON strings. Tags also occasionally
    arrive as one plain string; treating that as a single tag is useful.
    A malformed string in ``outcomes`` is intentionally left intact so
    Pydantic rejects and audits the broken market instead of inventing
    outcome rows.
    """
    cleaned = _coerce_market_value(value)
    if cleaned is None:
        return []
    if isinstance(cleaned, str):
        text = cleaned.strip()
        if not text:
            return []
        try:
            cleaned = json.loads(text)
        except json.JSONDecodeError:
            return [text] if field_name == "tags" else cleaned
    if isinstance(cleaned, tuple | set):
        cleaned = list(cleaned)
    if not isinstance(cleaned, list):
        return cleaned
    items: list[Any] = []
    for item in cleaned:
        normalized = _coerce_market_value(item)
        if field_name == "outcomes" and isinstance(normalized, str):
            with suppress(json.JSONDecodeError):
                normalized = _coerce_market_value(json.loads(normalized))
        items.append(normalized)
    return items


def _clean_market_record(row: dict[str, Any]) -> dict[str, Any]:
    """Run :func:`_coerce_market_value` over every cell in one row.

    Nested ``outcomes`` entries are walked one level deep so the inner
    dicts also lose their NaN / numpy types.
    """
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key in {"outcomes", "tags"}:
            cleaned = _coerce_market_list(value, field_name=key)
        else:
            cleaned = _coerce_market_value(value)
        if key == "outcomes" and isinstance(cleaned, list):
            cleaned = [_coerce_market_value(outcome) for outcome in cleaned]
        elif key == "tags" and isinstance(cleaned, list):
            cleaned = [t for t in cleaned if t is not None]
        out[key] = cleaned
    return out


def load_markets_from_parquet(
    markets_path: Path,
    *,
    limit: int | None = None,
) -> list[Market]:
    """Load markets from ``markets.parquet`` with defensive cleaning.

    The cleaning is necessary because the upstream parquet may have
    been written by either of:

    * Branch 03's writer, which serializes datetimes as ISO strings and
      missing optional fields as Python ``None`` (clean);
    * a future / external writer that uses native pandas ``datetime64``
      and stores missing optionals as ``NaT`` / ``NaN`` (needs cleanup).

    We normalize the second shape into the first before handing rows
    to Pydantic. The function returns an empty list when the file is
    empty; it raises ``FileNotFoundError`` only when the file does not
    exist.
    """
    if not markets_path.exists():
        raise FileNotFoundError(
            f"markets parquet not found at {markets_path}. "
            "Run `polysignal polymarket ingest` first to produce it."
        )
    df = read_dataframe(markets_path)
    if df.empty:
        return []
    if limit is not None and limit > 0:
        df = df.head(limit)

    cleaned_records = [_clean_market_record(row) for row in df.to_dict(orient="records")]
    markets: list[Market] = []
    for raw_row in cleaned_records:
        try:
            markets.append(Market.model_validate(raw_row))
        except Exception as exc:
            logger.warning(
                "social: skipping market %r — failed to validate: %s",
                raw_row.get("market_id") or raw_row.get("slug"),
                exc,
            )
            continue
    return markets


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_social_ingestion(
    *,
    ingestor: SocialIngestor,
    markets: Sequence[Market] | None = None,
    markets_path: Path | None = None,
    limit_markets: int | None = 10,
    queries_per_market: int = 6,
    posts_per_query: int = 20,
    save_raw: bool = True,
    dry_run: bool = False,
    since: datetime | None = None,
    until: datetime | None = None,
    settings: Settings | None = None,
    paths: DataPaths | None = None,
    run_id: str | None = None,
    now: datetime | None = None,
) -> SocialIngestionResult:
    """Run an end-to-end social ingestion against a single platform.

    Parameters
    ----------
    ingestor:
        Concrete :class:`SocialIngestor` (real or fake). The pipeline
        does not instantiate platforms itself — the caller decides.
    markets:
        Optional pre-loaded list of markets. When ``None``, the pipeline
        reads ``markets_path`` (or the default
        ``data/normalized/polymarket/markets.parquet``).
    save_raw:
        Persist raw per-post snapshots under
        ``data/raw/social/<platform>/<run-dir>/``.
    dry_run:
        Generate queries and persist ``social_queries.parquet``, but
        **skip every platform API call** and skip the social-posts
        Parquet write. Dry-run never requires platform credentials —
        the operator can preview the planned workload before standing
        up a Reddit application.
    """
    settings = settings or get_settings()
    paths = paths or get_data_paths()
    paths.ensure()
    rid = run_id or generate_social_run_id()
    moment = (now or datetime.now(tz=timezone.utc)).astimezone(timezone.utc)

    result = SocialIngestionResult(platform=ingestor.platform, run_id=rid)

    # Credentials are required for *live* ingestion only. Dry-run plans
    # the workload offline; the operator can preview queries without
    # ever standing up a Reddit application.
    if not dry_run and not ingestor.is_available():
        raise RuntimeError(
            f"Ingestor for {ingestor.platform.value} is not available. "
            "Check the credentials referenced in .env.example, "
            "or pass --dry-run to preview queries without contacting the API."
        )

    if markets is None:
        path = markets_path or paths.polymarket_markets_parquet
        markets = load_markets_from_parquet(path, limit=limit_markets)
    else:
        markets = list(markets)
        if limit_markets is not None and limit_markets > 0:
            markets = markets[:limit_markets]
    result.markets_seen = len(markets)

    queries = generate_queries(
        markets,
        queries_per_market=queries_per_market,
        now=moment,
    )
    result.queries = queries
    result.queries_generated = len(queries)

    # Generated queries are *always* persisted — even when downstream
    # search returns zero posts, or when running with --dry-run. The
    # queries themselves are an artifact worth keeping (an audit log
    # of "what was the system going to ask Reddit on this run").
    if queries:
        write_parquet(list(queries), paths.social_queries_parquet)
        result.queries_parquet = paths.social_queries_parquet

    # Dry-run short-circuits before any platform call. We surface the
    # planned output paths so the CLI can render them honestly.
    if dry_run:
        result.social_posts_parquet = None
        result.raw_dir = None
        result.records = []
        result.unique_posts = 0
        result.duplicates = 0
        return result

    writer: SocialRawWriter | None = None
    if save_raw and queries:
        run_dir = paths.raw_run_dir(f"social/{ingestor.platform.value}", rid, when=moment)
        writer = SocialRawWriter(directory=run_dir, platform=ingestor.platform.value)
        writer.extra.update(
            {
                "limit_markets": limit_markets,
                "queries_per_market": queries_per_market,
                "posts_per_query": posts_per_query,
                "markets_seen": len(markets),
                "since": since.isoformat() if since else None,
                "until": until.isoformat() if until else None,
            }
        )

    new_records: list[SocialPostRecord] = []
    for query in queries:
        attempt = QueryAttempt(
            market_id=query.market_id,
            query=query.query,
            query_type=query.query_type,
            rank=query.rank,
        )
        try:
            posts, raws = _execute_query(
                ingestor,
                query=query,
                posts_per_query=posts_per_query,
                since=since,
                until=until,
            )
        except RuntimeError as exc:
            attempt.error = str(exc)
            result.attempts.append(attempt)
            if writer is not None:
                writer.record_query(
                    query=query.query,
                    market_id=query.market_id,
                    post_count=0,
                    error=str(exc),
                )
            logger.warning(
                "%s: query %r failed: %s",
                ingestor.platform.value,
                query.query,
                exc,
            )
            continue

        if writer is not None:
            for post, raw in zip(posts, raws, strict=False):
                writer.write_post(
                    query=query.query,
                    market_id=query.market_id,
                    post_id=post.post_id,
                    payload=raw,
                )
            writer.record_query(
                query=query.query,
                market_id=query.market_id,
                post_count=len(posts),
            )

        attempt.post_count = len(posts)
        result.attempts.append(attempt)
        result.queries_executed += 1
        result.posts_collected += len(posts)

        for post in posts:
            raw_path = None
            if writer is not None:
                raw_path = str(
                    (
                        writer.directory
                        / _safe_raw_name(query.market_id, post.post_id, query.query)
                    ).resolve()
                )
            new_records.append(
                build_record(
                    post=post,
                    query=query.query,
                    market_id_candidate=query.market_id,
                    raw_path=raw_path,
                    collected_at=moment,
                    query_type=query.query_type,
                )
            )

    if writer is not None:
        writer.write_manifest()
        result.raw_dir = writer.directory

    # Persist social-post records when the merged dataset is non-empty.
    # Generated queries were already written above, including for
    # zero-result runs, so downstream code can inspect an empty search
    # plan without needing an empty schema-less posts parquet.
    existing = read_parquet(paths.social_posts_parquet, SocialPostRecord)
    deduped = merge_with_existing(new_records, existing)
    if new_records or existing:
        write_parquet(deduped, paths.social_posts_parquet)
        result.social_posts_parquet = paths.social_posts_parquet
    result.records = list(deduped)
    result.unique_posts = sum(1 for r in deduped if not r.is_duplicate)
    result.duplicates = sum(1 for r in deduped if r.is_duplicate)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _execute_query(
    ingestor: SocialIngestor,
    *,
    query: MarketQuery,
    posts_per_query: int,
    since: datetime | None,
    until: datetime | None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Run one query through the ingestor and return ``(posts, raws)``.

    ``search_raw`` is preferred when the ingestor exposes it (the
    Reddit ingestor does, so the pipeline can preserve raw snapshots).
    Otherwise we fall back to ``search`` and synthesize an empty raw
    list — the manifest will say so.
    """
    search_query = SearchQuery(
        text=query.query,
        market_id=query.market_id,
        since=since,
        until=until,
        limit=posts_per_query,
    )
    search_raw = getattr(ingestor, "search_raw", None)
    if callable(search_raw):
        posts, raws = search_raw(search_query)
        return list(posts), list(raws)
    posts = list(ingestor.search(search_query))
    return posts, [{} for _ in posts]


def _safe_raw_name(market_id: str, post_id: str, query: str = "") -> str:
    """Return the raw-payload filename for one retrieval event.

    Embedding a short query hash in the filename avoids collisions when
    the same Reddit post is surfaced by multiple queries for the same
    market — each retrieval gets its own file, so query lineage is
    preserved on disk (not only inside the JSON wrapper).
    """

    def _slug(value: str) -> str:
        return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in (value or ""))[:96]

    query_hash = hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:8]
    return f"{_slug(market_id)}__{_slug(post_id)}__{query_hash}.json"


def records_to_dataframe(records: Iterable[SocialPostRecord]) -> pd.DataFrame:
    """Helper used by the CLI inspector. Returns a flat dataframe."""
    rows = [r.model_dump(mode="json") for r in records]
    return pd.DataFrame(rows)


__all__ = [
    "QueryAttempt",
    "SocialIngestionResult",
    "load_markets_from_parquet",
    "records_to_dataframe",
    "run_social_ingestion",
]

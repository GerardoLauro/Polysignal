"""Resumable Branch 12B market-aware public-web social enrichment.

This module deliberately stays inside the responsible public-web boundary:
plain Scrapling fetches, old.reddit.com listings/search, and Telegram public
preview pages. Every fetched page is flushed before its checkpoint advances.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
from pydantic import BaseModel, Field

from polysignal.domain import Market, SocialPost, SocialPostRecord
from polysignal.ingestion.social.enrichment import (
    HtmlFetcher,
    RateLimiter,
    ScraplingFetcher,
    persist_records,
)
from polysignal.ingestion.social.enrichment_config import (
    EnrichmentConfig,
    RedditSource,
)
from polysignal.ingestion.social.enrichment_pipeline import (
    merge_enriched_datasets,
    write_collection_summary,
)
from polysignal.ingestion.social.normalizer import build_record
from polysignal.ingestion.social.query_gen import generate_queries_for_market
from polysignal.ingestion.social.raw_store import SocialRawWriter, generate_social_run_id
from polysignal.ingestion.social.reddit_scrapling_adapter import (
    find_next_page_url,
    parse_old_reddit_listing,
    reddit_raw_to_social_post,
)
from polysignal.ingestion.social.telegram_public_web_adapter import (
    find_before_cursor,
    parse_telegram_public_page,
    telegram_public_raw_to_social_post,
)
from polysignal.storage.parquet_store import read_parquet
from polysignal.storage.paths import DataPaths

_TELEGRAM_PUBLIC_RE = re.compile(r"^https://t\.me/s/[A-Za-z0-9_]+/?$")
_GENERIC_OUTCOMES = {"yes", "no", "true", "false"}
_QUERY_ALIASES = (
    ("bitcoin", "btc"),
    ("ethereum", "eth"),
)
_DOMAIN_SUFFIXES: dict[str, tuple[str, ...]] = {
    "politics": ("election", "poll"),
    "sports": ("odds", "injury"),
    "culture": ("box office", "release"),
    "science": ("technology", "ai"),
    "crypto": ("crypto",),
    "economy": ("markets",),
}
_REDDIT_CIRCUIT_COOLDOWN_SECONDS = 6 * 60 * 60
_OBJECTIVE_TARGETS = {"total": 50_000, "reddit": 35_000, "telegram": 15_000}


def _utc(value: datetime | None = None) -> datetime:
    moment = value or datetime.now(tz=timezone.utc)
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return _utc(value).astimezone(timezone.utc).isoformat()


def _safe_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)[:80]


def _work_id(*parts: Any) -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _config_fingerprint(config: EnrichmentConfig) -> str:
    payload = json.dumps(_jsonable(asdict(config)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonable(item) for item in value]
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write one JSON checkpoint atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON checkpoint or return an empty mapping."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


@dataclass(frozen=True)
class LargeEnrichmentOptions:
    """Runtime limits for one resumable Branch 12B collection."""

    target_total_records: int = 50_000
    target_reddit_records: int = 35_000
    target_telegram_records: int = 15_000
    max_pages_per_source: int = 50
    max_pages_per_query: int = 5
    max_results_per_query: int = 100
    max_markets: int | None = None
    max_queries_per_market: int = 10
    fallback_window_days: int = 180
    resume: bool = True
    stop_when_target_reached: bool = True
    dry_run: bool = False
    rate_limit_seconds: float = 4.0

    def validate(self) -> None:
        if self.target_total_records < 1:
            raise ValueError("target_total_records must be at least 1")
        if min(self.target_reddit_records, self.target_telegram_records) < 0:
            raise ValueError("platform targets cannot be negative")
        if self.max_pages_per_source < 1 or self.max_pages_per_query < 1:
            raise ValueError("page caps must be at least 1")
        if self.max_results_per_query < 1:
            raise ValueError("max_results_per_query must be at least 1")
        if not 1 <= self.max_queries_per_market <= 20:
            raise ValueError("max_queries_per_market must be between 1 and 20")
        if self.fallback_window_days < 1:
            raise ValueError("fallback_window_days must be at least 1")
        if self.rate_limit_seconds < 0:
            raise ValueError("rate_limit_seconds cannot be negative")


class MarketAwareQuery(BaseModel):
    """One deterministic Reddit-search work item derived from a saved market."""

    market_id: str
    condition_id: str | None = None
    question: str
    slug: str
    domain: str
    safe_outcome_terms: list[str] = Field(default_factory=list)
    close_time: datetime
    window_start: datetime
    window_source: str
    generated_query: str
    query_type: str
    selected_sources: list[str] = Field(default_factory=list)
    stable_work_id: str
    generated_at: datetime


@dataclass(frozen=True)
class TemporalClassification:
    relation: str
    leakage_eligible: bool | None
    exclusion_reason: str | None


@dataclass
class LargeEnrichmentResult:
    """Summary returned by :func:`collect_large_enriched`."""

    summary: dict[str, Any]
    plan: list[MarketAwareQuery]
    dry_run: bool


def validate_telegram_public_url(url: str) -> bool:
    """Whether a URL is an allowed public Telegram preview page."""
    return bool(_TELEGRAM_PUBLIC_RE.fullmatch(url.strip()))


def reddit_global_search_url(query: str) -> str:
    """Build one old Reddit global search URL."""
    return f"https://old.reddit.com/search/?q={quote_plus(query)}&sort=new&t=all"


def reddit_restricted_search_url(subreddit: str, query: str) -> str:
    """Build one old Reddit subreddit-restricted search URL."""
    return (
        f"https://old.reddit.com/r/{quote_plus(subreddit)}/search/"
        f"?q={quote_plus(query)}&restrict_sr=on&sort=new&t=all"
    )


def market_window(
    market: Market,
    *,
    fallback_window_days: int = 180,
) -> tuple[datetime, datetime, str] | None:
    """Return a trustworthy pre-close collection window for one market."""
    if market.close_time is None:
        return None
    end = _utc(market.close_time).astimezone(timezone.utc)
    start = _utc(market.created_at).astimezone(timezone.utc)
    source = "market_created_at"
    earliest = end - timedelta(days=fallback_window_days)
    if start >= end or start < earliest:
        start = earliest
        source = f"fallback_close_minus_{fallback_window_days}_days"
    return start, end, source


def classify_temporal_relation(
    created_at: datetime | None,
    *,
    window_start: datetime | None,
    close_time: datetime | None,
    parse_status: str = "parsed",
) -> TemporalClassification:
    """Classify one retrieval relative to its market window."""
    if created_at is None or parse_status != "parsed":
        reason = (
            "post_timestamp_invalid"
            if parse_status in {"invalid", "uncertain"}
            else "post_timestamp_missing"
        )
        return TemporalClassification("missing_timestamp", False, reason)
    if close_time is None:
        return TemporalClassification("unknown", None, "market_timeline_missing")
    moment = _utc(created_at)
    if moment > _utc(close_time):
        return TemporalClassification("post_close", False, "post_close")
    if window_start is not None and moment < _utc(window_start):
        return TemporalClassification("outside_window", False, "outside_window")
    if window_start is None:
        return TemporalClassification("pre_close", True, None)
    return TemporalClassification("inside_window", True, None)


def _safe_outcome_terms(market: Market) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for outcome in market.outcomes:
        label = (outcome.label or "").strip()
        lowered = label.lower()
        if not label or lowered in _GENERIC_OUTCOMES or lowered in seen:
            continue
        seen.add(lowered)
        values.append(label)
    return values[:4]


def _query_variants(query: str, domain: str) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = [(query, "generated")]
    lowered = query.lower()
    for long_name, short_name in _QUERY_ALIASES:
        if long_name in lowered:
            values.append((re.sub(long_name, short_name, query, flags=re.IGNORECASE), "alias"))
        elif re.search(rf"\b{re.escape(short_name)}\b", lowered):
            values.append((re.sub(rf"\b{short_name}\b", long_name, query, flags=re.IGNORECASE), "alias"))
    for suffix in _DOMAIN_SUFFIXES.get(domain, ()):
        if suffix not in lowered:
            values.append((f"{query} {suffix}", "domain_suffix"))
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value, kind in values:
        cleaned = " ".join(value.split())[:96].strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            out.append((cleaned, kind))
    return out


def _selected_reddit_sources(market: Market, sources: Sequence[RedditSource]) -> list[str]:
    tokens = {market.domain.value.lower(), *(tag.lower() for tag in market.tags)}
    ranked: list[tuple[int, str]] = []
    for source in sources:
        overlap = len(tokens & {domain.lower() for domain in source.domains})
        if overlap:
            ranked.append((-overlap, source.name))
    ranked.sort()
    return ["global", *(name for _, name in ranked[:3])]


def build_market_query_plan(
    markets: Sequence[Market],
    reddit_sources: Sequence[RedditSource],
    *,
    max_markets: int | None = None,
    max_queries_per_market: int = 10,
    fallback_window_days: int = 180,
    now: datetime | None = None,
) -> list[MarketAwareQuery]:
    """Build the stable, bounded Reddit-search plan for Branch 12B."""
    if not 1 <= max_queries_per_market <= 20:
        raise ValueError("max_queries_per_market must be between 1 and 20")
    from polysignal.dataset.selection import rank_markets, select_high_signal_markets

    generated_at = _utc(now)
    selected = select_high_signal_markets(
        markets,
        target=max_markets if max_markets is not None else min(130, len(markets)),
        now=generated_at,
        window_days=fallback_window_days,
        resolved_only=False,
        exclude_micro_markets=True,
        include_other=True,
    )
    selected_ids = {market.market_id for market in selected.markets}
    ranked = rank_markets([market for market in markets if market.market_id in selected_ids])
    plan: list[MarketAwareQuery] = []
    for market, _, _, _, _ in ranked:
        window = market_window(market, fallback_window_days=fallback_window_days)
        if window is None:
            continue
        window_start, close_time, window_source = window
        selected_sources = _selected_reddit_sources(market, reddit_sources)
        seen: set[str] = set()
        candidates: list[tuple[str, str]] = []
        for query in generate_queries_for_market(market, max_queries=max_queries_per_market):
            for value, variant in _query_variants(query.query, market.domain.value):
                if value.lower() in seen:
                    continue
                seen.add(value.lower())
                candidates.append((value, f"{query.query_type}:{variant}"))
                if len(candidates) >= max_queries_per_market:
                    break
            if len(candidates) >= max_queries_per_market:
                break
        for value, query_type in candidates:
            plan.append(
                MarketAwareQuery(
                    market_id=market.market_id,
                    condition_id=market.condition_id,
                    question=market.question,
                    slug=market.slug,
                    domain=market.domain.value,
                    safe_outcome_terms=_safe_outcome_terms(market),
                    close_time=close_time,
                    window_start=window_start,
                    window_source=window_source,
                    generated_query=value,
                    query_type=query_type,
                    selected_sources=selected_sources,
                    stable_work_id=_work_id("reddit-query", market.market_id, value),
                    generated_at=generated_at,
                )
            )
    return plan


def write_market_query_plan(plan: Sequence[MarketAwareQuery], paths: DataPaths) -> None:
    """Persist the market-aware plan in Parquet and CSV mirrors."""
    rows = [row.model_dump(mode="json") for row in plan]
    frame = pd.DataFrame(rows)
    paths.market_queries_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(paths.market_queries_parquet, index=False)
    frame.to_csv(paths.market_queries_csv, index=False)


class _Collector:
    """Mutable per-run collector with checkpointed page flushing."""

    def __init__(
        self,
        *,
        config: EnrichmentConfig,
        plan: Sequence[MarketAwareQuery],
        reddit_paths: DataPaths,
        telegram_paths: DataPaths,
        output_paths: DataPaths,
        options: LargeEnrichmentOptions,
        reddit_fetcher: HtmlFetcher | None,
        telegram_fetcher: HtmlFetcher | None,
        now: datetime,
        sleep: Callable[[float], None],
    ) -> None:
        self.config = config
        self.plan = list(plan)
        self.reddit_paths = reddit_paths
        self.telegram_paths = telegram_paths
        self.output_paths = output_paths
        self.options = options
        self.now = now
        self.run_id = generate_social_run_id(now)
        reddit_section = config.reddit
        telegram_section = config.telegram
        user_agent = (
            reddit_section.user_agent
            if reddit_section is not None
            else "PolySignalUniversityResearch/0.1"
        )
        telegram_agent = telegram_section.user_agent if telegram_section is not None else user_agent
        self.reddit_fetcher = reddit_fetcher or ScraplingFetcher(user_agent=user_agent)
        self.telegram_fetcher = telegram_fetcher or ScraplingFetcher(user_agent=telegram_agent)
        self.reddit_limiter = RateLimiter(options.rate_limit_seconds, sleep=sleep)
        self.telegram_limiter = RateLimiter(options.rate_limit_seconds, sleep=sleep)
        self.reddit_writer = SocialRawWriter(
            directory=reddit_paths.raw_run_dir("social/reddit_scrapling", self.run_id, when=now),
            platform="reddit_scrapling",
        )
        self.telegram_writer = SocialRawWriter(
            directory=telegram_paths.raw_run_dir("social/telegram_public_web", self.run_id, when=now),
            platform="telegram_public_web",
        )
        self.source_progress = read_json(output_paths.social_source_progress_json)
        self.query_progress = read_json(output_paths.social_query_progress_json)
        self.state = self._load_state()
        self.reddit_circuit_open = self._reddit_circuit_is_open()
        self.baseline_platform_summaries = self.state.setdefault(
            "baseline_12a_platform_summaries",
            {
                "reddit": read_json(
                    reddit_paths.analytical / "enrichment" / "collection_summary.json"
                ),
                "telegram": read_json(
                    telegram_paths.analytical / "enrichment" / "collection_summary.json"
                ),
            },
        )
        self.reddit_ids = self._existing_ids(reddit_paths)
        self.telegram_ids = self._existing_ids(telegram_paths)
        self.reddit_record_count = self._existing_record_count(reddit_paths)
        self.telegram_record_count = self._existing_record_count(telegram_paths)
        baseline_was_present = "baseline_12a_unique" in self.state
        legacy_baseline = {
            "reddit": self._legacy_unique_total(reddit_paths, len(self.reddit_ids)),
            "telegram": self._legacy_unique_total(telegram_paths, len(self.telegram_ids)),
        }
        self.baseline = self.state.setdefault("baseline_12a_unique", legacy_baseline)
        self.baseline_retrieval_rows = self.state.setdefault(
            "baseline_12a_persisted_retrieval_rows",
            (
                dict(self.baseline)
                if baseline_was_present
                else {
                    "reddit": self.reddit_record_count,
                    "telegram": self.telegram_record_count,
                }
            ),
        )
        self.deduplication = self.state.setdefault("deduplication", {})
        for platform in ("reddit", "telegram"):
            self.deduplication.setdefault(
                platform,
                {
                    "exact_duplicates_dropped": self._estimated_exact_duplicates_dropped(
                        platform
                    ),
                },
            )

    def _reddit_circuit_is_open(self) -> bool:
        circuit = self.state.get("reddit_circuit_breaker")
        if not circuit:
            blocked = [
                row
                for row in self.query_progress.values()
                if isinstance(row, dict)
                and row.get("platform") == "reddit"
                and row.get("reason") in {"http_status=403", "http_status=429"}
            ]
            if blocked:
                circuit = {
                    "opened_at": _iso(),
                    "reason": blocked[-1]["reason"],
                    "cooldown_seconds": _REDDIT_CIRCUIT_COOLDOWN_SECONDS,
                }
                self.state["reddit_circuit_breaker"] = circuit
        if not circuit:
            return False
        try:
            opened_at = datetime.fromisoformat(str(circuit["opened_at"]))
        except (KeyError, TypeError, ValueError):
            return True
        elapsed = (_utc() - _utc(opened_at)).total_seconds()
        if elapsed < int(circuit.get("cooldown_seconds", _REDDIT_CIRCUIT_COOLDOWN_SECONDS)):
            return True
        self.state.pop("reddit_circuit_breaker", None)
        return False

    def _legacy_unique_total(self, paths: DataPaths, default: int) -> int:
        summary = read_json(paths.analytical / "enrichment" / "collection_summary.json")
        return int(summary.get("normalized_total", default))

    def _existing_ids(self, paths: DataPaths) -> set[str]:
        records = read_parquet(paths.social_posts_parquet, SocialPostRecord)
        return {f"{record.platform.value}:{record.post_id}" for record in records}

    def _existing_record_count(self, paths: DataPaths) -> int:
        return len(read_parquet(paths.social_posts_parquet, SocialPostRecord))

    def _estimated_exact_duplicates_dropped(self, platform: str) -> int:
        rows = [
            row
            for progress in (self.source_progress, self.query_progress)
            for row in progress.values()
            if isinstance(row, dict) and row.get("platform") == platform
        ]
        attempted_rows = sum(int(row.get("results", 0)) for row in rows)
        persisted_rows = (
            self.reddit_record_count if platform == "reddit" else self.telegram_record_count
        )
        persisted_additions = max(
            0,
            persisted_rows - int(self.baseline_retrieval_rows.get(platform, 0)),
        )
        return max(0, attempted_rows - persisted_additions)

    def _load_state(self) -> dict[str, Any]:
        previous = read_json(self.output_paths.social_collection_state_json)
        identity = {
            "config_path": str(self.config.path.resolve()),
            "config_sha256": _config_fingerprint(self.config),
            "reddit_root": str(self.reddit_paths.root),
            "telegram_root": str(self.telegram_paths.root),
            "output_root": str(self.output_paths.root),
        }
        if previous and self.options.resume:
            previous_identity = dict(previous.get("identity") or {})
            previous_fingerprint = previous_identity.pop("config_sha256", None)
            comparable_identity = dict(identity)
            comparable_identity.pop("config_sha256")
            if previous_identity != comparable_identity:
                raise ValueError("resume state roots or config do not match this collection")
            if previous_fingerprint and previous_fingerprint != identity["config_sha256"]:
                raise ValueError("resume state source config does not match this collection")
            previous["identity"] = identity
            return previous
        return {
            "version": 1,
            "identity": identity,
            "started_at": _iso(self.now),
            "fetched_urls": [],
            "completed_work_ids": [],
            "stable_post_ids": {"reddit": [], "telegram": []},
        }

    def _checkpoint(self) -> None:
        self.state["updated_at"] = _iso()
        self.state["options"] = asdict(self.options)
        self.state["stable_post_ids"] = {
            "reddit": sorted(self.reddit_ids),
            "telegram": sorted(self.telegram_ids),
        }
        self.state["unique_counts"] = self.unique_counts
        self.state["deduplication"] = self.deduplication
        atomic_write_json(self.output_paths.social_collection_state_json, self.state)
        atomic_write_json(self.output_paths.social_source_progress_json, self.source_progress)
        atomic_write_json(self.output_paths.social_query_progress_json, self.query_progress)
        for paths, platform in (
            (self.reddit_paths, "reddit"),
            (self.telegram_paths, "telegram"),
        ):
            platform_sources = {
                key: value
                for key, value in self.source_progress.items()
                if isinstance(value, dict) and value.get("platform") == platform
            }
            platform_queries = {
                key: value
                for key, value in self.query_progress.items()
                if isinstance(value, dict) and value.get("platform") == platform
            }
            atomic_write_json(
                paths.social_collection_state_json,
                {
                    "version": 1,
                    "platform": platform,
                    "run_id": self.run_id,
                    "updated_at": self.state["updated_at"],
                    "unique_count": self.unique_counts[platform],
                    "stable_post_ids": self.state["stable_post_ids"][platform],
                },
            )
            atomic_write_json(paths.social_source_progress_json, platform_sources)
            atomic_write_json(paths.social_query_progress_json, platform_queries)

    @property
    def unique_counts(self) -> dict[str, int]:
        return {
            "reddit": len(self.reddit_ids),
            "telegram": len(self.telegram_ids),
            "total": len(self.reddit_ids) + len(self.telegram_ids),
        }

    @property
    def effective_reddit_target(self) -> int:
        return max(
            self.options.target_reddit_records,
            self.options.target_total_records - len(self.telegram_ids),
        )

    def _targets_reached(self) -> bool:
        if not self.options.stop_when_target_reached:
            return False
        counts = self.unique_counts
        return (
            counts["total"] >= self.options.target_total_records
            and counts["reddit"] >= self.effective_reddit_target
        )

    def _flush_records(self, records: Sequence[SocialPostRecord], paths: DataPaths) -> None:
        merged, rows_added, exact_duplicates_dropped = persist_records(
            records,
            posts_path=paths.social_posts_parquet,
        )
        platform = "reddit" if paths.root == self.reddit_paths.root else "telegram"
        stats = self.deduplication.setdefault(platform, {})
        stats["exact_duplicates_dropped"] = int(
            stats.get("exact_duplicates_dropped", 0)
        ) + exact_duplicates_dropped
        stats["retrieval_rows_added"] = int(stats.get("retrieval_rows_added", 0)) + rows_added
        stats["last_flush_at"] = _iso()
        target = self.reddit_ids if paths.root == self.reddit_paths.root else self.telegram_ids
        target.update(f"{record.platform.value}:{record.post_id}" for record in merged)
        if platform == "reddit":
            self.reddit_record_count = len(merged)
        else:
            self.telegram_record_count = len(merged)

    def _snapshot(self, writer: SocialRawWriter, work_id: str, page_no: int, html: str) -> str:
        path = writer.directory / f"{_safe_slug(work_id)}__p{page_no:03d}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        return str(path.resolve())

    def _finish_work(
        self,
        work_id: str,
        progress: dict[str, Any],
        status: str,
        *,
        reason: str | None = None,
    ) -> None:
        progress["status"] = status
        progress["finished_at"] = _iso()
        if reason:
            progress["reason"] = reason
        if status in {"completed", "exhausted"} and work_id not in self.state["completed_work_ids"]:
            self.state["completed_work_ids"].append(work_id)
        self._checkpoint()

    def _record(
        self,
        post: SocialPost,
        *,
        raw_path: str,
        query: MarketAwareQuery | None,
        work_id: str,
        mode: str,
    ) -> SocialPostRecord:
        classification = classify_temporal_relation(
            post.created_at,
            window_start=query.window_start if query else None,
            close_time=query.close_time if query else None,
            parse_status=str(post.extra.get("created_at_parse_status") or "parsed"),
        )
        if query is None and post.created_at is not None:
            classification = TemporalClassification("unknown", None, None)
        return build_record(
            post=post,
            query=query.generated_query if query else None,
            query_type=query.query_type if query else "public_source_listing",
            market_id_candidate=query.market_id if query else None,
            market_question_hint=query.question if query else None,
            raw_path=raw_path,
            collected_at=_utc(),
            retrieval_run_id=self.run_id,
            retrieval_work_id=work_id,
            retrieval_window_start=query.window_start if query else None,
            retrieval_window_end=query.close_time if query else None,
            retrieval_lookback_days=(
                (query.close_time - query.window_start).days if query is not None else None
            ),
            retrieval_window_label=query.window_source if query else None,
            retrieval_mode=mode,
            temporal_valid_candidate=classification.leakage_eligible,
            query_provenance=query.generated_query if query else None,
            temporal_relation_to_market=classification.relation,
            leakage_eligible=classification.leakage_eligible,
            exclusion_reason=classification.exclusion_reason,
        )

    def collect_telegram(self) -> None:
        section = self.config.telegram
        if section is None:
            return
        for source in section.sources:
            if len(self.telegram_ids) >= self.options.target_telegram_records or self._targets_reached():
                break
            if not validate_telegram_public_url(source.url):
                self.source_progress[source.name] = {
                    "platform": "telegram",
                    "source": source.name,
                    "status": "invalid_public_url",
                    "url": source.url,
                }
                self._checkpoint()
                continue
            work_id = _work_id("telegram-source", source.url)
            progress = self.source_progress.setdefault(
                work_id,
                {
                    "platform": "telegram",
                    "source": source.name,
                    "pages": 0,
                    "results": 0,
                    "started_at": _iso(),
                    "fetched_urls": [],
                },
            )
            if progress.get("status") == "completed":
                progress["status"] = "capped"
                if work_id in self.state["completed_work_ids"]:
                    self.state["completed_work_ids"].remove(work_id)
            if progress.get("status") in {"exhausted", "blocked"}:
                continue
            cursor = progress.get("cursor")
            for _ in range(self.options.max_pages_per_source - int(progress.get("pages", 0))):
                url = source.url if not cursor else f"{source.url}?before={cursor}"
                if url in self.state["fetched_urls"]:
                    self._finish_work(work_id, progress, "exhausted", reason="page_already_fetched")
                    break
                self.telegram_limiter.wait()
                fetched = self.telegram_fetcher.get(url)
                if not fetched.ok or not fetched.html:
                    self._finish_work(
                        work_id,
                        progress,
                        "blocked",
                        reason=fetched.error or f"http_status={fetched.status}",
                    )
                    break
                page_no = int(progress.get("pages", 0)) + 1
                snapshot = self._snapshot(self.telegram_writer, work_id, page_no, fetched.html)
                records: list[SocialPostRecord] = []
                for raw in parse_telegram_public_page(fetched.html, channel=source.channel):
                    post = telegram_public_raw_to_social_post(
                        raw,
                        source=source,
                        collection_method=section.collection_method,
                        source_url=source.url,
                        html_snapshot_path=snapshot,
                    )
                    if post is None:
                        continue
                    raw_path = self.telegram_writer.write_post("", work_id, post.post_id, raw)
                    records.append(
                        self._record(
                            post,
                            raw_path=str(raw_path),
                            query=None,
                            work_id=work_id,
                            mode=section.collection_method,
                        )
                    )
                self._flush_records(records, self.telegram_paths)
                self.state["fetched_urls"].append(url)
                next_cursor = find_before_cursor(fetched.html)
                progress.setdefault("fetched_urls", []).append(url)
                progress.update(
                    pages=page_no,
                    results=int(progress.get("results", 0)) + len(records),
                    cursor=next_cursor,
                    last_fetched_at=_iso(),
                )
                self._checkpoint()
                if len(self.telegram_ids) >= self.options.target_telegram_records:
                    self._finish_work(
                        work_id,
                        progress,
                        "capped",
                        reason="telegram_target_reached",
                    )
                    break
                if not records or not next_cursor or next_cursor == cursor:
                    reason = (
                        "no_records"
                        if not records
                        else "pagination_missing"
                        if not next_cursor
                        else "pagination_stalled"
                    )
                    self._finish_work(work_id, progress, "exhausted", reason=reason)
                    break
                cursor = next_cursor
            else:
                self._finish_work(work_id, progress, "capped", reason="page_cap_reached")

    def _collect_reddit_pages(
        self,
        *,
        source: RedditSource,
        start_url: str,
        work_id: str,
        progress: dict[str, Any],
        max_pages: int,
        max_results: int,
        query: MarketAwareQuery | None,
    ) -> None:
        section = self.config.reddit
        if progress.get("status") == "completed":
            progress["status"] = "capped"
            if work_id in self.state["completed_work_ids"]:
                self.state["completed_work_ids"].remove(work_id)
        if section is None or progress.get("status") in {"exhausted", "blocked"}:
            return
        url: str | None = progress.get("next_url") or start_url
        for _ in range(max_pages - int(progress.get("pages", 0))):
            if not url or self._targets_reached():
                break
            if url in self.state["fetched_urls"]:
                self._finish_work(work_id, progress, "exhausted", reason="page_already_fetched")
                break
            self.reddit_limiter.wait()
            fetched = self.reddit_fetcher.get(url)
            if not fetched.ok or not fetched.html:
                if fetched.status in {403, 429}:
                    self.state["reddit_circuit_breaker"] = {
                        "opened_at": _iso(),
                        "reason": f"http_status={fetched.status}",
                        "cooldown_seconds": _REDDIT_CIRCUIT_COOLDOWN_SECONDS,
                    }
                    self.reddit_circuit_open = True
                self._finish_work(
                    work_id,
                    progress,
                    "blocked",
                    reason=fetched.error or f"http_status={fetched.status}",
                )
                break
            page_no = int(progress.get("pages", 0)) + 1
            snapshot = self._snapshot(self.reddit_writer, work_id, page_no, fetched.html)
            records: list[SocialPostRecord] = []
            for raw in parse_old_reddit_listing(fetched.html, default_subreddit=source.subreddit):
                if int(progress.get("results", 0)) + len(records) >= max_results:
                    break
                post = reddit_raw_to_social_post(
                    raw,
                    source=source,
                    collection_method=section.collection_method,
                    source_url=start_url,
                    html_snapshot_path=snapshot,
                )
                if post is None:
                    continue
                raw_path = self.reddit_writer.write_post(
                    query.generated_query if query else "",
                    query.market_id if query else work_id,
                    post.post_id,
                    raw,
                    context={"retrieval_work_id": work_id},
                )
                records.append(
                    self._record(
                        post,
                        raw_path=str(raw_path),
                        query=query,
                        work_id=work_id,
                        mode=section.collection_method,
                    )
                )
            self._flush_records(records, self.reddit_paths)
            self.state["fetched_urls"].append(url)
            next_url = find_next_page_url(fetched.html)
            progress.setdefault("fetched_urls", []).append(url)
            progress.update(
                pages=page_no,
                results=int(progress.get("results", 0)) + len(records),
                next_url=next_url,
                last_fetched_at=_iso(),
            )
            self._checkpoint()
            if (
                not records
                or not next_url
                or next_url == url
                or int(progress["results"]) >= max_results
            ):
                reason = (
                    "no_records"
                    if not records
                    else "pagination_missing"
                    if not next_url
                    else "pagination_stalled"
                    if next_url == url
                    else "result_cap_reached"
                )
                self._finish_work(work_id, progress, "exhausted", reason=reason)
                break
            url = next_url
        else:
            self._finish_work(work_id, progress, "capped", reason="page_cap_reached")

    def collect_reddit(self) -> None:
        section = self.config.reddit
        if section is None or self.reddit_circuit_open:
            return
        source_by_name = {source.name: source for source in section.sources}
        for source in section.sources:
            if self._targets_reached() or self.reddit_circuit_open:
                return
            work_id = _work_id("reddit-source", source.url)
            progress = self.source_progress.setdefault(
                work_id,
                {
                    "platform": "reddit",
                    "source": source.name,
                    "pages": 0,
                    "results": 0,
                    "started_at": _iso(),
                    "fetched_urls": [],
                },
            )
            self._collect_reddit_pages(
                source=source,
                start_url=source.url,
                work_id=work_id,
                progress=progress,
                max_pages=self.options.max_pages_per_source,
                max_results=section.max_items_per_source,
                query=None,
            )
        for query in self.plan:
            if self._targets_reached() or self.reddit_circuit_open:
                return
            for selected in query.selected_sources:
                source = source_by_name.get(selected)
                if selected == "global":
                    source = RedditSource(
                        name="reddit_global_search",
                        url="https://old.reddit.com/search/",
                        subreddit="",
                        domains=(query.domain,),
                    )
                    url = reddit_global_search_url(query.generated_query)
                elif source is not None:
                    url = reddit_restricted_search_url(source.subreddit, query.generated_query)
                else:
                    continue
                work_id = _work_id(query.stable_work_id, selected)
                progress = self.query_progress.setdefault(
                    work_id,
                    {
                        "platform": "reddit",
                        "market_id": query.market_id,
                        "query": query.generated_query,
                        "source": selected,
                        "pages": 0,
                        "results": 0,
                        "started_at": _iso(),
                        "fetched_urls": [],
                    },
                )
                self._collect_reddit_pages(
                    source=source,
                    start_url=url,
                    work_id=work_id,
                    progress=progress,
                    max_pages=self.options.max_pages_per_query,
                    max_results=self.options.max_results_per_query,
                    query=query,
                )

    def finish(self) -> dict[str, Any]:
        self.reddit_writer.write_manifest()
        self.telegram_writer.write_manifest()
        self._checkpoint()
        merge = merge_enriched_datasets(
            reddit_paths=self.reddit_paths,
            telegram_paths=self.telegram_paths,
            output_paths=self.output_paths,
        )
        merge_summary = merge.summary()
        merge_path = self.output_paths.analytical / "enrichment" / "merge_summary.json"
        atomic_write_json(merge_path, merge_summary)
        for paths, platform in (
            (self.reddit_paths, "reddit"),
            (self.telegram_paths, "telegram"),
        ):
            platform_summary = self._platform_summary(paths, platform)
            atomic_write_json(paths.social_collection_summary_json, platform_summary)
            write_collection_summary(platform_summary, paths)
        records = read_parquet(self.output_paths.social_posts_parquet, SocialPostRecord)
        counts = self.unique_counts
        stage_reached = (
            counts["total"] >= self.options.target_total_records
            and counts["reddit"] >= self.effective_reddit_target
        )
        objective_reached = (
            counts["total"] >= _OBJECTIVE_TARGETS["total"]
            and counts["reddit"]
            >= max(_OBJECTIVE_TARGETS["reddit"], _OBJECTIVE_TARGETS["total"] - counts["telegram"])
        )
        progress = self._collection_progress()
        usefulness = self._usefulness_metrics(records)
        circuit_breakers = {"reddit": self.state.get("reddit_circuit_breaker")}
        summary = {
            "branch": "12B",
            "run_id": self.run_id,
            "generated_at": _iso(),
            "baseline_12a_unique": self.baseline,
            "final_unique": counts,
            "added_12b_unique": {
                "reddit": counts["reddit"] - self.baseline["reddit"],
                "telegram": counts["telegram"] - self.baseline["telegram"],
                "total": counts["total"] - sum(self.baseline.values()),
            },
            "persisted_retrieval_rows": len(records),
            "deduplication": self.deduplication,
            "objective_targets": {
                **_OBJECTIVE_TARGETS,
                "reached": objective_reached,
                "remaining_total": max(0, _OBJECTIVE_TARGETS["total"] - counts["total"]),
                "progress_ratio": round(counts["total"] / _OBJECTIVE_TARGETS["total"], 4),
            },
            "targets": {
                **asdict(self.options),
                "effective_reddit_target": self.effective_reddit_target,
                "telegram_shortfall_shifted_to_reddit": max(
                    0, self.options.target_telegram_records - counts["telegram"]
                ),
                "reached": stage_reached,
            },
            "fetched_pages": len(self.state["fetched_urls"]),
            "collection_progress": progress,
            "source_progress": self.source_progress,
            "query_progress": self.query_progress,
            "market_query_artifacts": {
                "parquet": str(self.output_paths.market_queries_parquet),
                "csv": str(self.output_paths.market_queries_csv),
                "rows": len(self.plan),
            },
            "usefulness": usefulness,
            "temporal_relation_counts": usefulness["temporal_relation_counts"],
            "circuit_breakers": circuit_breakers,
            "completion": self._completion_status(
                stage_reached=stage_reached,
                objective_reached=objective_reached,
                progress=progress,
                circuit_breakers=circuit_breakers,
            ),
            "output_paths": {
                "combined_parquet": str(self.output_paths.social_posts_parquet),
                "combined_csv": str(self.output_paths.social_posts_parquet.with_suffix(".csv")),
                "reddit_parquet": str(self.reddit_paths.social_posts_parquet),
                "reddit_csv": str(self.reddit_paths.social_posts_parquet.with_suffix(".csv")),
                "telegram_parquet": str(self.telegram_paths.social_posts_parquet),
                "telegram_csv": str(
                    self.telegram_paths.social_posts_parquet.with_suffix(".csv")
                ),
                "market_queries_parquet": str(self.output_paths.market_queries_parquet),
                "market_queries_csv": str(self.output_paths.market_queries_csv),
                "collection_state": str(self.output_paths.social_collection_state_json),
                "source_progress": str(self.output_paths.social_source_progress_json),
                "query_progress": str(self.output_paths.social_query_progress_json),
                "collection_summary": str(self.output_paths.social_collection_summary_json),
            },
            "merge": merge_summary,
        }
        atomic_write_json(self.output_paths.social_collection_summary_json, summary)
        write_collection_summary(summary, self.output_paths)
        return summary

    def _collection_progress(self) -> dict[str, Any]:
        source_rows = [row for row in self.source_progress.values() if isinstance(row, dict)]
        query_rows = [row for row in self.query_progress.values() if isinstance(row, dict)]

        def _status_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
            return dict(sorted(Counter(str(row.get("status") or "pending") for row in rows).items()))

        def _platform(platform: str) -> dict[str, Any]:
            sources = [row for row in source_rows if row.get("platform") == platform]
            queries = [row for row in query_rows if row.get("platform") == platform]
            return {
                "source_work_items": len(sources),
                "source_status_counts": _status_counts(sources),
                "source_pages_fetched": sum(int(row.get("pages", 0)) for row in sources),
                "source_rows_extracted": sum(int(row.get("results", 0)) for row in sources),
                "query_work_items": len(queries),
                "query_status_counts": _status_counts(queries),
                "query_pages_fetched": sum(int(row.get("pages", 0)) for row in queries),
                "query_rows_extracted": sum(int(row.get("results", 0)) for row in queries),
                "failed_sources": sorted(
                    {
                        str(row.get("source") or "unknown")
                        for row in sources
                        if row.get("status") in {"blocked", "invalid_public_url"}
                    }
                ),
                "exhausted_sources": sorted(
                    {
                        str(row.get("source") or "unknown")
                        for row in sources
                        if row.get("status") == "exhausted"
                    }
                ),
            }

        return {
            "reddit": _platform("reddit"),
            "telegram": _platform("telegram"),
            "fetched_urls": len(self.state["fetched_urls"]),
            "completed_work_ids": len(self.state["completed_work_ids"]),
        }

    def _usefulness_metrics(self, records: Sequence[SocialPostRecord]) -> dict[str, Any]:
        parse_status = Counter(record.created_at_parse_status or "unknown" for record in records)
        temporal = Counter(record.temporal_relation_to_market or "unspecified" for record in records)
        return {
            "timestamp_parse_status_counts": dict(sorted(parse_status.items())),
            "timestamp_parsed": sum(
                1
                for record in records
                if record.created_at is not None and record.created_at_parse_status == "parsed"
            ),
            "timestamp_missing_or_uncertain": sum(
                1
                for record in records
                if record.created_at is None
                or record.created_at_parse_status in {"missing", "invalid", "uncertain"}
            ),
            "market_aware_rows": sum(1 for record in records if record.market_id_hint),
            "general_context_rows": sum(1 for record in records if not record.market_id_hint),
            "pre_close_eligible": sum(1 for record in records if record.leakage_eligible is True),
            "post_close_excluded": sum(
                1 for record in records if record.exclusion_reason == "post_close"
            ),
            "outside_window_excluded": sum(
                1 for record in records if record.exclusion_reason == "outside_window"
            ),
            "missing_timestamp_excluded": sum(
                1
                for record in records
                if record.exclusion_reason in {"post_timestamp_missing", "post_timestamp_invalid"}
            ),
            "temporal_relation_counts": dict(sorted(temporal.items())),
        }

    def _completion_status(
        self,
        *,
        stage_reached: bool,
        objective_reached: bool,
        progress: dict[str, Any],
        circuit_breakers: dict[str, Any],
    ) -> dict[str, Any]:
        reddit_circuit = circuit_breakers.get("reddit")
        resume_after: str | None = None
        if reddit_circuit:
            opened_at = datetime.fromisoformat(str(reddit_circuit["opened_at"]))
            resume_after = _iso(
                _utc(opened_at)
                + timedelta(
                    seconds=int(
                        reddit_circuit.get(
                            "cooldown_seconds",
                            _REDDIT_CIRCUIT_COOLDOWN_SECONDS,
                        )
                    )
                )
            )
        pending_queries = int(
            progress["reddit"]["query_status_counts"].get("pending", 0)
        )
        if objective_reached:
            reason = "objective_target_reached"
        elif reddit_circuit:
            reason = "reddit_circuit_breaker"
        elif stage_reached:
            reason = "staged_target_reached"
        elif pending_queries:
            reason = "resume_available"
        else:
            reason = "configured_work_exhausted"
        return {
            "objective_reached": objective_reached,
            "stage_reached": stage_reached,
            "reason": reason,
            "resume_after": resume_after,
            "next_steps": (
                "No additional collection is required."
                if objective_reached
                else (
                    "Resume collection after the Reddit cooldown expires."
                    if reddit_circuit
                    else (
                        "Resume with the next staged target."
                        if stage_reached
                        else "Resume with higher page caps or add compliant public sources."
                    )
                )
            ),
        }

    def _platform_summary(self, paths: DataPaths, platform: str) -> dict[str, Any]:
        records = read_parquet(paths.social_posts_parquet, SocialPostRecord)
        previous = self.baseline_platform_summaries.get(platform, {})
        source_rows = [
            progress
            for progress in self.source_progress.values()
            if isinstance(progress, dict) and progress.get("platform") == platform
        ]
        query_rows = (
            [
                progress
                for progress in self.query_progress.values()
                if isinstance(progress, dict) and progress.get("platform") == platform
            ]
            if platform == "reddit"
            else []
        )
        failed = [
            *(previous.get("sources_failed") or []),
            *[
                {
                    "name": row.get("source", "unknown"),
                    "status": row.get("status", "unknown"),
                    "reason": row.get("reason"),
                }
                for row in source_rows
                if row.get("status") in {"blocked", "invalid_public_url"}
            ],
        ]
        failed_names = {str(row.get("name") or "") for row in failed}
        succeeded = sorted(
            {
                *(previous.get("sources_succeeded") or []),
                *(
                    str(row["source"])
                    for row in source_rows
                    if row.get("source")
                    and row.get("status") not in {"blocked", "invalid_public_url"}
                ),
            }
            - failed_names
        )
        unique_total = len(self.reddit_ids if platform == "reddit" else self.telegram_ids)
        rows_added = unique_total - int(self.baseline[platform])
        return {
            "branch": "12B",
            "platform": platform,
            "collection_method": "large_market_aware_public_web",
            "run_id": self.run_id,
            "started_at": _iso(self.now),
            "finished_at": _iso(),
            "raw_dir": str(
                self.reddit_writer.directory
                if platform == "reddit"
                else self.telegram_writer.directory
            ),
            "posts_path": str(paths.social_posts_parquet),
            "csv_path": str(paths.social_posts_parquet.with_suffix(".csv")),
            "sources_succeeded": succeeded,
            "sources_failed": failed,
            "raw_html_snapshots": int(previous.get("raw_html_snapshots", 0))
            + sum(int(row.get("pages", 0)) for row in [*source_rows, *query_rows]),
            "raw_extracted": int(previous.get("raw_extracted", 0))
            + sum(int(row.get("results", 0)) for row in [*source_rows, *query_rows]),
            "new_records_built": max(0, rows_added),
            "rows_added": max(0, rows_added),
            "exact_duplicates_dropped": int(
                self.deduplication.get(platform, {}).get("exact_duplicates_dropped", 0)
            ),
            "duplicate_rows_flagged": sum(1 for record in records if record.is_duplicate),
            "normalized_total": unique_total,
        }


def collect_large_enriched(
    *,
    config: EnrichmentConfig,
    markets: Sequence[Market],
    reddit_paths: DataPaths,
    telegram_paths: DataPaths,
    output_paths: DataPaths,
    options: LargeEnrichmentOptions | None = None,
    reddit_fetcher: HtmlFetcher | None = None,
    telegram_fetcher: HtmlFetcher | None = None,
    now: datetime | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> LargeEnrichmentResult:
    """Run or preview the resumable Branch 12B collector."""
    resolved = options or LargeEnrichmentOptions()
    resolved.validate()
    moment = _utc(now)
    reddit_sources = config.reddit.sources if config.reddit is not None else ()
    plan = build_market_query_plan(
        markets,
        reddit_sources,
        max_markets=resolved.max_markets,
        max_queries_per_market=resolved.max_queries_per_market,
        fallback_window_days=resolved.fallback_window_days,
        now=moment,
    )
    preview = {
        "branch": "12B",
        "dry_run": True,
        "markets_input": len(markets),
        "market_queries": len(plan),
        "reddit_sources": len(reddit_sources),
        "telegram_sources": len(config.telegram.sources) if config.telegram else 0,
        "targets": asdict(resolved),
    }
    if resolved.dry_run:
        return LargeEnrichmentResult(summary=preview, plan=plan, dry_run=True)
    write_market_query_plan(plan, output_paths)
    collector = _Collector(
        config=config,
        plan=plan,
        reddit_paths=reddit_paths,
        telegram_paths=telegram_paths,
        output_paths=output_paths,
        options=resolved,
        reddit_fetcher=reddit_fetcher,
        telegram_fetcher=telegram_fetcher,
        now=moment,
        sleep=sleep,
    )
    collector.collect_telegram()
    collector.collect_reddit()
    return LargeEnrichmentResult(summary=collector.finish(), plan=plan, dry_run=False)


__all__ = [
    "LargeEnrichmentOptions",
    "LargeEnrichmentResult",
    "MarketAwareQuery",
    "TemporalClassification",
    "atomic_write_json",
    "build_market_query_plan",
    "classify_temporal_relation",
    "collect_large_enriched",
    "market_window",
    "read_json",
    "reddit_global_search_url",
    "reddit_restricted_search_url",
    "validate_telegram_public_url",
    "write_market_query_plan",
]

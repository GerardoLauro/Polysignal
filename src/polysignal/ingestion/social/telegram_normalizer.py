"""Raw persistence, config loading, and parquet normalization for Telegram."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from polysignal.domain import MarketQuery, Platform, SocialPostRecord
from polysignal.ingestion.social.base import SearchQuery
from polysignal.ingestion.social.normalizer import build_record, merge_with_existing
from polysignal.ingestion.social.raw_store import SocialRawWriter, generate_social_run_id
from polysignal.ingestion.social.telegram import (
    TelegramAPIError,
    TelegramIngestor,
    raw_message_to_social_post,
)
from polysignal.storage.parquet_store import (
    CorruptParquetError,
    backup_parquet,
    quarantine_corrupt_parquet,
    read_parquet,
    read_parquet_strict,
    safe_write_parquet,
)
from polysignal.storage.paths import DataPaths, get_data_paths

if TYPE_CHECKING:
    from polysignal.dataset.models import TelegramSearchPlanEntry


@dataclass(frozen=True)
class TelegramChannel:
    username: str
    name: str | None = None
    domain: str | None = None
    domain_tags: tuple[str, ...] = ()
    language: str | None = None
    limit: int | None = None  # per-channel message cap (overrides the global limit)
    access_status: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        """Keep the legacy primary-domain field aligned with rich tags."""
        if self.domain and not self.domain_tags:
            object.__setattr__(self, "domain_tags", (self.domain,))
        elif not self.domain and self.domain_tags:
            object.__setattr__(self, "domain", self.domain_tags[0])


@dataclass
class TelegramFetchResult:
    run_id: str
    channels: list[str] = field(default_factory=list)
    posts_collected: int = 0
    raw_dir: Path | None = None
    raw_paths: list[Path] = field(default_factory=list)
    # Per-channel outcomes — honest provenance of what was actually read.
    succeeded: list[tuple[str, int]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    date_ranges: dict[str, tuple[datetime, datetime]] = field(default_factory=dict)
    manifest_path: Path | None = None


@dataclass
class TelegramMarketSearchResult:
    """Outcome envelope for a targeted market-window Telegram run."""

    run_id: str
    work_items_attempted: int = 0
    work_items_completed: int = 0
    work_items_empty: int = 0
    work_items_skipped: int = 0
    work_items_failed: int = 0
    posts_collected: int = 0
    raw_dir: Path | None = None
    manifest_path: Path | None = None
    raw_manifest_path: Path | None = None
    outcomes: list[dict[str, object]] = field(default_factory=list)


def load_telegram_channels(path: Path) -> list[TelegramChannel]:
    """Load enabled explicit Telegram targets from a YAML file.

    Recognised per-channel keys include the legacy ``username`` / ``name`` /
    ``domain`` shape and the richer ``handle`` / ``source_name`` /
    ``domain_tags`` / ``access_status`` aliases.
    """
    if not path.exists():
        hint = ""
        default = Path("config/telegram_channels.yml")
        if path != default and default.exists():
            hint = (
                f" Did you mean '{default}'? The directory is 'config/' (singular), "
                "not 'configs/'."
            )
        raise FileNotFoundError(f"Telegram channels config not found at {path}.{hint}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    channels = payload.get("channels")
    if not isinstance(channels, list):
        raise ValueError("Telegram channels config must contain a `channels` list.")
    out: list[TelegramChannel] = []
    for row in channels:
        if not isinstance(row, dict) or not row.get("enabled", True):
            continue
        username = str(row.get("handle") or row.get("username") or "").strip().removeprefix("@")
        if not username:
            raise ValueError("Each enabled Telegram channel requires `handle` or `username`.")
        raw_limit = row.get("limit")
        raw_domains = row.get("domain_tags", row.get("domain", []))
        if isinstance(raw_domains, str):
            raw_domains = [raw_domains]
        if not isinstance(raw_domains, list):
            raise ValueError(f"Telegram channel {username!r} requires list-valued `domain_tags`.")
        out.append(
            TelegramChannel(
                username=username,
                name=str(row.get("source_name") or row.get("name") or "") or None,
                domain=str(raw_domains[0]) if raw_domains else None,
                domain_tags=tuple(str(value) for value in raw_domains if str(value).strip()),
                language=str(row.get("language") or "") or None,
                limit=int(raw_limit) if raw_limit not in (None, "") else None,
                access_status=str(row.get("access_status") or row.get("access") or "") or None,
                notes=str(row.get("notes") or "") or None,
            )
        )
    return out


def fetch_telegram_channels(
    channels: list[TelegramChannel],
    *,
    ingestor: TelegramIngestor | None = None,
    query: str | None = None,
    limit: int = 200,
    max_total: int | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    dry_run: bool = False,
    paths: DataPaths | None = None,
    run_id: str | None = None,
    now: datetime | None = None,
    config_path: Path | None = None,
) -> TelegramFetchResult:
    """Fetch explicit channels into one run-scoped raw directory.

    Each channel is fetched independently: a channel that is inaccessible,
    private, rate-limited, empty, or unauthorized is recorded in
    ``result.skipped`` with its reason and the run continues. Successful
    channels and their message counts are recorded in ``result.succeeded``.
    A per-channel ``limit`` overrides the global one; ``max_total`` caps the
    whole run. Nothing is fabricated — only messages actually read are saved.
    """
    paths = paths or get_data_paths()
    rid = run_id or generate_social_run_id()
    result = TelegramFetchResult(run_id=rid, channels=[channel.username for channel in channels])
    if dry_run:
        return result

    moment = (now or datetime.now(tz=timezone.utc)).astimezone(timezone.utc)
    run_dir = paths.raw_run_dir("social/telegram", rid, when=moment)
    writer = SocialRawWriter(directory=run_dir, platform="telegram")
    writer.extra["channels"] = result.channels
    writer.extra["limit_per_channel"] = limit
    writer.extra["config_path"] = str(config_path) if config_path is not None else None
    writer.extra["run_timestamp"] = moment
    client = ingestor or TelegramIngestor()
    try:
        for channel in channels:
            if max_total is not None and result.posts_collected >= max_total:
                result.skipped.append((channel.username, "max_total reached"))
                continue
            channel_limit = channel.limit or limit
            if max_total is not None:
                channel_limit = min(channel_limit, max_total - result.posts_collected)
            try:
                posts, raws = client.fetch_channel_raw(
                    channel.username,
                    SearchQuery(
                        text=query or "", since=since, until=until, limit=channel_limit
                    ),
                )
            except TelegramAPIError as exc:
                # Per-channel access failure (private / not found / rate-limited /
                # not a broadcast). Record the reason and keep going. A global
                # config error (missing credentials) raises RuntimeError from the
                # client and is intentionally NOT caught here — it must fail loudly.
                reason = f"{type(exc).__name__}: {exc}"[:200]
                result.skipped.append((channel.username, reason))
                writer.record_query(
                    query or "",
                    f"telegram-channel-{channel.username}",
                    0,
                    error=reason,
                )
                continue
            raw_paths: list[Path] = []
            for post, raw in zip(posts, raws, strict=True):
                raw["source_config"] = {
                    "source_name": channel.name,
                    "domain_tags": list(channel.domain_tags),
                    "language": channel.language,
                    "access_status": channel.access_status,
                    "notes": channel.notes,
                }
                raw_paths.append(
                    writer.write_post(
                        query=query or "",
                        market_id=f"telegram-channel-{channel.username}",
                        post_id=post.post_id,
                        payload=raw,
                    )
                )
            result.posts_collected += len(posts)
            result.raw_paths.extend(raw_paths)
            if posts:
                result.succeeded.append((channel.username, len(posts)))
                result.date_ranges[channel.username] = (
                    min(post.created_at for post in posts),
                    max(post.created_at for post in posts),
                )
                writer.record_query(query or "", f"telegram-channel-{channel.username}", len(posts))
            else:
                reason = "empty: no usable messages returned"
                result.skipped.append((channel.username, reason))
                writer.record_query(
                    query or "",
                    f"telegram-channel-{channel.username}",
                    0,
                    error=reason,
                )
    finally:
        client.close()
    writer.extra["channels_attempted"] = result.channels
    writer.extra["channels_succeeded"] = [
        {
            "handle": username,
            "messages_collected": count,
            "date_range": {
                "start": result.date_ranges[username][0],
                "end": result.date_ranges[username][1],
            },
        }
        for username, count in result.succeeded
    ]
    writer.extra["channels_skipped"] = [
        {"handle": username, "reason": reason} for username, reason in result.skipped
    ]
    writer.extra["total_messages"] = result.posts_collected
    writer.extra["dedupe_count"] = writer.duplicate_writes
    result.manifest_path = writer.write_manifest()
    result.raw_dir = run_dir
    return result


def _entry_work_id(entry: TelegramSearchPlanEntry) -> str:
    if entry.work_id:
        return entry.work_id
    payload = "|".join(
        (
            entry.market_id,
            entry.channel_id or "<unassigned>",
            entry.query,
            entry.search_window_start.astimezone(timezone.utc).isoformat(),
            entry.search_window_end.astimezone(timezone.utc).isoformat(),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _targeted_context(
    entry: TelegramSearchPlanEntry,
    *,
    run_id: str,
    retrieval_mode: str = "server_side_search",
) -> dict[str, object]:
    return {
        "retrieval_run_id": run_id,
        "retrieval_work_id": _entry_work_id(entry),
        "query_type": entry.query_type,
        "retrieval_window_start": entry.search_window_start,
        "retrieval_window_end": entry.search_window_end,
        "retrieval_lookback_days": entry.lookback_days,
        "retrieval_window_label": entry.window_label,
        "retrieval_mode": retrieval_mode,
        "temporal_valid_candidate": True,
    }


def _read_search_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"runs": [], "work_items": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"runs": [], "work_items": {}}
    payload.setdefault("runs", [])
    payload.setdefault("work_items", {})
    return payload


def _write_search_manifest(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _is_inaccessible_channel_error(reason: str) -> bool:
    lowered = reason.lower()
    return any(
        marker in lowered
        for marker in (
            "channel specified is private",
            "lack permission",
            "username not occupied",
            "could not find the input entity",
        )
    )


def search_telegram_market_plan(
    entries: Sequence[TelegramSearchPlanEntry],
    channels: Sequence[TelegramChannel],
    *,
    ingestor: TelegramIngestor | None = None,
    limit_per_query: int = 100,
    max_markets: int | None = None,
    resume: bool = True,
    dry_run: bool = False,
    paths: DataPaths | None = None,
    run_id: str | None = None,
    now: datetime | None = None,
) -> TelegramMarketSearchResult:
    """Execute persisted market-window rows with Telegram server-side search."""
    paths = paths or get_data_paths()
    rid = run_id or generate_social_run_id()
    result = TelegramMarketSearchResult(run_id=rid)
    channel_by_username = {channel.username.lower(): channel for channel in channels}
    market_ids: list[str] = []
    for entry in entries:
        if entry.market_id not in market_ids:
            market_ids.append(entry.market_id)
    allowed_markets = set(market_ids[:max_markets] if max_markets else market_ids)
    selected = [entry for entry in entries if entry.market_id in allowed_markets]
    if dry_run:
        result.work_items_attempted = len(selected)
        return result

    moment = (now or datetime.now(tz=timezone.utc)).astimezone(timezone.utc)
    run_dir = paths.raw_run_dir("social/telegram", rid, when=moment)
    writer = SocialRawWriter(directory=run_dir, platform="telegram")
    status_path = paths.telegram_search_manifest_json
    manifest = _read_search_manifest(status_path)
    prior_items = manifest.get("work_items")
    if not isinstance(prior_items, dict):
        prior_items = {}
        manifest["work_items"] = prior_items
    blocked_channels: dict[str, str] = {}

    def checkpoint() -> None:
        if len(result.outcomes) % 25 == 0:
            _write_search_manifest(manifest, status_path)

    client = ingestor or TelegramIngestor()
    try:
        for entry in selected:
            work_id = _entry_work_id(entry)
            prior = prior_items.get(work_id)
            if (
                resume
                and isinstance(prior, dict)
                and prior.get("status") in {"completed", "empty"}
            ):
                result.work_items_skipped += 1
                continue
            context = _targeted_context(entry, run_id=rid)
            base_outcome: dict[str, object] = {
                "work_id": work_id,
                "market_id": entry.market_id,
                "channel_id": entry.channel_id,
                "query": entry.query,
                "query_type": entry.query_type,
                "window_start": entry.search_window_start.isoformat(),
                "window_end": entry.search_window_end.isoformat(),
                "lookback_days": entry.lookback_days,
                "window_label": entry.window_label,
                "retrieval_mode": "server_side_search",
                "run_id": rid,
            }
            if entry.channel_id is None:
                outcome = {**base_outcome, "status": "skipped", "reason": "unassigned channel"}
                result.work_items_skipped += 1
                result.outcomes.append(outcome)
                prior_items[work_id] = outcome
                checkpoint()
                continue
            channel = channel_by_username.get(entry.channel_id.lower())
            if channel is None:
                outcome = {
                    **base_outcome,
                    "status": "skipped",
                    "reason": "channel absent from enabled config",
                }
                result.work_items_skipped += 1
                result.outcomes.append(outcome)
                prior_items[work_id] = outcome
                checkpoint()
                continue
            blocked_reason = blocked_channels.get(channel.username.lower())
            if blocked_reason is not None:
                outcome = {
                    **base_outcome,
                    "status": "skipped",
                    "reason": f"known inaccessible channel: {blocked_reason}",
                }
                result.work_items_skipped += 1
                result.outcomes.append(outcome)
                prior_items[work_id] = outcome
                checkpoint()
                continue
            result.work_items_attempted += 1
            try:
                posts, raws = client.fetch_channel_raw(
                    channel.username,
                    SearchQuery(
                        text=entry.query,
                        market_id=entry.market_id,
                        since=entry.search_window_start,
                        until=entry.search_window_end,
                        limit=limit_per_query,
                    ),
                )
            except TelegramAPIError as exc:
                reason = f"{type(exc).__name__}: {exc}"[:500]
                outcome = {**base_outcome, "status": "error", "reason": reason}
                result.work_items_failed += 1
                result.outcomes.append(outcome)
                prior_items[work_id] = outcome
                if _is_inaccessible_channel_error(reason):
                    blocked_channels[channel.username.lower()] = reason
                writer.record_query(entry.query, entry.market_id, 0, error=reason, context=context)
                checkpoint()
                continue
            for post, raw in zip(posts, raws, strict=True):
                raw["source_config"] = {
                    "source_name": channel.name,
                    "domain_tags": list(channel.domain_tags),
                    "language": channel.language,
                    "access_status": channel.access_status,
                    "notes": channel.notes,
                }
                writer.write_post(
                    entry.query,
                    entry.market_id,
                    post.post_id,
                    raw,
                    context=context,
                )
            count = len(posts)
            status = "completed" if count else "empty"
            outcome = {**base_outcome, "status": status, "post_count": count}
            result.outcomes.append(outcome)
            prior_items[work_id] = outcome
            writer.record_query(entry.query, entry.market_id, count, context=context)
            result.posts_collected += count
            if count:
                result.work_items_completed += 1
            else:
                result.work_items_empty += 1
            checkpoint()
    finally:
        client.close()
    writer.extra["targeted_market_search"] = True
    writer.extra["work_items_attempted"] = result.work_items_attempted
    writer.extra["work_items_completed"] = result.work_items_completed
    writer.extra["work_items_empty"] = result.work_items_empty
    writer.extra["work_items_skipped"] = result.work_items_skipped
    writer.extra["work_items_failed"] = result.work_items_failed
    writer.extra["posts_collected"] = result.posts_collected
    result.raw_manifest_path = writer.write_manifest()
    result.raw_dir = run_dir
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        runs = []
        manifest["runs"] = runs
    runs.append(
        {
            "run_id": rid,
            "created_at": moment.isoformat(),
            "raw_dir": str(run_dir),
            "work_items_attempted": result.work_items_attempted,
            "work_items_completed": result.work_items_completed,
            "work_items_empty": result.work_items_empty,
            "work_items_skipped": result.work_items_skipped,
            "work_items_failed": result.work_items_failed,
            "posts_collected": result.posts_collected,
        }
    )
    _write_search_manifest(manifest, status_path)
    result.manifest_path = status_path
    return result


def _records_from_raw_dir(
    raw_dir: Path,
    *,
    channel_by_username: dict[str, TelegramChannel],
    provenance: dict[str | None, MarketQuery],
    require_text: bool,
) -> list[SocialPostRecord]:
    """Parse every message JSON in one raw run dir into ``SocialPostRecord`` rows.

    Pure of the parquet — used by both the incremental normalizer and the
    raw-chunk recovery path so they agree on field mapping byte-for-byte.
    """
    records: list[SocialPostRecord] = []
    for raw_path in sorted(raw_dir.glob("*.json")):
        if raw_path.name == "_manifest.json":
            continue
        wrapper = json.loads(raw_path.read_text(encoding="utf-8"))
        payload = wrapper.get("payload") or {}
        channel_payload = payload.get("channel") or {}
        username = str(channel_payload.get("username") or "").lower()
        configured_channel = channel_by_username.get(username)
        if configured_channel is not None:
            payload["source_config"] = {
                "source_name": configured_channel.name,
                "domain_tags": list(configured_channel.domain_tags),
                "language": configured_channel.language,
                "access_status": configured_channel.access_status,
                "notes": configured_channel.notes,
            }
        post = raw_message_to_social_post(payload)
        if post is None:
            continue
        if require_text and not post.text.strip():
            continue
        query = str(wrapper.get("query") or "").strip() or None
        market_query = provenance.get(query) if query else None
        wrapper_market_id = str(wrapper.get("market_id") or "").strip() or None
        targeted_market_id = (
            wrapper_market_id
            if wrapper.get("retrieval_work_id")
            and wrapper_market_id
            and not wrapper_market_id.startswith("telegram-channel-")
            else None
        )
        collected_at = datetime.fromisoformat(str(wrapper["collected_at"]).replace("Z", "+00:00"))
        window_start = _optional_datetime(wrapper.get("retrieval_window_start"))
        window_end = _optional_datetime(wrapper.get("retrieval_window_end"))
        query_type = str(wrapper.get("query_type") or "").strip() or (
            market_query.query_type
            if market_query
            else "telegram_channel_search"
            if query
            else "telegram_channel"
        )
        records.append(
            build_record(
                post=post,
                query=query,
                query_type=query_type,
                market_id_candidate=(
                    targeted_market_id
                    or (market_query.market_id if market_query else None)
                ),
                raw_path=str(raw_path.resolve()),
                collected_at=collected_at,
                retrieval_run_id=str(wrapper.get("retrieval_run_id") or "") or None,
                retrieval_work_id=str(wrapper.get("retrieval_work_id") or "") or None,
                retrieval_window_start=window_start,
                retrieval_window_end=window_end,
                retrieval_lookback_days=wrapper.get("retrieval_lookback_days"),
                retrieval_window_label=str(wrapper.get("retrieval_window_label") or "") or None,
                retrieval_mode=str(wrapper.get("retrieval_mode") or "") or None,
                temporal_valid_candidate=wrapper.get("temporal_valid_candidate"),
            )
        )
    return records


def _filter_existing_for_channels(
    existing: list[SocialPostRecord],
    *,
    channel_by_username: dict[str, TelegramChannel],
    channels: Sequence[TelegramChannel] | None,
    require_text: bool,
) -> list[SocialPostRecord]:
    """Restrict prior rows to enabled channels and backfill channel language."""
    if channels is not None:
        allowed_channels = {channel.username.lower() for channel in channels}
        existing = [
            record
            for record in existing
            if record.platform is not Platform.TELEGRAM
            or (
                record.subreddit_or_channel is not None
                and record.subreddit_or_channel.lower() in allowed_channels
            )
        ]
        for record in existing:
            if record.platform is not Platform.TELEGRAM or record.subreddit_or_channel is None:
                continue
            configured_channel = channel_by_username.get(record.subreddit_or_channel.lower())
            if configured_channel is not None and not record.language:
                record.language = configured_channel.language
    if require_text:
        existing = [record for record in existing if record.text.strip()]
    return existing


def _read_existing_social_posts(output: Path, *, append: bool) -> list[SocialPostRecord]:
    """Strictly read the prior social parquet for a read-modify-write.

    A *missing* file is an empty dataset (fine). A file that is present but
    unreadable (e.g. a 0-byte ``social_posts.parquet`` left by an interrupted
    or duplicated writer) is quarantined and raised as
    :class:`CorruptParquetError`, so the caller recovers from the raw chunks
    instead of overwriting the dataset with whatever tiny batch is in hand.
    """
    if not append:
        return []
    try:
        return read_parquet_strict(output, SocialPostRecord)
    except CorruptParquetError as exc:
        quarantine_corrupt_parquet(output)
        raise CorruptParquetError(output, exc.original) from exc


def normalize_telegram_raw_directory(
    raw_dir: Path,
    *,
    posts_path: Path | None = None,
    paths: DataPaths | None = None,
    channels: Sequence[TelegramChannel] | None = None,
    require_text: bool = False,
    append: bool = True,
) -> list[SocialPostRecord]:
    """Append one Telegram raw run to the unified social parquet dataset.

    Data-safety: the existing dataset is read *strictly*. If it is present but
    corrupt, it is quarantined and :class:`CorruptParquetError` is raised
    rather than silently overwritten — the historical cause of a 600k-message
    loss. The write goes through :func:`safe_write_parquet`, which backs up the
    prior file and refuses a catastrophic row-count shrink.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"Telegram raw directory not found at {raw_dir}.")
    paths = paths or get_data_paths()
    output = posts_path or paths.social_posts_parquet
    provenance = {
        market_query.query: market_query
        for market_query in read_parquet(paths.social_queries_parquet, MarketQuery)
    }
    channel_by_username = {
        channel.username.lower(): channel for channel in channels or []
    }
    new_records = _records_from_raw_dir(
        raw_dir,
        channel_by_username=channel_by_username,
        provenance=provenance,
        require_text=require_text,
    )
    existing = _read_existing_social_posts(output, append=append)
    existing = _filter_existing_for_channels(
        existing,
        channel_by_username=channel_by_username,
        channels=channels,
        require_text=require_text,
    )
    merged = merge_with_existing(new_records, existing)
    if new_records or existing:
        safe_write_parquet(merged, output, label="social_posts.parquet")
    return merged


def recover_social_posts_from_raw(
    *,
    paths: DataPaths | None = None,
    channels: Sequence[TelegramChannel] | None = None,
    require_text: bool = False,
    posts_path: Path | None = None,
    platform_dir: str = "telegram",
    merge_existing: bool = True,
) -> dict[str, object]:
    """Rebuild ``social_posts.parquet`` from every preserved raw chunk dir.

    Walks ``data/<root>/raw/social/<platform>/**`` (the per-round directories
    of message JSONs the collector flushes *before* normalization) and rebuilds
    the canonical parquet from them. This recovers a dataset whose parquet was
    truncated/overwritten while its raw chunks survived — exactly the 0-byte
    ``social_posts.parquet`` incident. Idempotent: re-running merges and
    de-duplicates rather than double-counting.

    Returns a summary dict (raw dirs scanned, records parsed, unique rows,
    rows written, output path, backup path of any prior parquet).
    """
    paths = paths or get_data_paths()
    output = posts_path or paths.social_posts_parquet
    provenance = {
        market_query.query: market_query
        for market_query in read_parquet(paths.social_queries_parquet, MarketQuery)
    }
    channel_by_username = {channel.username.lower(): channel for channel in channels or []}

    raw_root = paths.raw_social(platform_dir)
    raw_dirs = sorted({p.parent for p in raw_root.glob("**/*.json")})
    parsed = 0
    all_records: list[SocialPostRecord] = []
    for raw_dir in raw_dirs:
        records = _records_from_raw_dir(
            raw_dir,
            channel_by_username=channel_by_username,
            provenance=provenance,
            require_text=require_text,
        )
        parsed += len(records)
        all_records.extend(records)

    # Fold in whatever survives in the current parquet (e.g. the tiny post-loss
    # batch, or any rows whose raw chunks were pruned). A corrupt current file
    # is quarantined and skipped — recovery never aborts on it.
    existing: list[SocialPostRecord] = []
    if merge_existing:
        try:
            existing = read_parquet_strict(output, SocialPostRecord)
        except CorruptParquetError:
            quarantine_corrupt_parquet(output)
            existing = []

    merged = merge_with_existing(all_records, existing)
    backup = None
    if merged:
        # Recovery legitimately grows the dataset by orders of magnitude, but a
        # prior good file is still backed up. allow_shrink stays False so a
        # recovery that somehow parsed *fewer* rows than the live file is
        # refused rather than silently shrinking it.
        backup = backup_parquet(output)
        safe_write_parquet(merged, output, label="social_posts.parquet (recovery)")
    return {
        "raw_root": str(raw_root),
        "raw_dirs_scanned": len(raw_dirs),
        "records_parsed": parsed,
        "existing_rows_merged": len(existing),
        "unique_rows_written": len(merged),
        "output": str(output),
        "backup": str(backup) if backup else None,
    }


def _optional_datetime(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def latest_telegram_raw_directory(paths: DataPaths | None = None) -> Path:
    """Return the newest Telegram raw run directory."""
    paths = paths or get_data_paths()
    manifests = sorted(paths.raw_social("telegram").glob("*/*/_manifest.json"))
    if not manifests:
        raise FileNotFoundError("No Telegram raw runs found. Fetch a channel first.")
    return manifests[-1].parent


__all__ = [
    "TelegramChannel",
    "TelegramFetchResult",
    "TelegramMarketSearchResult",
    "fetch_telegram_channels",
    "latest_telegram_raw_directory",
    "load_telegram_channels",
    "normalize_telegram_raw_directory",
    "recover_social_posts_from_raw",
    "search_telegram_market_plan",
]

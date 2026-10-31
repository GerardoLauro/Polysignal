"""Staged, resumable, logged Telegram API collection for large datasets.

This is the Telegram counterpart to the Branch 12B large Scrapling
collector. It uses the real Telegram API (Telethon, via
:class:`~polysignal.ingestion.social.telegram.TelegramIngestor`) to walk
the **full history** of the configured public channels, paging backwards
across many runs so an overnight collection can accumulate up to and
beyond a million messages without ever re-reading what it already saved.

Design
------
* **Resumable.** A JSON state file
  (``data/state/telegram_collection_state.json``) records, per channel,
  the oldest message id collected so far and the running message count.
  Each round fetches a batch of messages *older* than that id
  (``offset_id``), so a killed-and-restarted run continues exactly where
  it stopped. A channel that returns zero new messages is marked
  ``exhausted`` and skipped on later rounds.
* **Rate-limit aware.** Telethon transparently honours Telegram's
  ``FloodWait`` between requests; a configurable ``pause_seconds`` adds
  extra spacing between channels.
* **Logged.** A per-run log file is written under ``data/reports/`` with
  one line per channel-round (channel, batch size, oldest id, totals).
* **Incremental + deduplicated.** Every round's raw snapshots are
  normalized straight into the canonical ``social_posts.parquet`` via the
  existing append/merge path, so the dataset grows monotonically and
  re-runs never double-count a message.
* **No fabrication.** Only messages actually returned by the API are
  persisted; nothing is synthesized to hit a target.

The collector stops when any of these is true: the dataset reaches the
cumulative ``target_total`` (messages already saved by earlier resumed
runs count toward it, so resuming a 600k dataset with
``target_total=1_000_000`` adds ~400k more); every channel is exhausted;
or ``max_rounds`` is hit.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polysignal.ingestion.social.base import SearchQuery
from polysignal.ingestion.social.raw_store import SocialRawWriter, generate_social_run_id
from polysignal.ingestion.social.telegram import TelegramAPIError, TelegramIngestor
from polysignal.ingestion.social.telegram_normalizer import (
    TelegramChannel,
    normalize_telegram_raw_directory,
)
from polysignal.storage.parquet_store import CorruptParquetError
from polysignal.storage.paths import DataPaths, get_data_paths

logger = logging.getLogger(__name__)


class StateCorruptError(RuntimeError):
    """The resume-state JSON is present but unreadable.

    Resuming would otherwise restart every channel from the top of its
    history and re-collect (then overwrite) what was already saved, so the
    collector refuses to start and points the user at recovery instead of
    silently treating the dataset as empty.
    """


@dataclass
class TelegramLargeOptions:
    """Knobs for one staged Telegram collection run."""

    target_total: int = 1_000_000
    batch_per_channel: int = 1000
    max_rounds: int = 1000
    pause_seconds: float = 1.0
    require_text: bool = True
    resume: bool = True


@dataclass
class TelegramLargeResult:
    run_id: str
    rounds_executed: int = 0
    new_messages: int = 0
    total_messages_state: int = 0
    channels_exhausted: int = 0
    channels_active: int = 0
    per_channel: dict[str, dict[str, Any]] = field(default_factory=dict)
    log_path: Path | None = None
    state_path: Path | None = None
    posts_path: Path | None = None
    target_reached: bool = False
    stop_reason: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "rounds_executed": self.rounds_executed,
            "new_messages": self.new_messages,
            "total_messages_state": self.total_messages_state,
            "channels_exhausted": self.channels_exhausted,
            "channels_active": self.channels_active,
            "target_reached": self.target_reached,
            "stop_reason": self.stop_reason,
            "per_channel": self.per_channel,
        }


def _state_path(paths: DataPaths) -> Path:
    return paths.social_enrichment_state / "telegram_collection_state.json"


def _load_state(path: Path) -> dict[str, dict[str, Any]]:
    """Load resume state, distinguishing *missing* (fresh) from *corrupt*.

    A missing file means a brand-new dataset -> empty state. A present but
    unreadable/empty file is quarantined and :class:`StateCorruptError` is
    raised, so a resumed run never silently restarts from scratch (which would
    re-collect the top of every channel and overwrite the saved dataset).
    """
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            raise ValueError("state file is empty (0 bytes)")
        payload = json.loads(raw)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        quarantined = path.with_name(f"{path.name}.corrupt-{_state_timestamp()}")
        with contextlib.suppress(OSError):
            os.replace(path, quarantined)
        raise StateCorruptError(
            f"resume state {path} is present but unreadable ({type(exc).__name__}: {exc}); "
            f"quarantined to {quarantined}. Rebuild it with "
            "`polysignal social telegram recover` (which also restores the parquet from "
            "the raw chunks), or pass --no-resume to deliberately start fresh."
        ) from exc
    channels = payload.get("channels") if isinstance(payload, dict) else None
    return channels if isinstance(channels, dict) else {}


def _state_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_state(path: Path, channels: dict[str, dict[str, Any]]) -> None:
    """Persist state atomically (tmp file + ``os.replace``).

    A non-atomic ``write_text`` that is interrupted (Ctrl-C, kill, a second
    racing collector) can leave a truncated/0-byte state file — which is what
    silently reset this dataset. Writing to a sibling temp file and renaming
    means a reader only ever sees the old or the new state, never a partial.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        "channels": channels,
    }
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    try:
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()


def rebuild_state_from_records(records: Any) -> dict[str, dict[str, Any]]:
    """Reconstruct per-channel resume state from recovered ``SocialPostRecord`` rows.

    For each Telegram channel: ``oldest_id`` = the smallest message id seen
    (so the next run pages further back, never re-reading) and ``total`` = the
    number of recovered messages. Channels are left ``exhausted=False`` — the
    next round re-checks and re-marks any that truly have no older history.
    """
    from polysignal.domain import Platform  # local import avoids a cycle at module load

    state: dict[str, dict[str, Any]] = {}
    for record in records:
        if getattr(record, "platform", None) is not Platform.TELEGRAM:
            continue
        channel = record.subreddit_or_channel
        if not channel:
            continue
        mid = _message_id(record.post_id)
        entry = state.setdefault(channel, {"oldest_id": None, "total": 0, "exhausted": False})
        entry["total"] += 1
        if mid is not None:
            current = entry["oldest_id"]
            entry["oldest_id"] = mid if current is None else min(current, mid)
    return state


def _message_id(post_id: str) -> int | None:
    """Extract the numeric Telegram message id from ``telegram:<chan>:<id>``."""
    try:
        return int(post_id.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return None


def collect_telegram_large(
    channels: list[TelegramChannel],
    *,
    options: TelegramLargeOptions | None = None,
    ingestor: TelegramIngestor | None = None,
    paths: DataPaths | None = None,
    run_id: str | None = None,
) -> TelegramLargeResult:
    """Run a staged, resumable Telegram history collection.

    ``channels`` are the enabled targets (typically from
    ``config/telegram_channels.yml``). Credentials must already be
    configured (``POLYSIGNAL_TELEGRAM_API_ID`` / ``_API_HASH``) and the
    Telethon session authenticated (``polysignal social telegram auth``).
    """
    options = options or TelegramLargeOptions()
    paths = paths or get_data_paths()
    paths.ensure()
    rid = run_id or generate_social_run_id()
    result = TelegramLargeResult(run_id=rid)

    state_path = _state_path(paths)
    state = _load_state(state_path) if options.resume else {}
    result.state_path = state_path

    log_dir = paths.social_enrichment_reports
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"telegram_large_{rid}.log"
    result.log_path = log_path
    log_handle = log_path.open("a", encoding="utf-8")

    def emit(message: str) -> None:
        line = f"{datetime.now(tz=timezone.utc).isoformat()} {message}"
        log_handle.write(line + "\n")
        log_handle.flush()
        logger.info(message)

    client = ingestor or TelegramIngestor()
    if not client.is_available():
        log_handle.close()
        raise RuntimeError(
            "Telegram API credentials are not configured. Set "
            "POLYSIGNAL_TELEGRAM_API_ID and POLYSIGNAL_TELEGRAM_API_HASH in .env "
            "and authenticate with `polysignal social telegram auth`."
        )

    # Seed per-channel state.
    for channel in channels:
        state.setdefault(
            channel.username,
            {"oldest_id": None, "total": 0, "exhausted": False},
        )

    # ``target_total`` is the cumulative dataset goal: messages already saved
    # by earlier resumed runs count toward it. ``baseline_total`` captures that
    # pre-existing count so a resumed run only collects what is still missing.
    baseline_total = sum(int(s.get("total", 0)) for s in state.values())

    emit(
        f"start run={rid} channels={len(channels)} target_total={options.target_total} "
        f"baseline_total={baseline_total} batch_per_channel={options.batch_per_channel} "
        f"resume={options.resume}"
    )

    try:
        for round_index in range(options.max_rounds):
            active = [
                channel
                for channel in channels
                if not state[channel.username]["exhausted"]
            ]
            if not active:
                result.stop_reason = "all_channels_exhausted"
                break
            if baseline_total + result.new_messages >= options.target_total:
                result.stop_reason = "target_reached"
                result.target_reached = True
                break

            round_run_id = f"{rid}-r{round_index:04d}"
            run_dir = paths.raw_run_dir("social/telegram", round_run_id)
            writer = SocialRawWriter(directory=run_dir, platform="telegram")
            writer.extra["staged_collection"] = True
            writer.extra["round"] = round_index
            round_new = 0

            for channel in active:
                channel_state = state[channel.username]
                remaining = options.target_total - (baseline_total + result.new_messages)
                if remaining <= 0:
                    break
                batch_limit = min(options.batch_per_channel, remaining)
                offset_id = channel_state.get("oldest_id")
                try:
                    posts, raws = client.fetch_channel_raw(
                        channel.username,
                        SearchQuery(text="", limit=batch_limit, offset_id=offset_id),
                    )
                except TelegramAPIError as exc:
                    reason = f"{type(exc).__name__}: {exc}"[:200]
                    emit(f"round={round_index} channel={channel.username} skipped: {reason}")
                    channel_state["last_error"] = reason
                    continue

                if not posts:
                    channel_state["exhausted"] = True
                    emit(
                        f"round={round_index} channel={channel.username} "
                        f"exhausted (total={channel_state['total']})"
                    )
                    continue

                ids = [mid for mid in (_message_id(p.post_id) for p in posts) if mid is not None]
                new_oldest = min(ids) if ids else channel_state.get("oldest_id")
                for post, raw in zip(posts, raws, strict=True):
                    raw["source_config"] = {
                        "source_name": channel.name,
                        "domain_tags": list(channel.domain_tags),
                        "language": channel.language,
                        "access_status": channel.access_status,
                        "notes": channel.notes,
                    }
                    writer.write_post(
                        query="",
                        market_id=f"telegram-channel-{channel.username}",
                        post_id=post.post_id,
                        payload=raw,
                    )
                channel_state["oldest_id"] = new_oldest
                channel_state["total"] += len(posts)
                result.new_messages += len(posts)
                round_new += len(posts)
                emit(
                    f"round={round_index} channel={channel.username} "
                    f"batch={len(posts)} oldest_id={new_oldest} "
                    f"channel_total={channel_state['total']} run_total={result.new_messages}"
                )
                if options.pause_seconds > 0:
                    time.sleep(options.pause_seconds)

            # Persist raw manifest + normalize this round into the canonical parquet.
            if round_new > 0:
                writer.extra["round_new_messages"] = round_new
                writer.write_manifest()
                try:
                    records = normalize_telegram_raw_directory(
                        run_dir,
                        paths=paths,
                        channels=channels,
                        require_text=options.require_text,
                    )
                except CorruptParquetError as exc:
                    # The existing dataset is unreadable. The raw chunk for this
                    # round is already flushed to disk, so nothing is lost — but
                    # we must NOT overwrite the dataset with this round alone.
                    # Stop cleanly and tell the user to recover from raw chunks.
                    result.stop_reason = "existing_parquet_corrupt"
                    emit(
                        f"round={round_index} ABORT: {exc} "
                        "Raw chunks are intact; run `polysignal social telegram recover` "
                        "to rebuild the parquet, then resume."
                    )
                    break
                result.posts_path = paths.social_posts_parquet
                emit(
                    f"round={round_index} normalized canonical_rows={len(records)} "
                    f"round_new={round_new}"
                )
            result.rounds_executed += 1
            _write_state(state_path, state)
        else:
            result.stop_reason = result.stop_reason or "max_rounds_reached"
    finally:
        client.close()
        _write_state(state_path, state)
        result.per_channel = state
        result.total_messages_state = sum(int(s.get("total", 0)) for s in state.values())
        result.channels_exhausted = sum(1 for s in state.values() if s.get("exhausted"))
        result.channels_active = len(channels) - result.channels_exhausted
        if not result.stop_reason:
            result.stop_reason = "completed"
        emit(
            f"stop reason={result.stop_reason} rounds={result.rounds_executed} "
            f"new_messages={result.new_messages} total_state={result.total_messages_state}"
        )
        summary_path = log_dir / f"telegram_large_{rid}_summary.json"
        summary_path.write_text(json.dumps(result.summary(), indent=2), encoding="utf-8")
        log_handle.close()

    return result


def recover_telegram_dataset(
    channels: list[TelegramChannel],
    *,
    paths: DataPaths | None = None,
    require_text: bool = True,
    rebuild_state: bool = True,
) -> dict[str, Any]:
    """Rebuild the canonical parquet AND the resume state from raw chunks.

    This is the operator-facing recovery entry point for the 0-byte-parquet
    incident: the per-round raw message JSONs survive even when the parquet was
    truncated/overwritten, so we re-normalize them into a fresh
    ``social_posts.parquet`` (backed up + shrink-guarded) and reconstruct
    ``telegram_collection_state.json`` so the next ``collect-large`` resumes
    from the deep history instead of the top of each channel.

    Returns a summary dict combining the parquet recovery stats and the
    rebuilt per-channel state.
    """
    from polysignal.domain import SocialPostRecord
    from polysignal.ingestion.social.telegram_normalizer import (
        recover_social_posts_from_raw,
    )
    from polysignal.storage.parquet_store import read_parquet

    paths = paths or get_data_paths()
    paths.ensure()
    summary = recover_social_posts_from_raw(
        paths=paths,
        channels=channels,
        require_text=require_text,
    )
    state_summary: dict[str, Any] = {}
    if rebuild_state:
        records = read_parquet(paths.social_posts_parquet, SocialPostRecord)
        rebuilt = rebuild_state_from_records(records)
        # Seed any configured channel that produced no recovered rows so the
        # state file lists every target (the collector also seeds at run time).
        for channel in channels:
            rebuilt.setdefault(
                channel.username, {"oldest_id": None, "total": 0, "exhausted": False}
            )
        state_path = _state_path(paths)
        _write_state(state_path, rebuilt)
        summary["state_path"] = str(state_path)
        summary["channels_in_state"] = len(rebuilt)
        summary["state_total"] = sum(int(v.get("total", 0)) for v in rebuilt.values())
        state_summary = rebuilt
    summary["per_channel_state"] = state_summary
    return summary


__all__ = [
    "StateCorruptError",
    "TelegramLargeOptions",
    "TelegramLargeResult",
    "collect_telegram_large",
    "rebuild_state_from_records",
    "recover_telegram_dataset",
]

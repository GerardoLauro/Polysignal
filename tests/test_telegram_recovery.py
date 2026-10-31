"""End-to-end recovery and state-safety for the Telegram collector.

Reproduces the production incident in miniature: collect a dataset, destroy
its parquet (0 bytes), and prove that (a) the normalizer refuses to silently
overwrite the corrupt file, and (b) ``recover_telegram_dataset`` rebuilds both
the parquet and the resume state from the surviving raw chunks.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from polysignal.domain import Platform, SocialPost
from polysignal.ingestion.social.base import SearchQuery
from polysignal.ingestion.social.telegram_large import (
    StateCorruptError,
    TelegramLargeOptions,
    _load_state,
    collect_telegram_large,
    rebuild_state_from_records,
    recover_telegram_dataset,
)
from polysignal.ingestion.social.telegram_normalizer import (
    TelegramChannel,
    normalize_telegram_raw_directory,
)
from polysignal.storage.parquet_store import (
    CorruptParquetError,
    parquet_is_readable,
    parquet_row_count,
)
from polysignal.storage.paths import get_data_paths


class _FakeHistoryIngestor:
    """Serves a fixed message id range per channel, newest-first, by offset_id."""

    def __init__(self, history: dict[str, int]) -> None:
        self.history = history
        self.closed = False

    def is_available(self) -> bool:
        return True

    def fetch_channel_raw(self, channel: str, query: SearchQuery):
        total = self.history.get(channel, 0)
        ceiling = query.offset_id if query.offset_id else total + 1
        ids = [i for i in range(total, 0, -1) if i < ceiling][: (query.limit or 100)]
        posts, raws = [], []
        for mid in ids:
            posts.append(
                SocialPost(
                    post_id=f"telegram:{channel}:{mid}",
                    platform=Platform.TELEGRAM,
                    author_id=channel,
                    author_handle=channel,
                    created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    text=f"{channel} message {mid}",
                )
            )
            raws.append(
                {
                    "channel": {"username": channel, "id": channel, "title": channel},
                    "message": {
                        "id": mid,
                        "text": f"{channel} message {mid}",
                        "date": "2026-01-01T00:00:00+00:00",
                    },
                }
            )
        return posts, raws

    def close(self) -> None:
        self.closed = True


def _collect(tmp_path: Path, history: dict[str, int]):
    paths = get_data_paths(root=tmp_path)
    channels = [TelegramChannel(username=name) for name in history]
    collect_telegram_large(
        channels,
        options=TelegramLargeOptions(
            target_total=10_000, batch_per_channel=10, max_rounds=200, pause_seconds=0.0
        ),
        ingestor=_FakeHistoryIngestor(history),
        paths=paths,
    )
    return paths, channels


def test_normalizer_refuses_to_overwrite_corrupt_parquet(tmp_path: Path) -> None:
    paths, channels = _collect(tmp_path, {"alpha": 40})
    posts_path = paths.social_posts_parquet
    good_count = parquet_row_count(posts_path)
    assert good_count == 40

    # Destroy the parquet (the incident signature: a 0-byte file).
    posts_path.write_bytes(b"")
    assert parquet_is_readable(posts_path) is False

    # A fresh round must NOT silently overwrite the corrupt file with a tiny
    # batch. It raises and quarantines the bad file instead.
    raw_dirs = sorted((paths.raw_social("telegram")).glob("**/_manifest.json"))
    any_round_dir = raw_dirs[0].parent
    with pytest.raises(CorruptParquetError):
        normalize_telegram_raw_directory(
            any_round_dir, paths=paths, channels=channels, require_text=True
        )
    assert list((posts_path.parent / "_corrupt").glob("*.corrupt"))


def test_recover_rebuilds_parquet_and_state_from_raw(tmp_path: Path) -> None:
    paths, channels = _collect(tmp_path, {"alpha": 50, "beta": 30})
    posts_path = paths.social_posts_parquet
    assert parquet_row_count(posts_path) == 80

    # Wipe the parquet AND the state - the worst case.
    posts_path.write_bytes(b"")
    state_path = paths.social_enrichment_state / "telegram_collection_state.json"
    state_path.write_bytes(b"")

    summary = recover_telegram_dataset(channels, paths=paths, require_text=True)

    # Parquet fully rebuilt from the surviving raw chunks.
    assert parquet_is_readable(posts_path)
    assert parquet_row_count(posts_path) == 80
    assert summary["unique_rows_written"] == 80
    assert summary["state_total"] == 80

    # State restored to the deep history so a resumed run does not re-collect.
    rebuilt = _load_state(state_path)
    assert rebuilt["alpha"]["total"] == 50
    assert rebuilt["alpha"]["oldest_id"] == 1
    assert rebuilt["beta"]["oldest_id"] == 1

    second = collect_telegram_large(
        channels,
        options=TelegramLargeOptions(
            target_total=10_000, batch_per_channel=10, max_rounds=50, pause_seconds=0.0
        ),
        ingestor=_FakeHistoryIngestor({"alpha": 50, "beta": 30}),
        paths=paths,
    )
    # Everything was already collected; resume adds nothing and exhausts.
    assert second.new_messages == 0


def test_load_state_raises_on_corrupt_but_empty_on_missing(tmp_path: Path) -> None:
    state_path = tmp_path / "telegram_collection_state.json"
    # Missing -> fresh empty state (no error).
    assert _load_state(state_path) == {}

    # Present but 0-byte -> refuse (quarantine + raise) rather than reset.
    state_path.write_bytes(b"")
    with pytest.raises(StateCorruptError):
        _load_state(state_path)
    assert list(tmp_path.glob("telegram_collection_state.json.corrupt-*"))


def test_rebuild_state_from_records_computes_oldest_and_totals() -> None:
    from polysignal.domain import SocialPostRecord

    records = [
        SocialPostRecord(
            post_id=f"telegram:alpha:{i}",
            platform=Platform.TELEGRAM,
            source_id=str(i),
            text=f"m{i}",
            subreddit_or_channel="alpha",
            collected_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        for i in (5, 6, 7)
    ]
    state = rebuild_state_from_records(records)
    assert state["alpha"]["total"] == 3
    assert state["alpha"]["oldest_id"] == 5
    assert state["alpha"]["exhausted"] is False

"""Tests for the staged, resumable Telegram API collector.

A fake ingestor serves a finite, deterministic message history per
channel so the resume / exhaustion / offset-paging logic can be exercised
offline without any network or Telethon dependency.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from polysignal.domain import Platform, SocialPost
from polysignal.ingestion.social.base import SearchQuery
from polysignal.ingestion.social.telegram_large import (
    TelegramLargeOptions,
    collect_telegram_large,
)
from polysignal.ingestion.social.telegram_normalizer import TelegramChannel
from polysignal.storage.paths import get_data_paths


class _FakeHistoryIngestor:
    """Serves a fixed message id range per channel, newest-first, by offset_id."""

    def __init__(self, history: dict[str, int]) -> None:
        # channel -> number of messages (ids 1..N)
        self.history = history
        self.closed = False

    def is_available(self) -> bool:
        return True

    def fetch_channel_raw(self, channel: str, query: SearchQuery):
        total = self.history.get(channel, 0)
        # Newest-first: ids total..1. offset_id returns ids strictly below it.
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


def test_collect_telegram_large_walks_full_history(tmp_path: Path) -> None:
    paths = get_data_paths(root=tmp_path)
    fake = _FakeHistoryIngestor({"alpha": 5, "beta": 3})
    channels = [TelegramChannel(username="alpha"), TelegramChannel(username="beta")]

    result = collect_telegram_large(
        channels,
        options=TelegramLargeOptions(
            target_total=1000,
            batch_per_channel=2,
            max_rounds=50,
            pause_seconds=0.0,
        ),
        ingestor=fake,
        paths=paths,
    )

    # All 8 messages collected, both channels exhausted, nothing re-read.
    assert result.new_messages == 8
    assert result.channels_exhausted == 2
    assert result.stop_reason == "all_channels_exhausted"
    assert fake.closed is True
    assert result.log_path is not None and result.log_path.exists()
    assert result.state_path is not None and result.state_path.exists()


def test_collect_telegram_large_is_resumable(tmp_path: Path) -> None:
    paths = get_data_paths(root=tmp_path)
    channels = [TelegramChannel(username="alpha")]

    # First run: stop early after 2 messages.
    collect_telegram_large(
        channels,
        options=TelegramLargeOptions(
            target_total=2, batch_per_channel=2, max_rounds=50, pause_seconds=0.0
        ),
        ingestor=_FakeHistoryIngestor({"alpha": 6}),
        paths=paths,
    )

    # Second run resumes from saved offset and collects the remaining 4.
    second = collect_telegram_large(
        channels,
        options=TelegramLargeOptions(
            target_total=1000, batch_per_channel=2, max_rounds=50, pause_seconds=0.0
        ),
        ingestor=_FakeHistoryIngestor({"alpha": 6}),
        paths=paths,
    )
    assert second.new_messages == 4
    assert second.per_channel["alpha"]["total"] == 6
    assert second.per_channel["alpha"]["exhausted"] is True


def test_target_total_is_cumulative_across_resumed_runs(tmp_path: Path) -> None:
    """``target_total`` counts the whole dataset, so resume tops up to it."""
    paths = get_data_paths(root=tmp_path)
    channels = [TelegramChannel(username="alpha")]

    # First run caps the dataset at 10 messages (channel has plenty more).
    first = collect_telegram_large(
        channels,
        options=TelegramLargeOptions(
            target_total=10, batch_per_channel=5, max_rounds=50, pause_seconds=0.0
        ),
        ingestor=_FakeHistoryIngestor({"alpha": 100}),
        paths=paths,
    )
    assert first.new_messages == 10
    assert first.target_reached is True

    # Resuming with a higher cumulative target adds only the missing 5, not a
    # fresh 15 (which the old per-run semantics would have collected).
    second = collect_telegram_large(
        channels,
        options=TelegramLargeOptions(
            target_total=15, batch_per_channel=5, max_rounds=50, pause_seconds=0.0
        ),
        ingestor=_FakeHistoryIngestor({"alpha": 100}),
        paths=paths,
    )
    assert second.new_messages == 5
    assert second.target_reached is True
    assert second.stop_reason == "target_reached"
    assert second.per_channel["alpha"]["total"] == 15
    assert second.per_channel["alpha"]["exhausted"] is False

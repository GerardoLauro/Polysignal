"""Data-safety guards for the parquet store.

These lock in the fixes for the incident where a 0-byte / corrupt
``social_posts.parquet`` was silently treated as empty and then overwritten,
destroying ~600k recovered-from-raw messages.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from polysignal.domain import Platform, SocialPostRecord
from polysignal.storage.parquet_store import (
    CorruptParquetError,
    DatasetShrinkError,
    backup_parquet,
    parquet_is_readable,
    parquet_row_count,
    quarantine_corrupt_parquet,
    read_parquet_strict,
    safe_write_parquet,
    write_parquet,
)


def _records(n: int) -> list[SocialPostRecord]:
    return [
        SocialPostRecord(
            post_id=f"telegram:chan:{i}",
            platform=Platform.TELEGRAM,
            source_id=str(i),
            text=f"message {i}",
            subreddit_or_channel="chan",
            collected_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        for i in range(n)
    ]


def test_safe_write_refuses_catastrophic_shrink(tmp_path: Path) -> None:
    path = tmp_path / "social_posts.parquet"
    write_parquet(_records(100), path)

    with pytest.raises(DatasetShrinkError):
        safe_write_parquet(_records(3), path, label="social_posts")

    # The existing file is left completely intact ...
    assert parquet_row_count(path) == 100
    # ... and the rejected payload is preserved for inspection.
    assert list(tmp_path.glob("social_posts.parquet.rejected-*.parquet"))


def test_safe_write_allows_growth_and_backs_up(tmp_path: Path) -> None:
    path = tmp_path / "social_posts.parquet"
    write_parquet(_records(100), path)

    safe_write_parquet(_records(250), path, label="social_posts")
    assert parquet_row_count(path) == 250
    # The prior good file was snapshotted before being replaced.
    assert list((tmp_path / "_backups").glob("social_posts.parquet.*.bak"))


def test_safe_write_shrink_allowed_when_opted_in(tmp_path: Path) -> None:
    path = tmp_path / "social_posts.parquet"
    write_parquet(_records(100), path)
    safe_write_parquet(_records(2), path, allow_shrink=True)
    assert parquet_row_count(path) == 2


def test_read_strict_distinguishes_missing_from_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "social_posts.parquet"
    # Missing == empty dataset, never an error.
    assert read_parquet_strict(path, SocialPostRecord) == []

    write_parquet(_records(5), path)
    assert len(read_parquet_strict(path, SocialPostRecord)) == 5

    # A present-but-0-byte file must raise, not silently read as empty.
    path.write_bytes(b"")
    assert parquet_is_readable(path) is False
    with pytest.raises(CorruptParquetError):
        read_parquet_strict(path, SocialPostRecord)


def test_quarantine_moves_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "social_posts.parquet"
    path.write_bytes(b"")  # 0-byte / corrupt
    dest = quarantine_corrupt_parquet(path)
    assert dest is not None and dest.exists()
    assert not path.exists()
    assert dest.parent.name == "_corrupt"


def test_row_count_and_readable_helpers(tmp_path: Path) -> None:
    path = tmp_path / "social_posts.parquet"
    assert parquet_row_count(path) is None  # missing
    write_parquet(_records(7), path)
    assert parquet_row_count(path) == 7
    assert parquet_is_readable(path) is True
    path.write_bytes(b"PAR1")  # truncated
    assert parquet_row_count(path) is None
    assert parquet_is_readable(path) is False


def test_backup_only_snapshots_readable_files(tmp_path: Path) -> None:
    path = tmp_path / "social_posts.parquet"
    assert backup_parquet(path) is None  # nothing to back up
    write_parquet(_records(3), path)
    assert backup_parquet(path) is not None

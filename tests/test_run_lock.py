"""Tests for the duplicate-run lock used by the long-running collectors."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from polysignal.storage import run_lock as run_lock_module
from polysignal.storage.run_lock import RunLockError, read_lock, run_lock


def test_run_lock_creates_and_releases(tmp_path: Path) -> None:
    lock = tmp_path / "telegram_api.lock"
    with run_lock(lock, label="telegram-collect-large"):
        assert lock.exists()
        payload = read_lock(lock)
        assert payload is not None
        assert payload["pid"] == os.getpid()
        assert payload["label"] == "telegram-collect-large"
    # Released on normal exit.
    assert not lock.exists()


def test_run_lock_blocks_a_live_holder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock = tmp_path / "scrapling_web.lock"
    lock.write_text(json.dumps({"pid": 999999, "started_at": "earlier"}), encoding="utf-8")
    monkeypatch.setattr(run_lock_module, "_pid_alive", lambda pid: True)

    with pytest.raises(RunLockError, match="already holds"), run_lock(
        lock, label="scrapling-collect-large-enriched"
    ):
        pass

    # A blocked attempt must not clobber the live holder's lock file.
    assert read_lock(lock) == {"pid": 999999, "started_at": "earlier"}


def test_run_lock_reclaims_a_dead_holder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock = tmp_path / "telegram_api.lock"
    lock.write_text(json.dumps({"pid": 999999, "started_at": "earlier"}), encoding="utf-8")
    monkeypatch.setattr(run_lock_module, "_pid_alive", lambda pid: False)

    entered = False
    with run_lock(lock, label="telegram-collect-large"):
        entered = True
        assert read_lock(lock)["pid"] == os.getpid()
    assert entered
    assert not lock.exists()


def test_run_lock_released_on_exception(tmp_path: Path) -> None:
    lock = tmp_path / "telegram_api.lock"
    with pytest.raises(RuntimeError, match="boom"), run_lock(
        lock, label="telegram-collect-large"
    ):
        raise RuntimeError("boom")
    assert not lock.exists()

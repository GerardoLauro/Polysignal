"""Cross-process run lock to stop duplicate long-running collectors.

A long Telegram/Scrapling collection must never run twice in parallel against
the same staging root — duplicate runs race on the same checkpoint/parquet
files and waste rate-limit budget. This module provides a tiny PID-aware lock
file: a second process that finds a *live* holder refuses to start; a lock left
behind by a dead process (e.g. a killed run) is treated as stale and taken over.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunLockError(RuntimeError):
    """Raised when another live process already holds the run lock."""


def _pid_alive(pid: int) -> bool:
    """Whether ``pid`` refers to a currently running process.

    Deliberately avoids ``os.kill(pid, 0)`` on Windows, where ``os.kill``
    terminates the target for any non-CTRL signal. Uses ``WaitForSingleObject``
    instead, which only reports liveness.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        process_query_limited_information = 0x1000
        synchronize = 0x00100000
        wait_timeout = 0x00000102
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            process_query_limited_information | synchronize, False, pid
        )
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == wait_timeout
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_lock(path: Path) -> dict[str, Any] | None:
    """Return the lock payload, or ``None`` if missing/unreadable."""
    try:
        # utf-8-sig transparently tolerates a BOM if the file was written by
        # another tool, so a real holder is never mistaken for an absent lock.
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def lock_is_active(path: Path) -> bool:
    """Return whether ``path`` names a lock currently held by a live PID."""
    payload = read_lock(path)
    if not payload:
        return False
    return _pid_alive(int(payload.get("pid", 0) or 0))


@contextmanager
def run_lock(path: Path, *, label: str) -> Iterator[Path]:
    """Hold an exclusive run lock at ``path`` for the duration of the block.

    Raises :class:`RunLockError` if a different, still-running process already
    holds it. A lock owned by a dead PID is reclaimed. The lock is always
    released on exit (normal, exception, or KeyboardInterrupt).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_lock(path) if path.exists() else None
    if existing:
        holder_pid = int(existing.get("pid", 0) or 0)
        if holder_pid and holder_pid != os.getpid() and _pid_alive(holder_pid):
            raise RunLockError(
                f"another {label} run (pid {holder_pid}, started "
                f"{existing.get('started_at', 'unknown')}) already holds {path}. "
                "Refusing to launch a duplicate; stop it first or wait for it to finish."
            )
    payload = {
        "pid": os.getpid(),
        "label": label,
        "started_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        yield path
    finally:
        current = read_lock(path)
        if current and int(current.get("pid", 0) or 0) == os.getpid():
            with suppress(OSError):
                path.unlink()


__all__ = ["RunLockError", "lock_is_active", "read_lock", "run_lock"]

"""Resilient supervisor for the long social-collection session.

Keeps ``run_long_social_session.py`` collecting for a total wall-clock budget,
auto-restarting it whenever it exits early, is killed, or hangs (stale heartbeat),
so a multi-hour cross-platform scrape survives the recurring orchestrator deaths.

Why this exists
---------------
The orchestrator and the collectors work correctly, but the orchestrator process is
terminated externally after ~75-85 min when launched from a transient shell (the
process is reaped together with the shell's job). This supervisor is meant to be
launched by the **Windows Task Scheduler** (``schtasks /run``) so that it is a child
of the scheduler service rather than of any shell, and it then keeps a child
orchestrator alive for the whole budget. Collection still flows entirely through the
unchanged ``run_long_social_session.py`` and its collectors; this module only
supervises them and never touches the pipeline.

It restarts the child on three conditions:
  * the child process exits (clean finish of its ``--hours`` slice, crash, or kill);
  * the child's heartbeat in ``session_state.json`` goes stale (a hang);
  * a stale lock from a previous child is cleared before each (re)launch.

Stop it by creating ``data/staging/long_social_session/STOP_SUPERVISOR`` or sending
Ctrl-C / SIGTERM.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SESSION_ROOT = ROOT / "data" / "staging" / "long_social_session"
# IMPORTANT: the supervisor's own logs must live OUTSIDE the orchestrator's ``logs/``
# directory, because the orchestrator archives everything in ``logs/`` on startup and
# would deadlock on files this supervisor holds open (WinError 32).
SUP_LOG_DIR = SESSION_ROOT / "supervisor_logs"
ORCH_STATE = SESSION_ROOT / "session_state.json"
SUP_STATE = SESSION_ROOT / "supervisor_state.json"
SUP_LOG = SUP_LOG_DIR / "supervisor.log"
CHILD_OUT = SUP_LOG_DIR / "child_stdout.log"
STOP = SESSION_ROOT / "STOP_SUPERVISOR"
LOCK = SESSION_ROOT / "long_session.lock"
CHILD_LOCKS = (
    ROOT / "data" / "staging" / "enriched" / "scrapling_web.lock",
    ROOT / "data" / "staging" / "social_comments" / "state" / "comment_backfill.lock",
)

# Tuned high-yield parameters (cross-platform: maximise Reddit/X posts + replies).
# Reddit *posts* are saturated for the market set, so time is shifted to comments and
# X-account rotation, which are the growth areas for cross-platform balance.
# Efficient config: fast, frequent cycles maximise throughput. Posts are supply-capped
# for resolved markets (more accounts/subreddits add little), so this favours quick cycles
# and high comment sampling — the one stream with real remaining supply. The widened
# 60-account / 40-subreddit pool (rotated 12/cycle) still casts the broadest net for posts.
TUNED = [
    "--cycle-minutes", "25",
    "--reddit-target", "500000",
    "--x-target-per-pass", "6000",
    "--x-accounts-per-pass", "12",
    "--reddit-pass-minutes", "8",
    "--reddit-comment-posts-per-pass", "100",
    "--x-comment-posts-per-pass", "50",
    "--comments-per-post", "20",
    "--reddit-rate-limit", "3",
    "--comment-rate-limit", "3",
]

HEARTBEAT_STALE_S = 360.0   # restart the child if its heartbeat freezes this long
LAUNCH_GRACE_S = 150.0      # don't apply the stale-heartbeat check right after launch
POLL_S = 30.0               # supervisor poll cadence
BACKOFF_S = 20.0            # pause before relaunch after an unexpected death
MIN_CHILD_HOURS = 0.05      # stop if less than this remains

_STOP = False


def _log(msg: str) -> None:
    line = f"[{datetime.now(tz=timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    with suppress(Exception):
        SUP_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with SUP_LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _sig(_signum: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def _pid_alive(pid: int) -> bool:
    with suppress(Exception):
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True,
        )
        return str(pid) in out.stdout
    return False


def _orch_state() -> dict:
    with suppress(Exception):
        return json.loads(ORCH_STATE.read_text(encoding="utf-8-sig"))
    return {}


def _heartbeat_age() -> float:
    hb = _orch_state().get("heartbeat_at")
    if not hb:
        return 1e9
    with suppress(Exception):
        return (datetime.now(tz=timezone.utc) - datetime.fromisoformat(hb)).total_seconds()
    return 1e9


def _clear_stale_locks() -> None:
    for path in (LOCK, *CHILD_LOCKS):
        if not path.exists():
            continue
        pid = None
        with suppress(Exception):
            pid = json.loads(path.read_text(encoding="utf-8")).get("pid")
        if not (pid and _pid_alive(int(pid))):
            with suppress(Exception):
                path.unlink()
                _log(f"cleared stale lock {path.name}")


def _kill_orphans(self_pid: int) -> int:
    """Kill leftover orchestrator/collector processes (never this supervisor).

    A crashed or killed orchestrator can orphan its collectors, which keep the stage
    log files (``comments.log`` etc.) open; the next orchestrator then cannot archive
    them and exits. Clearing orphans before each (re)launch keeps exactly one collector
    set running and lets the new orchestrator start cleanly.
    """
    pat = ("run_long_social_session|collect_twitter_playwright|"
           "collect_social_comments|collect-large-enriched")
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        f"Where-Object {{ $_.CommandLine -match '{pat}' }} | "
        "Select-Object -ExpandProperty ProcessId"
    )
    killed = 0
    with suppress(Exception):
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=30,
        )
        for tok in out.stdout.split():
            with suppress(Exception):
                pid = int(tok.strip())
                if pid != self_pid:
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    killed += 1
    return killed


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    with suppress(Exception):
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    with suppress(Exception):
        proc.wait(timeout=15)


def _write_state(**kw: Any) -> None:
    SUP_STATE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "supervisor_pid": os.getpid(),
        "heartbeat_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    payload.update(kw)
    tmp = SUP_STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, SUP_STATE)


def main() -> int:
    ap = argparse.ArgumentParser(description="Supervise the long social session.")
    ap.add_argument("--total-hours", type=float, default=10.0)
    ap.add_argument("--max-restarts", type=int, default=300)
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _sig)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sig)
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    SUP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    if STOP.exists():
        STOP.unlink()
    if not PY.exists():
        _log(f"FATAL: venv python not found at {PY}")
        return 1

    start = time.monotonic()
    deadline = start + args.total_hours * 3600.0
    started_at = datetime.now(tz=timezone.utc).isoformat()
    restarts = 0
    _log(f"SUPERVISOR START pid={os.getpid()} total_hours={args.total_hours}")
    _write_state(status="starting", started_at=started_at,
                 total_hours=args.total_hours, restarts=0)

    while not _STOP and time.monotonic() < deadline and restarts < args.max_restarts:
        if STOP.exists():
            _log("STOP_SUPERVISOR present -> exiting")
            break
        remaining_h = (deadline - time.monotonic()) / 3600.0
        if remaining_h < MIN_CHILD_HOURS:
            break

        orphans = _kill_orphans(os.getpid())
        if orphans:
            _log(f"cleared {orphans} orphaned collector/orchestrator process(es)")
            time.sleep(3.0)  # let Windows release the log-file handles
        _clear_stale_locks()
        cmd = [str(PY), "scripts/run_long_social_session.py",
               "--hours", f"{remaining_h:.4f}", *TUNED]
        _log(f"launch child #{restarts + 1} remaining={remaining_h:.2f}h")
        child_out = CHILD_OUT.open("a", encoding="utf-8")
        child_out.write(f"\n===== child #{restarts + 1} @ {started_at} =====\n")
        child_out.flush()
        proc = subprocess.Popen(cmd, cwd=str(ROOT),
                                stdout=child_out, stderr=subprocess.STDOUT)
        launched = time.monotonic()
        reason = "?"

        while True:
            time.sleep(POLL_S)
            if _STOP or STOP.exists():
                reason = "stop-requested"
                _terminate(proc)
                break
            if time.monotonic() >= deadline:
                reason = "deadline-reached"
                _terminate(proc)
                break
            rc = proc.poll()
            if rc is not None:
                reason = f"exit={rc}"
                break
            age = _heartbeat_age()
            if time.monotonic() - launched > LAUNCH_GRACE_S and age > HEARTBEAT_STALE_S:
                reason = f"hung(heartbeat={age:.0f}s)"
                _terminate(proc)
                break
            _write_state(status="running", started_at=started_at,
                         total_hours=args.total_hours, restarts=restarts,
                         child_pid=proc.pid, remaining_hours=round(remaining_h, 3),
                         orch_heartbeat_age_s=round(age, 1),
                         counts=_orch_state().get("counts", {}))

        with suppress(Exception):
            child_out.close()
        restarts += 1
        _log(f"child #{restarts} ended: {reason} | counts={_orch_state().get('counts', {})}")
        if reason in ("stop-requested", "deadline-reached"):
            break
        if time.monotonic() < deadline and not _STOP:
            time.sleep(BACKOFF_S)

    _kill_orphans(os.getpid())
    _write_state(status="stopped", started_at=started_at,
                 finished_at=datetime.now(tz=timezone.utc).isoformat(),
                 total_hours=args.total_hours, restarts=restarts,
                 counts=_orch_state().get("counts", {}))
    _log(f"SUPERVISOR END restarts={restarts} counts={_orch_state().get('counts', {})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

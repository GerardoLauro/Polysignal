"""Timed Reddit + X post-and-comment collection session.

One Reddit discovery process runs continuously while the orchestrator cycles
authenticated X profile collection and resumable Reddit/X comment backfills.
All writes stay under ``data/staging``. At the deadline, Ctrl-C, or a STOP-file
request, the process trees are terminated and a final summary is persisted.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from polysignal.ingestion.social.enrichment_config import load_enrichment_config
from polysignal.storage.run_lock import lock_is_active, run_lock

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
SESSION_ROOT = ROOT / "data" / "staging" / "long_social_session"
LOG_ROOT = SESSION_ROOT / "logs"
STATE_PATH = SESSION_ROOT / "session_state.json"
SUMMARY_PATH = SESSION_ROOT / "session_summary.json"
STOP_PATH = SESSION_ROOT / "STOP"
LOCK_PATH = SESSION_ROOT / "long_session.lock"
CHILD_LOCKS = (
    ROOT / "data" / "staging" / "enriched" / "scrapling_web.lock",
    ROOT
    / "data"
    / "staging"
    / "social_comments"
    / "state"
    / "comment_backfill.lock",
)

DATASETS = {
    "reddit_posts": ROOT
    / "data"
    / "staging"
    / "reddit"
    / "normalized"
    / "social"
    / "social_posts.parquet",
    "twitter_posts": ROOT / "data" / "staging" / "twitter_web" / "social_posts.parquet",
    "reddit_comments": ROOT
    / "data"
    / "staging"
    / "reddit"
    / "normalized"
    / "social"
    / "post_comments.parquet",
    "twitter_comments": ROOT
    / "data"
    / "staging"
    / "twitter_web"
    / "post_comments.parquet",
    "combined_comments": ROOT
    / "data"
    / "staging"
    / "social_comments"
    / "normalized"
    / "social"
    / "post_comments.parquet",
}

_STOP_REQUESTED = False
_ACTIVE_PROCESS: subprocess.Popen[str] | None = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bounded long social scrape.")
    parser.add_argument("--hours", type=float, default=10.0)
    parser.add_argument("--cycle-minutes", type=float, default=30.0)
    parser.add_argument("--reddit-target", type=int, default=500_000)
    parser.add_argument("--x-target-per-pass", type=int, default=6_000)
    parser.add_argument("--x-accounts-per-pass", type=int, default=8)
    parser.add_argument("--reddit-pass-minutes", type=float, default=12.0)
    parser.add_argument("--reddit-comment-posts-per-pass", type=int, default=150)
    parser.add_argument("--x-comment-posts-per-pass", type=int, default=75)
    parser.add_argument("--comments-per-post", type=int, default=10)
    parser.add_argument("--reddit-rate-limit", type=float, default=3.0)
    parser.add_argument("--comment-rate-limit", type=float, default=3.0)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _configure_logging() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_ROOT / "session.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def _archive_previous_logs() -> None:
    """Move prior current-session logs aside before a new run."""
    if not LOG_ROOT.exists():
        return
    current = [
        path
        for path in LOG_ROOT.glob("*.log")
        if path.is_file() and path.stat().st_size > 0
    ]
    if not current:
        return
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = LOG_ROOT / "archive" / stamp
    archive.mkdir(parents=True, exist_ok=True)
    for path in current:
        # A log still held open by an orphaned collector from a previous run cannot be
        # moved on Windows (WinError 32). Skip it rather than crashing the whole run;
        # logging opens in append mode, so leaving it in place is harmless.
        with suppress(OSError):
            os.replace(path, archive / path.name)


def _signal_handler(_signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _load_env_file() -> dict[str, str]:
    env = os.environ.copy()
    path = ROOT / ".env"
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    return env


def _row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return -1


def _counts() -> dict[str, int]:
    return {name: _row_count(path) for name, path in DATASETS.items()}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _stop_requested(deadline: float) -> bool:
    return _STOP_REQUESTED or STOP_PATH.exists() or time.monotonic() >= deadline


def _terminate_tree(process: subprocess.Popen[str] | None, label: str) -> None:
    if process is None or process.poll() is not None:
        return
    logging.info("Stopping %s process tree pid=%s", label, process.pid)
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=10)
    else:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def _clear_stale_child_locks() -> None:
    for path in CHILD_LOCKS:
        if path.exists() and not lock_is_active(path):
            path.unlink()
            logging.info("Removed stale child lock %s", path)


def _start_logged(
    command: list[str],
    log_path: Path,
    *,
    env: dict[str, str] | None = None,
) -> tuple[subprocess.Popen[str], Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    handle.write(
        f"\n[{datetime.now(tz=timezone.utc).isoformat()}] START "
        f"{subprocess.list2cmdline(command)}\n"
    )
    handle.flush()
    creation_flags = (
        subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    )
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        creationflags=creation_flags,
    )
    return process, handle


def _run_logged(
    command: list[str],
    log_path: Path,
    *,
    deadline: float,
    timeout_seconds: float,
    env: dict[str, str] | None = None,
    label: str,
    heartbeat: Callable[[], None] | None = None,
) -> str:
    global _ACTIVE_PROCESS
    process, handle = _start_logged(command, log_path, env=env)
    _ACTIVE_PROCESS = process
    started = time.monotonic()
    status = "unknown"
    try:
        if heartbeat:
            heartbeat()
        next_heartbeat = time.monotonic()
        while process.poll() is None:
            if _stop_requested(deadline):
                _terminate_tree(process, label)
                status = "stopped"
                break
            if time.monotonic() - started >= timeout_seconds:
                _terminate_tree(process, label)
                status = "timeout"
                break
            if heartbeat and time.monotonic() >= next_heartbeat:
                heartbeat()
                next_heartbeat = time.monotonic() + 5
            time.sleep(2)
        if status == "unknown":
            status = f"exit={process.returncode}"
    finally:
        handle.write(
            f"[{datetime.now(tz=timezone.utc).isoformat()}] END {label} {status}\n"
        )
        handle.close()
        _ACTIVE_PROCESS = None
    return status


def _reddit_command(args: argparse.Namespace) -> list[str]:
    return [
        str(PYTHON),
        "-m",
        "polysignal",
        "social",
        "collect-large-enriched",
        "--config",
        "configs/social_enrichment_sources.example.yaml",
        "--markets-path",
        "data/normalized/polymarket/markets.parquet",
        "--reddit-root",
        "data/staging/reddit",
        "--telegram-root",
        "data/staging/telegram_web",
        "--output-root",
        "data/staging/enriched",
        "--target-total-records",
        str(args.reddit_target),
        "--target-reddit-records",
        str(args.reddit_target),
        "--target-telegram-records",
        "0",
        "--max-pages-per-source",
        "2" if args.smoke else "200",
        "--max-pages-per-query",
        "1" if args.smoke else "10",
        "--max-results-per-query",
        "100",
        "--rate-limit-seconds",
        str(args.reddit_rate_limit),
        "--resume",
    ]


def _x_command(args: argparse.Namespace) -> list[str]:
    return [
        str(PYTHON),
        "scripts/collect_twitter_playwright.py",
        str(args.x_target_per_pass),
    ]


def _comments_command(args: argparse.Namespace) -> list[str]:
    reddit_posts = 1 if args.smoke else args.reddit_comment_posts_per_pass
    twitter_posts = 1 if args.smoke else args.x_comment_posts_per_pass
    comments_per_post = 2 if args.smoke else args.comments_per_post
    return [
        str(PYTHON),
        "scripts/collect_social_comments.py",
        "--platform",
        "all",
        "--reddit-max-posts",
        str(reddit_posts),
        "--twitter-max-posts",
        str(twitter_posts),
        "--comments-per-post",
        str(comments_per_post),
        "--rate-limit-seconds",
        str(args.comment_rate_limit),
    ]


def _metadata_command() -> list[str]:
    return [
        str(PYTHON),
        "scripts/collect_social_comments.py",
        "--platform",
        "all",
        "--metadata-only",
    ]


def _validate(args: argparse.Namespace, env: dict[str, str]) -> dict[str, int]:
    if not PYTHON.exists():
        raise RuntimeError(f"virtualenv Python not found: {PYTHON}")
    if args.hours <= 0 or args.cycle_minutes <= 0:
        raise ValueError("--hours and --cycle-minutes must be positive")
    if args.x_accounts_per_pass <= 0 or args.reddit_pass_minutes <= 0:
        raise ValueError("account/pass limits must be positive")
    if not env.get("X_AUTH_TOKEN") or not env.get("X_CT0"):
        raise RuntimeError("X_AUTH_TOKEN and X_CT0 are required in .env")
    config = load_enrichment_config(
        ROOT / "configs" / "social_enrichment_sources.example.yaml"
    )
    if not config.reddit or not config.reddit.parse_comments:
        raise RuntimeError("Reddit comment collection is disabled in config")
    if not config.twitter or not config.twitter.parse_comments:
        raise RuntimeError("X reply collection is disabled in config")
    return {
        "reddit_sources": len(config.reddit_sources),
        "twitter_config_sources": len(config.twitter_sources),
        "twitter_rotating_accounts": 40,
    }


def _sleep_until(
    next_cycle: float,
    deadline: float,
    heartbeat: Callable[[], None] | None = None,
) -> None:
    while time.monotonic() < next_cycle and not _stop_requested(deadline):
        if heartbeat:
            heartbeat()
        time.sleep(min(5.0, next_cycle - time.monotonic()))


def main() -> int:
    args = _parser().parse_args()
    _archive_previous_logs()
    _configure_logging()
    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    env = _load_env_file()
    source_counts = _validate(args, env)
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    if STOP_PATH.exists():
        STOP_PATH.unlink()
    start_wall = datetime.now(tz=timezone.utc)
    duration_seconds = 90.0 if args.smoke else args.hours * 3600
    deadline = time.monotonic() + duration_seconds
    baseline = _counts()
    cycles: list[dict[str, Any]] = []
    active_stage = "starting"
    active_detail = ""
    active_cycle = 0

    def heartbeat() -> None:
        _write_json(
            STATE_PATH,
            {
                "status": "running",
                "pid": os.getpid(),
                "started_at": start_wall.isoformat(),
                "heartbeat_at": datetime.now(tz=timezone.utc).isoformat(),
                "elapsed_minutes": round(
                    (datetime.now(tz=timezone.utc) - start_wall).total_seconds() / 60,
                    1,
                ),
                "requested_hours": args.hours,
                "source_counts": source_counts,
                "cycle": active_cycle,
                "active_stage": active_stage,
                "active_detail": active_detail,
                "counts": _counts(),
            },
        )

    with run_lock(LOCK_PATH, label="long-social-session"):
        logging.info(
            "SESSION START hours=%s sources=%s cycle_minutes=%s baseline=%s",
            args.hours,
            source_counts,
            args.cycle_minutes,
            baseline,
        )
        heartbeat()

        cycle = 0
        try:
            while not _stop_requested(deadline):
                cycle += 1
                active_cycle = cycle
                cycle_start = time.monotonic()
                offset = ((cycle - 1) * 9) % source_counts["twitter_rotating_accounts"]

                active_stage = "reddit_posts"
                active_detail = (
                    f"bounded discovery pass, up to {args.reddit_pass_minutes:g} minutes"
                )
                reddit_status = _run_logged(
                    _reddit_command(args),
                    LOG_ROOT / "reddit_posts.log",
                    deadline=deadline,
                    timeout_seconds=45 if args.smoke else args.reddit_pass_minutes * 60,
                    env=env,
                    label=f"reddit-posts-cycle-{cycle}",
                    heartbeat=heartbeat,
                )
                _clear_stale_child_locks()
                if _stop_requested(deadline):
                    break

                active_stage = "x_posts"
                active_detail = (
                    f"offset={offset}, accounts={args.x_accounts_per_pass}"
                )
                x_env = env.copy()
                x_env["POLYSIGNAL_X_ACCOUNT_OFFSET"] = str(offset)
                x_env["POLYSIGNAL_X_MAX_ACCOUNTS"] = str(
                    2 if args.smoke else args.x_accounts_per_pass
                )
                x_status = _run_logged(
                    _x_command(args),
                    LOG_ROOT / "x_posts.log",
                    deadline=deadline,
                    timeout_seconds=60 if args.smoke else 1800,
                    env=x_env,
                    label=f"x-posts-cycle-{cycle}",
                    heartbeat=heartbeat,
                )
                if _stop_requested(deadline):
                    break
                active_stage = "comments"
                active_detail = (
                    f"reddit_posts={args.reddit_comment_posts_per_pass}, "
                    f"x_posts={args.x_comment_posts_per_pass}"
                )
                comments_status = _run_logged(
                    _comments_command(args),
                    LOG_ROOT / "comments.log",
                    deadline=deadline,
                    timeout_seconds=60 if args.smoke else 3000,
                    env=env,
                    label=f"comments-cycle-{cycle}",
                    heartbeat=heartbeat,
                )
                counts = _counts()
                record = {
                    "cycle": cycle,
                    "finished_at": datetime.now(tz=timezone.utc).isoformat(),
                    "x_account_offset": offset,
                    "reddit_status": reddit_status,
                    "x_status": x_status,
                    "comments_status": comments_status,
                    "counts": counts,
                }
                cycles.append(record)
                logging.info("Cycle %s complete: %s", cycle, record)
                active_stage = "waiting"
                active_detail = "waiting for next cycle"
                heartbeat()
                interval = 5.0 if args.smoke else args.cycle_minutes * 60
                _sleep_until(cycle_start + interval, deadline, heartbeat)
        finally:
            _terminate_tree(_ACTIVE_PROCESS, "active-cycle-worker")
            _clear_stale_child_locks()

        if not args.smoke and not STOP_PATH.exists() and not _STOP_REQUESTED:
            metadata_status = _run_logged(
                _metadata_command(),
                LOG_ROOT / "comments.log",
                deadline=time.monotonic() + 600,
                timeout_seconds=600,
                env=env,
                label="final-comment-metadata",
                heartbeat=heartbeat,
            )
        else:
            metadata_status = "skipped"

        final_counts = _counts()
        finished = datetime.now(tz=timezone.utc)
        summary = {
            "status": "completed",
            "active_stage": "completed",
            "active_detail": "session finished",
            "pid": os.getpid(),
            "started_at": start_wall.isoformat(),
            "finished_at": finished.isoformat(),
            "heartbeat_at": finished.isoformat(),
            "elapsed_hours": round(
                (finished - start_wall).total_seconds() / 3600, 3
            ),
            "requested_hours": args.hours,
            "stopped_by_file": STOP_PATH.exists(),
            "source_counts": source_counts,
            "baseline_counts": baseline,
            "final_counts": final_counts,
            "deltas": {
                key: final_counts[key] - baseline.get(key, 0) for key in final_counts
            },
            "cycles": cycles,
            "metadata_status": metadata_status,
            "logs": {
                "session": str(LOG_ROOT / "session.log"),
                "reddit_posts": str(LOG_ROOT / "reddit_posts.log"),
                "x_posts": str(LOG_ROOT / "x_posts.log"),
                "comments": str(LOG_ROOT / "comments.log"),
            },
        }
        _write_json(SUMMARY_PATH, summary)
        _write_json(STATE_PATH, summary)
        if STOP_PATH.exists():
            STOP_PATH.unlink()
        logging.info("SESSION END summary=%s deltas=%s", SUMMARY_PATH, summary["deltas"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

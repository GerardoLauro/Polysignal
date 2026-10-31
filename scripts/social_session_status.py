"""Print a compact status snapshot for the long social collection session."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data/staging/long_social_session/session_state.json"
PATHS = {
    "reddit_posts": ROOT
    / "data/staging/reddit/normalized/social/social_posts.parquet",
    "twitter_posts": ROOT / "data/staging/twitter_web/social_posts.parquet",
    "reddit_comments": ROOT
    / "data/staging/reddit/normalized/social/post_comments.parquet",
    "twitter_comments": ROOT / "data/staging/twitter_web/post_comments.parquet",
    "combined_comments": ROOT
    / "data/staging/social_comments/normalized/social/post_comments.parquet",
}


def _count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return -1


def main() -> int:
    state = {}
    if STATE.exists():
        try:
            state = json.loads(STATE.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            state = {"status": "unreadable state"}
    print(f"status: {state.get('status', 'not started')}")
    print(f"pid: {state.get('pid', '-')}")
    print(f"started_at: {state.get('started_at', '-')}")
    print(f"heartbeat_at: {state.get('heartbeat_at', '-')}")
    heartbeat = state.get("heartbeat_at")
    if heartbeat:
        try:
            age = datetime.now(tz=timezone.utc) - datetime.fromisoformat(heartbeat)
            print(f"heartbeat_age_seconds: {int(age.total_seconds())}")
        except ValueError:
            pass
    print(f"cycle: {state.get('cycle', len(state.get('cycles', [])))}")
    print(f"active_stage: {state.get('active_stage', '-')}")
    print(f"active_detail: {state.get('active_detail', '-')}")
    for name, path in PATHS.items():
        print(f"{name}: {_count(path):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

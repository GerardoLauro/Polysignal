"""Offline tests for the social raw store."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from polysignal.ingestion.social.raw_store import (
    SocialRawWriter,
    generate_social_run_id,
)


def test_generate_run_id_is_short_and_sortable():
    now = datetime(2026, 5, 31, 12, 34, 56, tzinfo=timezone.utc)
    run_id = generate_social_run_id(now=now)
    assert run_id.startswith("123456Z-")
    assert len(run_id) <= 24


def test_writer_persists_post_and_manifest(tmp_path):
    writer = SocialRawWriter(directory=tmp_path / "run", platform="reddit")
    path = writer.write_post(
        query="trump 25th amendment",
        market_id="cond-1",
        post_id="abc123",
        payload={"hello": "world"},
    )
    writer.record_query(query="trump 25th amendment", market_id="cond-1", post_count=1)
    manifest_path = writer.write_manifest()
    assert path.exists()
    assert manifest_path.exists()

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["query"] == "trump 25th amendment"
    assert raw["market_id"] == "cond-1"
    assert raw["post_id"] == "abc123"
    assert raw["payload"] == {"hello": "world"}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["platform"] == "reddit"
    assert manifest["object_count"] == 1
    assert manifest["queries"] == [
        {
            "query": "trump 25th amendment",
            "market_id": "cond-1",
            "post_count": 1,
            "error": None,
        }
    ]


def test_writer_sanitizes_unsafe_post_id_for_filename(tmp_path):
    writer = SocialRawWriter(directory=tmp_path / "run", platform="reddit")
    path = writer.write_post(
        query="anything",
        market_id="cond/with/slashes",
        post_id="abc/../escape",
        payload={"ok": True},
    )
    # No traversal, no slashes in the filename.
    assert "/" not in path.name
    assert "\\" not in path.name
    assert path.parent == writer.directory


def test_writer_records_query_with_error_payload(tmp_path):
    writer = SocialRawWriter(directory=tmp_path / "run", platform="reddit")
    writer.record_query(
        query="failing query",
        market_id="cond-2",
        post_count=0,
        error="HTTP 429 — rate limit",
    )
    path = writer.write_manifest()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["queries"][0]["error"].startswith("HTTP 429")


def test_writer_uses_separate_files_when_same_post_is_hit_by_two_queries(tmp_path):
    """Fix 4 — query lineage is preserved on disk, not only in JSON."""
    writer = SocialRawWriter(directory=tmp_path / "run", platform="reddit")
    p1 = writer.write_post(
        query="trump 25th amendment",
        market_id="cond-1",
        post_id="abc123",
        payload={"hello": "from query A"},
    )
    p2 = writer.write_post(
        query="trump resignation",
        market_id="cond-1",
        post_id="abc123",
        payload={"hello": "from query B"},
    )
    # Distinct files — no collision.
    assert p1 != p2
    assert p1.exists()
    assert p2.exists()
    # Each file carries its own query in the wrapper.
    assert json.loads(p1.read_text(encoding="utf-8"))["query"] == "trump 25th amendment"
    assert json.loads(p2.read_text(encoding="utf-8"))["query"] == "trump resignation"


def test_writer_repeats_same_triple_overwrite_same_file(tmp_path):
    """Re-issuing the exact same (query, market, post) is idempotent on disk."""
    writer = SocialRawWriter(directory=tmp_path / "run", platform="reddit")
    p1 = writer.write_post(
        query="q",
        market_id="m",
        post_id="abc",
        payload={"version": 1},
    )
    p2 = writer.write_post(
        query="q",
        market_id="m",
        post_id="abc",
        payload={"version": 2},
    )
    assert p1 == p2
    # The latest payload wins (within the same run).
    assert json.loads(p1.read_text(encoding="utf-8"))["payload"] == {"version": 2}
    # object_count describes snapshots on disk, not attempted writes.
    manifest = json.loads(writer.write_manifest().read_text(encoding="utf-8"))
    assert manifest["object_count"] == 1


def test_writer_preserves_targeted_retrieval_context(tmp_path):
    writer = SocialRawWriter(directory=tmp_path / "run", platform="telegram")
    path = writer.write_post(
        query="bitcoin",
        market_id="cond-1",
        post_id="telegram:c:1",
        payload={"ok": True},
        context={"retrieval_work_id": "work-1", "retrieval_window_label": "0-7d"},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["retrieval_work_id"] == "work-1"
    assert payload["retrieval_window_label"] == "0-7d"

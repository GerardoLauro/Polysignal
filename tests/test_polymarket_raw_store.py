"""Tests for ``polysignal.ingestion.polymarket.raw_store``."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from polysignal.ingestion.polymarket.raw_store import RawRunWriter, generate_run_id
from polysignal.storage.paths import get_data_paths


def test_generate_run_id_is_unique() -> None:
    a = generate_run_id(datetime(2025, 1, 1, 12, 34, 56, tzinfo=timezone.utc))
    b = generate_run_id(datetime(2025, 1, 1, 12, 34, 56, tzinfo=timezone.utc))
    # Both encode the same wall-clock time but the random suffix
    # differs.
    assert a.startswith("123456Z-")
    assert b.startswith("123456Z-")
    assert a != b


def test_write_page_creates_json_file(tmp_path: Path) -> None:
    writer = RawRunWriter(
        directory=tmp_path / "run",
        endpoint="GET /markets",
        params={"limit": 10},
        user_agent="PolySignal/test",
    )
    path = writer.write_page(0, [{"id": "m1"}, {"id": "m2"}])
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == [{"id": "m1"}, {"id": "m2"}]
    assert writer.page_count == 1
    assert writer.object_count == 2


def test_write_named_persists_single_object(tmp_path: Path) -> None:
    writer = RawRunWriter(
        directory=tmp_path / "run",
        endpoint="GET /prices-history",
        params={},
    )
    path = writer.write_named("market_abc__token_xyz", {"history": []})
    assert path.exists()
    assert path.name == "market_abc__token_xyz.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"history": []}


def test_write_named_rejects_path_separators(tmp_path: Path) -> None:
    writer = RawRunWriter(directory=tmp_path / "run", endpoint="x", params={})
    with pytest.raises(ValueError, match="path separators"):
        writer.write_named("evil/../escape", {"k": "v"})


def test_write_manifest_records_run_metadata(tmp_path: Path) -> None:
    writer = RawRunWriter(
        directory=tmp_path / "run",
        endpoint="GET /markets",
        params={"limit": 5},
        user_agent="PolySignal/test",
    )
    writer.write_page(0, [{"id": "m1"}])
    writer.write_page(1, [{"id": "m2"}, {"id": "m3"}])
    manifest_path = writer.write_manifest()
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["endpoint"] == "GET /markets"
    assert manifest["params"] == {"limit": 5}
    assert manifest["page_count"] == 2
    assert manifest["object_count"] == 3
    assert manifest["user_agent"] == "PolySignal/test"
    # Timestamps must be timezone-aware UTC ISO strings.
    assert manifest["started_at"].endswith("+00:00")
    assert manifest["finished_at"].endswith("+00:00")


def test_raw_run_dir_layout_is_run_scoped(tmp_path: Path) -> None:
    paths = get_data_paths(root=tmp_path)
    when = datetime(2025, 5, 15, 12, 34, 56, tzinfo=timezone.utc)
    directory = paths.raw_run_dir("polymarket/markets", "abc123", when=when)
    assert directory.parts[-2:] == ("2025-05-15", "20250515T123456Z-abc123")
    assert str(directory).startswith(str(paths.raw_polymarket_markets))

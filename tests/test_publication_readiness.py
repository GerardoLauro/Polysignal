"""Integration regressions for the source-only GitHub package."""
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import sys
import threading
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location("publication_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def archive_fixture(tmp_path, data):
    archive = tmp_path / "01-dashboard.zip"
    with zipfile.ZipFile(archive, "w") as target:
        for name, content in data.items():
            target.writestr(name, content)
    return {"archives": [{"name": archive.name, "bytes": archive.stat().st_size,
                         "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                         "files": [{"path": name, "bytes": len(content),
                                    "sha256": hashlib.sha256(content).hexdigest()}
                                   for name, content in data.items()]}]}


def test_restore_and_verify_are_idempotent(tmp_path):
    module = load_script("restore_bundle_data")
    manifest = archive_fixture(tmp_path, {"data/example.txt": b"example"})
    destination = tmp_path / "project"
    assert module.restore(tmp_path, destination, manifest, verify_only=True) == 1
    assert not destination.exists()
    assert module.restore(tmp_path, destination, manifest) == 1
    assert module.restore(tmp_path, destination, manifest) == 0


def test_all_restore_conflicts_are_checked_before_writes(tmp_path):
    module = load_script("restore_bundle_data")
    manifest = archive_fixture(tmp_path, {"data/first.txt": b"one", "data/last.txt": b"two"})
    destination = tmp_path / "project"
    (destination / "data").mkdir(parents=True)
    existing = destination / "data/last.txt"
    existing.write_bytes(b"user-owned")
    with pytest.raises(FileExistsError):
        module.restore(tmp_path, destination, manifest)
    assert not (destination / "data/first.txt").exists()
    assert existing.read_bytes() == b"user-owned"


@pytest.mark.parametrize("name", ["../escape", "data/../../escape", "C:/escape", "data\\escape"])
def test_restore_rejects_unsafe_paths(tmp_path, name):
    module = load_script("restore_bundle_data")
    with pytest.raises(ValueError):
        module.safe_destination(tmp_path, name)


def test_default_release_excludes_large_data(tmp_path, monkeypatch):
    module = load_script("make_release")
    (tmp_path / "data/dataset_export").mkdir(parents=True)
    (tmp_path / "data/dataset_export/private.parquet").write_bytes(b"private")
    (tmp_path / "data/examples").mkdir()
    (tmp_path / "data/examples/public.csv").write_text("value\n1\n")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    files = module._iter_files()
    assert tmp_path / "data/examples/public.csv" in files
    assert not any("dataset_export" in str(p) for p in files)


def test_missing_data_returns_actionable_setup_message(monkeypatch):
    module = load_script("serve_app")
    monkeypatch.setattr(module, "build_ps_data", lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError()))
    data, error = module.load_dashboard_data()
    assert data is None
    assert "outside GitHub" in error
    assert module.main(["--check"]) == 2


def test_server_still_serves_ui_without_private_dataset():
    module = load_script("serve_app")
    server = module.ThreadingHTTPServer(("127.0.0.1", 0), module.Handler)
    server.ps_data = None
    server.data_error = "Dataset not installed"
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("GET", "/")
        response = connection.getresponse()
        assert response.status == 200
        assert b"PolySignal" in response.read()
        connection.request("GET", module.API_PATH)
        response = connection.getresponse()
        assert response.status == 503
        assert json.loads(response.read())["error"] == "Dataset not installed"
        assert response.getheader("Access-Control-Allow-Origin") is None
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def test_nested_settings_read_local_env_file(tmp_path, monkeypatch):
    from polysignal.config.settings import Neo4jSettings, PolymarketSettings, RedditSettings
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "NEO4J_URI=bolt://localhost:9999\nPOLYMARKET_REQUEST_TIMEOUT_SECONDS=12\nREDDIT_USER_AGENT=fixture-only\n",
        encoding="utf-8",
    )
    for name in ("NEO4J_URI", "POLYMARKET_REQUEST_TIMEOUT_SECONDS", "REDDIT_USER_AGENT"):
        monkeypatch.delenv(name, raising=False)
    assert Neo4jSettings().uri == "bolt://localhost:9999"
    assert PolymarketSettings().request_timeout_seconds == 12
    assert RedditSettings().user_agent == "fixture-only"


def test_raw_names_are_bounded_and_remain_traceable(tmp_path):
    from polysignal.ingestion.polymarket.raw_store import RawRunWriter
    writer = RawRunWriter(directory=tmp_path, endpoint="fixture", params={})
    original = "market_" + "a" * 64 + "__token_" + "b" * 80
    result = writer.write_named(original, {"fixture": True})
    assert len(result.name) <= 80
    assert writer.extra["named_files"][result.name] == original + ".json"
    assert json.loads(result.read_text())["fixture"] is True


def test_optional_fetcher_dependency_is_loaded_only_when_used(monkeypatch):
    from polysignal.ingestion.social.enrichment import ScraplingFetcher
    def unavailable():
        raise RuntimeError("optional library absent")
    monkeypatch.setattr(ScraplingFetcher, "_import_fetcher", staticmethod(unavailable))
    fetcher = ScraplingFetcher(user_agent="unit-test")
    with pytest.raises(RuntimeError, match="optional library"):
        fetcher.get("https://example.invalid")

"""Smoke tests for the Branch 05 linking CLI commands."""

from __future__ import annotations

import csv
from pathlib import Path

from typer.testing import CliRunner

from polysignal.cli.main import app
from polysignal.config import get_settings
from polysignal.linking import fixture_markets, fixture_posts, run_linking_cross_product
from polysignal.linking.linking_pipeline import write_linked_posts
from polysignal.storage.paths import get_data_paths

runner = CliRunner()


def test_linking_help_lists_branch05_commands() -> None:
    result = runner.invoke(app, ["linking", "--help"])
    assert result.exit_code == 0, result.output
    for cmd in ("score-candidates", "inspect-links", "evaluate"):
        assert cmd in result.output
    # Rich handles its own box drawing on Windows, but these arrows in
    # command descriptions crash the default CP1252 console renderer.
    assert "→" not in result.output
    assert "↔" not in result.output


def test_linking_score_candidates_dry_run_with_fixtures(tmp_path: Path) -> None:
    """--use-fixtures + --dry-run must succeed without parquet inputs."""
    get_settings.cache_clear()
    try:
        result = runner.invoke(
            app,
            [
                "linking",
                "score-candidates",
                "--use-fixtures",
                "--dry-run",
            ],
        )
    finally:
        get_settings.cache_clear()
    assert result.exit_code == 0, result.output
    assert "pairs scored" in result.output
    assert "dry-run" in result.output.lower()


def test_linking_score_candidates_writes_linked_posts_parquet(monkeypatch, tmp_path: Path) -> None:
    """--use-fixtures end-to-end produces a parquet file at the expected path."""
    monkeypatch.setenv("POLYSIGNAL_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        result = runner.invoke(
            app,
            ["linking", "score-candidates", "--use-fixtures"],
        )
        paths = get_data_paths()
        assert result.exit_code == 0, result.output
        assert paths.linked_posts_parquet.exists()
    finally:
        get_settings.cache_clear()


def test_linking_score_candidates_missing_posts_falls_back_to_fixtures(
    monkeypatch, tmp_path: Path
) -> None:
    """Missing social_posts.parquet must surface a user-friendly fallback,
    not a crash."""
    monkeypatch.setenv("POLYSIGNAL_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        result = runner.invoke(
            app,
            ["linking", "score-candidates", "--dry-run"],
        )
    finally:
        get_settings.cache_clear()
    assert result.exit_code == 0, result.output
    # The CLI explicitly announces the fallback.
    assert "fixture" in result.output.lower()


def test_linking_inspect_links_friendly_when_file_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("POLYSIGNAL_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        result = runner.invoke(app, ["linking", "inspect-links"])
    finally:
        get_settings.cache_clear()
    assert result.exit_code != 0
    assert "linked_posts" in result.output.lower() or "no linked_posts" in result.output.lower()


def test_linking_inspect_links_shows_telegram_suffix_and_exports_csv(tmp_path: Path) -> None:
    row = run_linking_cross_product(fixture_posts()[:1], fixture_markets()[:1])[0].model_copy(
        update={
            "post_id": "telegram:107272312345:9876",
            "post_text_preview": "🚨 readable alert",
            "relevance_score": 0.99,
        }
    )
    links_path = write_linked_posts([row], tmp_path / "linked_posts.parquet")
    csv_path = tmp_path / "linked_posts_sorted.csv"
    result = runner.invoke(
        app,
        [
            "linking",
            "inspect-links",
            "--links-path",
            str(links_path),
            "--limit",
            "1",
            "--export-csv",
            str(csv_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert ":9876" in result.output
    assert csv_path.exists()
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        exported = list(csv.DictReader(fh))
    assert exported[0]["post_id"] == "telegram:107272312345:9876"
    assert exported[0]["market_question"]
    assert exported[0]["post_text_preview"] == "🚨 readable alert"
    assert "llm_judged" in exported[0]


def test_linking_evaluate_reports_metrics() -> None:
    result = runner.invoke(app, ["linking", "evaluate", "--use-fixtures", "--no-report"])
    assert result.exit_code == 0, result.output
    for needle in ("accuracy", "macro_f1", "confusion matrix", "top errors"):
        assert needle.lower() in result.output.lower()


def test_linking_evaluate_writes_json_report(monkeypatch, tmp_path: Path) -> None:
    """--write-report (default) writes evaluation_report.json under analytical/."""
    monkeypatch.setenv("POLYSIGNAL_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        result = runner.invoke(app, ["linking", "evaluate", "--use-fixtures"])
        report_path = get_data_paths().linking_evaluation_report
        assert result.exit_code == 0, result.output
        assert report_path.exists()
        text = report_path.read_text(encoding="utf-8")
        assert "macro_f1" in text
        assert "false_positives" in text
    finally:
        get_settings.cache_clear()


def test_linking_score_candidates_with_no_embeddings(monkeypatch, tmp_path: Path) -> None:
    """--no-embeddings keeps deterministic behavior (embedding_enabled=False rows)."""
    monkeypatch.setenv("POLYSIGNAL_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        result = runner.invoke(
            app,
            ["linking", "score-candidates", "--use-fixtures", "--no-embeddings"],
        )
        assert result.exit_code == 0, result.output
        assert "disabled" in result.output
    finally:
        get_settings.cache_clear()


def test_linking_score_candidates_use_embeddings_falls_back_gracefully(
    monkeypatch, tmp_path: Path
) -> None:
    """--use-embeddings without sentence-transformers must NOT crash."""
    monkeypatch.setenv("POLYSIGNAL_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        result = runner.invoke(
            app,
            [
                "linking",
                "score-candidates",
                "--use-fixtures",
                "--use-embeddings",
                "--embedding-model",
                "nonexistent/model-does-not-exist",
            ],
        )
        # Either ST is installed and runs, or it's missing and we fall back —
        # both paths must exit zero. Empty-vector cosine collapses to 0 so
        # the deterministic baseline is preserved either way.
        assert result.exit_code == 0, result.output
    finally:
        get_settings.cache_clear()

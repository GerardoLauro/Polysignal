"""Smoke tests for the Typer CLI.

These run the CLI in-process via Typer's ``CliRunner``. No subprocess,
no network, no Neo4j, no LLM provider.
"""

from __future__ import annotations

from typer.testing import CliRunner

from polysignal.cli.main import app
from polysignal.graph.loaders import GraphLoadPlan

runner = CliRunner()


def test_root_help_works():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "PolySignal" in result.output


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "PolySignal" in result.output


def test_config_show_renders_a_table():
    result = runner.invoke(app, ["config-show"])
    assert result.exit_code == 0
    # The Rich table always includes at least one of the configured
    # endpoints — Polymarket is enabled by default.
    assert "polymarket" in result.output.lower()


def test_refresh_status_fast_lists_light_steps():
    result = runner.invoke(app, ["refresh", "status", "--mode", "fast"])
    assert result.exit_code == 0
    # Fast mode always plans the cheap refresh steps on a fresh state.
    assert "fetch_latest_prices" in result.output
    assert "compute_probability" in result.output


def test_refresh_status_unknown_mode_returns_error():
    result = runner.invoke(app, ["refresh", "status", "--mode", "nonsense"])
    assert result.exit_code != 0


def test_graph_init_dry_run_prints_cypher():
    result = runner.invoke(app, ["graph", "init", "--dry-run"])
    assert result.exit_code == 0
    assert "CREATE CONSTRAINT" in result.output
    assert "CREATE INDEX" in result.output


def test_graph_load_dry_run_does_not_contact_neo4j(monkeypatch):
    plan = GraphLoadPlan(
        datasets={},
        operations=[],
        planned_node_counts={"Contract": 0},
        planned_relationship_counts={"LINKED_TO": 0},
    )
    monkeypatch.setattr("polysignal.cli.main.build_load_plan", lambda **_: plan)
    result = runner.invoke(app, ["graph", "load", "--dry-run"])
    assert result.exit_code == 0
    assert "Neo4j was not contacted" in result.output


def test_graph_inspect_uses_offline_plan(monkeypatch):
    plan = GraphLoadPlan(
        datasets={},
        operations=[],
        planned_node_counts={"Contract": 0},
        planned_relationship_counts={"LINKED_TO": 0},
    )
    monkeypatch.setattr("polysignal.cli.main.build_load_plan", lambda **_: plan)
    result = runner.invoke(app, ["graph", "inspect"])
    assert result.exit_code == 0
    assert "planned graph" in result.output


def test_polymarket_help_lists_real_commands():
    """The polymarket subcommands are now real (no longer stubs)."""
    result = runner.invoke(app, ["polymarket", "--help"])
    assert result.exit_code == 0
    for cmd in ("fetch-markets", "inspect-market", "fetch-prices", "ingest"):
        assert cmd in result.output


def test_social_help_lists_branch04_commands():
    """Branch 04 surfaces real social subcommands (no longer a stub)."""
    result = runner.invoke(app, ["social", "--help"])
    assert result.exit_code == 0
    for cmd in ("generate-queries", "ingest", "reddit-search", "inspect-posts"):
        assert cmd in result.output


def test_social_ingest_live_without_credentials_exits_non_zero(monkeypatch):
    """Live ingestion (no --dry-run) exits non-zero when no creds are set."""
    for var in (
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "TWITTER_BEARER_TOKEN",
        "TELEGRAM_API_ID",
        "TELEGRAM_API_HASH",
    ):
        monkeypatch.delenv(var, raising=False)
    # Settings are cached; reset so the env change is picked up.
    from polysignal.config import get_settings

    get_settings.cache_clear()
    try:
        result = runner.invoke(app, ["social", "ingest", "--platform", "reddit"])
    finally:
        get_settings.cache_clear()
    assert result.exit_code != 0
    assert "credentials" in result.output.lower() or "skipping" in result.output.lower()


def test_social_ingest_dry_run_without_credentials_succeeds(monkeypatch, tmp_path):
    """Critical Fix 3 — dry-run must succeed even with no Reddit credentials."""
    for var in (
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "TWITTER_BEARER_TOKEN",
        "TELEGRAM_API_ID",
        "TELEGRAM_API_HASH",
    ):
        monkeypatch.delenv(var, raising=False)
    from polysignal.config import get_settings
    from polysignal.domain import Market, MarketDomain, Outcome
    from polysignal.storage.parquet_store import write_parquet
    from polysignal.storage.paths import get_data_paths

    # Seed a markets.parquet under tmp_path so the loader has something
    # to consume during the dry-run.
    paths = get_data_paths(root=tmp_path)
    paths.ensure()
    from datetime import datetime, timezone

    write_parquet(
        [
            Market(
                market_id="cond-1",
                slug="test",
                question="Will Trump be out as President before GTA VI?",
                domain=MarketDomain.POLITICS,
                created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                outcomes=[
                    Outcome(outcome_id="yes", label="Yes", index=0),
                    Outcome(outcome_id="no", label="No", index=1),
                ],
            )
        ],
        paths.polymarket_markets_parquet,
    )

    get_settings.cache_clear()
    try:
        result = runner.invoke(
            app,
            [
                "social",
                "ingest",
                "--platform",
                "reddit",
                "--limit-markets",
                "1",
                "--queries-per-market",
                "3",
                "--no-save-raw",
                "--dry-run",
                "--output-dir",
                str(tmp_path),
            ],
        )
    finally:
        get_settings.cache_clear()
    assert result.exit_code == 0, result.output
    # The summary header marks the run as a dry-run plan.
    assert "dry-run" in result.output.lower()
    # social_queries.parquet is still produced.
    assert paths.social_queries_parquet.exists()
    # social_posts.parquet is NOT produced.
    assert not paths.social_posts_parquet.exists()

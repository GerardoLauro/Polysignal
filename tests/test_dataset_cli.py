"""CLI smoke tests for the high-signal dataset commands.

The tests bypass the live Polymarket and Telegram APIs by writing
fixture parquet files into a temporary data root, then invoking the
Typer app against that root.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from polysignal.cli.main import app
from polysignal.dataset import HighSignalMarket, TelegramSearchPlanEntry
from polysignal.domain import Market, MarketDomain, Outcome
from polysignal.storage.parquet_store import read_parquet, write_parquet
from polysignal.storage.paths import get_data_paths


def _seed_polymarket_markets(root: Path) -> None:
    markets: list[Market] = []
    for i in range(20):
        domain = (
            MarketDomain.POLITICS
            if i < 7
            else MarketDomain.CRYPTO
            if i < 14
            else MarketDomain.SPORTS
        )
        markets.append(
            Market(
                market_id=f"m{i:02d}",
                slug=f"m-{i:02d}",
                question=f"Will entity{i} win the {2026} race by end of 2026?",
                domain=domain,
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                close_time=datetime(2026, 12, 31, tzinfo=timezone.utc),
                is_resolved=True,
                is_closed=True,
                outcomes=[
                    Outcome(outcome_id=f"y{i}", label="Yes", index=0),
                    Outcome(outcome_id=f"n{i}", label="No", index=1),
                ],
                volume=300_000.0,
                liquidity=50_000.0,
            )
        )
    paths = get_data_paths(root=root)
    paths.ensure()
    paths.polymarket_markets_parquet.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(markets, paths.polymarket_markets_parquet)


def _write_channels_yaml(tmp_path: Path) -> Path:
    target = tmp_path / "channels.yml"
    target.write_text(
        "channels:\n"
        "  - username: cointelegraph\n"
        "    domain: crypto\n"
        "    enabled: true\n"
        "  - username: politico\n"
        "    domain: politics\n"
        "    enabled: true\n",
        encoding="utf-8",
    )
    return target


def test_select_high_signal_markets_writes_parquet(tmp_path: Path) -> None:
    _seed_polymarket_markets(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "dataset",
            "select-high-signal-markets",
            "--target-markets",
            "10",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    paths = get_data_paths(root=tmp_path)
    rows = read_parquet(paths.high_signal_markets_parquet, HighSignalMarket)
    assert 0 < len(rows) <= 10
    assert all(0.0 <= row.market_priority_score <= 1.0 for row in rows)


def test_select_dry_run_does_not_write(tmp_path: Path) -> None:
    _seed_polymarket_markets(tmp_path)
    paths = get_data_paths(root=tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "dataset",
            "select-high-signal-markets",
            "--target-markets",
            "5",
            "--output-dir",
            str(tmp_path),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert not paths.high_signal_markets_parquet.exists()


def test_build_search_plan_works_without_credentials(tmp_path: Path) -> None:
    _seed_polymarket_markets(tmp_path)
    paths = get_data_paths(root=tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "dataset",
            "select-high-signal-markets",
            "--target-markets",
            "6",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output

    channels = _write_channels_yaml(tmp_path)
    result = runner.invoke(
        app,
        [
            "social",
            "telegram",
            "build-search-plan",
            "--channels-config",
            str(channels),
            "--queries-per-market",
            "3",
            "--max-queries-per-channel",
            "2",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    plan = read_parquet(paths.telegram_search_plan_parquet, TelegramSearchPlanEntry)
    assert plan, "plan should contain at least one entry"
    assert any(entry.channel_id == "cointelegraph" for entry in plan)
    assert paths.telegram_search_plan_json.exists()


def test_build_search_plan_warns_when_no_channels_configured(tmp_path: Path) -> None:
    _seed_polymarket_markets(tmp_path)
    paths = get_data_paths(root=tmp_path)
    runner = CliRunner()
    runner.invoke(
        app,
        [
            "dataset",
            "select-high-signal-markets",
            "--target-markets",
            "3",
            "--output-dir",
            str(tmp_path),
        ],
    )
    missing = tmp_path / "missing_channels.yml"
    result = runner.invoke(
        app,
        [
            "social",
            "telegram",
            "build-search-plan",
            "--channels-config",
            str(missing),
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "no channels config" in result.output.lower()
    plan = read_parquet(paths.telegram_search_plan_parquet, TelegramSearchPlanEntry)
    assert all(entry.channel_id is None for entry in plan)


def test_coverage_report_runs_without_messages_or_segments(tmp_path: Path) -> None:
    _seed_polymarket_markets(tmp_path)
    paths = get_data_paths(root=tmp_path)
    channels = _write_channels_yaml(tmp_path)
    runner = CliRunner()
    runner.invoke(
        app,
        [
            "dataset",
            "select-high-signal-markets",
            "--target-markets",
            "4",
            "--output-dir",
            str(tmp_path),
        ],
    )
    runner.invoke(
        app,
        [
            "social",
            "telegram",
            "build-search-plan",
            "--channels-config",
            str(channels),
            "--output-dir",
            str(tmp_path),
        ],
    )
    result = runner.invoke(
        app,
        [
            "social",
            "telegram",
            "coverage-report",
            "--channels-config",
            str(channels),
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(paths.telegram_coverage_report_json.read_text(encoding="utf-8"))
    assert payload["selected_markets_total"] > 0
    assert payload["raw_messages_collected"] == 0
    assert payload["limitations"]  # documents the missing telegram corpus


def test_build_manifest_collects_artefacts(tmp_path: Path) -> None:
    _seed_polymarket_markets(tmp_path)
    paths = get_data_paths(root=tmp_path)
    channels = _write_channels_yaml(tmp_path)
    runner = CliRunner()
    runner.invoke(
        app,
        [
            "dataset",
            "select-high-signal-markets",
            "--target-markets",
            "4",
            "--output-dir",
            str(tmp_path),
        ],
    )
    runner.invoke(
        app,
        [
            "social",
            "telegram",
            "build-search-plan",
            "--channels-config",
            str(channels),
            "--output-dir",
            str(tmp_path),
        ],
    )
    result = runner.invoke(
        app,
        [
            "dataset",
            "build-manifest",
            "--output-dir",
            str(tmp_path),
            "--notes",
            "smoke test",
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = json.loads(paths.high_signal_telegram_manifest_json.read_text(encoding="utf-8"))
    assert manifest["dataset_name"] == "high_signal_telegram"
    assert manifest["artefacts"]["selected_markets_parquet"].endswith("high_signal_markets.parquet")
    assert manifest["notes"] == ["smoke test"]

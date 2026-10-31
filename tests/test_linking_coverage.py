"""Tests for linking coverage statistics (Prompt 1 Task F)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from polysignal.cli.main import app
from polysignal.linking.coverage import (
    build_linking_coverage_report,
    write_linking_coverage_report,
)
from polysignal.storage.paths import get_data_paths

runner = CliRunner()


def _markets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"market_id": "m1", "thematic_category": "politics"},
            {"market_id": "m2", "thematic_category": "politics"},
            {"market_id": "m3", "thematic_category": "finance"},
        ]
    )


def _linked(rows: list[dict]) -> pd.DataFrame:
    cols = ["market_id", "platform", "relevance_label", "final_relevance_label"]
    return pd.DataFrame(rows, columns=cols)


def test_coverage_counts_accepted_links_case_insensitive():
    markets = _markets()
    linked = _linked(
        [
            # m1: 6 accepted reddit links (>=5)
            *[{"market_id": "m1", "platform": "reddit", "relevance_label": "relevant",
               "final_relevance_label": None} for _ in range(6)],
            # m2: 1 accepted telegram link, plus a rejected one (ambiguous excluded)
            {"market_id": "m2", "platform": "telegram", "relevance_label": "WEAKLY_RELEVANT",
             "final_relevance_label": None},
            {"market_id": "m2", "platform": "telegram", "relevance_label": "ambiguous",
             "final_relevance_label": None},
            # m3: zero accepted links
        ]
    )
    report = build_linking_coverage_report(markets, linked, thresholds=(1, 5))
    assert report.total_contracts == 3
    assert report.total_linked_posts == 7  # 6 + 1 accepted (ambiguous dropped)
    assert report.overall.counts_at_least_k["1"] == 2  # m1, m2
    assert report.overall.counts_at_least_k["5"] == 1  # only m1
    assert report.overall.fraction_at_least_k["1"] == round(2 / 3, 6)


def test_coverage_breaks_down_by_domain_and_platform():
    markets = _markets()
    linked = _linked(
        [
            {"market_id": "m1", "platform": "reddit", "relevance_label": "relevant",
             "final_relevance_label": None},
            {"market_id": "m3", "platform": "telegram", "relevance_label": "relevant",
             "final_relevance_label": None},
        ]
    )
    report = build_linking_coverage_report(markets, linked, thresholds=(1,))
    # politics has 2 contracts, 1 with a link -> 0.5
    assert report.by_domain["politics"].total_contracts == 2
    assert report.by_domain["politics"].fraction_at_least_k["1"] == 0.5
    # finance has 1 contract, 1 with a link -> 1.0
    assert report.by_domain["finance"].fraction_at_least_k["1"] == 1.0
    assert set(report.by_platform) == {"reddit", "telegram"}
    assert report.by_platform["reddit"].counts_at_least_k["1"] == 1


def test_coverage_final_label_overrides_baseline():
    markets = _markets()
    linked = _linked(
        [
            # baseline says relevant, but LLM final says not_relevant -> excluded
            {"market_id": "m1", "platform": "reddit", "relevance_label": "relevant",
             "final_relevance_label": "not_relevant"},
        ]
    )
    report = build_linking_coverage_report(markets, linked, thresholds=(1,))
    assert report.total_linked_posts == 0
    assert report.overall.counts_at_least_k["1"] == 0


def test_coverage_empty_linked_is_safe():
    report = build_linking_coverage_report(_markets(), pd.DataFrame(), thresholds=(1, 5))
    assert report.total_linked_posts == 0
    assert report.overall.fraction_at_least_k["1"] == 0.0
    assert report.by_platform == {}


def test_write_coverage_report_roundtrip(tmp_path: Path):
    report = build_linking_coverage_report(_markets(), pd.DataFrame(), thresholds=(1,))
    out = tmp_path / "coverage_report.json"
    write_linking_coverage_report(report, out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["total_contracts"] == 3
    assert "overall" in loaded and "by_domain" in loaded and "by_platform" in loaded
    assert loaded["thresholds"] == [1]


def test_cli_linking_coverage_writes_json(tmp_path: Path):
    paths = get_data_paths(root=tmp_path)
    paths.polymarket_markets_parquet.parent.mkdir(parents=True, exist_ok=True)
    paths.linked_posts_parquet.parent.mkdir(parents=True, exist_ok=True)
    _markets().to_parquet(paths.polymarket_markets_parquet, index=False)
    _linked(
        [{"market_id": "m1", "platform": "reddit", "relevance_label": "relevant",
          "final_relevance_label": None}]
    ).to_parquet(paths.linked_posts_parquet, index=False)
    result = runner.invoke(
        app, ["linking", "coverage", "--output-dir", str(tmp_path), "--thresholds", "1,5"]
    )
    assert result.exit_code == 0, result.output
    assert paths.linking_coverage_report_json.exists()
    data = json.loads(paths.linking_coverage_report_json.read_text(encoding="utf-8"))
    assert data["thresholds"] == [1, 5]
    assert data["overall"]["counts_at_least_k"]["1"] == 1

"""Branch 07 upstream data-quality tests.

Covers the resolved-market evaluation labels, the canonical stance
output path (no-LLM), the human review export, and the offline
real-data integration check. Everything here is fully offline: no live
APIs, no live LLMs, no Docker, and no Neo4j.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

from polysignal.cli.main import (
    _end_date_extra_params,
    _validate_discovery,
    _validate_market_status,
    app,
)
from polysignal.domain import (
    LinkedPost,
    Market,
    MarketDomain,
    Outcome,
    Platform,
    PricePoint,
    SocialPostRecord,
)
from polysignal.domain.enums import (
    ConstraintMatch,
    LinkMethod,
    RelevanceLabel,
    ResolutionResult,
    StanceLabel,
    StanceMethod,
)
from polysignal.evaluation import (
    build_resolved_labels,
    read_resolved_labels,
    write_resolved_labels,
)
from polysignal.graph.loaders import build_load_plan
from polysignal.linking.linking_pipeline import write_linked_posts
from polysignal.stance import read_stance_records, run_stance_pipeline
from polysignal.stance.models import StanceRecord, write_stance_records
from polysignal.storage.parquet_store import write_parquet
from polysignal.storage.paths import get_data_paths

UTC = timezone.utc


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _market(
    market_id: str,
    *,
    question: str,
    resolution: ResolutionResult | None,
    is_resolved: bool,
    is_closed: bool,
    domain: MarketDomain = MarketDomain.POLITICS,
    resolved_at: datetime | None = None,
) -> Market:
    return Market(
        market_id=market_id,
        slug=market_id,
        question=question,
        domain=domain,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        close_time=datetime(2024, 6, 1, tzinfo=UTC),
        resolved_at=resolved_at,
        is_resolved=is_resolved,
        is_closed=is_closed,
        resolution=resolution,
        condition_id=market_id,
        outcomes=[
            Outcome(market_id=market_id, outcome_id=f"{market_id}-yes", label="Yes", index=0),
            Outcome(market_id=market_id, outcome_id=f"{market_id}-no", label="No", index=1),
        ],
    )


def _outcomes_for(markets: list[Market]) -> list[Outcome]:
    out: list[Outcome] = []
    for market in markets:
        out.extend(market.outcomes)
    return out


@pytest.fixture
def mixed_markets() -> list[Market]:
    return [
        _market("m-yes", question="Will A happen?", resolution=ResolutionResult.YES,
                is_resolved=True, is_closed=True),
        _market("m-no", question="Will B happen?", resolution=ResolutionResult.NO,
                is_resolved=True, is_closed=True),
        _market("m-active", question="Will C happen?", resolution=None,
                is_resolved=False, is_closed=False),
        _market("m-invalid", question="Will D happen?", resolution=ResolutionResult.INVALID,
                is_resolved=True, is_closed=True),
    ]


# --------------------------------------------------------------------------- #
# Resolved labels — pure builder
# --------------------------------------------------------------------------- #
def test_resolved_labels_filters_to_resolved_only(mixed_markets: list[Market]) -> None:
    labels = build_resolved_labels(mixed_markets, _outcomes_for(mixed_markets))
    ids = {label.market_id for label in labels}
    # The active market must never appear in the evaluation set.
    assert "m-active" not in ids
    # YES, NO, and INVALID resolved markets are all retained as targets.
    assert ids == {"m-yes", "m-no", "m-invalid"}


def test_resolved_labels_map_winning_outcome(mixed_markets: list[Market]) -> None:
    labels = {label.market_id: label for label in
              build_resolved_labels(mixed_markets, _outcomes_for(mixed_markets))}
    yes = labels["m-yes"]
    assert yes.binary_yes_no == "YES"
    assert yes.winning_outcome_label == "Yes"
    assert yes.winning_outcome_id == "m-yes-yes"
    assert yes.winning_outcome_index == 0
    no = labels["m-no"]
    assert no.binary_yes_no == "NO"
    assert no.winning_outcome_label == "No"
    assert no.winning_outcome_id == "m-no-no"
    # INVALID has no winner — we never guess.
    invalid = labels["m-invalid"]
    assert invalid.winning_outcome_id is None
    assert invalid.binary_yes_no is None


def test_resolved_labels_dedup_by_market_id() -> None:
    # Gamma pagination can return the same market twice; the resolved set
    # must carry exactly one row per market_id.
    dup = [
        _market("m-dup", question="Will A happen?", resolution=ResolutionResult.YES,
                is_resolved=True, is_closed=True),
        _market("m-dup", question="Will A happen?", resolution=ResolutionResult.YES,
                is_resolved=True, is_closed=True),
    ]
    labels = build_resolved_labels(dup)
    assert len(labels) == 1
    assert labels[0].market_id == "m-dup"


def test_resolved_labels_record_source_fields(mixed_markets: list[Market]) -> None:
    labels = {label.market_id: label for label in
              build_resolved_labels(mixed_markets, _outcomes_for(mixed_markets))}
    assert "resolution" in labels["m-yes"].source_fields
    assert "is_resolved" in labels["m-yes"].source_fields


def test_resolved_labels_uses_inline_outcomes_when_outcomes_omitted(
    mixed_markets: list[Market],
) -> None:
    labels = build_resolved_labels(mixed_markets)  # no separate outcomes arg
    yes = next(label for label in labels if label.market_id == "m-yes")
    assert yes.winning_outcome_id == "m-yes-yes"
    assert yes.n_outcomes == 2


def test_resolved_labels_roundtrip_utc(tmp_path: Path, mixed_markets: list[Market]) -> None:
    labels = build_resolved_labels(mixed_markets, _outcomes_for(mixed_markets))
    path = tmp_path / "resolved.parquet"
    write_resolved_labels(labels, path)
    loaded = read_resolved_labels(path)
    assert len(loaded) == 3
    sample = loaded[0]
    assert sample.created_at is not None
    # Timestamps parse back as timezone-aware UTC.
    assert sample.created_at.tzinfo is not None
    assert sample.created_at.utcoffset() == UTC.utcoffset(None)


def test_resolved_labels_empty_writes_stable_schema(tmp_path: Path) -> None:
    active_only = [
        _market("m-active", question="Will C happen?", resolution=None,
                is_resolved=False, is_closed=False)
    ]
    labels = build_resolved_labels(active_only)
    assert labels == []
    path = tmp_path / "empty.parquet"
    write_resolved_labels(labels, path)
    frame = pd.read_parquet(path)
    assert len(frame) == 0
    # The column schema is stable even with zero resolved markets.
    assert "winning_outcome_id" in frame.columns
    assert read_resolved_labels(path) == []


# --------------------------------------------------------------------------- #
# Stance canonical path — no-LLM
# --------------------------------------------------------------------------- #
def _seed_linked_directional(root: Path) -> None:
    paths = get_data_paths(root=root)
    rows = [
        LinkedPost(
            link_id="l-yes",
            market_id="m-btc",
            post_id="p-yes",
            platform=Platform.TELEGRAM,
            relevance_score=0.8,
            relevance_label=RelevanceLabel.RELEVANT,
            confidence=0.8,
            method=LinkMethod.DETERMINISTIC_BASELINE,
            explanation="seed",
            matched_terms=["Bitcoin", "150k"],
            constraint_match=ConstraintMatch.UNKNOWN,
            created_at=datetime(2026, 5, 1, tzinfo=UTC),
            post_created_at=datetime(2026, 4, 1, tzinfo=UTC),
            market_question="Will Bitcoin hit $150k by December 2026?",
            post_text_preview="Yes, Bitcoin will definitely hit 150k — strong upside confirmed.",
            anchor_score=0.6,
        ),
        LinkedPost(
            link_id="l-unrel",
            market_id="m-btc",
            post_id="p-unrel",
            platform=Platform.TELEGRAM,
            relevance_score=0.05,
            relevance_label=RelevanceLabel.UNRELATED,
            confidence=0.2,
            method=LinkMethod.DETERMINISTIC_BASELINE,
            explanation="seed",
            constraint_match=ConstraintMatch.UNKNOWN,
            created_at=datetime(2026, 5, 1, tzinfo=UTC),
            market_question="Will Bitcoin hit $150k by December 2026?",
            post_text_preview="unrelated weather chatter",
        ),
    ]
    write_linked_posts(rows, paths.linked_posts_parquet)


def test_stance_pipeline_writes_canonical_path_without_llm(tmp_path: Path) -> None:
    _seed_linked_directional(tmp_path)
    paths = get_data_paths(root=tmp_path)
    result = run_stance_pipeline(paths=paths)  # llm_judge defaults to None
    assert result.output_path == paths.stance_labels_parquet
    assert paths.stance_labels_parquet.exists()
    records = read_stance_records(paths.stance_labels_parquet)
    # The unrelated row is filtered out by the eligibility gate.
    assert {r.market_id for r in records} == {"m-btc"}
    # No accidental LLM usage in the default heuristic path.
    assert all(r.method is StanceMethod.HEURISTIC for r in records)
    assert all(r.used_llm is False for r in records)


def test_stance_records_schema_is_graph_loader_compatible(tmp_path: Path) -> None:
    _seed_linked_directional(tmp_path)
    paths = get_data_paths(root=tmp_path)
    run_stance_pipeline(paths=paths)
    # The graph loader reads the same canonical parquet without error.
    records = read_stance_records(paths.stance_labels_parquet)
    assert records
    sample = records[0]
    for field in ("stance_id", "market_id", "post_id", "stance_label", "confidence"):
        assert hasattr(sample, field)


# --------------------------------------------------------------------------- #
# CLI — export-resolved-labels, inspect, export-review
# --------------------------------------------------------------------------- #
def _seed_polymarket(root: Path, markets: list[Market]) -> None:
    paths = get_data_paths(root=root)
    write_parquet(markets, paths.polymarket_markets_parquet)
    write_parquet(_outcomes_for(markets), paths.polymarket_outcomes_parquet)
    prices = [
        PricePoint(
            market_id=markets[0].market_id,
            outcome_id=markets[0].outcomes[0].outcome_id,
            timestamp=datetime(2024, 3, 1, tzinfo=UTC),
            price=0.6,
        )
    ]
    write_parquet(prices, paths.polymarket_price_points_parquet)


def test_cli_flag_validators() -> None:
    # Discovery + market-status accept their documented values and reject junk.
    assert _validate_discovery("markets") == "markets"
    assert _validate_discovery("EVENTS") == "events"
    assert _validate_market_status("closed") == "closed"
    with pytest.raises(typer.Exit):
        _validate_discovery("bogus")
    with pytest.raises(typer.Exit):
        _validate_market_status("settled")
    # end-date-min builds the Gamma filter from an ISO date, or rejects junk.
    assert _end_date_extra_params(None) is None
    assert _end_date_extra_params("2024-06-01") == {"end_date_min": "2024-06-01"}
    with pytest.raises(typer.Exit):
        _end_date_extra_params("last-tuesday")


def test_cli_export_resolved_labels(tmp_path: Path, mixed_markets: list[Market]) -> None:
    _seed_polymarket(tmp_path, mixed_markets)
    out = tmp_path / "resolved_market_labels.parquet"
    result = CliRunner().invoke(
        app,
        ["polymarket", "export-resolved-labels", "--source-dir", str(tmp_path),
         "--output-path", str(out)],
    )
    assert result.exit_code == 0, result.output
    labels = read_resolved_labels(out)
    assert {label.market_id for label in labels} == {"m-yes", "m-no", "m-invalid"}


def test_cli_polymarket_inspect_resolved_only(tmp_path: Path, mixed_markets: list[Market]) -> None:
    _seed_polymarket(tmp_path, mixed_markets)
    result = CliRunner().invoke(
        app,
        ["polymarket", "inspect", "--resolved-only", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "resolved markets" in result.output
    # 3 of 4 fixture markets are resolved.
    assert "unresolved markets" in result.output


def test_cli_polymarket_inspect_missing_dataset(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["polymarket", "inspect", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 2


def test_cli_export_review_writes_csv_and_parquet(tmp_path: Path) -> None:
    _seed_linked_directional(tmp_path)
    paths = get_data_paths(root=tmp_path)
    # Provide a market for domain join.
    write_parquet(
        [_market("m-btc", question="Will Bitcoin hit $150k by December 2026?",
                 resolution=None, is_resolved=False, is_closed=False,
                 domain=MarketDomain.CRYPTO)],
        paths.polymarket_markets_parquet,
    )
    # Seed a stance record to exercise the join.
    run_stance_pipeline(paths=paths)
    csv_out = tmp_path / "review.csv"
    parquet_out = tmp_path / "review.parquet"
    result = CliRunner().invoke(
        app,
        ["linking", "export-review", "--output-dir", str(tmp_path),
         "--csv-output", str(csv_out), "--parquet-output", str(parquet_out)],
    )
    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(csv_out.open("r", encoding="utf-8-sig")))
    assert rows
    header = rows[0].keys()
    for col in ("market_id", "market_question", "domain", "retrieval_query",
                "relevance_score", "stance_label", "text", "temporal_valid"):
        assert col in header
    # Only the meaningful (relevant) row is exported, not the unrelated one.
    assert all(r["relevance_label"] != "unrelated" for r in rows)
    assert pd.read_parquet(parquet_out).shape[0] == len(rows)


def test_cli_polymarket_inspect_deduplicates_market_and_outcome_shape(tmp_path: Path) -> None:
    market = _market(
        "m-dup",
        question="Will the duplicate market resolve?",
        resolution=ResolutionResult.YES,
        is_resolved=True,
        is_closed=True,
    )
    paths = get_data_paths(root=tmp_path)
    write_parquet([market, market], paths.polymarket_markets_parquet)
    write_parquet([*market.outcomes, *market.outcomes], paths.polymarket_outcomes_parquet)
    result = CliRunner().invoke(
        app,
        ["polymarket", "inspect", "--resolved-only", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "duplicate market rows removed" in result.output
    assert "1" in result.output
    assert "duplicate outcome rows removed" in result.output


# --------------------------------------------------------------------------- #
# Real-data integration check — graph load plan
# --------------------------------------------------------------------------- #
def _seed_graph_inputs(root: Path) -> None:
    paths = get_data_paths(root=root)
    market = _market("m-btc", question="Will Bitcoin hit $150k by December 2026?",
                     resolution=None, is_resolved=False, is_closed=False,
                     domain=MarketDomain.CRYPTO)
    write_parquet([market], paths.polymarket_markets_parquet)
    write_parquet(market.outcomes, paths.polymarket_outcomes_parquet)
    post = SocialPostRecord(
        post_id="parent-1",
        platform=Platform.TELEGRAM,
        source_id="parent-1",
        author_id="a1",
        author_name="Cointelegraph",
        created_at=datetime(2026, 4, 1, tzinfo=UTC),
        text="Bitcoin will hit 150k",
        collected_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    write_parquet([post], paths.social_posts_parquet)
    link = LinkedPost(
        link_id="lk1",
        market_id="m-btc",
        post_id="parent-1",
        platform=Platform.TELEGRAM,
        relevance_score=0.8,
        relevance_label=RelevanceLabel.RELEVANT,
        confidence=0.8,
        method=LinkMethod.DETERMINISTIC_BASELINE,
        constraint_match=ConstraintMatch.UNKNOWN,
        created_at=datetime(2026, 5, 1, tzinfo=UTC),
        market_question="Will Bitcoin hit $150k by December 2026?",
        parent_post_id="parent-1",
    )
    write_linked_posts([link], paths.linked_posts_parquet)


def test_integration_graph_plan_with_directional_and_neutral_stance(tmp_path: Path) -> None:
    _seed_graph_inputs(tmp_path)
    paths = get_data_paths(root=tmp_path)
    stance_rows = [
        StanceRecord(
            stance_id="s-dir",
            market_id="m-btc",
            post_id="parent-1",
            parent_post_id="parent-1",
            stance_label=StanceLabel.SUPPORTS_YES,
            stance_score=0.7,
            confidence=0.7,
            method=StanceMethod.HEURISTIC,
            created_at=datetime(2026, 5, 1, tzinfo=UTC),
            source_platform=Platform.TELEGRAM,
            source_id="parent-1",
        ),
        StanceRecord(
            stance_id="s-neutral",
            market_id="m-btc",
            post_id="parent-1",
            parent_post_id="parent-1",
            stance_label=StanceLabel.NEUTRAL,
            stance_score=0.0,
            confidence=0.3,
            method=StanceMethod.HEURISTIC,
            created_at=datetime(2026, 5, 1, tzinfo=UTC),
            source_platform=Platform.TELEGRAM,
            source_id="parent-1",
        ),
    ]
    write_stance_records(stance_rows, paths.stance_labels_parquet)

    plan = build_load_plan(paths=paths)
    # Core graph plans from real datasets.
    assert plan.planned_node_counts["Contract"] == 1
    assert plan.planned_relationship_counts["LINKED_TO"] == 1
    # Directional stance becomes a SUPPORTS edge; the neutral row is skipped.
    assert plan.planned_relationship_counts["SUPPORTS"] == 1
    assert plan.skipped_rows.get("stance_non_directional") == 1


def test_integration_graph_plan_skips_stance_when_absent(tmp_path: Path) -> None:
    _seed_graph_inputs(tmp_path)
    paths = get_data_paths(root=tmp_path)
    plan = build_load_plan(paths=paths)
    assert plan.planned_relationship_counts["SUPPORTS"] == 0
    assert any("stance" in reason.lower() for reason in plan.skipped_components)


# --------------------------------------------------------------------------- #
# Graph query CLI — dry-run prints parameterized Cypher without Neo4j
# --------------------------------------------------------------------------- #
def test_cli_graph_queries_list_and_dry_run() -> None:
    runner = CliRunner()
    listing = runner.invoke(app, ["graph", "queries", "list"])
    assert listing.exit_code == 0, listing.output
    assert "contract-coverage" in listing.output
    assert "stance-summary" in listing.output

    top = runner.invoke(app, ["graph", "queries", "top-contracts", "--dry-run", "--limit", "7"])
    assert top.exit_code == 0, top.output
    assert "$limit" in top.output
    assert "Neo4j was not contacted" in top.output

    nbr = runner.invoke(
        app, ["graph", "queries", "contract-neighborhood", "--market-id", "m-btc", "--dry-run"]
    )
    assert nbr.exit_code == 0, nbr.output
    assert "$market_id" in nbr.output

    health = runner.invoke(app, ["graph", "queries", "health", "--dry-run"])
    assert health.exit_code == 0, health.output
    assert "posts_without_author" in health.output

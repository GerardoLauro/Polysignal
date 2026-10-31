"""Offline Branch 09 evaluation/backtest tests.

No live API, Neo4j, or LLM provider is involved. The fixture writes the same
canonical Parquet artefacts that the real pipeline consumes.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from polysignal.cli.main import app
from polysignal.domain import Market, MarketDomain, Outcome, PricePoint, ResolutionResult
from polysignal.evaluation.backtest import (
    FORBIDDEN_SOCIAL_FEATURE_COLUMNS,
    SOCIAL_MODEL_FEATURES,
    BacktestConfig,
    build_evaluation_dataset,
    resolve_backtest_profile_paths,
    run_backtest,
    write_backtest_artifacts,
)
from polysignal.evaluation.benchmark_profiles import (
    REAL_OPTIONAL,
    REAL_SOCIAL_ENRICHED_MAIN,
    get_benchmark_profile,
    resolve_benchmark_profile,
)
from polysignal.evaluation.dashboard_view import (
    CLAIM_DISCLAIMER,
    DASHBOARD_VIEW_COLUMNS,
    build_dashboard_prediction_view,
    write_dashboard_prediction_view,
)
from polysignal.features.models import SocialPressureFeatures, write_social_pressure_features
from polysignal.storage.parquet_store import write_parquet
from polysignal.storage.paths import DataPaths, get_data_paths

ANCHOR = datetime(2025, 6, 1, tzinfo=timezone.utc)
runner = CliRunner()


def _feature(
    market_id: str,
    *,
    target_yes: bool,
    evidence: int = 6,
    dataset_mode: str = "real",
) -> SocialPressureFeatures:
    balance = 0.8 if target_yes else -0.8
    yes_count = evidence if target_yes else 0
    no_count = evidence if not target_yes else 0
    return SocialPressureFeatures(
        market_id=market_id,
        condition_id=market_id,
        question=f"Will {market_id} happen?",
        domain="crypto",
        close_time=ANCHOR,
        dataset_mode=dataset_mode,
        accepted_link_count=evidence,
        linked_post_count=evidence,
        unique_post_count=evidence,
        unique_author_count=min(evidence, 3),
        unique_channel_count=min(evidence, 2),
        relevant_count=evidence,
        supports_yes_count=yes_count,
        supports_no_count=no_count,
        directional_stance_count=evidence,
        yes_pressure_raw=float(yes_count),
        no_pressure_raw=float(no_count),
        net_pressure_raw=float(yes_count - no_count),
        yes_pressure_weighted=0.8 * yes_count,
        no_pressure_weighted=0.8 * no_count,
        net_pressure_weighted=0.8 * (yes_count - no_count),
        pressure_direction="YES" if target_yes else "NO",
        pressure_strength=abs(balance),
        stance_balance=balance,
        confidence_mean=0.8 if evidence else 0.0,
        confidence_weighted_score=(balance + 1.0) / 2.0,
        temporal_decay_pressure=balance,
        late_pressure=balance,
        evidence_quality_score=0.8 if evidence else 0.0,
        source_diversity_score=0.7 if evidence else 0.0,
        confidence_coverage=1.0 if evidence else 0.0,
        accepted_evidence_ratio=1.0 if evidence else 0.0,
        graph_linked_post_count=evidence,
        graph_supports_count=yes_count,
        graph_opposes_count=no_count,
        graph_author_count=min(evidence, 3),
        graph_channel_count=min(evidence, 2),
        graph_neighborhood_size=evidence * 2,
        engagement_total=float(evidence * 10),
        engagement_mean=10.0 if evidence else 0.0,
        engagement_max=10.0 if evidence else 0.0,
        engagement_log_total=float(evidence),
    )


def _seed_dataset(
    root: Path,
    *,
    n_markets: int = 12,
    with_prices: bool = True,
    evidence: int = 6,
    dataset_mode: str = "real",
    target_yes_every: int = 2,
) -> None:
    paths = get_data_paths(root=root)
    markets: list[Market] = []
    outcomes: list[Outcome] = []
    prices: list[PricePoint] = []
    features: list[SocialPressureFeatures] = []
    for index in range(n_markets):
        market_id = f"m-{index:02d}"
        target_yes = index % target_yes_every == 0
        yes = Outcome(market_id=market_id, outcome_id=f"{market_id}-yes", label="Yes", index=0)
        no = Outcome(market_id=market_id, outcome_id=f"{market_id}-no", label="No", index=1)
        markets.append(
            Market(
                market_id=market_id,
                condition_id=market_id,
                slug=market_id,
                question=f"Will {market_id} happen?",
                domain=MarketDomain.CRYPTO,
                created_at=ANCHOR - timedelta(days=30),
                close_time=ANCHOR,
                is_resolved=True,
                is_closed=True,
                resolution=ResolutionResult.YES if target_yes else ResolutionResult.NO,
                outcomes=[yes, no],
            )
        )
        outcomes.extend([yes, no])
        if with_prices:
            prices.append(
                PricePoint(
                    market_id=market_id,
                    outcome_id=yes.outcome_id,
                    timestamp=ANCHOR - timedelta(hours=1),
                    price=0.7 if target_yes else 0.3,
                )
            )
            # Explicit leakage trap: a later settled price must never win.
            prices.append(
                PricePoint(
                    market_id=market_id,
                    outcome_id=yes.outcome_id,
                    timestamp=ANCHOR + timedelta(hours=1),
                    price=1.0 if target_yes else 0.0,
                )
            )
        features.append(
            _feature(
                market_id,
                target_yes=target_yes,
                evidence=evidence,
                dataset_mode=dataset_mode,
            )
        )
    write_parquet(markets, paths.polymarket_markets_parquet)
    write_parquet(outcomes, paths.polymarket_outcomes_parquet)
    if prices:
        write_parquet(prices, paths.polymarket_price_points_parquet)
    write_social_pressure_features(features, paths.social_pressure_features_parquet)


def test_evaluation_dataset_joins_canonical_targets_without_leakage(tmp_path: Path):
    _seed_dataset(tmp_path)
    dataset = build_evaluation_dataset(
        paths=get_data_paths(root=tmp_path),
        dataset_mode="real",
    )
    assert len(dataset.frame) == 12
    assert dataset.metadata["join_keys"] == ["market_id"]
    assert "build_resolved_labels" in dataset.metadata["target_source"]
    assert dataset.metadata["row_counts"]["rows_with_preclose_market_probability"] == 12
    assert set(dataset.frame["market_probability"]) == {0.3, 0.7}
    assert set(SOCIAL_MODEL_FEATURES).isdisjoint(FORBIDDEN_SOCIAL_FEATURE_COLUMNS)
    assert "target_yes" not in SOCIAL_MODEL_FEATURES
    assert "winning_outcome_id" not in SOCIAL_MODEL_FEATURES


def test_social_feature_table_excludes_resolved_labels():
    fields = SocialPressureFeatures.model_fields
    assert FORBIDDEN_SOCIAL_FEATURE_COLUMNS.isdisjoint(fields)


def test_synthetic_expected_direction_is_rejected_as_a_feature(tmp_path: Path):
    _seed_dataset(tmp_path)
    paths = get_data_paths(root=tmp_path)
    frame = pd.read_parquet(paths.social_pressure_features_parquet)
    frame["synthetic_expected_direction"] = "YES"
    frame.to_parquet(paths.social_pressure_features_parquet, index=False)
    with pytest.raises(ValueError, match="forbidden evaluation-only"):
        build_evaluation_dataset(paths=paths)


def test_market_only_social_only_and_market_social_models(tmp_path: Path):
    _seed_dataset(tmp_path)
    config = BacktestConfig(dataset_mode="real")
    first = run_backtest(paths=get_data_paths(root=tmp_path), config=config)
    second = run_backtest(paths=get_data_paths(root=tmp_path), config=config)

    assert set(first.predictions) == {"constant", "market_only", "social_only", "market_social"}
    assert set(first.predictions["market_only"]["prediction"]) == {0.3, 0.7}
    assert first.predictions["social_only"]["model_status"].str.contains("oof_regularized").all()
    assert first.predictions["market_social"]["model_status"].str.contains("oof_regularized").all()
    assert first.metrics_summary["models"]["market_only"]["brier_score"] == pytest.approx(0.09)
    assert first.metrics_summary["models"]["social_only"]["brier_score"] < 0.25
    assert first.metrics_summary["models"]["market_social"]["brier_score"] < 0.25
    for model in first.predictions:
        pd.testing.assert_frame_equal(first.predictions[model], second.predictions[model])


def test_sparse_real_dataset_remains_compatible_and_honest(tmp_path: Path):
    _seed_dataset(tmp_path, n_markets=6, with_prices=False, evidence=0, dataset_mode="real")
    run = run_backtest(
        paths=get_data_paths(root=tmp_path),
        config=BacktestConfig(dataset_mode="real"),
    )
    assert run.predictions["market_only"].empty
    assert run.predictions["market_social"].empty
    assert run.metrics_summary["models"]["market_only"]["brier_score"] is None
    assert run.dataset.metadata["evidence_coverage"]["rows_with_sufficient_evidence"] == 0
    assert "insufficient for strong conclusions" in run.report["interpretation"]


def test_exclude_insufficient_evidence_filters_sparse_rows(tmp_path: Path):
    _seed_dataset(tmp_path, n_markets=4, with_prices=False, evidence=0, dataset_mode="real")
    dataset = build_evaluation_dataset(
        paths=get_data_paths(root=tmp_path),
        dataset_mode="real",
        exclude_insufficient=True,
    )
    assert dataset.frame.empty


def test_artifact_writer_and_cli_smoke(tmp_path: Path):
    _seed_dataset(tmp_path)
    paths = get_data_paths(root=tmp_path)
    run = run_backtest(paths=paths, config=BacktestConfig(dataset_mode="real"))
    written = write_backtest_artifacts(run, paths=paths)
    artifacts = paths.backtest_profile(REAL_OPTIONAL)
    assert artifacts.evaluation_dataset_parquet.exists()
    assert artifacts.predictions_market_only_parquet.exists()
    assert artifacts.predictions_social_only_parquet.exists()
    assert artifacts.predictions_market_social_parquet.exists()
    assert artifacts.metrics_summary_json.exists()
    assert artifacts.calibration_bins_csv.exists()
    assert artifacts.backtest_report_json.exists()
    assert artifacts.benchmark_profile_json.exists()
    assert artifacts.warnings_json.exists()
    assert not paths.evaluation_dataset_parquet.exists()
    assert "backtest_report_json" in written

    inspect = runner.invoke(
        app,
        [
            "evaluate",
            "inspect-backtest",
            "--output-dir",
            str(tmp_path),
            "--dataset-mode",
            "real",
        ],
    )
    assert inspect.exit_code == 0, inspect.output
    assert "Branch 09 backtest metrics" in inspect.output


def test_cli_backtest_dry_run_writes_nothing(tmp_path: Path):
    _seed_dataset(tmp_path)
    paths = get_data_paths(root=tmp_path)
    result = runner.invoke(
        app,
        [
            "evaluate",
            "backtest",
            "--output-dir",
            str(tmp_path),
            "--dataset-mode",
            "real",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "--dry-run set; no files written" in result.output
    assert not paths.backtest_profile(REAL_OPTIONAL).evaluation_dataset_parquet.exists()


def test_profile_resolution_defaults_and_legacy_compatibility(tmp_path: Path):
    paths = get_data_paths(root=tmp_path)
    profile, resolved_paths = resolve_backtest_profile_paths(paths=paths)
    assert profile.name == REAL_OPTIONAL
    assert resolved_paths == paths
    assert resolve_benchmark_profile(legacy_dataset_mode="real").name == REAL_OPTIONAL
    assert get_benchmark_profile(REAL_SOCIAL_ENRICHED_MAIN).default_root == "data_social_enriched_eval"
    # An explicit non-default real profile conflicts with the legacy ``real`` mode,
    # which always maps to real_optional.
    with pytest.raises(ValueError, match="conflicts"):
        resolve_benchmark_profile(
            benchmark_profile=REAL_SOCIAL_ENRICHED_MAIN,
            legacy_dataset_mode="real",
        )
    # The synthetic legacy mode was removed and is now rejected.
    with pytest.raises(ValueError, match="dataset_mode must be one of"):
        resolve_benchmark_profile(legacy_dataset_mode="synthetic")


def test_enriched_profile_is_deterministic_and_preserves_branch08(tmp_path: Path):
    _seed_dataset(tmp_path, n_markets=64, target_yes_every=8)
    paths = get_data_paths(root=tmp_path)
    before = paths.social_pressure_features_parquet.read_bytes()
    config = BacktestConfig(benchmark_profile=REAL_SOCIAL_ENRICHED_MAIN)

    first = run_backtest(paths=paths, config=config)
    second = run_backtest(paths=paths, config=config)

    assert paths.social_pressure_features_parquet.read_bytes() == before
    assert first.dataset.metadata["benchmark_profile"] == REAL_SOCIAL_ENRICHED_MAIN
    assert first.dataset.metadata["market_probability_source"] == (
        "latest_valid_preclose_yes_price"
    )
    assert first.dataset.metadata["row_counts"]["rows_with_preclose_market_probability"] == 64
    # No synthetic overlay exists: market priors are the canonical pre-close prices.
    assert first.dataset.controlled_social_features is None
    assert set(first.dataset.frame["market_probability"]) == {0.3, 0.7}
    pd.testing.assert_frame_equal(first.dataset.frame, second.dataset.frame)
    social_metrics = first.metrics_summary["models"]["social_only"]
    assert 0.5 < social_metrics["auc"] <= 1.0

    written = write_backtest_artifacts(first, paths=paths, export_csv=True)
    assert paths.social_pressure_features_parquet.read_bytes() == before
    assert "controlled_social_pressure_features" not in written
    assert paths.backtest_profile(REAL_SOCIAL_ENRICHED_MAIN).predictions_market_social_parquet.exists()
    assert paths.backtest_profile(REAL_SOCIAL_ENRICHED_MAIN).predictions_csv("market_social").exists()


def test_real_optional_keeps_original_features_and_canonical_prices(tmp_path: Path):
    _seed_dataset(tmp_path)
    paths = get_data_paths(root=tmp_path)
    dataset = build_evaluation_dataset(paths=paths, benchmark_profile=REAL_OPTIONAL)
    assert dataset.controlled_social_features is None
    assert dataset.metadata["input_paths"]["branch08_social_pressure_features"] == str(
        paths.social_pressure_features_parquet
    )
    assert dataset.metadata["input_paths"]["effective_social_pressure_features"] == str(
        paths.social_pressure_features_parquet
    )
    assert set(dataset.frame["market_probability"]) == {0.3, 0.7}


def test_real_optional_keeps_canonical_latest_preclose_prices(tmp_path: Path):
    _seed_dataset(tmp_path, dataset_mode="real")
    dataset = build_evaluation_dataset(
        paths=get_data_paths(root=tmp_path),
        benchmark_profile=REAL_OPTIONAL,
    )
    assert dataset.metadata["dataset_mode"] == "real"
    assert dataset.metadata["market_probability_source"] == "latest_valid_preclose_yes_price"
    assert set(dataset.frame["market_probability"]) == {0.3, 0.7}


def test_profile_metadata_columns_are_rejected_from_branch08_input(tmp_path: Path):
    _seed_dataset(tmp_path)
    paths = get_data_paths(root=tmp_path)
    frame = pd.read_parquet(paths.social_pressure_features_parquet)
    frame["benchmark_profile"] = REAL_SOCIAL_ENRICHED_MAIN
    frame.to_parquet(paths.social_pressure_features_parquet, index=False)
    with pytest.raises(ValueError, match="forbidden evaluation-only"):
        build_evaluation_dataset(paths=paths, benchmark_profile=REAL_OPTIONAL)


def test_cli_default_profile_inspection_and_conflict_rejection(tmp_path: Path):
    _seed_dataset(tmp_path)
    inspected = runner.invoke(
        app,
        ["evaluate", "inspect-profile", "--output-dir", str(tmp_path)],
    )
    assert inspected.exit_code == 0, inspected.output
    assert REAL_OPTIONAL in inspected.output
    assert "real empirical claims allowed" in inspected.output

    default_backtest = runner.invoke(
        app,
        ["evaluate", "backtest", "--output-dir", str(tmp_path), "--dry-run"],
    )
    assert default_backtest.exit_code == 0, default_backtest.output
    assert REAL_OPTIONAL in default_backtest.output

    conflict = runner.invoke(
        app,
        [
            "evaluate",
            "backtest",
            "--output-dir",
            str(tmp_path),
            "--benchmark-profile",
            REAL_SOCIAL_ENRICHED_MAIN,
            "--dataset-mode",
            "real",
            "--dry-run",
        ],
    )
    assert conflict.exit_code == 1
    assert "conflicts" in conflict.output


def _persist_enriched_backtest(root: Path) -> tuple[str, DataPaths]:
    _seed_dataset(root, n_markets=64, target_yes_every=8)
    paths = get_data_paths(root=root)
    before_hash = hashlib.sha256(paths.social_pressure_features_parquet.read_bytes()).hexdigest()
    run = run_backtest(
        paths=paths,
        config=BacktestConfig(benchmark_profile=REAL_SOCIAL_ENRICHED_MAIN),
    )
    write_backtest_artifacts(run, paths=paths)
    return before_hash, paths


def test_dashboard_prediction_view_joins_persisted_artifacts_without_touching_branch08(
    tmp_path: Path,
):
    before_hash, paths = _persist_enriched_backtest(tmp_path)
    fixed_now = datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc)

    view = build_dashboard_prediction_view(
        paths=paths,
        benchmark_profile=REAL_SOCIAL_ENRICHED_MAIN,
        generated_at=fixed_now,
    )
    profile_paths = paths.backtest_profile(REAL_SOCIAL_ENRICHED_MAIN)
    evaluation = pd.read_parquet(profile_paths.evaluation_dataset_parquet)
    market = pd.read_parquet(profile_paths.predictions_market_only_parquet).set_index("market_id")
    social = pd.read_parquet(profile_paths.predictions_social_only_parquet).set_index("market_id")
    combined = pd.read_parquet(profile_paths.predictions_market_social_parquet).set_index("market_id")
    joined = view.frame.set_index("market_id")

    assert len(view.frame) == len(evaluation) == 64
    assert list(view.frame.columns) == list(DASHBOARD_VIEW_COLUMNS)
    assert joined["market_only_probability"].equals(market["prediction"])
    assert joined["social_only_probability"].equals(social["prediction"])
    assert joined["market_social_probability"].equals(combined["prediction"])
    assert joined["social_adjustment_delta"].equals(
        combined["prediction"] - market["prediction"]
    )
    assert set(view.frame["horizon_name"]) == {"final_pre_close"}
    assert set(view.frame["market_probability_age_hours"]) == {1.0}
    assert view.frame["used_market_probability"].all()
    assert view.frame["used_social_features"].all()
    assert view.frame["market_only_model_status"].notna().all()
    assert view.frame["social_only_model_status"].notna().all()
    assert view.frame["market_social_model_status"].notna().all()
    assert view.manifest["generated_at"] == "2026-06-02T12:00:00Z"
    assert view.manifest["claim_disclaimer"] == CLAIM_DISCLAIMER
    after_hash = hashlib.sha256(paths.social_pressure_features_parquet.read_bytes()).hexdigest()
    assert after_hash == before_hash


def test_dashboard_prediction_view_tolerates_missing_optional_inputs(tmp_path: Path):
    _, paths = _persist_enriched_backtest(tmp_path)
    profile_paths = paths.backtest_profile(REAL_SOCIAL_ENRICHED_MAIN)
    evaluation = pd.read_parquet(profile_paths.evaluation_dataset_parquet)
    evaluation = evaluation.drop(columns=["late_pressure", "source_diversity_score"])
    evaluation.to_parquet(profile_paths.evaluation_dataset_parquet, index=False)
    profile_paths.predictions_social_only_parquet.unlink()

    view = build_dashboard_prediction_view(
        paths=paths,
        benchmark_profile=REAL_SOCIAL_ENRICHED_MAIN,
    )

    assert len(view.frame) == len(evaluation)
    assert view.frame["late_pressure"].isna().all()
    assert view.frame["source_diversity_score"].isna().all()
    assert view.frame["social_only_probability"].isna().all()
    assert view.frame["social_only_model_status"].isna().all()
    assert any("Optional evaluation columns unavailable" in item for item in view.manifest["warnings"])
    assert any("predictions_social_only.parquet" in item for item in view.manifest["warnings"])


def test_dashboard_prediction_view_writer_manifest_csv_and_profile_isolation(tmp_path: Path):
    before_hash, paths = _persist_enriched_backtest(tmp_path)
    profile_paths = paths.backtest_profile(REAL_SOCIAL_ENRICHED_MAIN)
    view = build_dashboard_prediction_view(
        paths=paths,
        benchmark_profile=REAL_SOCIAL_ENRICHED_MAIN,
        generated_at=datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc),
    )

    written = write_dashboard_prediction_view(view, paths=paths, export_csv=True)

    assert profile_paths.dashboard_prediction_view_parquet.exists()
    assert profile_paths.dashboard_prediction_view_csv.exists()
    assert profile_paths.dashboard_manifest_json.exists()
    assert "dashboard_prediction_view_csv" in written
    manifest = json.loads(profile_paths.dashboard_manifest_json.read_text(encoding="utf-8"))
    assert manifest["profile"] == REAL_SOCIAL_ENRICHED_MAIN
    assert manifest["row_count"] == 64
    assert manifest["csv_exported"] is True
    assert "Prediction Explorer" in manifest["available_panels"]
    assert manifest["claim_disclaimer"] == CLAIM_DISCLAIMER
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_dashboard_prediction_view(view, paths=paths)
    assert not paths.backtest_profile(REAL_OPTIONAL).dashboard_prediction_view_parquet.exists()
    assert not (tmp_path / "analytical" / "backtests" / REAL_OPTIONAL).exists()
    after_hash = hashlib.sha256(paths.social_pressure_features_parquet.read_bytes()).hexdigest()
    assert after_hash == before_hash


def test_cli_build_dashboard_view_dry_run_writes_nothing(tmp_path: Path):
    _, paths = _persist_enriched_backtest(tmp_path)
    profile_paths = paths.backtest_profile(REAL_SOCIAL_ENRICHED_MAIN)

    result = runner.invoke(
        app,
        [
            "evaluate",
            "build-dashboard-view",
            "--output-dir",
            str(tmp_path),
            "--benchmark-profile",
            REAL_SOCIAL_ENRICHED_MAIN,
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "--dry-run set; no files written" in result.output
    assert not profile_paths.dashboard_prediction_view_parquet.exists()
    assert not profile_paths.dashboard_manifest_json.exists()

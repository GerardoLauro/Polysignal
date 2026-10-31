"""Leakage-free final-pre-close forecasting and backtesting for Branch 09.

Branch 08 emits one interpretable social-pressure row per market. This module
joins those rows to canonical Branch 07 resolved labels strictly for scoring,
selects the latest valid pre-close YES price, and evaluates simple baselines.

The first implementation deliberately stays small:

* constant/base-rate baseline;
* market-only baseline using the latest valid price at or before close;
* social-only regularized logistic regression;
* market + social regularized logistic regression.

Learned predictions are out-of-fold. A fold that cannot fit a binary model
falls back to a smoothed training base rate instead of leaking held-out labels
or fabricating confidence.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from polysignal.domain import Market, Outcome, ProbabilityEstimate
from polysignal.evaluation.benchmark_profiles import (
    BenchmarkProfile,
    resolve_benchmark_profile,
)
from polysignal.evaluation.metrics import (
    accuracy,
    brier_score,
    calibration_bins,
    expected_calibration_error,
    log_loss,
)
from polysignal.evaluation.resolved_labels import (
    ResolvedMarketLabel,
    build_resolved_labels,
    read_resolved_labels,
)
from polysignal.features.spi import detect_dataset_mode
from polysignal.storage.parquet_store import read_dataframe, read_parquet
from polysignal.storage.paths import BacktestPaths, DataPaths, get_data_paths

SOCIAL_MODEL_FEATURES = (
    "stance_balance",
    "pressure_strength",
    "temporal_decay_pressure",
    "pressure_acceleration",
    "evidence_quality_score",
    "source_diversity_score",
    "accepted_evidence_ratio",
    "engagement_log_total",
)

FORBIDDEN_SOCIAL_FEATURE_COLUMNS = {
    "binary_yes_no",
    "benchmark_profile",
    "benchmark_type",
    "controlled_social_signal",
    "expected_net_pressure_direction",
    "feature_source",
    "is_resolved",
    "market_probability_source",
    "real_resolution",
    "resolution",
    "resolved_label",
    "synthetic_expected_direction",
    "supports_dashboard_demo",
    "supports_real_empirical_claims",
    "target",
    "target_yes",
    "winning_outcome_id",
    "winning_outcome_index",
    "winning_outcome_label",
}

PREDICTION_COLUMNS = (
    "market_id",
    "dataset_mode",
    "benchmark_profile",
    "benchmark_type",
    "controlled_social_signal",
    "market_probability_source",
    "supports_dashboard_demo",
    "supports_real_empirical_claims",
    "model",
    "model_status",
    "prediction",
    "target_yes",
    "binary_yes_no",
    "market_probability",
    "market_probability_timestamp",
    "pressure_direction",
    "directional_stance_count",
    "evidence_sufficient",
    "evidence_bucket",
)

class Forecaster(Protocol):
    """Legacy point-forecast interface retained for dashboard callers."""

    def forecast_at(
        self,
        market_id: str,
        outcome_id: str,
        timestamp: datetime,
    ) -> ProbabilityEstimate: ...


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for the conservative final-pre-close backtest."""

    benchmark_profile: str | None = None
    dataset_mode: str | None = None
    target: str = "yes"
    min_evidence: int = 3
    exclude_insufficient: bool = False
    n_calibration_bins: int = 10
    n_splits: int = 5
    random_state: int = 42
    anchor_offsets: tuple[timedelta, ...] = ()


@dataclass(frozen=True)
class BacktestResult:
    """Compatibility result for callers interested in compact metric maps."""

    by_metric: dict[str, float | int | None]
    by_slice: dict[str, dict[str, float | int | None]]


@dataclass
class EvaluationDataset:
    """Joined evaluation-only frame plus construction audit metadata."""

    frame: pd.DataFrame
    metadata: dict[str, Any]
    controlled_social_features: pd.DataFrame | None = None


@dataclass
class BacktestRun:
    """Complete in-memory Branch 09 run, ready to persist."""

    dataset: EvaluationDataset
    predictions: dict[str, pd.DataFrame]
    metrics_summary: dict[str, Any]
    calibration: pd.DataFrame
    feature_importance: pd.DataFrame
    report: dict[str, Any]


def resolve_backtest_profile_paths(
    *,
    paths: DataPaths | None = None,
    benchmark_profile: str | None = None,
    dataset_mode: str | None = None,
) -> tuple[BenchmarkProfile, DataPaths]:
    """Resolve profile selection and its default root while retaining legacy modes."""
    candidate_paths = paths
    legacy_mode = dataset_mode
    if legacy_mode == "auto":
        candidate_paths = candidate_paths or get_data_paths()
        legacy_mode = detect_dataset_mode(candidate_paths)
    profile = resolve_benchmark_profile(
        benchmark_profile=benchmark_profile,
        legacy_dataset_mode=legacy_mode,
    )
    resolved_paths = candidate_paths or get_data_paths(root=Path(profile.default_root))
    return profile, resolved_paths


def build_evaluation_dataset(
    *,
    paths: DataPaths | None = None,
    benchmark_profile: str | None = None,
    dataset_mode: str | None = None,
    target: str = "yes",
    min_evidence: int = 3,
    exclude_insufficient: bool = False,
) -> EvaluationDataset:
    """Join Branch 08 features to canonical targets and valid pre-close prices."""
    profile, paths = resolve_backtest_profile_paths(
        paths=paths,
        benchmark_profile=benchmark_profile,
        dataset_mode=dataset_mode,
    )
    mode = profile.dataset_mode
    if target not in {"yes", "resolved"}:
        raise ValueError("target must be one of: yes, resolved")
    if min_evidence < 1:
        raise ValueError("min_evidence must be >= 1")

    feature_path = paths.social_pressure_features_parquet
    if not feature_path.exists():
        raise FileNotFoundError(
            f"missing Branch 08 feature table: {feature_path}. "
            "Run `polysignal features compute-social-pressure` first."
        )
    raw_features = read_dataframe(feature_path)
    _assert_social_feature_schema(raw_features)
    features, duplicate_feature_rows = _deduplicate_features(raw_features)

    labels, target_source = _load_canonical_targets(paths)
    label_frame = _binary_label_frame(labels)
    controlled_social_features: pd.DataFrame | None = None
    preclose_prices = _latest_preclose_yes_prices(paths, label_frame)

    joined = features.merge(label_frame, on="market_id", how="inner", validate="one_to_one")
    joined = joined.merge(preclose_prices, on="market_id", how="left", validate="one_to_one")
    joined["target_yes"] = (joined["binary_yes_no"] == "YES").astype(int)
    joined["has_preclose_market_probability"] = joined["market_probability"].notna()
    joined["evidence_sufficient"] = joined["directional_stance_count"] >= min_evidence
    joined["evidence_bucket"] = joined["directional_stance_count"].map(_evidence_bucket)
    joined["prediction_cutoff"] = joined["close_time"]
    joined["benchmark_profile"] = profile.name
    joined["benchmark_type"] = profile.benchmark_type
    joined["controlled_social_signal"] = profile.controlled_social_signal
    joined["market_probability_source"] = profile.market_probability_source
    joined["supports_dashboard_demo"] = profile.supports_dashboard_demo
    joined["supports_real_empirical_claims"] = profile.supports_real_empirical_claims
    joined = joined.sort_values("market_id").reset_index(drop=True)

    joined_before_filter = len(joined)
    if exclude_insufficient:
        joined = joined[joined["evidence_sufficient"]].reset_index(drop=True)

    metadata = {
        "dataset_mode": mode,
        "benchmark_profile": profile.name,
        "benchmark_type": profile.benchmark_type,
        "benchmark_profile_metadata": {
            **profile.to_metadata(),
            "controlled_scenario": None,
        },
        "target": "YES resolution probability",
        "input_paths": {
            "branch08_social_pressure_features": str(feature_path),
            "effective_social_pressure_features": str(feature_path),
            "normalized_markets": str(paths.polymarket_markets_parquet),
            "normalized_outcomes": str(paths.polymarket_outcomes_parquet),
            "normalized_price_points": str(paths.polymarket_price_points_parquet),
            "resolved_labels_export": str(paths.resolved_market_labels_parquet),
        },
        "market_probability_source": profile.market_probability_source,
        "target_source": target_source,
        "join_keys": ["market_id"],
        "row_counts": {
            "feature_rows_raw": len(raw_features),
            "feature_rows_unique": len(features),
            "duplicate_feature_rows_dropped": duplicate_feature_rows,
            "canonical_resolved_labels": len(labels),
            "binary_resolved_targets": len(label_frame),
            "joined_rows_before_evidence_filter": joined_before_filter,
            "evaluation_rows": len(joined),
            "rows_with_preclose_market_probability": int(
                joined["has_preclose_market_probability"].sum()
            ),
        },
        "target_distribution": _target_distribution(joined),
        "evidence_coverage": _evidence_coverage(joined),
        "model_feature_columns": list(SOCIAL_MODEL_FEATURES),
        "leakage_safeguards": [
            "Branch 08 feature input is rejected if any known target/outcome column appears.",
            "Resolved labels are joined only after social features are loaded and validated.",
            "Models receive an explicit predictor whitelist; target columns are never predictors.",
            "Market-only probability is the latest YES price with timestamp <= close_time.",
            "Post-resolution Outcome.current_price values are never used.",
            "The Branch 08 feature parquet is consumed read-only.",
        ],
    }
    return EvaluationDataset(
        frame=joined,
        metadata=metadata,
        controlled_social_features=controlled_social_features,
    )


def run_backtest(
    *,
    paths: DataPaths | None = None,
    config: BacktestConfig | None = None,
    model: str = "all",
) -> BacktestRun:
    """Build the evaluation frame and run the requested offline baselines."""
    cfg = config or BacktestConfig()
    profile, paths = resolve_backtest_profile_paths(
        paths=paths,
        benchmark_profile=cfg.benchmark_profile,
        dataset_mode=cfg.dataset_mode,
    )
    requested = _requested_models(model)
    dataset = build_evaluation_dataset(
        paths=paths,
        benchmark_profile=profile.name,
        target=cfg.target,
        min_evidence=cfg.min_evidence,
        exclude_insufficient=cfg.exclude_insufficient,
    )
    frame = dataset.frame
    priced = frame[frame["has_preclose_market_probability"]].copy()

    predictions: dict[str, pd.DataFrame] = {}
    base_prediction, base_status = _base_rate_predictions(frame["target_yes"].tolist(), cfg)
    predictions["constant"] = _prediction_frame(frame, "constant", base_prediction, base_status)

    importance_frames: list[pd.DataFrame] = []
    if "market_only" in requested:
        predictions["market_only"] = _prediction_frame(
            priced,
            "market_only",
            priced["market_probability"].astype(float).to_numpy(),
            profile.market_probability_source,
        )
    if "social_only" in requested:
        social_prediction, social_status, social_importance = _logistic_predictions(
            frame, list(SOCIAL_MODEL_FEATURES), cfg, model="social_only"
        )
        predictions["social_only"] = _prediction_frame(
            frame, "social_only", social_prediction, social_status
        )
        importance_frames.append(social_importance)
    if "market_social" in requested:
        combined_features = ["market_probability", *SOCIAL_MODEL_FEATURES]
        combined_prediction, combined_status, combined_importance = _logistic_predictions(
            priced, combined_features, cfg, model="market_social"
        )
        predictions["market_social"] = _prediction_frame(
            priced, "market_social", combined_prediction, combined_status
        )
        importance_frames.append(combined_importance)

    feature_importance = _concat_nonempty(importance_frames, _importance_columns())
    metrics_summary = _metrics_summary(predictions, cfg.n_calibration_bins)
    calibration = _all_calibration_bins(predictions, cfg.n_calibration_bins)
    diagnostics = _social_diagnostics(frame, predictions.get("social_only"))
    report = _build_report(paths, cfg, dataset, predictions, metrics_summary, diagnostics)
    return BacktestRun(
        dataset=dataset,
        predictions=predictions,
        metrics_summary=metrics_summary,
        calibration=calibration,
        feature_importance=feature_importance,
        report=report,
    )


def write_backtest_artifacts(
    run: BacktestRun,
    *,
    paths: DataPaths | None = None,
    overwrite: bool = False,
    export_csv: bool = False,
) -> dict[str, Path]:
    """Persist one Branch 09 run without silently replacing prior artefacts."""
    paths = _default_paths_for_dataset(run.dataset, paths)
    artifact_paths = _profile_artifact_paths(run.dataset, paths)
    targets: dict[str, Path] = {
        **_dataset_artifact_targets(run.dataset, paths, export_csv=export_csv),
        "metrics_summary_json": artifact_paths.metrics_summary_json,
        "metrics_summary_csv": artifact_paths.metrics_summary_csv,
        "calibration_bins_csv": artifact_paths.calibration_bins_csv,
        "backtest_report_json": artifact_paths.backtest_report_json,
    }
    for name in run.predictions:
        targets[f"predictions_{name}"] = artifact_paths.predictions_parquet(name)
        if export_csv:
            targets[f"predictions_{name}_csv"] = artifact_paths.predictions_csv(name)
    if not run.feature_importance.empty:
        targets["feature_importance_csv"] = artifact_paths.feature_importance_csv

    _assert_overwrite_allowed(targets, overwrite=overwrite)

    artifact_paths.root.mkdir(parents=True, exist_ok=True)
    _write_dataset_artifacts(run.dataset, paths, artifact_paths, export_csv=export_csv)
    for name, frame in run.predictions.items():
        frame.to_parquet(artifact_paths.predictions_parquet(name), index=False)
        if export_csv:
            frame.to_csv(artifact_paths.predictions_csv(name), index=False)

    artifact_paths.metrics_summary_json.write_text(
        json.dumps(run.metrics_summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    _metrics_csv_frame(run.metrics_summary).to_csv(artifact_paths.metrics_summary_csv, index=False)
    run.calibration.to_csv(artifact_paths.calibration_bins_csv, index=False)
    if not run.feature_importance.empty:
        run.feature_importance.to_csv(artifact_paths.feature_importance_csv, index=False)
    artifact_paths.backtest_report_json.write_text(
        json.dumps(run.report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return targets


def write_evaluation_dataset_artifacts(
    dataset: EvaluationDataset,
    *,
    paths: DataPaths | None = None,
    overwrite: bool = False,
    export_csv: bool = False,
) -> dict[str, Path]:
    """Persist a profile-scoped evaluation dataset and its audit sidecars."""
    paths = _default_paths_for_dataset(dataset, paths)
    artifact_paths = _profile_artifact_paths(dataset, paths)
    targets = _dataset_artifact_targets(dataset, paths, export_csv=export_csv)
    _assert_overwrite_allowed(targets, overwrite=overwrite)
    artifact_paths.root.mkdir(parents=True, exist_ok=True)
    _write_dataset_artifacts(dataset, paths, artifact_paths, export_csv=export_csv)
    return targets


def _dataset_artifact_targets(
    dataset: EvaluationDataset,
    paths: DataPaths,
    *,
    export_csv: bool,
) -> dict[str, Path]:
    artifact_paths = _profile_artifact_paths(dataset, paths)
    targets = {
        "evaluation_dataset": artifact_paths.evaluation_dataset_parquet,
        "benchmark_profile_json": artifact_paths.benchmark_profile_json,
        "warnings_json": artifact_paths.warnings_json,
    }
    if export_csv:
        targets["evaluation_dataset_csv"] = artifact_paths.evaluation_dataset_csv
    return targets


def _write_dataset_artifacts(
    dataset: EvaluationDataset,
    paths: DataPaths,
    artifact_paths: BacktestPaths,
    *,
    export_csv: bool,
) -> None:
    dataset.frame.to_parquet(artifact_paths.evaluation_dataset_parquet, index=False)
    if export_csv:
        dataset.frame.to_csv(artifact_paths.evaluation_dataset_csv, index=False)
    profile_metadata = dataset.metadata["benchmark_profile_metadata"]
    artifact_paths.benchmark_profile_json.write_text(
        json.dumps(profile_metadata, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    artifact_paths.warnings_json.write_text(
        json.dumps(
            {
                "benchmark_profile": dataset.metadata["benchmark_profile"],
                "warnings": profile_metadata["warnings"],
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _profile_artifact_paths(dataset: EvaluationDataset, paths: DataPaths) -> BacktestPaths:
    return paths.backtest_profile(dataset.metadata["benchmark_profile"])


def _default_paths_for_dataset(
    dataset: EvaluationDataset,
    paths: DataPaths | None,
) -> DataPaths:
    if paths is not None:
        return paths
    default_root = dataset.metadata["benchmark_profile_metadata"]["default_root"]
    return get_data_paths(root=Path(default_root))


def _assert_overwrite_allowed(targets: dict[str, Path], *, overwrite: bool) -> None:
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing backtest artefacts: {joined}")


def _assert_social_feature_schema(frame: pd.DataFrame) -> None:
    forbidden = sorted(set(frame.columns) & FORBIDDEN_SOCIAL_FEATURE_COLUMNS)
    if forbidden:
        raise ValueError(
            "Branch 08 feature table contains forbidden evaluation-only columns: "
            + ", ".join(forbidden)
        )
    required = {"market_id", "directional_stance_count", "pressure_direction"}
    missing = sorted((required | set(SOCIAL_MODEL_FEATURES)) - set(frame.columns))
    if missing:
        raise ValueError("Branch 08 feature table is missing required columns: " + ", ".join(missing))


def _deduplicate_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep one deterministic feature row per market and report duplicates."""
    if frame.empty:
        return frame.copy(), 0
    duplicate_count = int(frame.duplicated(subset=["market_id"]).sum())
    return (
        frame.sort_values("market_id").drop_duplicates(subset=["market_id"], keep="first").copy(),
        duplicate_count,
    )


def _load_canonical_targets(paths: DataPaths) -> tuple[list[ResolvedMarketLabel], str]:
    export = paths.resolved_market_labels_parquet
    if export.exists():
        return read_resolved_labels(export), str(export)
    markets = read_parquet(paths.polymarket_markets_parquet, Market)
    outcomes = read_parquet(paths.polymarket_outcomes_parquet, Outcome)
    return (
        build_resolved_labels(markets, outcomes),
        (
            f"{paths.polymarket_markets_parquet} + {paths.polymarket_outcomes_parquet} "
            "via polysignal.evaluation.resolved_labels.build_resolved_labels"
        ),
    )


def _binary_label_frame(labels: Sequence[ResolvedMarketLabel]) -> pd.DataFrame:
    rows = [
        {
            "market_id": label.market_id,
            "binary_yes_no": label.binary_yes_no,
            "target_close_time": label.close_time,
            "target_resolution": label.resolution,
            "winning_outcome_id": label.winning_outcome_id,
        }
        for label in labels
        if label.binary_yes_no in {"YES", "NO"}
    ]
    columns = [
        "market_id",
        "binary_yes_no",
        "target_close_time",
        "target_resolution",
        "winning_outcome_id",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("market_id")
        .drop_duplicates(subset=["market_id"], keep="first")
    )


def _latest_preclose_yes_prices(paths: DataPaths, labels: pd.DataFrame) -> pd.DataFrame:
    output_columns = ["market_id", "market_probability", "market_probability_timestamp"]
    prices = read_dataframe(paths.polymarket_price_points_parquet)
    outcomes = read_dataframe(paths.polymarket_outcomes_parquet)
    if prices.empty or outcomes.empty or labels.empty:
        return pd.DataFrame(columns=output_columns)

    yes = outcomes[outcomes["label"].astype(str).str.strip().str.lower() == "yes"].copy()
    yes = yes[["market_id", "outcome_id"]].drop_duplicates()
    ambiguous = yes.groupby("market_id")["outcome_id"].nunique()
    if (ambiguous > 1).any():
        bad = ", ".join(sorted(ambiguous[ambiguous > 1].index.astype(str)))
        raise ValueError(f"multiple YES outcome ids found for market(s): {bad}")
    yes = yes.drop_duplicates(subset=["market_id"], keep="first")

    candidates = prices.merge(yes, on=["market_id", "outcome_id"], how="inner")
    candidates = candidates.merge(
        labels[["market_id", "target_close_time"]], on="market_id", how="inner"
    )
    candidates["timestamp_dt"] = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    candidates["close_dt"] = pd.to_datetime(
        candidates["target_close_time"], utc=True, errors="coerce"
    )
    candidates["price"] = pd.to_numeric(candidates["price"], errors="coerce")
    candidates = candidates[
        candidates["timestamp_dt"].notna()
        & candidates["close_dt"].notna()
        & candidates["price"].between(0.0, 1.0)
        & (candidates["timestamp_dt"] <= candidates["close_dt"])
    ]
    if candidates.empty:
        return pd.DataFrame(columns=output_columns)
    latest = candidates.sort_values(["market_id", "timestamp_dt"]).drop_duplicates(
        subset=["market_id"], keep="last"
    )
    return latest.rename(
        columns={"price": "market_probability", "timestamp": "market_probability_timestamp"}
    )[list(output_columns)]


def _requested_models(model: str) -> set[str]:
    if model == "all":
        return {"market_only", "social_only", "market_social"}
    if model == "market-social":
        return {"market_only", "social_only", "market_social"}
    if model == "market-only":
        return {"market_only"}
    if model == "social-only":
        return {"social_only"}
    raise ValueError("model must be one of: market-only, social-only, market-social, all")


def _pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    max_iter=1000,
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _stratified_folds(outcomes: Sequence[int], cfg: BacktestConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    counts = Counter(int(value) for value in outcomes)
    if len(counts) < 2 or min(counts.values()) < 2:
        return []
    n_splits = min(cfg.n_splits, min(counts.values()))
    if n_splits < 2:
        return []
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=cfg.random_state
    )
    dummy = np.zeros((len(outcomes), 1))
    return list(splitter.split(dummy, np.asarray(outcomes, dtype=int)))


def _smoothed_rate(outcomes: Sequence[int]) -> float:
    """Laplace-smoothed rate; defined even for an empty training fold."""
    return (sum(int(value) for value in outcomes) + 1.0) / (len(outcomes) + 2.0)


def _base_rate_predictions(
    outcomes: Sequence[int], cfg: BacktestConfig
) -> tuple[np.ndarray, str]:
    values = [int(value) for value in outcomes]
    if not values:
        return np.asarray([], dtype=float), "no_rows"
    folds = _stratified_folds(values, cfg)
    predictions = np.zeros(len(values), dtype=float)
    if folds:
        for train, test in folds:
            rate = _smoothed_rate([values[index] for index in train])
            predictions[test] = rate
        return predictions, f"oof_smoothed_base_rate_{len(folds)}_fold"
    for index in range(len(values)):
        training = [value for position, value in enumerate(values) if position != index]
        predictions[index] = _smoothed_rate(training)
    return predictions, "leave_one_out_smoothed_base_rate"


def _logistic_predictions(
    frame: pd.DataFrame,
    features: list[str],
    cfg: BacktestConfig,
    *,
    model: str,
) -> tuple[np.ndarray, str, pd.DataFrame]:
    if frame.empty:
        return np.asarray([], dtype=float), "no_rows", pd.DataFrame(columns=_importance_columns())
    outcomes = frame["target_yes"].astype(int).tolist()
    folds = _stratified_folds(outcomes, cfg)
    if not folds:
        fallback, status = _base_rate_predictions(outcomes, cfg)
        return fallback, f"{status}; logistic_fallback_insufficient_class_support", pd.DataFrame(
            columns=_importance_columns()
        )

    matrix = frame[features].apply(pd.to_numeric, errors="coerce")
    predictions = np.zeros(len(frame), dtype=float)
    for train, test in folds:
        estimator = _pipeline()
        estimator.fit(matrix.iloc[train], np.asarray(outcomes, dtype=int)[train])
        predictions[test] = estimator.predict_proba(matrix.iloc[test])[:, 1]

    importance = _fit_feature_importance(matrix, outcomes, features, model)
    return predictions, f"oof_regularized_logistic_{len(folds)}_fold", importance


def _fit_feature_importance(
    matrix: pd.DataFrame,
    outcomes: Sequence[int],
    features: Sequence[str],
    model: str,
) -> pd.DataFrame:
    if len(set(outcomes)) < 2:
        return pd.DataFrame(columns=_importance_columns())
    estimator = _pipeline()
    estimator.fit(matrix, np.asarray(outcomes, dtype=int))
    fitted = estimator.named_steps["model"]
    rows = [
        {
            "model": model,
            "feature": feature,
            "coefficient": float(coefficient),
            "absolute_coefficient": abs(float(coefficient)),
            "note": "descriptive full-dataset fit; scored predictions are out-of-fold",
        }
        for feature, coefficient in zip(features, fitted.coef_[0], strict=True)
    ]
    return pd.DataFrame(rows).sort_values("absolute_coefficient", ascending=False)


def _prediction_frame(
    frame: pd.DataFrame,
    model: str,
    prediction: Sequence[float],
    status: str,
) -> pd.DataFrame:
    result = frame[
        [
            "market_id",
            "dataset_mode",
            "benchmark_profile",
            "benchmark_type",
            "controlled_social_signal",
            "market_probability_source",
            "supports_dashboard_demo",
            "supports_real_empirical_claims",
            "target_yes",
            "binary_yes_no",
            "market_probability",
            "market_probability_timestamp",
            "pressure_direction",
            "directional_stance_count",
            "evidence_sufficient",
            "evidence_bucket",
        ]
    ].copy()
    result.insert(8, "model", model)
    result.insert(9, "model_status", status)
    result.insert(10, "prediction", np.asarray(prediction, dtype=float))
    return result[list(PREDICTION_COLUMNS)]


def _score_predictions(frame: pd.DataFrame, *, model: str, scope: str, n_bins: int) -> dict[str, Any]:
    if frame.empty:
        return {
            "model": model,
            "scope": scope,
            "rows_evaluated": 0,
            "target_yes": 0,
            "target_no": 0,
            "brier_score": None,
            "log_loss": None,
            "accuracy": None,
            "auc": None,
            "expected_calibration_error": None,
            "confusion_matrix_0_5": None,
        }
    probabilities = frame["prediction"].astype(float).tolist()
    outcomes = frame["target_yes"].astype(int).tolist()
    classified = [int(probability >= 0.5) for probability in probabilities]
    tn = sum(pred == 0 and actual == 0 for pred, actual in zip(classified, outcomes, strict=True))
    fp = sum(pred == 1 and actual == 0 for pred, actual in zip(classified, outcomes, strict=True))
    fn = sum(pred == 0 and actual == 1 for pred, actual in zip(classified, outcomes, strict=True))
    tp = sum(pred == 1 and actual == 1 for pred, actual in zip(classified, outcomes, strict=True))
    return {
        "model": model,
        "scope": scope,
        "rows_evaluated": len(frame),
        "target_yes": sum(outcomes),
        "target_no": len(outcomes) - sum(outcomes),
        "brier_score": brier_score(probabilities, outcomes),
        "log_loss": log_loss(probabilities, outcomes),
        "accuracy": accuracy(probabilities, outcomes),
        "auc": float(roc_auc_score(outcomes, probabilities)) if len(set(outcomes)) == 2 else None,
        "expected_calibration_error": expected_calibration_error(
            probabilities, outcomes, n_bins=n_bins
        ),
        "confusion_matrix_0_5": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _metrics_summary(predictions: dict[str, pd.DataFrame], n_bins: int) -> dict[str, Any]:
    models = {
        name: _score_predictions(frame, model=name, scope="available_rows", n_bins=n_bins)
        for name, frame in predictions.items()
    }
    common_ids = set(predictions.get("market_only", pd.DataFrame()).get("market_id", []))
    common: dict[str, dict[str, Any]] = {}
    for name, frame in predictions.items():
        subset = frame[frame["market_id"].isin(common_ids)].copy()
        common[name] = _score_predictions(
            subset, model=name, scope="common_preclose_price_rows", n_bins=n_bins
        )
    market = common.get("market_only", {})
    combined = common.get("market_social", {})
    delta_brier = _metric_delta(market, combined, "brier_score")
    delta_log_loss = _metric_delta(market, combined, "log_loss")
    return {
        "models": models,
        "common_comparison": common,
        "lift_over_market_only": {
            "delta_brier": delta_brier,
            "delta_log_loss": delta_log_loss,
            "definition": "market_only metric - market_social metric; positive means market_social improved",
        },
    }


def _metric_delta(
    baseline: dict[str, Any], candidate: dict[str, Any], metric: str
) -> float | None:
    left = baseline.get(metric)
    right = candidate.get(metric)
    if left is None or right is None:
        return None
    return float(left - right)


def _all_calibration_bins(predictions: dict[str, pd.DataFrame], n_bins: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, frame in predictions.items():
        if frame.empty:
            continue
        values = calibration_bins(
            frame["prediction"].astype(float).tolist(),
            frame["target_yes"].astype(int).tolist(),
            n_bins=n_bins,
        )
        for row in values:
            rows.append({"model": model, "scope": "available_rows", **row})
    columns = [
        "model",
        "scope",
        "bin_index",
        "bin_lower",
        "bin_upper",
        "count",
        "mean_probability",
        "observed_rate",
        "absolute_error",
    ]
    return pd.DataFrame(rows, columns=columns)


def _social_diagnostics(
    frame: pd.DataFrame, social_predictions: pd.DataFrame | None
) -> dict[str, Any]:
    directional = frame[frame["pressure_direction"].isin(["YES", "NO"])].copy()
    directional["pressure_target_yes"] = (directional["pressure_direction"] == "YES").astype(int)
    relationship = pd.crosstab(frame["pressure_direction"], frame["target_yes"])
    diagnostics: dict[str, Any] = {
        "directional_rows": len(directional),
        "directional_agreement": (
            float((directional["pressure_target_yes"] == directional["target_yes"]).mean())
            if not directional.empty
            else None
        ),
        "pressure_direction_vs_target": {
            str(direction): {
                "target_no": int(values.get(0, 0)),
                "target_yes": int(values.get(1, 0)),
            }
            for direction, values in relationship.iterrows()
        },
        "correlations_with_target": {
            column: _safe_correlation(frame[column], frame["target_yes"])
            for column in ("stance_balance", "net_pressure_weighted")
        },
        "coverage_by_evidence_bucket": {
            str(bucket): int(count)
            for bucket, count in frame["evidence_bucket"].value_counts().sort_index().items()
        },
        "performance_by_evidence_sufficiency": {},
    }
    if social_predictions is not None:
        for sufficient, label in ((True, "sufficient"), (False, "insufficient")):
            subset = social_predictions[social_predictions["evidence_sufficient"] == sufficient]
            diagnostics["performance_by_evidence_sufficiency"][label] = _score_predictions(
                subset,
                model="social_only",
                scope=f"evidence_{label}",
                n_bins=10,
            )
    return diagnostics


def _build_report(
    paths: DataPaths,
    cfg: BacktestConfig,
    dataset: EvaluationDataset,
    predictions: dict[str, pd.DataFrame],
    metrics_summary: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    mode = dataset.metadata["dataset_mode"]
    profile_name = dataset.metadata["benchmark_profile"]
    profile_metadata = dataset.metadata["benchmark_profile_metadata"]
    artifact_paths = _profile_artifact_paths(dataset, paths)
    comparison = metrics_summary["lift_over_market_only"]
    interpretation = _interpretation(profile_name, metrics_summary)
    return {
        "branch": "09 - Forecasting, Backtesting & Beating Polymarket",
        "scientific_framing": (
            "Social-pressure features provide incremental signal only if market+social "
            "improves over market-only under no-leakage evaluation."
        ),
        "dataset_mode": mode,
        "benchmark_profile": profile_metadata,
        "warnings": profile_metadata["warnings"],
        "evaluation_policy": "per-market final pre-close evaluation",
        "config": {
            "benchmark_profile": profile_name,
            "target": cfg.target,
            "min_evidence": cfg.min_evidence,
            "exclude_insufficient": cfg.exclude_insufficient,
            "n_calibration_bins": cfg.n_calibration_bins,
            "n_splits": cfg.n_splits,
            "random_state": cfg.random_state,
        },
        "dataset_construction": dataset.metadata,
        "model_definitions": {
            "constant": "Out-of-fold Laplace-smoothed training base-rate probability.",
            "market_only": profile_metadata["market_probability_source"],
            "social_only": (
                "Out-of-fold regularized logistic regression over the explicit Branch 08 "
                "social-feature whitelist."
            ),
            "market_social": (
                "Out-of-fold regularized logistic regression over market_probability plus "
                "the same explicit social-feature whitelist."
            ),
        },
        "prediction_rows": {name: len(frame) for name, frame in predictions.items()},
        "metrics": metrics_summary,
        "comparison": comparison,
        "social_signal_diagnostics": diagnostics,
        "interpretation": interpretation,
        "known_limitations": [
            *profile_metadata["warnings"],
            "This first implementation evaluates the final pre-close cut only; time slices are deferred.",
            "Feature-importance coefficients are descriptive full-dataset fits; scored predictions are out-of-fold.",
        ],
        "output_paths": {
            "root": str(artifact_paths.root),
            "evaluation_dataset": str(artifact_paths.evaluation_dataset_parquet),
            "predictions_constant": str(artifact_paths.predictions_constant_parquet),
            "predictions_market_only": str(artifact_paths.predictions_market_only_parquet),
            "predictions_social_only": str(artifact_paths.predictions_social_only_parquet),
            "predictions_market_social": str(artifact_paths.predictions_market_social_parquet),
            "metrics_summary_json": str(artifact_paths.metrics_summary_json),
            "metrics_summary_csv": str(artifact_paths.metrics_summary_csv),
            "calibration_bins_csv": str(artifact_paths.calibration_bins_csv),
            "feature_importance_csv": str(artifact_paths.feature_importance_csv),
            "backtest_report_json": str(artifact_paths.backtest_report_json),
            "benchmark_profile_json": str(artifact_paths.benchmark_profile_json),
            "warnings_json": str(artifact_paths.warnings_json),
            "controlled_social_pressure_features": None,
        },
    }


def _interpretation(profile_name: str, metrics_summary: dict[str, Any]) -> str:
    comparison = metrics_summary["lift_over_market_only"]
    common = metrics_summary["common_comparison"].get("market_only", {})
    enough_classes = common.get("target_yes", 0) > 0 and common.get("target_no", 0) > 0
    improves = (
        comparison.get("delta_brier") is not None
        and comparison["delta_brier"] > 0
        and comparison.get("delta_log_loss") is not None
        and comparison["delta_log_loss"] > 0
    )
    if enough_classes and improves:
        return (
            "Preliminary real-data results suggest possible incremental signal, "
            "but the sample is too sparse for strong claims."
        )
    return (
        "On the current sparse Telegram-only real dataset, evidence coverage is "
        "insufficient for strong conclusions."
    )


def _target_distribution(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {"YES": 0, "NO": 0}
    counts = frame["binary_yes_no"].value_counts()
    return {"YES": int(counts.get("YES", 0)), "NO": int(counts.get("NO", 0))}


def _evidence_coverage(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {
            "rows_with_any_accepted_link": 0,
            "rows_with_any_directional_stance": 0,
            "rows_with_sufficient_evidence": 0,
            "rows_with_insufficient_evidence": 0,
        }
    return {
        "rows_with_any_accepted_link": int((frame["accepted_link_count"] > 0).sum()),
        "rows_with_any_directional_stance": int((frame["directional_stance_count"] > 0).sum()),
        "rows_with_sufficient_evidence": int(frame["evidence_sufficient"].sum()),
        "rows_with_insufficient_evidence": int((~frame["evidence_sufficient"]).sum()),
    }


def _evidence_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count < 3:
        return "1-2"
    if count < 10:
        return "3-9"
    return "10+"


def _safe_correlation(left: pd.Series, right: pd.Series) -> float | None:
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return None
    value = left.corr(right)
    return float(value) if value is not None and math.isfinite(value) else None


def _concat_nonempty(frames: Sequence[pd.DataFrame], columns: Sequence[str]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame(columns=columns)


def _importance_columns() -> list[str]:
    return ["model", "feature", "coefficient", "absolute_coefficient", "note"]


def _metrics_csv_frame(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for section in ("models", "common_comparison"):
        for metrics in summary[section].values():
            row = dict(metrics)
            if row.get("confusion_matrix_0_5") is not None:
                row["confusion_matrix_0_5"] = json.dumps(row["confusion_matrix_0_5"], sort_keys=True)
            row["section"] = section
            rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "FORBIDDEN_SOCIAL_FEATURE_COLUMNS",
    "SOCIAL_MODEL_FEATURES",
    "BacktestConfig",
    "BacktestResult",
    "BacktestRun",
    "EvaluationDataset",
    "Forecaster",
    "build_evaluation_dataset",
    "resolve_backtest_profile_paths",
    "run_backtest",
    "write_backtest_artifacts",
    "write_evaluation_dataset_artifacts",
]

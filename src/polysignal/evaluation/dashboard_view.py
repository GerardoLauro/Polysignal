"""Dashboard-ready read layer over persisted Branch 09 backtest artifacts.

This module does not recompute features or forecasts. It joins existing
profile-scoped outputs into one stable presentation table for Branch 10.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from polysignal.storage.paths import BacktestPaths, DataPaths

CLAIM_DISCLAIMER = (
    "Real-data evaluation profile. Results are preliminary: at the current "
    "single-platform (Telegram-heavy) coverage, market price is the strong "
    "predictor and social signal adds no measured lift."
)

DASHBOARD_VIEW_COLUMNS = (
    "market_id",
    "condition_id",
    "question",
    "domain",
    "dataset_mode",
    "benchmark_profile",
    "benchmark_type",
    "controlled_social_signal",
    "market_probability_source",
    "supports_dashboard_demo",
    "supports_real_empirical_claims",
    "horizon_name",
    "prediction_cutoff",
    "close_time",
    "target_yes",
    "binary_yes_no",
    "market_probability",
    "market_probability_timestamp",
    "market_probability_age_hours",
    "market_only_probability",
    "social_only_probability",
    "market_social_probability",
    "social_adjustment_delta",
    "market_only_error",
    "social_only_error",
    "market_social_error",
    "best_model_for_row",
    "pressure_direction",
    "directional_stance_count",
    "evidence_sufficient",
    "evidence_bucket",
    "accepted_link_count",
    "stance_balance",
    "confidence_weighted_score",
    "net_pressure_weighted",
    "pressure_strength",
    "temporal_decay_pressure",
    "early_pressure",
    "late_pressure",
    "pressure_acceleration",
    "evidence_quality_score",
    "source_diversity_score",
    "used_market_probability",
    "used_social_features",
    "market_only_model_status",
    "social_only_model_status",
    "market_social_model_status",
)

_PROFILE_COLUMNS = (
    "dataset_mode",
    "benchmark_profile",
    "benchmark_type",
    "controlled_social_signal",
    "market_probability_source",
    "supports_dashboard_demo",
    "supports_real_empirical_claims",
)

_EVALUATION_COLUMNS = tuple(
    column
    for column in DASHBOARD_VIEW_COLUMNS
    if column
    not in {
        "horizon_name",
        "market_probability_age_hours",
        "market_only_probability",
        "social_only_probability",
        "market_social_probability",
        "social_adjustment_delta",
        "market_only_error",
        "social_only_error",
        "market_social_error",
        "best_model_for_row",
        "used_market_probability",
        "used_social_features",
        "market_only_model_status",
        "social_only_model_status",
        "market_social_model_status",
    }
)

_MODEL_ALIASES = {
    "market_only": "market_only_probability",
    "social_only": "social_only_probability",
    "market_social": "market_social_probability",
}

_AVAILABLE_PANELS = (
    "Benchmark Overview",
    "Model Comparison",
    "Lift Panel",
    "Prediction Explorer",
    "Social Signal Inspector",
    "Calibration View",
    "Feature Importance",
    "Benchmark Validity / Warnings",
    "Artifact Browser",
)


@dataclass(frozen=True)
class DashboardPredictionView:
    """One in-memory Branch 10 presentation table plus its manifest."""

    frame: pd.DataFrame
    manifest: dict[str, Any]


def build_dashboard_prediction_view(
    *,
    paths: DataPaths,
    benchmark_profile: str,
    generated_at: datetime | None = None,
) -> DashboardPredictionView:
    """Join persisted Branch 09 artifacts without recomputing forecasting or SPI."""
    artifacts = paths.backtest_profile(benchmark_profile)
    if not artifacts.evaluation_dataset_parquet.exists():
        raise FileNotFoundError(
            f"missing evaluation dataset: {artifacts.evaluation_dataset_parquet}. "
            "Run `polysignal evaluate backtest` first."
        )

    warnings: list[str] = []
    profile_metadata = _read_json_object(
        artifacts.benchmark_profile_json,
        warnings=warnings,
        label="benchmark profile",
    )
    warning_sidecar = _read_json_object(
        artifacts.warnings_json,
        warnings=warnings,
        label="warnings",
    )
    _extend_unique(warnings, profile_metadata.get("warnings", []))
    _extend_unique(warnings, warning_sidecar.get("warnings", []))

    frame = pd.read_parquet(artifacts.evaluation_dataset_parquet)
    _assert_unique_market_ids(frame, artifacts.evaluation_dataset_parquet)
    frame = frame.copy()
    _add_profile_defaults(frame, benchmark_profile, profile_metadata)

    missing_evaluation_columns = [
        column for column in _EVALUATION_COLUMNS if column not in frame.columns
    ]
    for column in missing_evaluation_columns:
        frame[column] = pd.NA
    if missing_evaluation_columns:
        warnings.append(
            "Optional evaluation columns unavailable and emitted as null: "
            + ", ".join(missing_evaluation_columns)
        )

    for model, probability_column in _MODEL_ALIASES.items():
        frame = _join_prediction_artifact(
            frame,
            artifacts=artifacts,
            model=model,
            probability_column=probability_column,
            warnings=warnings,
        )

    if "horizon_name" not in frame.columns:
        frame["horizon_name"] = "final_pre_close"
    else:
        frame["horizon_name"] = frame["horizon_name"].fillna("final_pre_close")

    cutoff = _datetime_series(frame, "prediction_cutoff")
    probability_timestamp = _datetime_series(frame, "market_probability_timestamp")
    frame["market_probability_age_hours"] = (
        cutoff - probability_timestamp
    ).dt.total_seconds() / 3600.0

    target = pd.to_numeric(frame["target_yes"], errors="coerce")
    for model, probability_column in _MODEL_ALIASES.items():
        frame[probability_column] = pd.to_numeric(frame[probability_column], errors="coerce")
        frame[f"{model}_error"] = (frame[probability_column] - target).abs()
    frame["social_adjustment_delta"] = (
        frame["market_social_probability"] - frame["market_only_probability"]
    )
    frame["best_model_for_row"] = _best_model_for_row(frame)
    frame["used_market_probability"] = frame[
        ["market_only_probability", "market_social_probability"]
    ].notna().any(axis="columns")
    frame["used_social_features"] = frame[
        ["social_only_probability", "market_social_probability"]
    ].notna().any(axis="columns")

    frame = frame[list(DASHBOARD_VIEW_COLUMNS)].copy()
    manifest = _build_manifest(
        artifacts=artifacts,
        benchmark_profile=benchmark_profile,
        frame=frame,
        warnings=warnings,
        generated_at=generated_at,
    )
    return DashboardPredictionView(frame=frame, manifest=manifest)


def write_dashboard_prediction_view(
    view: DashboardPredictionView,
    *,
    paths: DataPaths,
    overwrite: bool = False,
    export_csv: bool = False,
) -> dict[str, Path]:
    """Persist a dashboard view and profile-level manifest."""
    benchmark_profile = str(view.manifest["profile"])
    artifacts = paths.backtest_profile(benchmark_profile)
    targets = {
        "dashboard_prediction_view": artifacts.dashboard_prediction_view_parquet,
        "dashboard_manifest_json": artifacts.dashboard_manifest_json,
    }
    if export_csv:
        targets["dashboard_prediction_view_csv"] = artifacts.dashboard_prediction_view_csv
    _assert_overwrite_allowed(targets, overwrite=overwrite)

    artifacts.root.mkdir(parents=True, exist_ok=True)
    view.frame.to_parquet(artifacts.dashboard_prediction_view_parquet, index=False)
    if export_csv:
        view.frame.to_csv(artifacts.dashboard_prediction_view_csv, index=False)
    manifest = {**view.manifest, "csv_exported": export_csv}
    artifacts.dashboard_manifest_json.write_text(
        json.dumps(manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return targets


def _add_profile_defaults(
    frame: pd.DataFrame,
    benchmark_profile: str,
    profile_metadata: dict[str, Any],
) -> None:
    defaults = {column: profile_metadata.get(column, pd.NA) for column in _PROFILE_COLUMNS}
    defaults["benchmark_profile"] = benchmark_profile
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default


def _join_prediction_artifact(
    frame: pd.DataFrame,
    *,
    artifacts: BacktestPaths,
    model: str,
    probability_column: str,
    warnings: list[str],
) -> pd.DataFrame:
    status_column = f"{model}_model_status"
    frame = frame.drop(columns=[probability_column, status_column], errors="ignore")
    prediction_path = artifacts.predictions_parquet(model)
    if not prediction_path.exists():
        warnings.append(f"Optional prediction artifact unavailable: {prediction_path}")
        frame[probability_column] = pd.NA
        frame[status_column] = pd.NA
        return frame

    prediction = pd.read_parquet(prediction_path)
    if "market_id" not in prediction.columns:
        warnings.append(f"Optional prediction artifact has no market_id column: {prediction_path}")
        frame[probability_column] = pd.NA
        frame[status_column] = pd.NA
        return frame
    _assert_unique_market_ids(prediction, prediction_path)

    columns = ["market_id"]
    rename: dict[str, str] = {}
    if "prediction" in prediction.columns:
        columns.append("prediction")
        rename["prediction"] = probability_column
    else:
        warnings.append(f"Optional prediction column unavailable and emitted as null: {prediction_path}")
    if "model_status" in prediction.columns:
        columns.append("model_status")
        rename["model_status"] = status_column
    else:
        warnings.append(f"Optional model_status unavailable and emitted as null: {prediction_path}")

    selected = prediction[columns].rename(columns=rename)
    if probability_column not in selected.columns:
        selected[probability_column] = pd.NA
    if status_column not in selected.columns:
        selected[status_column] = pd.NA
    return frame.merge(
        selected[["market_id", probability_column, status_column]],
        on="market_id",
        how="left",
        validate="one_to_one",
    )


def _assert_unique_market_ids(frame: pd.DataFrame, source: Path) -> None:
    if "market_id" not in frame.columns:
        raise ValueError(f"dashboard input has no market_id column: {source}")
    if frame["market_id"].isna().any():
        raise ValueError(f"dashboard input contains null market_id values: {source}")
    if frame["market_id"].duplicated().any():
        raise ValueError(f"dashboard input contains duplicate market_id values: {source}")


def _datetime_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    return pd.to_datetime(frame[column], utc=True, errors="coerce")


def _best_model_for_row(frame: pd.DataFrame) -> pd.Series:
    errors = frame[["market_only_error", "social_only_error", "market_social_error"]]
    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    available = errors.notna().any(axis="columns")
    result.loc[available] = (
        errors.loc[available]
        .idxmin(axis="columns")
        .map(
            {
                "market_only_error": "market_only",
                "social_only_error": "social_only",
                "market_social_error": "market_social",
            }
        )
    )
    return result


def _build_manifest(
    *,
    artifacts: BacktestPaths,
    benchmark_profile: str,
    frame: pd.DataFrame,
    warnings: list[str],
    generated_at: datetime | None,
) -> dict[str, Any]:
    return {
        "profile": benchmark_profile,
        "artifact_paths": {
            "dashboard_prediction_view_parquet": _display_path(
                artifacts.dashboard_prediction_view_parquet
            ),
            "dashboard_prediction_view_csv": _display_path(
                artifacts.dashboard_prediction_view_csv
            ),
            "dashboard_manifest_json": _display_path(artifacts.dashboard_manifest_json),
            "metrics_summary_json": _display_path(artifacts.metrics_summary_json),
            "metrics_summary_csv": _display_path(artifacts.metrics_summary_csv),
            "calibration_bins_csv": _display_path(artifacts.calibration_bins_csv),
            "feature_importance_csv": _display_path(artifacts.feature_importance_csv),
            "benchmark_profile_json": _display_path(artifacts.benchmark_profile_json),
            "warnings_json": _display_path(artifacts.warnings_json),
            "project_artifact_index_json": "PROJECT_PRE_BRANCH10_ARTIFACT_INDEX.json",
        },
        "row_count": len(frame),
        "available_panels": list(_AVAILABLE_PANELS),
        "key_columns": list(DASHBOARD_VIEW_COLUMNS),
        "warnings": warnings,
        "generated_at": _generated_at_iso(generated_at),
        "claim_disclaimer": CLAIM_DISCLAIMER,
    }


def _read_json_object(path: Path, *, warnings: list[str], label: str) -> dict[str, Any]:
    if not path.exists():
        warnings.append(f"Optional {label} sidecar unavailable: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        warnings.append(f"Optional {label} sidecar could not be read: {path} ({exc})")
        return {}
    if not isinstance(value, dict):
        warnings.append(f"Optional {label} sidecar is not a JSON object: {path}")
        return {}
    return value


def _extend_unique(target: list[str], values: Any) -> None:
    if not isinstance(values, list | tuple):
        return
    for value in values:
        text = str(value)
        if text not in target:
            target.append(text)


def _generated_at_iso(value: datetime | None) -> str:
    moment = value or datetime.now(tz=timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _assert_overwrite_allowed(targets: dict[str, Path], *, overwrite: bool) -> None:
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing dashboard artifacts: {joined}")


__all__ = [
    "CLAIM_DISCLAIMER",
    "DASHBOARD_VIEW_COLUMNS",
    "DashboardPredictionView",
    "build_dashboard_prediction_view",
    "write_dashboard_prediction_view",
]

"""Evaluation — metrics, resolved-market labels, and backtesting."""

from polysignal.evaluation.backtest import (
    BacktestConfig,
    BacktestRun,
    EvaluationDataset,
    build_evaluation_dataset,
    resolve_backtest_profile_paths,
    run_backtest,
    write_backtest_artifacts,
    write_evaluation_dataset_artifacts,
)
from polysignal.evaluation.benchmark_profiles import (
    DEFAULT_BENCHMARK_PROFILE,
    REAL_OPTIONAL,
    REAL_SOCIAL_ENRICHED_MAIN,
    BenchmarkProfile,
    benchmark_profiles,
    get_benchmark_profile,
    resolve_benchmark_profile,
)
from polysignal.evaluation.dashboard_view import (
    CLAIM_DISCLAIMER,
    DASHBOARD_VIEW_COLUMNS,
    DashboardPredictionView,
    build_dashboard_prediction_view,
    write_dashboard_prediction_view,
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
    write_resolved_labels,
)
from polysignal.evaluation.temporal_admissibility import (
    TemporalAdmissibility,
    assess_post_admissibility,
    is_historical_market,
    market_temporal_anchor,
)
from polysignal.evaluation.usefulness import (
    DEFAULT_STAGING_ROOTS,
    build_usefulness_report,
)

__all__ = [
    "CLAIM_DISCLAIMER",
    "DASHBOARD_VIEW_COLUMNS",
    "DEFAULT_BENCHMARK_PROFILE",
    "DEFAULT_STAGING_ROOTS",
    "REAL_OPTIONAL",
    "REAL_SOCIAL_ENRICHED_MAIN",
    "BacktestConfig",
    "BacktestRun",
    "BenchmarkProfile",
    "DashboardPredictionView",
    "EvaluationDataset",
    "ResolvedMarketLabel",
    "TemporalAdmissibility",
    "accuracy",
    "assess_post_admissibility",
    "benchmark_profiles",
    "brier_score",
    "build_dashboard_prediction_view",
    "build_evaluation_dataset",
    "build_resolved_labels",
    "build_usefulness_report",
    "calibration_bins",
    "expected_calibration_error",
    "get_benchmark_profile",
    "is_historical_market",
    "log_loss",
    "market_temporal_anchor",
    "read_resolved_labels",
    "resolve_backtest_profile_paths",
    "resolve_benchmark_profile",
    "run_backtest",
    "write_backtest_artifacts",
    "write_dashboard_prediction_view",
    "write_evaluation_dataset_artifacts",
    "write_resolved_labels",
]

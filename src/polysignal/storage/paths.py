"""Canonical filesystem layout for PolySignal data.

Every module that needs a path should go through ``DataPaths`` so the
layout can evolve in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from polysignal.config import get_settings


@dataclass(frozen=True)
class BacktestPaths:
    """Profile-scoped Branch 09 artifact paths."""

    root: Path

    @property
    def evaluation_dataset_parquet(self) -> Path:
        return self.root / "evaluation_dataset.parquet"

    @property
    def evaluation_dataset_csv(self) -> Path:
        return self.root / "evaluation_dataset.csv"

    @property
    def predictions_constant_parquet(self) -> Path:
        return self.root / "predictions_constant.parquet"

    @property
    def predictions_market_only_parquet(self) -> Path:
        return self.root / "predictions_market_only.parquet"

    @property
    def predictions_social_only_parquet(self) -> Path:
        return self.root / "predictions_social_only.parquet"

    @property
    def predictions_market_social_parquet(self) -> Path:
        return self.root / "predictions_market_social.parquet"

    @property
    def metrics_summary_json(self) -> Path:
        return self.root / "metrics_summary.json"

    @property
    def metrics_summary_csv(self) -> Path:
        return self.root / "metrics_summary.csv"

    @property
    def calibration_bins_csv(self) -> Path:
        return self.root / "calibration_bins.csv"

    @property
    def feature_importance_csv(self) -> Path:
        return self.root / "feature_importance.csv"

    @property
    def backtest_report_json(self) -> Path:
        return self.root / "backtest_report.json"

    @property
    def benchmark_profile_json(self) -> Path:
        return self.root / "benchmark_profile.json"

    @property
    def warnings_json(self) -> Path:
        return self.root / "warnings.json"

    @property
    def dashboard_prediction_view_parquet(self) -> Path:
        return self.root / "dashboard_prediction_view.parquet"

    @property
    def dashboard_prediction_view_csv(self) -> Path:
        return self.root / "dashboard_prediction_view.csv"

    @property
    def dashboard_manifest_json(self) -> Path:
        return self.root / "dashboard_manifest.json"

    def predictions_parquet(self, model: str) -> Path:
        return self.root / f"predictions_{model}.parquet"

    def predictions_csv(self, model: str) -> Path:
        return self.root / f"predictions_{model}.csv"


@dataclass(frozen=True)
class DataPaths:
    root: Path
    raw: Path
    normalized: Path
    analytical: Path
    cache: Path
    graph_exports: Path

    # ------------------------------------------------------------------ raw
    @property
    def raw_polymarket(self) -> Path:
        return self.raw / "polymarket"

    @property
    def raw_polymarket_markets(self) -> Path:
        return self.raw / "polymarket" / "markets"

    @property
    def raw_polymarket_events(self) -> Path:
        return self.raw / "polymarket" / "events"

    @property
    def raw_polymarket_prices(self) -> Path:
        return self.raw / "polymarket" / "prices"

    def raw_social(self, platform: str) -> Path:
        """Raw directory for a given social platform.

        Branch 04 groups social raw payloads under ``data/raw/social/<platform>/``
        so the layout matches the conceptual split between Polymarket and the
        social sources documented in ``docs/DATA_MODEL.md``.
        """
        return self.raw / "social" / platform

    @property
    def normalized_social(self) -> Path:
        return self.normalized / "social"

    @property
    def social_posts_parquet(self) -> Path:
        return self.normalized / "social" / "social_posts.parquet"

    @property
    def post_comments_parquet(self) -> Path:
        """Traccia_2 §2 sampled comments/replies (sibling of social_posts)."""
        return self.normalized / "social" / "post_comments.parquet"

    @property
    def social_queries_parquet(self) -> Path:
        return self.normalized / "social" / "social_queries.parquet"

    @property
    def social_enrichment_state(self) -> Path:
        """Branch 12B resumable collection checkpoints."""
        return self.root / "state"

    @property
    def social_enrichment_queries(self) -> Path:
        """Branch 12B market-aware query artifacts."""
        return self.root / "queries"

    @property
    def social_enrichment_reports(self) -> Path:
        """Branch 12B run reports."""
        return self.root / "reports"

    @property
    def social_collection_state_json(self) -> Path:
        return self.social_enrichment_state / "collection_state.json"

    @property
    def social_source_progress_json(self) -> Path:
        return self.social_enrichment_state / "source_progress.json"

    @property
    def social_query_progress_json(self) -> Path:
        return self.social_enrichment_state / "query_progress.json"

    @property
    def market_queries_parquet(self) -> Path:
        return self.social_enrichment_queries / "market_queries.parquet"

    @property
    def market_queries_csv(self) -> Path:
        return self.social_enrichment_queries / "market_queries.csv"

    @property
    def social_collection_summary_json(self) -> Path:
        return self.social_enrichment_reports / "collection_summary.json"

    @property
    def linked_posts_parquet(self) -> Path:
        """Branch 05 output — semantic-relevance verdicts per (market, post)."""
        return self.normalized / "social" / "linked_posts.parquet"

    @property
    def social_post_segments_parquet(self) -> Path:
        """Branch 05G derived dataset — multi-topic digest segments."""
        return self.normalized / "social" / "social_post_segments.parquet"

    # ----------------------------------------------------- high-signal dataset
    @property
    def analytical_dataset(self) -> Path:
        """Root for the curated high-signal Telegram dataset artefacts."""
        return self.analytical / "dataset"

    @property
    def high_signal_markets_parquet(self) -> Path:
        return self.analytical_dataset / "high_signal_markets.parquet"

    @property
    def telegram_search_plan_parquet(self) -> Path:
        return self.analytical_dataset / "telegram_search_plan.parquet"

    @property
    def telegram_search_plan_json(self) -> Path:
        return self.analytical_dataset / "telegram_search_plan.json"

    @property
    def telegram_search_manifest_json(self) -> Path:
        return self.analytical_dataset / "telegram_search_manifest.json"

    @property
    def telegram_coverage_report_json(self) -> Path:
        return self.analytical_dataset / "telegram_coverage_report.json"

    @property
    def high_signal_telegram_manifest_json(self) -> Path:
        return self.analytical_dataset / "high_signal_telegram_manifest.json"

    # --------------------------------------------------- Branch 06 — stance
    @property
    def stance_labels_parquet(self) -> Path:
        """Branch 06A output — outcome-aware stance labels."""
        return self.normalized / "social" / "stance_labels.parquet"

    @property
    def analytical_stance(self) -> Path:
        return self.analytical / "stance"

    @property
    def manual_stance_review_csv(self) -> Path:
        """Curated stance rows for human spot-checking."""
        return self.analytical_stance / "manual_stance_review.csv"

    @property
    def stance_evaluation_report_json(self) -> Path:
        """Stance evaluation report against human labels (when available)."""
        return self.analytical_stance / "stance_evaluation_report.json"

    @property
    def stance_llm_cache_dir(self) -> Path:
        """Branch 06B — on-disk cache for the optional LLM stance judge."""
        return self.llm_cache / "stance"

    @property
    def manual_review_candidates_csv(self) -> Path:
        """Branch 05 manual-review CSV (curated linked rows for human eyes)."""
        return self.analytical / "linking" / "manual_review_candidates.csv"

    @property
    def linking_coverage_report_json(self) -> Path:
        """Traccia_2 linking coverage statistics (fraction of contracts >=k posts)."""
        return self.analytical / "linking" / "coverage_report.json"

    # ----------------------------------- Branch 07 — resolved-market evaluation
    @property
    def analytical_evaluation(self) -> Path:
        """Root for resolved-market evaluation artefacts (evaluation targets only)."""
        return self.analytical / "evaluation"

    @property
    def resolved_market_labels_parquet(self) -> Path:
        """Canonical resolved-only evaluation labels.

        Branch 07 upstream-data milestone. These rows are **evaluation
        targets only** and must never feed retrieval, query generation,
        linking, stance, the Social Pressure Index, or prediction inputs.
        """
        return self.analytical_evaluation / "resolved_market_labels.parquet"

    @property
    def analytical_review(self) -> Path:
        """Root for human-readable review packs (quality control / demo)."""
        return self.analytical / "review"

    @property
    def top_linked_social_evidence_csv(self) -> Path:
        """Human review pack: top linked social evidence per market (CSV)."""
        return self.analytical_review / "top_linked_social_evidence.csv"

    @property
    def top_linked_social_evidence_parquet(self) -> Path:
        """Human review pack: top linked social evidence per market (Parquet)."""
        return self.analytical_review / "top_linked_social_evidence.parquet"

    @property
    def linking_evaluation_report(self) -> Path:
        """Branch 05 fixture-evaluation report (JSON)."""
        return self.analytical / "linking" / "evaluation_report.json"

    @property
    def embedding_cache_dir(self) -> Path:
        """On-disk cache for Branch 05B embedding vectors."""
        return self.cache / "embeddings"

    @property
    def linking_llm_cache_dir(self) -> Path:
        """On-disk cache for optional Branch 05C relevance judgments."""
        return self.llm_cache / "linking"

    @property
    def social_cache(self) -> Path:
        return self.cache / "social"

    def raw_run_dir(
        self,
        source: str,
        run_id: str,
        when: datetime | None = None,
    ) -> Path:
        """Return a run-scoped subdirectory under the raw zone.

        Layout follows the design captured in ``docs/DATA_MODEL.md``:

            data/raw/<source>/<YYYY-MM-DD>/<YYYYMMDDTHHMMSSZ>-<run_id>/

        ``source`` may be a slash-separated path such as
        ``"polymarket/markets"``. ``when`` defaults to ``datetime.now(UTC)``.
        The directory is not created here — callers decide when to
        materialize it on disk.
        """
        moment = (when or datetime.now(tz=timezone.utc)).astimezone(timezone.utc)
        day = moment.strftime("%Y-%m-%d")
        stamp = moment.strftime("%Y%m%dT%H%M%SZ")
        return self.raw / source / day / f"{stamp}-{run_id}"

    # ----------------------------------------------------------- normalized
    @property
    def normalized_polymarket(self) -> Path:
        return self.normalized / "polymarket"

    @property
    def polymarket_markets_parquet(self) -> Path:
        return self.normalized / "polymarket" / "markets.parquet"

    @property
    def polymarket_outcomes_parquet(self) -> Path:
        return self.normalized / "polymarket" / "outcomes.parquet"

    @property
    def polymarket_price_points_parquet(self) -> Path:
        return self.normalized / "polymarket" / "price_points.parquet"

    @property
    def markets_parquet(self) -> Path:
        return self.normalized / "markets.parquet"

    @property
    def prices_parquet(self) -> Path:
        return self.normalized / "prices.parquet"

    @property
    def posts_parquet(self) -> Path:
        return self.normalized / "posts.parquet"

    @property
    def links_parquet(self) -> Path:
        return self.normalized / "links.parquet"

    @property
    def stances_parquet(self) -> Path:
        return self.normalized / "stances.parquet"

    @property
    def entities_parquet(self) -> Path:
        return self.normalized / "entities.parquet"

    # ----------------------------------------------------------- analytical
    @property
    def social_pressure_parquet(self) -> Path:
        return self.analytical / "social_pressure.parquet"

    # ------------------------------------------- Branch 08 — feature engineering
    @property
    def analytical_features(self) -> Path:
        """Root for Branch 08 feature-engineering artefacts."""
        return self.analytical / "features"

    @property
    def social_pressure_features_parquet(self) -> Path:
        """Branch 08 main output — per-market Social Pressure Index features."""
        return self.analytical_features / "social_pressure.parquet"

    @property
    def social_pressure_features_csv(self) -> Path:
        """Optional CSV mirror of the SPI feature table."""
        return self.analytical_features / "social_pressure.csv"

    @property
    def social_pressure_summary_json(self) -> Path:
        """Run-level summary for the SPI feature computation."""
        return self.analytical_features / "social_pressure_summary.json"

    @property
    def task3_feature_matrix_parquet(self) -> Path:
        """Task-3 prediction feature matrix (one row per resolved contract)."""
        return self.analytical_features / "task3_feature_matrix.parquet"

    @property
    def task3_feature_matrix_summary_json(self) -> Path:
        """Provenance/metadata for the Task-3 feature matrix."""
        return self.analytical_features / "task3_feature_matrix_summary.json"

    # ------------------------------------ Branch 09 — forecasting / backtests
    @property
    def analytical_backtests(self) -> Path:
        """Root for leakage-free forecasting and backtest artefacts."""
        return self.analytical / "backtests"

    def backtest_profile(self, profile: str) -> BacktestPaths:
        """Return isolated output paths for one Branch 09 benchmark profile."""
        if not profile or profile in {".", ".."} or "/" in profile or "\\" in profile:
            raise ValueError(f"invalid benchmark profile path component: {profile!r}")
        return BacktestPaths(root=self.analytical_backtests / profile)

    @property
    def analytical_benchmark_profiles(self) -> Path:
        """Root for derived Branch 09 benchmark inputs."""
        return self.analytical / "benchmark_profiles"

    def controlled_social_pressure_features_parquet(self, profile: str) -> Path:
        """Derived controlled SPI snapshot; never the canonical Branch 08 table."""
        return self.analytical_benchmark_profiles / profile / "social_pressure_controlled.parquet"

    @property
    def evaluation_dataset_parquet(self) -> Path:
        return self.analytical_backtests / "evaluation_dataset.parquet"

    @property
    def evaluation_dataset_csv(self) -> Path:
        return self.analytical_backtests / "evaluation_dataset.csv"

    @property
    def predictions_constant_parquet(self) -> Path:
        return self.analytical_backtests / "predictions_constant.parquet"

    @property
    def predictions_market_only_parquet(self) -> Path:
        return self.analytical_backtests / "predictions_market_only.parquet"

    @property
    def predictions_social_only_parquet(self) -> Path:
        return self.analytical_backtests / "predictions_social_only.parquet"

    @property
    def predictions_market_social_parquet(self) -> Path:
        return self.analytical_backtests / "predictions_market_social.parquet"

    @property
    def metrics_summary_json(self) -> Path:
        return self.analytical_backtests / "metrics_summary.json"

    @property
    def metrics_summary_csv(self) -> Path:
        return self.analytical_backtests / "metrics_summary.csv"

    @property
    def calibration_bins_csv(self) -> Path:
        return self.analytical_backtests / "calibration_bins.csv"

    @property
    def feature_importance_csv(self) -> Path:
        return self.analytical_backtests / "feature_importance.csv"

    @property
    def backtest_report_json(self) -> Path:
        return self.analytical_backtests / "backtest_report.json"

    @property
    def probability_estimates_parquet(self) -> Path:
        return self.analytical / "probability_estimates.parquet"

    # ---------------------------------------------------------------- cache
    @property
    def llm_cache(self) -> Path:
        return self.cache / "llm"

    @property
    def feature_cache(self) -> Path:
        return self.cache / "feature"

    @property
    def refresh_state(self) -> Path:
        return self.cache / "refresh_state.json"

    # ------------------------------------------------------------------ ops
    def ensure(self) -> None:
        """Create the directory layout if it doesn't already exist."""
        for path in (
            self.raw,
            self.normalized,
            self.analytical,
            self.cache,
            self.graph_exports,
            self.llm_cache,
            self.feature_cache,
        ):
            path.mkdir(parents=True, exist_ok=True)


def get_data_paths(root: Path | None = None) -> DataPaths:
    """Build a ``DataPaths`` instance, defaulting to the configured root."""
    base = (root or get_settings().data_dir).resolve()
    return DataPaths(
        root=base,
        raw=base / "raw",
        normalized=base / "normalized",
        analytical=base / "analytical",
        cache=base / "cache",
        graph_exports=base / "graph_exports",
    )

"""Tests for the Task-3 feature matrix and temporal-CV model runner."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from polysignal.evaluation.task3 import (
    ABLATIONS,
    MODEL_FAMILIES,
    evaluate_predictions,
    run_task3,
    write_task3_artifacts,
)
from polysignal.evaluation.task3_features import (
    NUMERIC_SOCIAL_FEATURES,
    TEXT_COLUMN,
    assemble_feature_matrix,
    evidence_social_features,
)


def _evidence_fixture() -> pd.DataFrame:
    base = datetime(2025, 3, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(6):
        rows.append(
            {
                "market_id": "m1",
                "parent_post_id": f"p{i}",
                "post_author": f"a{i % 2}",
                "post_channel": "chan_a",
                "post_engagement_score": float(i * 10),
                "post_created_at": base + timedelta(days=i),
                "post_text_preview": "election polls debate" if i % 2 else "rally crowd speech",
                "market_close_time": base + timedelta(days=10),
            }
        )
    rows.append(
        {
            "market_id": "m2",
            "parent_post_id": "q0",
            "post_author": "z0",
            "post_channel": "chan_b",
            "post_engagement_score": 5.0,
            "post_created_at": base,
            "post_text_preview": "rate decision inflation",
            "market_close_time": base + timedelta(days=10),
        }
    )
    return pd.DataFrame(rows)


def _enrichment_fixture() -> pd.DataFrame:
    rows = [
        {"post_id": f"p{i}", "sentiment_compound": 0.5 if i % 2 else -0.5,
         "sentiment_label": "positive" if i % 2 else "negative"}
        for i in range(6)
    ]
    rows.append({"post_id": "q0", "sentiment_compound": 0.0, "sentiment_label": "neutral"})
    return pd.DataFrame(rows)


class TestEvidenceSocialFeatures:
    def test_aggregates_per_market(self):
        feats = evidence_social_features(_evidence_fixture(), _enrichment_fixture())
        assert set(feats["market_id"]) == {"m1", "m2"}
        m1 = feats[feats["market_id"] == "m1"].iloc[0]
        assert m1["vol_total_posts"] == 6
        assert m1["vol_unique_authors"] == 2
        assert m1["vol_unique_channels"] == 1
        assert 0.0 <= m1["sent_frac_pos"] <= 1.0
        assert m1[TEXT_COLUMN]  # non-empty document
        assert m1["ling_doc_len"] > 0

    def test_empty_evidence_returns_columns(self):
        feats = evidence_social_features(pd.DataFrame(), pd.DataFrame())
        assert feats.empty
        assert TEXT_COLUMN in feats.columns


class TestAssembleFeatureMatrix:
    def test_missing_evidence_is_zero_filled(self):
        base = pd.DataFrame(
            {
                "market_id": ["m1", "m2", "m3"],
                "domain": ["politics", "finance", "sports"],
                "close_time": pd.to_datetime(["2025-03-10", "2025-03-11", "2025-03-12"], utc=True),
                "binary_yes_no": ["YES", "NO", "NO"],
                "target_yes": [1, 0, 0],
                "market_probability": [0.7, np.nan, 0.2],
                "engagement_total": [10.0, 0.0, 0.0],
                "engagement_mean": [2.0, 0.0, 0.0],
                "engagement_max": [5.0, 0.0, 0.0],
                "engagement_log_total": [2.3, 0.0, 0.0],
                "stance_balance": [0.1, 0.0, 0.0],
                "pressure_strength": [0.2, 0.0, 0.0],
                "net_pressure_weighted": [0.0, 0.0, 0.0],
                "spi_score_shrunk": [0.5, 0.5, 0.5],
                "temporal_decay_pressure": [0.0, 0.0, 0.0],
                "evidence_quality_score": [0.0, 0.0, 0.0],
                "source_diversity_score": [0.0, 0.0, 0.0],
                "accepted_evidence_ratio": [0.0, 0.0, 0.0],
                "directional_stance_count": [3, 0, 0],
            }
        )
        evidence_features = evidence_social_features(_evidence_fixture(), _enrichment_fixture())
        matrix = assemble_feature_matrix(base, evidence_features)
        assert len(matrix) == 3
        # No NaNs in numeric social features after the left join.
        assert matrix[list(NUMERIC_SOCIAL_FEATURES)].isna().sum().sum() == 0
        # The unpriced contract was imputed to a neutral 0.5.
        assert matrix.loc[matrix["market_id"] == "m2", "market_probability"].iloc[0] == 0.5
        # vol_total_comments is an honest zero (not collected yet).
        assert (matrix["vol_total_comments"] == 0).all()


def _synthetic_matrix(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    vocab = [f"w{i}" for i in range(30)]
    market_prob = rng.uniform(0, 1, n)
    target = (market_prob + rng.normal(0, 0.15, n) > 0.5).astype(int)
    docs = [" ".join(rng.choice(vocab, size=12)) for _ in range(n)]
    data = {
        "market_id": [f"m{i:03d}" for i in range(n)],
        "domain": rng.choice(["politics", "finance", "sports"], n),
        "close_time": [base + timedelta(days=i) for i in range(n)],
        "binary_yes_no": ["YES" if t else "NO" for t in target],
        "target_yes": target,
        "market_probability": market_prob,
        TEXT_COLUMN: docs,
        "has_social_evidence": rng.random(n) > 0.3,
    }
    for col in NUMERIC_SOCIAL_FEATURES:
        data[col] = rng.normal(0, 1, n)
    return pd.DataFrame(data)


class TestEvaluatePredictions:
    def test_perfect_separation_scores_one(self):
        preds = np.array([0.9, 0.1, 0.8, 0.2])
        mask = np.array([True, True, True, True])
        targets = np.array([1, 0, 1, 0])
        metrics = evaluate_predictions(preds, mask, targets)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["auc"] == 1.0

    def test_empty_scored_set(self):
        metrics = evaluate_predictions(np.array([np.nan]), np.array([False]), np.array([1]))
        assert metrics == {"n_eval": 0}


class TestRunTask3:
    def test_temporal_cv_ablation_runs(self):
        matrix = _synthetic_matrix()
        run = run_task3(matrix=matrix, feature_metadata={"source": "fixture"}, n_splits=3)
        results = run.results
        assert results["cross_validation"]["scheme"] == "expanding_window_temporal"
        assert results["cross_validation"]["n_folds"] >= 1
        # Every model family and ablation is reported.
        assert set(results["models"]) == set(MODEL_FAMILIES)
        for ablations in results["models"].values():
            assert set(ablations) == set(ABLATIONS)
        # The market arm should beat the constant baseline on this market-driven target.
        best_market = max(
            ablations["market_only"]["macro_f1"] for ablations in results["models"].values()
        )
        assert best_market > results["baseline_constant"]["macro_f1"]
        assert not run.feature_importance.empty

    def test_artifacts_round_trip(self, tmp_path):
        from polysignal.storage.paths import get_data_paths

        paths = get_data_paths(root=tmp_path)
        paths.ensure()
        run = run_task3(matrix=_synthetic_matrix(), feature_metadata={}, n_splits=3)
        written = write_task3_artifacts(run, paths=paths, overwrite=False)
        assert written["results_json"].exists()
        assert written["results_csv"].exists()
        assert paths.task3_feature_matrix_parquet.exists()
        # Refuses to clobber without --overwrite.
        with pytest.raises(FileExistsError):
            write_task3_artifacts(run, paths=paths, overwrite=False)

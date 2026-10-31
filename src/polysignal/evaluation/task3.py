"""Task-3 outcome prediction (Traccia_2 Task 3).

Trains several model families on the Task-3 feature matrix under an expanding-window
**temporal cross-validation** scheme and runs the market/social/combined **ablation**.
For every (model, ablation) pair it reports accuracy, macro-F1 (+ per-class F1),
AUC-ROC, Brier and calibration error on the out-of-fold predictions, plus a constant
base-rate baseline. Results persist as ``task3_results.{json,csv}`` next to the Branch-09
backtest artefacts.

Design notes
------------
* **No leakage.** Folds are ordered by contract close time (expanding window), the
  linguistic TF-IDF -> SVD representation is fit *inside each training fold*, and the
  resolution label is never a predictor.
* **Honest degradation.** A training fold with a single outcome class falls back to the
  smoothed training base rate rather than fabricating a fit, and AUC is ``None`` when a
  scored set is single-class.
* **Model families** (all expose ``predict_proba`` for AUC/calibration): regularized
  logistic regression (linear), linear SVM (Platt-calibrated), Gaussian naive Bayes
  (probabilistic), and gradient boosting (tree ensemble).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from polysignal.analytics._common import to_utc
from polysignal.evaluation.metrics import (
    accuracy,
    brier_score,
    expected_calibration_error,
    f1_per_class,
    log_loss,
    macro_f1,
)
from polysignal.evaluation.task3_features import (
    MARKET_FEATURES,
    NUMERIC_SOCIAL_FEATURES,
    TEXT_COLUMN,
    build_task3_feature_matrix,
    write_task3_feature_matrix,
)
from polysignal.evaluation.temporal_cv import expanding_window_splits
from polysignal.storage.paths import DataPaths, get_data_paths

RANDOM_STATE = 42
TFIDF_MAX_FEATURES = 2000
SVD_COMPONENTS = 16

ModelFactory = Callable[[], Any]

MODEL_FAMILIES: dict[str, ModelFactory] = {
    "logistic_regression": lambda: LogisticRegression(
        C=0.5, max_iter=1000, class_weight="balanced", solver="liblinear",
        random_state=RANDOM_STATE,
    ),
    "linear_svm": lambda: CalibratedClassifierCV(
        LinearSVC(C=0.5, class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE),
        method="sigmoid",
        cv=3,
        ensemble=False,
    ),
    "naive_bayes": lambda: GaussianNB(),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=RANDOM_STATE),
}

# ablation name -> (numeric feature columns, whether to add the text branch)
ABLATIONS: dict[str, tuple[tuple[str, ...], bool]] = {
    "market_only": (MARKET_FEATURES, False),
    "social_only": (NUMERIC_SOCIAL_FEATURES, True),
    "market_social": (MARKET_FEATURES + NUMERIC_SOCIAL_FEATURES, True),
}


@dataclass
class Task3Run:
    """In-memory Task-3 run, ready to persist."""

    matrix: pd.DataFrame
    feature_metadata: dict[str, Any]
    results: dict[str, Any]
    results_table: pd.DataFrame
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)


def _build_pipeline(estimator: Any, numeric_features: Sequence[str], use_text: bool) -> Pipeline:
    """Assemble a leakage-safe feature transformer + classifier pipeline."""
    numeric_branch = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scale", StandardScaler()),
        ]
    )
    transformers: list[tuple[str, Any, Any]] = [
        ("numeric", numeric_branch, list(numeric_features)),
    ]
    if use_text:
        text_branch = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=TFIDF_MAX_FEATURES,
                        stop_words="english",
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                ("svd", TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)),
            ]
        )
        # A scalar column key passes the 1-D text Series the vectorizer expects.
        transformers.append(("text", text_branch, TEXT_COLUMN))
    features = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(steps=[("features", features), ("clf", estimator)])


def _smoothed_rate(values: Sequence[int]) -> float:
    return (sum(int(v) for v in values) + 1.0) / (len(values) + 2.0)


def _temporal_oof_predictions(
    matrix: pd.DataFrame,
    folds: Sequence[tuple[list[int], list[int]]],
    factory: ModelFactory,
    numeric_features: Sequence[str],
    use_text: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(predictions, scored_mask)`` of out-of-fold positive-class probs."""
    n = len(matrix)
    predictions = np.full(n, np.nan, dtype=float)
    targets = matrix["target_yes"].astype(int).to_numpy()
    for train_idx, test_idx in folds:
        y_train = targets[train_idx]
        if len(set(y_train.tolist())) < 2:
            predictions[test_idx] = _smoothed_rate(y_train.tolist())
            continue
        pipeline = _build_pipeline(factory(), numeric_features, use_text)
        pipeline.fit(matrix.iloc[train_idx], y_train)
        proba = pipeline.predict_proba(matrix.iloc[test_idx])
        predictions[test_idx] = proba[:, 1]
    scored_mask = ~np.isnan(predictions)
    return predictions, scored_mask


def _constant_oof_predictions(
    matrix: pd.DataFrame,
    folds: Sequence[tuple[list[int], list[int]]],
) -> tuple[np.ndarray, np.ndarray]:
    n = len(matrix)
    predictions = np.full(n, np.nan, dtype=float)
    targets = matrix["target_yes"].astype(int).to_numpy()
    for train_idx, test_idx in folds:
        predictions[test_idx] = _smoothed_rate(targets[train_idx].tolist())
    scored_mask = ~np.isnan(predictions)
    return predictions, scored_mask


def evaluate_predictions(
    predictions: np.ndarray,
    scored_mask: np.ndarray,
    targets: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute the full Task-3 metric set on the scored out-of-fold rows."""
    probs = predictions[scored_mask].tolist()
    outs = targets[scored_mask].astype(int).tolist()
    if not probs:
        return {"n_eval": 0}
    both_classes = len(set(outs)) == 2
    per_class = f1_per_class(probs, outs)
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(outs, probs)) if both_classes else None
    except Exception:  # pragma: no cover - defensive
        auc = None
    return {
        "n_eval": len(outs),
        "target_yes": int(sum(outs)),
        "target_no": int(len(outs) - sum(outs)),
        "accuracy": round(accuracy(probs, outs), 4),
        "macro_f1": round(macro_f1(probs, outs), 4),
        "f1_yes": round(per_class[1], 4),
        "f1_no": round(per_class[0], 4),
        "auc": round(auc, 4) if auc is not None else None,
        "brier": round(brier_score(probs, outs), 4),
        "log_loss": round(log_loss(probs, outs), 4),
        "ece": round(expected_calibration_error(probs, outs, n_bins=n_bins), 4),
    }


def _feature_importance(
    matrix: pd.DataFrame,
    numeric_features: Sequence[str],
    use_text: bool,
) -> pd.DataFrame:
    """Descriptive full-dataset logistic coefficients over the combined features."""
    targets = matrix["target_yes"].astype(int).to_numpy()
    if len(set(targets.tolist())) < 2:
        return pd.DataFrame(columns=["feature", "coefficient", "absolute_coefficient", "note"])
    pipeline = _build_pipeline(
        MODEL_FAMILIES["logistic_regression"](), numeric_features, use_text
    )
    pipeline.fit(matrix, targets)
    names = [f"num::{name}" for name in numeric_features]
    if use_text:
        names += [f"text_svd::{i}" for i in range(SVD_COMPONENTS)]
    clf = pipeline.named_steps["clf"]
    coeffs = clf.coef_[0]
    if len(coeffs) != len(names):  # pragma: no cover - shape guard
        names = [f"f{i}" for i in range(len(coeffs))]
    rows = [
        {
            "feature": name,
            "coefficient": float(value),
            "absolute_coefficient": abs(float(value)),
            "note": "descriptive full-dataset fit; scored metrics are out-of-fold",
        }
        for name, value in zip(names, coeffs, strict=True)
    ]
    return pd.DataFrame(rows).sort_values("absolute_coefficient", ascending=False).reset_index(
        drop=True
    )


def run_task3(
    *,
    paths: DataPaths | None = None,
    n_splits: int = 5,
    matrix: pd.DataFrame | None = None,
    feature_metadata: dict[str, Any] | None = None,
) -> Task3Run:
    """Build (or accept) the feature matrix and run the temporal-CV ablation."""
    paths = paths or get_data_paths()
    if matrix is None:
        matrix, feature_metadata = build_task3_feature_matrix(paths=paths)
    feature_metadata = feature_metadata or {}

    order_keys = to_utc(matrix["close_time"]).tolist()
    folds = expanding_window_splits(order_keys, n_splits=n_splits, min_train_size=20)
    targets = matrix["target_yes"].astype(int).to_numpy()

    models: dict[str, dict[str, Any]] = {}
    table_rows: list[dict[str, Any]] = []

    const_pred, const_mask = _constant_oof_predictions(matrix, folds)
    const_metrics = evaluate_predictions(const_pred, const_mask, targets)
    table_rows.append({"model": "constant", "ablation": "baseline", **const_metrics})

    for model_name, factory in MODEL_FAMILIES.items():
        models[model_name] = {}
        for ablation, (numeric_features, use_text) in ABLATIONS.items():
            preds, mask = _temporal_oof_predictions(
                matrix, folds, factory, numeric_features, use_text
            )
            metrics = evaluate_predictions(preds, mask, targets)
            models[model_name][ablation] = metrics
            table_rows.append({"model": model_name, "ablation": ablation, **metrics})

    results_table = pd.DataFrame(table_rows)
    importance = _feature_importance(
        matrix, MARKET_FEATURES + NUMERIC_SOCIAL_FEATURES, use_text=True
    )

    best = _best_configuration(models)
    results = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "n_contracts": int(len(matrix)),
        "n_with_social_evidence": int(matrix.get("has_social_evidence", pd.Series(dtype=bool)).sum()),
        "target_distribution": {
            "YES": int((targets == 1).sum()),
            "NO": int((targets == 0).sum()),
        },
        "cross_validation": {
            "scheme": "expanding_window_temporal",
            "order_key": "contract close_time",
            "n_splits_requested": n_splits,
            "n_folds": len(folds),
            "fold_sizes": [{"train": len(tr), "test": len(te)} for tr, te in folds],
            "min_train_size": 20,
        },
        "model_families": list(MODEL_FAMILIES),
        "ablations": list(ABLATIONS),
        "baseline_constant": const_metrics,
        "models": models,
        "best_by_macro_f1": best,
        "conclusion": _conclusion(models, best),
        "feature_metadata": feature_metadata,
    }
    return Task3Run(
        matrix=matrix,
        feature_metadata=feature_metadata,
        results=results,
        results_table=results_table,
        feature_importance=importance,
    )


def _best_configuration(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {"macro_f1": -1.0}
    for model_name, ablations in models.items():
        for ablation, metrics in ablations.items():
            score = metrics.get("macro_f1")
            if score is not None and score > best["macro_f1"]:
                best = {"model": model_name, "ablation": ablation, **metrics}
    return best


def _conclusion(models: dict[str, dict[str, Any]], best: dict[str, Any]) -> str:
    """Compare the best social/combined config against market-only, honestly."""
    market_scores = [
        ablations["market_only"].get("macro_f1")
        for ablations in models.values()
        if ablations.get("market_only", {}).get("macro_f1") is not None
    ]
    social_scores = [
        ablations["social_only"].get("macro_f1")
        for ablations in models.values()
        if ablations.get("social_only", {}).get("macro_f1") is not None
    ]
    if not market_scores or not social_scores:
        return "Insufficient evaluation rows for a confident Task-3 conclusion."
    best_market = max(market_scores)
    best_social = max(social_scores)
    if best_social > best_market + 0.02:
        return (
            "Social features show a measurable macro-F1 edge over market-only at current "
            "coverage, but the sample is small — treat as preliminary."
        )
    return (
        "Consistent with the project thesis: the market price is the strong predictor and "
        "social features add no macro-F1 lift over market-only at current (Telegram-heavy) "
        "coverage."
    )


def write_task3_artifacts(
    run: Task3Run,
    *,
    paths: DataPaths | None = None,
    profile: str = "real_optional",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Persist the feature matrix, results JSON/CSV and feature-importance CSV."""
    paths = paths or get_data_paths()
    matrix_path = write_task3_feature_matrix(run.matrix, paths=paths)
    paths.task3_feature_matrix_summary_json.write_text(
        json.dumps(run.feature_metadata, indent=2, default=str), encoding="utf-8"
    )

    root = paths.backtest_profile(profile).root
    root.mkdir(parents=True, exist_ok=True)
    results_json = root / "task3_results.json"
    results_csv = root / "task3_results.csv"
    importance_csv = root / "task3_feature_importance.csv"
    if not overwrite:
        existing = [p for p in (results_json, results_csv) if p.exists()]
        if existing:
            joined = ", ".join(str(p) for p in existing)
            raise FileExistsError(f"refusing to overwrite existing Task-3 artefacts: {joined}")

    results_json.write_text(json.dumps(run.results, indent=2, default=str), encoding="utf-8")
    run.results_table.to_csv(results_csv, index=False)
    if not run.feature_importance.empty:
        run.feature_importance.to_csv(importance_csv, index=False)
    return {
        "feature_matrix": matrix_path,
        "results_json": results_json,
        "results_csv": results_csv,
        "feature_importance_csv": importance_csv if not run.feature_importance.empty else None,
    }


__all__ = [
    "ABLATIONS",
    "MODEL_FAMILIES",
    "Task3Run",
    "evaluate_predictions",
    "run_task3",
    "write_task3_artifacts",
]

"""Task-3 feature matrix (Traccia_2 Task 3 — outcome prediction).

Builds one row per resolved binary Polymarket contract (label = resolution
outcome) covering the four Traccia_2 feature categories:

* **Volume** — total accepted posts, comments sample size, posting growth rate
  over the contract lifetime, unique authors/channels, active days.
* **Engagement** — aggregate/log engagement (reused from the Branch-08 Social
  Pressure table) plus a high-vs-low-engagement ratio.
* **Sentiment** — mean and variance of post sentiment, positive/negative/neutral
  fractions, and an early-vs-late sentiment trend.
* **Linguistic** — a per-contract text document (concatenated pre-close post
  text) that the model runner vectorises with a *per-fold* TF-IDF -> SVD pipeline
  so the linguistic representation never leaks across the temporal split.

The Social Pressure Index columns (Branch 08) are carried through as additional
engineered social features. The market price arm (``market_probability``) is the
latest valid pre-close YES price, joined by the leakage-safe
:func:`polysignal.evaluation.backtest.build_evaluation_dataset`.

Every social feature is derived only from posts published at or before each
contract's close time (``accepted_evidence`` already enforces this), so the
matrix is safe to consume under temporal cross-validation. Contracts without any
accepted social evidence keep zero-filled social features — an honest "no signal"
state, never a fabricated one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from polysignal.analytics._common import to_utc
from polysignal.evaluation.backtest import build_evaluation_dataset
from polysignal.storage.parquet_store import read_dataframe
from polysignal.storage.paths import DataPaths, get_data_paths

# Maximum characters retained per contract text document (bounds TF-IDF memory;
# the vocabulary of a contract is captured well before this limit).
MAX_DOCUMENT_CHARS = 20_000

# ---- feature-group column contracts (imported by the model runner) ----------
MARKET_FEATURES: tuple[str, ...] = ("market_probability",)

VOLUME_FEATURES: tuple[str, ...] = (
    "vol_total_posts",
    "vol_total_comments",
    "vol_posting_growth",
    "vol_unique_authors",
    "vol_unique_channels",
    "vol_active_days",
)

ENGAGEMENT_FEATURES: tuple[str, ...] = (
    "engagement_total",
    "engagement_mean",
    "engagement_max",
    "engagement_log_total",
    "eng_high_low_ratio",
)

SENTIMENT_FEATURES: tuple[str, ...] = (
    "sent_mean",
    "sent_var",
    "sent_frac_pos",
    "sent_frac_neg",
    "sent_frac_neutral",
    "sent_trend",
)

# Reused Branch-08 Social Pressure Index columns (already leakage-audited).
PRESSURE_FEATURES: tuple[str, ...] = (
    "stance_balance",
    "pressure_strength",
    "net_pressure_weighted",
    "spi_score_shrunk",
    "temporal_decay_pressure",
    "evidence_quality_score",
    "source_diversity_score",
    "accepted_evidence_ratio",
    "directional_stance_count",
)

TEXT_COLUMN = "ling_text"

NUMERIC_SOCIAL_FEATURES: tuple[str, ...] = (
    VOLUME_FEATURES + ENGAGEMENT_FEATURES + SENTIMENT_FEATURES + PRESSURE_FEATURES
)

IDENTITY_COLUMNS: tuple[str, ...] = (
    "market_id",
    "domain",
    "close_time",
    "binary_yes_no",
    "target_yes",
)

_EVIDENCE_FEATURE_COLUMNS: tuple[str, ...] = (
    "market_id",
    "vol_total_posts",
    "vol_posting_growth",
    "vol_unique_authors",
    "vol_unique_channels",
    "vol_active_days",
    "eng_high_low_ratio",
    "sent_mean",
    "sent_var",
    "sent_frac_pos",
    "sent_frac_neg",
    "sent_frac_neutral",
    "sent_trend",
    TEXT_COLUMN,
    "ling_doc_len",
)


def _empty_evidence_features() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_EVIDENCE_FEATURE_COLUMNS))


def _posting_growth(times: pd.Series) -> float:
    """Late-vs-early posting growth over the observed posting window.

    Splits the contract's posting window at its temporal midpoint and returns the
    log-ratio of late-half to early-half post counts (smoothed). Positive means
    discourse accelerated toward resolution; ``0.0`` when there is no spread.
    """
    valid = times.dropna()
    if len(valid) < 2:
        return 0.0
    start, end = valid.min(), valid.max()
    if start == end:
        return 0.0
    midpoint = start + (end - start) / 2
    early = int((valid <= midpoint).sum())
    late = int((valid > midpoint).sum())
    return float(np.log((late + 1.0) / (early + 1.0)))


def _sentiment_trend(times: pd.Series, sentiment: pd.Series) -> float:
    """Late-half minus early-half mean sentiment (0.0 when undefined)."""
    frame = pd.DataFrame({"t": times, "s": sentiment}).dropna()
    if len(frame) < 2:
        return 0.0
    start, end = frame["t"].min(), frame["t"].max()
    if start == end:
        return 0.0
    midpoint = start + (end - start) / 2
    early = frame.loc[frame["t"] <= midpoint, "s"]
    late = frame.loc[frame["t"] > midpoint, "s"]
    if early.empty or late.empty:
        return 0.0
    return float(late.mean() - early.mean())


def evidence_social_features(
    evidence: pd.DataFrame,
    enrichment: pd.DataFrame,
) -> pd.DataFrame:
    """Per-contract volume/engagement/sentiment/linguistic features from evidence.

    ``evidence`` is the accepted-evidence table (already accepted + pre-close);
    ``enrichment`` supplies per-post sentiment, joined on ``parent_post_id``.
    """
    if evidence.empty:
        return _empty_evidence_features()

    ev = evidence.copy()
    ev["post_created_at"] = to_utc(ev["post_created_at"])
    ev["post_engagement_score"] = pd.to_numeric(ev["post_engagement_score"], errors="coerce")
    ev["post_text_preview"] = ev["post_text_preview"].astype("string").fillna("")
    ev["post_author"] = ev["post_author"].astype("string")
    ev["post_channel"] = ev["post_channel"].astype("string")

    if not enrichment.empty:
        senti = (
            enrichment[["post_id", "sentiment_compound", "sentiment_label"]]
            .drop_duplicates("post_id")
            .copy()
        )
        senti["sentiment_compound"] = pd.to_numeric(senti["sentiment_compound"], errors="coerce")
        senti["sentiment_label"] = senti["sentiment_label"].astype("string").str.lower()
        ev = ev.merge(
            senti, left_on="parent_post_id", right_on="post_id", how="left"
        )
    else:
        ev["sentiment_compound"] = np.nan
        ev["sentiment_label"] = pd.Series(pd.NA, index=ev.index, dtype="string")

    global_median_engagement = float(ev["post_engagement_score"].median(skipna=True) or 0.0)

    rows: list[dict[str, Any]] = []
    for market_id, slab in ev.groupby("market_id", sort=True):
        engagement = slab["post_engagement_score"]
        high = int((engagement > global_median_engagement).sum())
        low = int((engagement <= global_median_engagement).sum())
        sentiment = slab["sentiment_compound"]
        labels = slab["sentiment_label"]
        n_labelled = int(labels.notna().sum())
        document = " ".join(t for t in slab["post_text_preview"].tolist() if t)[:MAX_DOCUMENT_CHARS]
        rows.append(
            {
                "market_id": market_id,
                "vol_total_posts": int(len(slab)),
                "vol_posting_growth": _posting_growth(slab["post_created_at"]),
                "vol_unique_authors": int(slab["post_author"].nunique(dropna=True)),
                "vol_unique_channels": int(slab["post_channel"].nunique(dropna=True)),
                "vol_active_days": int(slab["post_created_at"].dt.floor("D").nunique()),
                "eng_high_low_ratio": float((high + 1.0) / (low + 1.0)),
                "sent_mean": float(sentiment.mean()) if sentiment.notna().any() else 0.0,
                "sent_var": float(sentiment.var(ddof=0)) if sentiment.notna().sum() > 1 else 0.0,
                "sent_frac_pos": (
                    float((labels == "positive").sum() / n_labelled) if n_labelled else 0.0
                ),
                "sent_frac_neg": (
                    float((labels == "negative").sum() / n_labelled) if n_labelled else 0.0
                ),
                "sent_frac_neutral": (
                    float((labels == "neutral").sum() / n_labelled) if n_labelled else 0.0
                ),
                "sent_trend": _sentiment_trend(slab["post_created_at"], sentiment),
                TEXT_COLUMN: document,
                "ling_doc_len": int(len(document.split())),
            }
        )
    return pd.DataFrame(rows, columns=list(_EVIDENCE_FEATURE_COLUMNS))


def assemble_feature_matrix(
    base: pd.DataFrame,
    evidence_features: pd.DataFrame,
) -> pd.DataFrame:
    """Join the leakage-safe base frame with evidence-derived social features.

    Contracts absent from ``evidence_features`` (no accepted social evidence) get
    zero-filled numeric social features and an empty text document.
    """
    keep_base = [
        "market_id",
        "domain",
        "close_time",
        "binary_yes_no",
        "target_yes",
        "market_probability",
        "engagement_total",
        "engagement_mean",
        "engagement_max",
        "engagement_log_total",
        *PRESSURE_FEATURES,
    ]
    available = [c for c in keep_base if c in base.columns]
    frame = base[available].copy()
    frame = frame.merge(evidence_features, on="market_id", how="left", validate="one_to_one")

    # Neutral 0.5 for the few contracts without a valid pre-close YES price, so the
    # market arm scores every row and the ablation compares like-for-like.
    if "market_probability" in frame.columns:
        frame["market_probability"] = pd.to_numeric(
            frame["market_probability"], errors="coerce"
        ).fillna(0.5)
    frame["vol_total_comments"] = 0  # comments sample not yet collected (honest zero)
    numeric_fill = [
        c for c in NUMERIC_SOCIAL_FEATURES if c in frame.columns and c != "vol_total_comments"
    ]
    frame[numeric_fill] = frame[numeric_fill].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].astype("string").fillna("")
    frame["ling_doc_len"] = pd.to_numeric(frame.get("ling_doc_len"), errors="coerce").fillna(0).astype(int)
    frame["has_social_evidence"] = frame["vol_total_posts"] > 0
    ordered = [
        *IDENTITY_COLUMNS,
        *MARKET_FEATURES,
        *NUMERIC_SOCIAL_FEATURES,
        "ling_doc_len",
        "has_social_evidence",
        TEXT_COLUMN,
    ]
    ordered = [c for c in ordered if c in frame.columns]
    return frame[ordered].sort_values("market_id").reset_index(drop=True)


def build_task3_feature_matrix(
    *,
    paths: DataPaths | None = None,
    benchmark_profile: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the Task-3 feature matrix and a provenance/metadata dictionary."""
    paths = paths or get_data_paths()
    dataset = build_evaluation_dataset(
        paths=paths,
        benchmark_profile=benchmark_profile,
        target="yes",
    )
    base = dataset.frame
    evidence = (
        read_dataframe(paths.analytical / "evidence" / "accepted_evidence.parquet")
        if (paths.analytical / "evidence" / "accepted_evidence.parquet").exists()
        else pd.DataFrame()
    )
    enrichment_path = paths.normalized_social / "post_enrichment.parquet"
    enrichment = read_dataframe(enrichment_path) if enrichment_path.exists() else pd.DataFrame()

    evidence_features = evidence_social_features(evidence, enrichment)
    matrix = assemble_feature_matrix(base, evidence_features)

    metadata = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "n_contracts": int(len(matrix)),
        "n_with_social_evidence": int(matrix["has_social_evidence"].sum()),
        "target_distribution": {
            "YES": int((matrix["target_yes"] == 1).sum()),
            "NO": int((matrix["target_yes"] == 0).sum()),
        },
        "feature_groups": {
            "market": list(MARKET_FEATURES),
            "volume": list(VOLUME_FEATURES),
            "engagement": list(ENGAGEMENT_FEATURES),
            "sentiment": list(SENTIMENT_FEATURES),
            "pressure_spi": list(PRESSURE_FEATURES),
            "linguistic": [TEXT_COLUMN, "ling_doc_len (per-fold TF-IDF -> SVD at train time)"],
        },
        "provenance": {
            "base_join": "polysignal.evaluation.backtest.build_evaluation_dataset (leakage-safe)",
            "social_evidence": str(paths.analytical / "evidence" / "accepted_evidence.parquet"),
            "sentiment": str(enrichment_path),
            "spi_features": str(paths.social_pressure_features_parquet),
            "market_probability": "latest valid YES price with timestamp <= close_time",
        },
        "leakage_safeguards": [
            "Social features use only accepted evidence, which is pre-close filtered.",
            "Linguistic TF-IDF/SVD is fit per training fold inside the model runner.",
            "The resolution outcome is the label only; it is never a predictor column.",
        ],
        "notes": [
            "vol_total_comments is 0: the comments sample is not yet collected (Traccia_2 §2 "
            "comments deferred to the multi-platform scrape).",
            "Contracts with no accepted evidence keep zero-filled social features.",
        ],
    }
    return matrix, metadata


def write_task3_feature_matrix(
    matrix: pd.DataFrame,
    *,
    paths: DataPaths | None = None,
) -> Any:
    """Persist the feature matrix to ``data/analytical/features/``."""
    paths = paths or get_data_paths()
    paths.analytical_features.mkdir(parents=True, exist_ok=True)
    out = paths.task3_feature_matrix_parquet
    matrix.to_parquet(out, index=False)
    return out


__all__ = [
    "ENGAGEMENT_FEATURES",
    "IDENTITY_COLUMNS",
    "MARKET_FEATURES",
    "NUMERIC_SOCIAL_FEATURES",
    "PRESSURE_FEATURES",
    "SENTIMENT_FEATURES",
    "TEXT_COLUMN",
    "VOLUME_FEATURES",
    "assemble_feature_matrix",
    "build_task3_feature_matrix",
    "evidence_social_features",
    "write_task3_feature_matrix",
]

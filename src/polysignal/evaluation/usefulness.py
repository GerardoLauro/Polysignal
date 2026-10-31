"""Usefulness / relevance accounting for the collected social dataset.

This answers the operator question "how many of the posts I collected are
actually useful for PolySignal?" by reading the artifacts the collection and
linking pipeline already produce and folding them into one honest report:

* **staging** — raw collected counts per source (Telegram API, Telegram web,
  Reddit web/API, enriched) so you can see what landed before promotion;
* **canonical posts** — total vs unique-after-dedup, per platform / channel /
  source, and how many carry usable text;
* **linking** — which posts were matched to a selected market and the
  relevance verdict split (accepted evidence vs weak/ambiguous/unrelated),
  with per-market coverage;
* **stance** — the stance-label distribution over the linked evidence;
* **headline** — the useful percentage, plus an explicit list of remaining
  data gaps (stages not yet run, sources that returned nothing, etc.).

Everything is read-only and defensive: a missing artifact becomes a recorded
gap, never a crash. Heavy parquet files are read as DataFrames (not rehydrated
into Pydantic models) so a ~600k-row dataset reports in seconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from polysignal.storage.parquet_store import parquet_row_count, read_dataframe
from polysignal.storage.paths import DataPaths, get_data_paths

# Staging roots written by the collectors, in promotion order. Each maps to the
# social parquet the source materializes (if any).
DEFAULT_STAGING_ROOTS: tuple[str, ...] = (
    "telegram_api",
    "telegram_web",
    "reddit",
    "enriched",
)

_ACCEPTED = "relevant"
_WEAK = "weakly_relevant"
_AMBIGUOUS = "ambiguous"
_UNRELATED = "unrelated"


def _top_counts(series: pd.Series, top_n: int) -> dict[str, int]:
    counts = series.dropna().astype(str).value_counts()
    return {str(k): int(v) for k, v in counts.head(top_n).items()}


def _staging_counts(data_root: Path, roots: tuple[str, ...]) -> dict[str, Any]:
    """Count rows in each staging source's social parquet, honestly.

    A source with no normalized parquet (e.g. Reddit when old.reddit.com
    returned 403/429) reports ``rows: 0`` with ``present: False`` rather than
    being silently omitted.
    """
    out: dict[str, Any] = {}
    for name in roots:
        parquet = data_root / "staging" / name / "normalized" / "social" / "social_posts.parquet"
        rows = parquet_row_count(parquet)
        out[name] = {
            "present": parquet.exists(),
            "rows": int(rows) if rows is not None else 0,
            "path": str(parquet),
        }
    return out


def _label_column(linked: pd.DataFrame) -> pd.Series:
    """Best available relevance label per row (LLM/final overlay over baseline)."""
    label = linked.get("relevance_label")
    if label is None:
        return pd.Series(["unknown"] * len(linked), index=linked.index)
    label = label.astype("string")
    final = linked.get("final_relevance_label")
    if final is not None:
        label = final.astype("string").fillna(label)
    return label.fillna("unknown")


def build_usefulness_report(
    *,
    paths: DataPaths | None = None,
    staging_roots: tuple[str, ...] = DEFAULT_STAGING_ROOTS,
    market_ids: list[str] | None = None,
    top_n: int = 15,
) -> dict[str, Any]:
    """Compute the usefulness report dict from on-disk artifacts."""
    paths = paths or get_data_paths()
    gaps: list[str] = []
    report: dict[str, Any] = {
        "data_root": str(paths.root),
        "staging": _staging_counts(paths.root, staging_roots),
    }

    # ----------------------------------------------------------- canonical posts
    posts = read_dataframe(paths.social_posts_parquet)
    if posts.empty:
        gaps.append(
            "canonical social_posts.parquet is empty/missing - run collection then "
            "`social promote-staging`."
        )
    collected_total = len(posts)
    if "is_duplicate" in posts.columns:
        unique_posts = int((~posts["is_duplicate"].fillna(False).astype(bool)).sum())
    else:
        unique_posts = int(posts["post_id"].nunique()) if "post_id" in posts.columns else 0
    with_text = (
        int((posts["text_length"].fillna(0) > 0).sum()) if "text_length" in posts.columns else 0
    )
    report["posts"] = {
        "collected_total": collected_total,
        "unique_after_dedup": unique_posts,
        "with_usable_text": with_text,
        "by_platform": _top_counts(posts["platform"], top_n) if "platform" in posts else {},
        "by_channel": (
            _top_counts(posts["subreddit_or_channel"], top_n)
            if "subreddit_or_channel" in posts
            else {}
        ),
    }

    # ----------------------------------------------------------------- linking
    linked = read_dataframe(paths.linked_posts_parquet)
    if linked.empty:
        gaps.append(
            "linked_posts.parquet is empty/missing - run `linking score-candidates` to "
            "match posts to markets."
        )
        report["linking"] = {"available": False}
    else:
        labels = _label_column(linked)
        post_ids = linked.get("post_id")
        matched_posts = int(post_ids.nunique()) if post_ids is not None else 0
        accepted_mask = labels == _ACCEPTED
        # "Social signal" = topically linked at >= weak relevance. This is what the
        # Social Pressure Index actually consumes, so it is the honest measure of
        # "useful for the project" — strict "accepted evidence" (relevant only) is
        # reported alongside it, never instead of it.
        signal_mask = accepted_mask | (labels == _WEAK)
        accepted_post_ids = (
            set(linked.loc[accepted_mask, "post_id"].astype(str)) if post_ids is not None else set()
        )
        signal_post_ids = (
            set(linked.loc[signal_mask, "post_id"].astype(str)) if post_ids is not None else set()
        )
        per_market_relevant: dict[str, int] = {}
        per_market_signal: dict[str, int] = {}
        if "market_id" in linked.columns:
            per_market_relevant = {
                str(k): int(v)
                for k, v in linked.loc[accepted_mask, "market_id"]
                .astype(str)
                .value_counts()
                .head(top_n)
                .items()
            }
            per_market_signal = {
                str(k): int(v)
                for k, v in linked.loc[signal_mask, "market_id"]
                .astype(str)
                .value_counts()
                .head(top_n)
                .items()
            }
        report["linking"] = {
            "available": True,
            "linked_pairs": len(linked),
            "posts_matched_to_markets": matched_posts,
            "markets_with_evidence": (
                int(linked["market_id"].nunique()) if "market_id" in linked.columns else 0
            ),
            "llm_judge_applied": (
                int(linked["llm_applied"].fillna(False).astype(bool).sum())
                if "llm_applied" in linked.columns
                else 0
            ),
            "verdict_split": {
                "accepted_evidence_pairs": int(accepted_mask.sum()),
                "weakly_relevant_pairs": int((labels == _WEAK).sum()),
                "ambiguous_pairs": int((labels == _AMBIGUOUS).sum()),
                "unrelated_pairs": int((labels == _UNRELATED).sum()),
            },
            "accepted_evidence_unique_posts": len(accepted_post_ids),
            "social_signal_unique_posts": len(signal_post_ids),
            "per_market_accepted_evidence": per_market_relevant,
            "per_market_social_signal": per_market_signal,
        }

    # ------------------------------------------------------------------ stance
    stance = read_dataframe(paths.stance_labels_parquet)
    if stance.empty:
        gaps.append("stance_labels.parquet is empty/missing - run `stance` scoring.")
        report["stance"] = {"available": False}
    else:
        stance_col = "stance_label" if "stance_label" in stance.columns else "label"
        report["stance"] = {
            "available": True,
            "scored_rows": len(stance),
            "distribution": (
                _top_counts(stance[stance_col], 20) if stance_col in stance.columns else {}
            ),
        }

    # ----------------------------------------------------------- gaps & headline
    reddit_staged = report["staging"].get("reddit", {}).get("rows", 0)
    if reddit_staged == 0:
        gaps.append(
            "Reddit collected 0 rows - old.reddit.com blocks automated fetches (HTTP "
            "403/429). Use the official Reddit OAuth API collector "
            "(`social ingest --platform reddit --output-dir data/staging/reddit`) "
            "once REDDIT_CLIENT_ID/SECRET are set."
        )
    accepted_unique = report.get("linking", {}).get("accepted_evidence_unique_posts", 0)
    signal_unique = report.get("linking", {}).get("social_signal_unique_posts", 0)
    denom = unique_posts or collected_total or 1
    report["headline"] = {
        "collected_total": collected_total,
        "unique_after_dedup": unique_posts,
        # Strict: decisive evidence (relevance label == relevant).
        "accepted_evidence_unique_posts": accepted_unique,
        "accepted_evidence_percentage": round(100.0 * accepted_unique / denom, 3),
        # Broad: topically-linked social signal (>= weakly relevant) — the SPI input.
        "social_signal_unique_posts": signal_unique,
        "social_signal_percentage": round(100.0 * signal_unique / denom, 3),
        # Back-compat alias kept pointing at the strict measure.
        "useful_percentage": round(100.0 * accepted_unique / denom, 3),
        "useful_percentage_basis": "accepted_evidence (relevant) and social_signal (>=weak) / unique",
    }
    report["data_gaps"] = gaps
    return report

"""The usefulness report's funnel + dual (evidence vs social-signal) metric."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from polysignal.evaluation.usefulness import build_usefulness_report
from polysignal.storage.parquet_store import write_dataframe
from polysignal.storage.paths import get_data_paths


def _seed(tmp_path: Path):
    paths = get_data_paths(root=tmp_path)
    paths.normalized_social.mkdir(parents=True, exist_ok=True)

    posts = pd.DataFrame(
        {
            "post_id": [f"telegram:c:{i}" for i in range(10)],
            "platform": ["telegram"] * 10,
            "subreddit_or_channel": ["cointelegraph"] * 10,
            "is_duplicate": [False] * 9 + [True],
            "text_length": [50] * 10,
        }
    )
    write_dataframe(posts, paths.social_posts_parquet)

    # 3 relevant, 2 weakly_relevant, 1 ambiguous, rest unrelated — across post ids.
    linked = pd.DataFrame(
        {
            "post_id": [f"telegram:c:{i}" for i in range(6)],
            "market_id": ["m1"] * 6,
            "relevance_label": [
                "relevant",
                "relevant",
                "relevant",
                "weakly_relevant",
                "weakly_relevant",
                "ambiguous",
            ],
            "final_relevance_label": [None] * 6,
            "llm_applied": [True, True, False, True, False, True],
        }
    )
    write_dataframe(linked, paths.linked_posts_parquet)
    return paths


def test_report_reports_both_evidence_and_social_signal(tmp_path: Path) -> None:
    paths = _seed(tmp_path)
    rep = build_usefulness_report(paths=paths)

    assert rep["posts"]["collected_total"] == 10
    assert rep["posts"]["unique_after_dedup"] == 9

    link = rep["linking"]
    assert link["available"] is True
    assert link["accepted_evidence_unique_posts"] == 3
    # social signal = relevant + weakly_relevant unique posts = 3 + 2.
    assert link["social_signal_unique_posts"] == 5
    assert link["llm_judge_applied"] == 4

    head = rep["headline"]
    # Strict evidence 3/9, broad social signal 5/9.
    assert head["accepted_evidence_percentage"] == round(100 * 3 / 9, 3)
    assert head["social_signal_percentage"] == round(100 * 5 / 9, 3)
    # Reddit-block gap surfaces (no reddit staging present).
    assert any("Reddit" in g for g in rep["data_gaps"])


def test_report_handles_missing_linking(tmp_path: Path) -> None:
    paths = get_data_paths(root=tmp_path)
    paths.normalized_social.mkdir(parents=True, exist_ok=True)
    write_dataframe(
        pd.DataFrame(
            {
                "post_id": ["telegram:c:1"],
                "platform": ["telegram"],
                "is_duplicate": [False],
                "text_length": [10],
            }
        ),
        paths.social_posts_parquet,
    )
    rep = build_usefulness_report(paths=paths)
    assert rep["linking"]["available"] is False
    assert rep["headline"]["social_signal_percentage"] == 0
    assert any("linked_posts" in g for g in rep["data_gaps"])

"""Tests for the PySpark/pandas per-market social aggregation job (Task E3).

The pandas reference path is always exercised. The Spark path is compared to
it only when a JVM-backed PySpark is available (``JAVA_HOME`` set); otherwise
that test is skipped, and the fallback-equivalence is asserted instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from polysignal.processing.social_aggregate import (
    _aggregate_pandas,
    _aggregate_spark,
    deduplicate_and_aggregate_posts,
    is_spark_available,
    run_dedup_aggregate_job,
)


def _posts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # m1: two distinct posts, one duplicated row
            {"market_id_candidate": "m1", "platform": "reddit", "post_id": "p1",
             "author_id": "a1", "engagement_score": 10.0, "is_duplicate": False},
            {"market_id_candidate": "m1", "platform": "reddit", "post_id": "p1",
             "author_id": "a1", "engagement_score": 10.0, "is_duplicate": True},
            {"market_id_candidate": "m1", "platform": "telegram", "post_id": "p2",
             "author_id": "a2", "engagement_score": 30.0, "is_duplicate": False},
            # m2: one post
            {"market_id_candidate": "m2", "platform": "reddit", "post_id": "p3",
             "author_id": "a1", "engagement_score": 5.0, "is_duplicate": False},
        ]
    )


def test_pandas_aggregate_dedups_and_summarizes():
    out = deduplicate_and_aggregate_posts(_posts(), use_spark=False)
    by_market = out.set_index("market_id")
    # p1 duplicate row dropped -> m1 has 2 posts (p1, p2)
    assert by_market.loc["m1", "post_count"] == 2
    assert by_market.loc["m1", "unique_authors"] == 2
    assert by_market.loc["m1", "platform_count"] == 2
    assert by_market.loc["m1", "total_engagement"] == 40.0
    assert by_market.loc["m1", "max_engagement"] == 30.0
    assert by_market.loc["m2", "post_count"] == 1


def test_resolves_market_via_links_when_candidate_missing():
    posts = _posts().copy()
    posts["market_id_candidate"] = None  # real-corpus condition
    links = pd.DataFrame(
        [
            {"post_id": "p1", "market_id": "mX"},
            {"post_id": "p2", "market_id": "mX"},
            {"post_id": "p3", "market_id": "mY"},
        ]
    )
    out = deduplicate_and_aggregate_posts(posts, links=links, use_spark=False)
    by_market = out.set_index("market_id")
    assert by_market.loc["mX", "post_count"] == 2
    assert by_market.loc["mY", "post_count"] == 1


def test_empty_posts_safe():
    out = deduplicate_and_aggregate_posts(pd.DataFrame(columns=["platform", "post_id"]), use_spark=False)
    assert out.empty


def test_run_job_writes_parquet(tmp_path: Path):
    posts_path = tmp_path / "posts.parquet"
    _posts().to_parquet(posts_path, index=False)
    out_path = tmp_path / "agg.parquet"
    info = run_dedup_aggregate_job(posts_path=posts_path, output_path=out_path, use_spark=False)
    assert info["engine"] == "pandas"
    assert out_path.exists()
    assert info["markets"] == 2


@pytest.mark.skipif(not is_spark_available(), reason="JVM/PySpark not available")
def test_spark_matches_pandas_when_available():
    posts = _posts()
    pdf = _aggregate_pandas(posts)
    sdf = _aggregate_spark(posts)
    merged = pdf.merge(sdf, on="market_id", suffixes=("_p", "_s"))
    assert len(merged) == len(pdf)
    assert (merged["post_count_p"] == merged["post_count_s"]).all()
    assert (merged["unique_authors_p"] == merged["unique_authors_s"]).all()
    assert np.allclose(
        merged["total_engagement_p"].astype(float),
        merged["total_engagement_s"].astype(float),
        rtol=1e-6,
    )

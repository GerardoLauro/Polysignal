"""Per-market social dedup + aggregation (PySpark with a pandas fallback).

This is a Big-Data "summarize + join" job over the full social corpus. The
pandas path is always available; the Spark path activates when ``use_spark``
is requested *and* a working PySpark/JVM is importable, partitioning the work
across cores. Both paths compute the identical aggregate schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# Columns the job reads (others are ignored if present).
_AGG_COLUMNS = ["market_id_candidate", "platform", "post_id", "author_id", "engagement_score"]


@dataclass
class PerMarketAggregate:
    """One per-market social aggregate row (Spark/pandas identical schema)."""

    market_id: str
    post_count: int
    unique_authors: int
    platform_count: int
    total_engagement: float
    mean_engagement: float
    max_engagement: float


_AGG_SCHEMA = [
    "market_id",
    "post_count",
    "unique_authors",
    "platform_count",
    "total_engagement",
    "mean_engagement",
    "max_engagement",
]


def is_spark_available() -> bool:
    """True when PySpark imports and a JVM can be reached (JAVA_HOME set)."""
    try:
        import pyspark  # noqa: F401
    except Exception:
        return False
    import os

    return bool(os.environ.get("JAVA_HOME") or os.environ.get("SPARK_HOME"))


def _dedup_key_frame(posts: pd.DataFrame, links: pd.DataFrame | None = None) -> pd.DataFrame:
    """Deduplicate to one row per (platform, post_id) and resolve market_id.

    Prefers an explicit ``is_duplicate == False`` row; otherwise keeps the
    first occurrence. ``market_id`` is taken from ``market_id_candidate`` when
    present, else resolved by joining ``links`` (post_id -> best market_id).
    Returns only the columns the aggregation needs.
    """
    work = posts.copy()
    for col in _AGG_COLUMNS:
        if col not in work.columns:
            work[col] = None
    if "is_duplicate" in work.columns:
        work = work.sort_values("is_duplicate")  # False (0) before True (1)
    deduped = work.drop_duplicates(subset=["platform", "post_id"], keep="first")
    out = deduped[_AGG_COLUMNS].rename(columns={"market_id_candidate": "market_id"})
    # If the candidate market is empty (real corpus) but links are supplied,
    # resolve the linked market via a post_id join (the scale-out join pattern).
    if links is not None and not links.empty and out["market_id"].notna().sum() == 0:
        best = (
            links[["post_id", "market_id"]]
            .dropna(subset=["post_id", "market_id"])
            .drop_duplicates(subset=["post_id"], keep="first")
        )
        out = out.drop(columns=["market_id"]).merge(best, on="post_id", how="left")
    return out


def _aggregate_pandas(posts: pd.DataFrame, links: pd.DataFrame | None = None) -> pd.DataFrame:
    deduped = _dedup_key_frame(posts, links)
    deduped = deduped[deduped["market_id"].notna()]
    if deduped.empty:
        return pd.DataFrame(columns=_AGG_SCHEMA)
    deduped["engagement_score"] = pd.to_numeric(
        deduped["engagement_score"], errors="coerce"
    ).fillna(0.0)
    grouped = deduped.groupby("market_id")
    out = pd.DataFrame(
        {
            "post_count": grouped["post_id"].size(),
            "unique_authors": grouped["author_id"].nunique(),
            "platform_count": grouped["platform"].nunique(),
            "total_engagement": grouped["engagement_score"].sum(),
            "mean_engagement": grouped["engagement_score"].mean(),
            "max_engagement": grouped["engagement_score"].max(),
        }
    ).reset_index()
    return out.sort_values("market_id").reset_index(drop=True)[_AGG_SCHEMA]


def _aggregate_spark(posts: pd.DataFrame, links: pd.DataFrame | None = None) -> pd.DataFrame:
    """Spark implementation of the same aggregate spec.

    Falls back to pandas if a SparkSession cannot be created (e.g. no JVM at
    runtime) so callers never crash on environment gaps.
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F  # noqa: N812 - canonical Spark alias
    except Exception:
        return _aggregate_pandas(posts, links)

    deduped = _dedup_key_frame(posts, links)
    deduped = deduped[deduped["market_id"].notna()].copy()
    if deduped.empty:
        return pd.DataFrame(columns=_AGG_SCHEMA)
    deduped["engagement_score"] = pd.to_numeric(
        deduped["engagement_score"], errors="coerce"
    ).fillna(0.0)

    spark: Any = None
    try:
        spark = (
            SparkSession.builder.master("local[*]")
            .appName("polysignal-social-aggregate")
            .config("spark.sql.shuffle.partitions", "16")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        sdf = spark.createDataFrame(deduped[_AGG_COLUMNS_RENAMED()])
        agg = (
            sdf.groupBy("market_id")
            .agg(
                F.count("post_id").alias("post_count"),
                F.countDistinct("author_id").alias("unique_authors"),
                F.countDistinct("platform").alias("platform_count"),
                F.sum("engagement_score").alias("total_engagement"),
                F.mean("engagement_score").alias("mean_engagement"),
                F.max("engagement_score").alias("max_engagement"),
            )
            .orderBy("market_id")
        )
        result = agg.toPandas()
    except Exception:
        return _aggregate_pandas(posts, links)
    finally:
        if spark is not None:
            spark.stop()
    # Normalize dtypes to match the pandas path.
    for col in ("post_count", "unique_authors", "platform_count"):
        result[col] = result[col].astype(int)
    return result.sort_values("market_id").reset_index(drop=True)[_AGG_SCHEMA]


def _AGG_COLUMNS_RENAMED() -> list[str]:  # noqa: N802 - internal helper
    # After _dedup_key_frame, market_id_candidate is already renamed to market_id.
    return ["market_id", "platform", "post_id", "author_id", "engagement_score"]


def deduplicate_and_aggregate_posts(
    posts: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    use_spark: bool = True,
) -> pd.DataFrame:
    """Dedup by (platform, post_id) and compute per-market social aggregates.

    Uses Spark when ``use_spark`` and a JVM-backed PySpark is available;
    otherwise the pandas reference path. ``links`` (post_id -> market_id)
    resolves the market when posts carry no ``market_id_candidate`` (the real
    Telegram corpus). Output schema is identical either way.
    """
    if use_spark and is_spark_available():
        return _aggregate_spark(posts, links)
    return _aggregate_pandas(posts, links)


def run_dedup_aggregate_job(
    *,
    posts_path: Path,
    output_path: Path,
    links_path: Path | None = None,
    use_spark: bool = True,
) -> dict[str, Any]:
    """Read posts parquet, run the job, write per-market aggregates parquet."""
    posts = pd.read_parquet(posts_path)
    links = (
        pd.read_parquet(links_path, columns=["post_id", "market_id"])
        if links_path is not None and Path(links_path).exists()
        else None
    )
    engine = "spark" if (use_spark and is_spark_available()) else "pandas"
    result = deduplicate_and_aggregate_posts(posts, links=links, use_spark=use_spark)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    return {
        "engine": engine,
        "input_posts": len(posts),
        "markets": len(result),
        "output_path": str(output_path),
    }


__all__ = [
    "PerMarketAggregate",
    "deduplicate_and_aggregate_posts",
    "is_spark_available",
    "run_dedup_aggregate_job",
]

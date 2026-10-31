"""Big-data processing jobs for PolySignal.

The headline social corpus is ~1.35M posts. These jobs express the heavy
"summarize + dedup + join" work in a way that scales out on Spark when a JVM
is available, with a pandas fallback so the package stays runnable single-node
without extra dependencies.

:func:`deduplicate_and_aggregate_posts` is the reference job: it deduplicates
posts by ``(platform, post_id)`` and produces per-market aggregates
(post counts, unique authors, engagement summaries, platform split). The Spark
and pandas paths share one specification and are asserted equivalent in tests.
"""

from __future__ import annotations

from polysignal.processing.social_aggregate import (
    PerMarketAggregate,
    deduplicate_and_aggregate_posts,
    is_spark_available,
    run_dedup_aggregate_job,
)

__all__ = [
    "PerMarketAggregate",
    "deduplicate_and_aggregate_posts",
    "is_spark_available",
    "run_dedup_aggregate_job",
]

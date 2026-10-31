"""Task-2 analytics — descriptive, discourse, correlation, cross-platform.

Pure-transform analysis modules plus a :mod:`runner` orchestrator that persists
artifacts under ``data/analytical/analytics/``. Wired to the CLI ``analytics``
command group.
"""

from polysignal.analytics import (
    correlation,
    cross_platform,
    descriptive,
    discourse,
    temporal,
)
from polysignal.analytics.runner import AnalyticsResult, run_all

__all__ = [
    "AnalyticsResult",
    "correlation",
    "cross_platform",
    "descriptive",
    "discourse",
    "run_all",
    "temporal",
]

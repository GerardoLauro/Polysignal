"""High-signal Polymarket + Telegram dataset preparation.

This subpackage builds a small, focused dataset on top of the existing
ingestion pipelines:

* :func:`select_high_signal_markets` selects ~50 Polymarket markets that
  are likely to be widely discussed on Telegram / social media.
* :func:`build_telegram_search_plan` turns those markets into a concrete
  plan of (channel, query, search-window) entries for Telegram ingestion.
* :func:`build_coverage_report` summarises how well the resulting Telegram
  corpus covers the selected markets.

The modules are intentionally pure functions over Pydantic models so the
selection / plan / coverage logic stays unit-testable without any I/O.
"""

from polysignal.dataset.coverage import build_coverage_report, write_coverage_report
from polysignal.dataset.export import ExportResult, export_dataset
from polysignal.dataset.manifest import build_manifest, write_manifest
from polysignal.dataset.models import (
    DATASET_TARGETS,
    DatasetTargets,
    HighSignalDomain,
    HighSignalMarket,
    TelegramCoverageEntry,
    TelegramCoverageReport,
    TelegramSearchPlanEntry,
)
from polysignal.dataset.scoring import (
    market_priority_score,
    social_searchability_score,
    telegram_searchability_score,
)
from polysignal.dataset.selection import (
    HighSignalSelectionResult,
    classify_high_signal_domain,
    rank_markets,
    select_high_signal_markets,
    write_high_signal_markets,
)
from polysignal.dataset.telegram_plan import (
    build_telegram_search_plan,
    write_telegram_search_plan,
)

__all__ = [
    "DATASET_TARGETS",
    "DatasetTargets",
    "ExportResult",
    "HighSignalDomain",
    "HighSignalMarket",
    "HighSignalSelectionResult",
    "TelegramCoverageEntry",
    "TelegramCoverageReport",
    "TelegramSearchPlanEntry",
    "build_coverage_report",
    "build_manifest",
    "build_telegram_search_plan",
    "classify_high_signal_domain",
    "export_dataset",
    "market_priority_score",
    "rank_markets",
    "select_high_signal_markets",
    "social_searchability_score",
    "telegram_searchability_score",
    "write_coverage_report",
    "write_high_signal_markets",
    "write_manifest",
    "write_telegram_search_plan",
]

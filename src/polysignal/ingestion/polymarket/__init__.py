"""Polymarket ingestion subpackage.

Public surface for callers (CLI, dashboard, tests). The scope of this
milestone is **binary YES/NO markets only** — see the module-level
docstrings in :mod:`binary_filter` and :mod:`normalizer`, and the
"Polymarket ingestion" section of ``README.md``.
"""

from polysignal.ingestion.polymarket.binary_filter import (
    BinaryFilterResult,
    BinaryMarket,
    classify_market,
    split_binary,
)
from polysignal.ingestion.polymarket.client import PolymarketAPIError, PolymarketClient
from polysignal.ingestion.polymarket.normalizer import (
    build_outcomes,
    normalize_market,
    normalize_price_history,
)
from polysignal.ingestion.polymarket.pipeline import (
    CuratedMarketReport,
    IngestionResult,
    PriceFetchOutcome,
    SkippedMarket,
    fetch_prices_for_market,
    ingest_curated_markets,
    run_market_ingestion,
)
from polysignal.ingestion.polymarket.raw_store import RawRunWriter, generate_run_id

__all__ = [
    "BinaryFilterResult",
    "BinaryMarket",
    "CuratedMarketReport",
    "IngestionResult",
    "PolymarketAPIError",
    "PolymarketClient",
    "PriceFetchOutcome",
    "RawRunWriter",
    "SkippedMarket",
    "build_outcomes",
    "classify_market",
    "fetch_prices_for_market",
    "generate_run_id",
    "ingest_curated_markets",
    "normalize_market",
    "normalize_price_history",
    "run_market_ingestion",
    "split_binary",
]

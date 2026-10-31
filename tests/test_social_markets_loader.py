"""Offline tests for :func:`load_markets_from_parquet`.

These tests pin the regression behind Critical Fix 1: the social
pipeline must read both writers' shapes of ``markets.parquet``
(Branch 03's ISO-string-and-None form *and* a future writer's
native-datetime64 + NaT/NaN form) without raising Pydantic
``ValidationError``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from polysignal.domain import Market, MarketDomain, Outcome
from polysignal.ingestion.social.pipeline import (
    _clean_market_record,
    _coerce_market_value,
    load_markets_from_parquet,
)
from polysignal.storage.parquet_store import write_parquet


def _polymarket_market() -> Market:
    return Market(
        market_id="cond-1",
        slug="example",
        question="Will Trump be out as President before GTA VI?",
        description="Resolution language.",
        domain=MarketDomain.POLITICS,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        outcomes=[
            Outcome(outcome_id="yes", label="Yes", index=0),
            Outcome(outcome_id="no", label="No", index=1),
        ],
    )


# ---------------------------------------------------------------------------
# Coercion helper unit tests
# ---------------------------------------------------------------------------


def test_coerce_value_normalizes_nat_to_none():
    assert _coerce_market_value(pd.NaT) is None


def test_coerce_value_normalizes_nan_to_none():
    assert _coerce_market_value(float("nan")) is None


def test_coerce_value_converts_pandas_timestamp_to_utc_datetime():
    ts = pd.Timestamp("2026-05-31T12:00:00Z")
    out = _coerce_market_value(ts)
    assert isinstance(out, datetime)
    assert out.tzinfo is not None
    assert out.year == 2026


def test_coerce_value_converts_numpy_array_to_python_list():
    arr = np.array([{"a": 1}, {"a": 2}], dtype=object)
    out = _coerce_market_value(arr)
    assert isinstance(out, list)
    assert out == [{"a": 1}, {"a": 2}]


def test_coerce_value_converts_numpy_scalars_to_python_scalars():
    assert _coerce_market_value(np.int64(4)) == 4
    assert _coerce_market_value(np.bool_(True)) is True
    assert _coerce_market_value(np.float64(1.5)) == 1.5
    assert _coerce_market_value(np.float64("nan")) is None


def test_coerce_value_passes_through_plain_dict_and_str():
    assert _coerce_market_value({"a": 1}) == {"a": 1}
    assert _coerce_market_value("hello") == "hello"
    assert _coerce_market_value(None) is None


def test_clean_market_record_walks_nested_outcome_dicts():
    """Inner outcome dicts must also have NaN cleaned out."""
    row = {
        "market_id": "m1",
        "slug": "s1",
        "question": "Q?",
        "description": None,
        "domain": "politics",
        "created_at": pd.Timestamp("2025-01-01T00:00:00Z"),
        "close_time": pd.NaT,
        "resolved_at": pd.NaT,
        "is_resolved": False,
        "is_closed": False,
        "resolution": None,
        "outcomes": [
            {
                "outcome_id": "yes",
                "label": "Yes",
                "index": 0,
                "current_price": float("nan"),
                "market_id": "m1",
            },
            {
                "outcome_id": "no",
                "label": "No",
                "index": 1,
                "current_price": 0.5,
                "market_id": "m1",
            },
        ],
        "tags": [],
        "source_url": None,
        "event_id": None,
        "condition_id": None,
        "volume": float("nan"),
        "liquidity": float("nan"),
    }
    cleaned = _clean_market_record(row)
    assert cleaned["close_time"] is None
    assert cleaned["volume"] is None
    assert cleaned["outcomes"][0]["current_price"] is None
    assert cleaned["outcomes"][1]["current_price"] == 0.5
    # Pydantic must accept the cleaned record.
    Market.model_validate(cleaned)


def test_clean_market_record_handles_json_and_tuple_list_fields():
    cleaned = _clean_market_record(
        {
            "outcomes": (
                '{"outcome_id": "yes", "label": "Yes", "index": 0}',
                '{"outcome_id": "no", "label": "No", "index": 1}',
            ),
            "tags": '["crypto", "bitcoin"]',
        }
    )
    assert cleaned["tags"] == ["crypto", "bitcoin"]
    assert isinstance(cleaned["outcomes"], list)
    assert cleaned["outcomes"][0]["outcome_id"] == "yes"


def test_clean_market_record_handles_json_encoded_outcomes():
    cleaned = _clean_market_record(
        {
            "outcomes": (
                '[{"outcome_id": "yes", "label": "Yes", "index": 0}, '
                '{"outcome_id": "no", "label": "No", "index": 1}]'
            ),
            "tags": "crypto",
        }
    )
    assert cleaned["outcomes"][0]["outcome_id"] == "yes"
    assert cleaned["tags"] == ["crypto"]


# ---------------------------------------------------------------------------
# End-to-end loader behaviour
# ---------------------------------------------------------------------------


def test_load_markets_handles_nat_and_nan_in_native_datetime_columns(tmp_path: Path) -> None:
    """The native-datetime64 + NaT/NaN shape must NOT raise ValidationError."""
    df = pd.DataFrame(
        [
            {
                "market_id": "cond-1",
                "slug": "slug-1",
                "question": "Q?",
                "description": "Body.",
                "domain": "politics",
                "created_at": pd.Timestamp("2025-01-01", tz="UTC"),
                "close_time": pd.NaT,
                "resolved_at": pd.NaT,
                "is_resolved": False,
                "is_closed": False,
                "resolution": None,
                "outcomes": [
                    {"outcome_id": "yes", "label": "Yes", "index": 0},
                    {"outcome_id": "no", "label": "No", "index": 1},
                ],
                "tags": [],
                "source_url": None,
                "event_id": None,
                "condition_id": None,
                "volume": float("nan"),
                "liquidity": float("nan"),
            },
        ]
    )
    target = tmp_path / "markets.parquet"
    df.to_parquet(target, index=False)
    out = load_markets_from_parquet(target)
    assert len(out) == 1
    assert out[0].close_time is None
    assert out[0].resolved_at is None
    assert out[0].volume is None
    assert out[0].liquidity is None


def test_load_markets_handles_branch03_iso_string_shape(tmp_path: Path) -> None:
    """The Branch-03 ISO-string shape continues to load cleanly."""
    market = _polymarket_market()
    target = tmp_path / "markets.parquet"
    write_parquet([market], target)
    out = load_markets_from_parquet(target)
    assert len(out) == 1
    assert out[0].market_id == market.market_id
    assert out[0].question == market.question


def test_load_markets_respects_limit_argument(tmp_path: Path) -> None:
    target = tmp_path / "markets.parquet"
    # Two markets — limit=1 must keep only the first.
    markets = [
        _polymarket_market(),
        _polymarket_market().model_copy(update={"market_id": "cond-2", "slug": "example-2"}),
    ]
    write_parquet(markets, target)
    out = load_markets_from_parquet(target, limit=1)
    assert len(out) == 1
    assert out[0].market_id == "cond-1"


def test_load_markets_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_markets_from_parquet(tmp_path / "nope.parquet")


def test_load_markets_skips_invalid_row_without_failing_the_run(tmp_path: Path) -> None:
    """One bad row must not poison the whole load."""
    df = pd.DataFrame(
        [
            {
                "market_id": "ok-1",
                "slug": "ok-1",
                "question": "Q?",
                "description": None,
                "domain": "politics",
                "created_at": pd.Timestamp("2025-01-01", tz="UTC"),
                "close_time": None,
                "resolved_at": None,
                "is_resolved": False,
                "is_closed": False,
                "resolution": None,
                "outcomes": [
                    {"outcome_id": "yes", "label": "Yes", "index": 0},
                    {"outcome_id": "no", "label": "No", "index": 1},
                ],
                "tags": [],
                "source_url": None,
                "event_id": None,
                "condition_id": None,
                "volume": None,
                "liquidity": None,
            },
            {
                # Pathologically broken — only one outcome (violates min_length=2)
                "market_id": "broken",
                "slug": "broken",
                "question": "Q?",
                "description": None,
                "domain": "politics",
                "created_at": pd.Timestamp("2025-01-01", tz="UTC"),
                "close_time": None,
                "resolved_at": None,
                "is_resolved": False,
                "is_closed": False,
                "resolution": None,
                "outcomes": [{"outcome_id": "only", "label": "Only", "index": 0}],
                "tags": [],
                "source_url": None,
                "event_id": None,
                "condition_id": None,
                "volume": None,
                "liquidity": None,
            },
        ]
    )
    target = tmp_path / "markets.parquet"
    df.to_parquet(target, index=False)
    out = load_markets_from_parquet(target)
    assert [m.market_id for m in out] == ["ok-1"]

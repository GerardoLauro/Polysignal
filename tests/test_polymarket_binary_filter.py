"""Tests for ``polysignal.ingestion.polymarket.binary_filter``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polysignal.ingestion.polymarket.binary_filter import (
    classify_market,
    split_binary,
)

FIXTURES = Path(__file__).parent / "fixtures" / "polymarket"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# ----------------------------------------------------------- accept variants


@pytest.mark.parametrize(
    "yes,no",
    [
        ("Yes", "No"),
        ("YES", "NO"),
        ("yes", "no"),
        ("Y", "N"),
    ],
)
def test_classify_accepts_yes_no_variants(yes: str, no: str) -> None:
    payload = {"outcomes": json.dumps([yes, no]), "outcomePrices": '["0.5", "0.5"]'}
    result = classify_market(payload)
    assert result.is_binary is True
    assert result.market is not None
    assert result.market.yes_label == yes
    assert result.market.no_label == no


def test_classify_extracts_token_ids_and_prices() -> None:
    payload = _load("binary_market.json")
    result = classify_market(payload)
    assert result.is_binary is True
    assert result.market is not None
    # The fixture lists "Yes" first.
    assert result.market.yes_token_id.startswith("712345")
    assert result.market.no_token_id.startswith("823456")
    assert result.market.yes_price == pytest.approx(0.62)
    assert result.market.no_price == pytest.approx(0.38)


def test_classify_handles_no_first_ordering() -> None:
    payload = {
        "outcomes": '["No", "Yes"]',
        "outcomePrices": '["0.31", "0.69"]',
        "clobTokenIds": '["NTK", "YTK"]',
    }
    result = classify_market(payload)
    assert result.is_binary is True
    assert result.market is not None
    # YES must always be returned as the yes_* fields regardless of
    # ordering in the upstream payload.
    assert result.market.yes_label == "Yes"
    assert result.market.no_label == "No"
    assert result.market.yes_token_id == "YTK"
    assert result.market.no_token_id == "NTK"
    assert result.market.yes_price == pytest.approx(0.69)
    assert result.market.no_price == pytest.approx(0.31)


# ----------------------------------------------------------- reject variants


def test_classify_rejects_multi_outcome() -> None:
    payload = _load("multi_outcome_market.json")
    result = classify_market(payload)
    assert result.is_binary is False
    assert "4 outcomes" in (result.reason or "")
    assert len(result.raw_outcomes) == 4


def test_classify_rejects_non_yes_no_binary() -> None:
    payload = _load("non_yes_no_binary.json")
    result = classify_market(payload)
    assert result.is_binary is False
    assert "non-YES/NO" in (result.reason or "")


def test_classify_rejects_missing_outcomes_field() -> None:
    result = classify_market({"id": "x"})
    assert result.is_binary is False
    assert "missing" in (result.reason or "")


def test_classify_rejects_malformed_outcomes_string() -> None:
    result = classify_market({"outcomes": "not-json"})
    assert result.is_binary is False
    assert "missing" in (result.reason or "")


def test_classify_accepts_native_list_outcomes() -> None:
    """Some older payloads expose the native list rather than a JSON string."""
    payload = {"outcomes": ["Yes", "No"]}
    result = classify_market(payload)
    assert result.is_binary is True


def test_classify_tolerates_missing_prices() -> None:
    payload = {"outcomes": '["Yes", "No"]'}
    result = classify_market(payload)
    assert result.is_binary is True
    assert result.market is not None
    assert result.market.yes_price is None
    assert result.market.no_price is None
    assert result.market.yes_token_id is None
    assert result.market.no_token_id is None


# ---------------------------------------------------------------- split_binary


def test_split_binary_partitions_payloads() -> None:
    payloads = [
        _load("binary_market.json"),
        _load("multi_outcome_market.json"),
        _load("non_yes_no_binary.json"),
        _load("resolved_binary_market.json"),
    ]
    kept, skipped = split_binary(payloads)
    assert len(kept) == 2
    assert len(skipped) == 2
    # The skip reasons mention "outcomes" so a CLI summary remains useful.
    assert all(reason for _, reason in skipped)


def test_split_binary_records_skip_reason_for_each_market(caplog) -> None:
    payloads = [_load("multi_outcome_market.json")]
    with caplog.at_level("INFO", logger="polysignal.ingestion.polymarket.binary_filter"):
        kept, skipped = split_binary(payloads)
    assert kept == []
    assert len(skipped) == 1
    assert "skipping market" in caplog.text

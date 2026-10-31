"""Binary YES/NO market detection for Polymarket payloads.

This milestone of PolySignal supports **only** binary YES/NO markets.
Everything downstream — stance detection, social pressure, social
adjusted probability, backtesting — is outcome-aware and currently
expects exactly two outcomes labelled ``Yes`` and ``No``. Multi-outcome
markets are explicitly out of scope and must be detected, skipped, and
logged with a clear reason.

The classifier is a pure function of the raw Gamma payload. It does
not raise: every market either passes (``is_binary=True``) or is
skipped with a structured reason. The reason string is what makes its
way into logs / the CLI summary so operators understand exactly why a
market was filtered out.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


YES_LABELS = frozenset({"yes", "y", "true", "yea", "yeah"})
NO_LABELS = frozenset({"no", "n", "false", "nay"})


@dataclass(frozen=True)
class BinaryMarket:
    """A successfully classified binary YES/NO market.

    Holds the de-stringified payload fields the normalizer needs. The
    raw payload itself is the source of truth for everything else.
    """

    yes_label: str
    no_label: str
    yes_token_id: str | None
    no_token_id: str | None
    yes_price: float | None
    no_price: float | None


@dataclass(frozen=True)
class BinaryFilterResult:
    """The outcome of running the binary filter on one market payload."""

    is_binary: bool
    reason: str | None = None  # populated when is_binary is False.
    market: BinaryMarket | None = None
    raw_outcomes: list[str] = field(default_factory=list)


def _maybe_json_list(value: Any) -> list[Any] | None:
    """Decode a Polymarket field that may be a list or a JSON-encoded list.

    The Gamma API serializes lists (outcomes, prices, token ids) as JSON
    strings; older payloads occasionally expose the native list. We
    accept either and reject anything else with ``None``.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(decoded, list):
            return decoded
    return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_market(payload: dict[str, Any]) -> BinaryFilterResult:
    """Decide whether ``payload`` describes a binary YES/NO market.

    The function never raises and never mutates ``payload``. When a
    market is rejected, ``reason`` is a short human-readable string
    suitable for logs and CLI summaries.
    """
    outcomes_raw = _maybe_json_list(payload.get("outcomes"))
    if outcomes_raw is None:
        return BinaryFilterResult(
            is_binary=False,
            reason="missing or unparsable 'outcomes' field",
        )

    outcomes = [str(o).strip() for o in outcomes_raw]
    if len(outcomes) != 2:
        return BinaryFilterResult(
            is_binary=False,
            reason=f"non-binary market: {len(outcomes)} outcomes",
            raw_outcomes=outcomes,
        )

    lowered = [o.lower() for o in outcomes]
    yes_idx: int | None = None
    no_idx: int | None = None
    for idx, label in enumerate(lowered):
        if label in YES_LABELS and yes_idx is None:
            yes_idx = idx
        elif label in NO_LABELS and no_idx is None:
            no_idx = idx

    if yes_idx is None or no_idx is None or yes_idx == no_idx:
        return BinaryFilterResult(
            is_binary=False,
            reason=f"non-YES/NO outcomes: {outcomes!r}",
            raw_outcomes=outcomes,
        )

    prices_raw = _maybe_json_list(payload.get("outcomePrices")) or []
    token_ids_raw = _maybe_json_list(payload.get("clobTokenIds")) or []

    yes_price = _safe_float(prices_raw[yes_idx]) if yes_idx < len(prices_raw) else None
    no_price = _safe_float(prices_raw[no_idx]) if no_idx < len(prices_raw) else None

    yes_token = None
    no_token = None
    if yes_idx < len(token_ids_raw):
        token = token_ids_raw[yes_idx]
        if token is not None and str(token).strip():
            yes_token = str(token).strip()
    if no_idx < len(token_ids_raw):
        token = token_ids_raw[no_idx]
        if token is not None and str(token).strip():
            no_token = str(token).strip()

    binary = BinaryMarket(
        yes_label=outcomes[yes_idx],
        no_label=outcomes[no_idx],
        yes_token_id=yes_token,
        no_token_id=no_token,
        yes_price=yes_price,
        no_price=no_price,
    )
    return BinaryFilterResult(is_binary=True, market=binary, raw_outcomes=outcomes)


def split_binary(
    payloads: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], BinaryMarket]], list[tuple[dict[str, Any], str]]]:
    """Partition ``payloads`` into (kept, skipped) lists.

    Each kept entry is the original payload paired with the parsed
    ``BinaryMarket``. Each skipped entry is the payload with the reason
    string. Skipped markets are logged at ``INFO`` level so operators
    can audit the filter behaviour without re-running the pipeline.
    """
    kept: list[tuple[dict[str, Any], BinaryMarket]] = []
    skipped: list[tuple[dict[str, Any], str]] = []
    for payload in payloads:
        result = classify_market(payload)
        if result.is_binary and result.market is not None:
            kept.append((payload, result.market))
        else:
            reason = result.reason or "unknown"
            skipped.append((payload, reason))
            market_ref = payload.get("conditionId") or payload.get("id") or payload.get("slug")
            logger.info("polymarket: skipping market %s — %s", market_ref, reason)
    return kept, skipped


__all__ = [
    "NO_LABELS",
    "YES_LABELS",
    "BinaryFilterResult",
    "BinaryMarket",
    "classify_market",
    "split_binary",
]

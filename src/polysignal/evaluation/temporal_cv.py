"""Expanding-window temporal cross-validation for Task-3 prediction.

Traccia_2 Task 3 requires a *temporal* cross-validation scheme that respects the
chronological ordering of contracts and prevents leakage: a fold may only train
on contracts that resolved strictly before the contracts it is scored on.

This module implements that as an expanding-window splitter over a caller-supplied
ordering key (the contract resolution/close date). It is a pure function over the
ordering keys — no DataFrame or scikit-learn dependency — so it unit-tests against
small fixtures and is reused by :mod:`polysignal.evaluation.task3`.

Given ``n`` rows ordered chronologically and ``k`` splits, the rows are partitioned
into ``k + 1`` contiguous blocks. Fold ``i`` (``0 <= i < k``) trains on every row in
blocks ``0..i`` and is scored on block ``i + 1``. Training windows therefore expand
monotonically and every test row is strictly later than its training rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

TemporalSplit = tuple[list[int], list[int]]


def _chronological_order(order_keys: Sequence[Any]) -> list[int]:
    """Return original indices sorted by ``order_keys`` (stable, deterministic).

    ``None`` keys are sorted last so undated contracts never contaminate an early
    training window. Ties keep their original relative order (stable sort) and are
    then broken by index for full determinism.
    """
    n = len(order_keys)
    decorated = [
        (key is None, key, index)
        for index, key in enumerate(order_keys)
    ]
    # Sort by (is_none, key, index). Mixed/None keys are guarded by the is_none flag
    # so the comparison never touches a None against a real key.
    decorated.sort(key=lambda triple: (triple[0], triple[1] if not triple[0] else 0, triple[2]))
    return [index for _, _, index in decorated]


def expanding_window_splits(
    order_keys: Sequence[Any],
    *,
    n_splits: int = 5,
    min_train_size: int = 1,
) -> list[TemporalSplit]:
    """Yield expanding-window ``(train_indices, test_indices)`` folds.

    Parameters
    ----------
    order_keys:
        One chronological sort key per row (e.g. contract close/resolution date).
        Indices in the returned folds refer to positions in this sequence.
    n_splits:
        Maximum number of folds. Reduced automatically when there are too few rows
        to form that many non-empty test blocks.
    min_train_size:
        A fold is only emitted once its training window has at least this many rows,
        so the earliest (tiny) windows are skipped rather than fit on one example.

    Returns
    -------
    A list of folds. Empty when the data cannot support even a single valid split
    (fewer than ``min_train_size + 1`` rows), so callers can fall back gracefully.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if min_train_size < 1:
        raise ValueError("min_train_size must be >= 1")

    order = _chronological_order(order_keys)
    n = len(order)
    if n < min_train_size + 1:
        return []

    # We need n_splits test blocks plus at least one leading training block.
    usable_splits = min(n_splits, n - 1)
    if usable_splits < 1:
        return []

    # Partition the n ordered positions into usable_splits + 1 contiguous blocks.
    boundaries = [round(i * n / (usable_splits + 1)) for i in range(usable_splits + 2)]

    folds: list[TemporalSplit] = []
    for i in range(usable_splits):
        train_end = boundaries[i + 1]
        test_end = boundaries[i + 2]
        train_idx = order[:train_end]
        test_idx = order[train_end:test_end]
        if len(train_idx) < min_train_size or not test_idx:
            continue
        folds.append((train_idx, test_idx))
    return folds


__all__ = ["TemporalSplit", "expanding_window_splits"]

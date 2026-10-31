"""Tests for ``polysignal.evaluation.temporal_cv``."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from polysignal.evaluation.temporal_cv import expanding_window_splits


def _dt(day: int) -> datetime:
    return datetime(2025, 1, day, tzinfo=timezone.utc)


class TestExpandingWindowSplits:
    def test_train_is_always_strictly_earlier_than_test(self):
        keys = [_dt(d) for d in range(1, 11)]  # already chronological
        folds = expanding_window_splits(keys, n_splits=4)
        assert folds, "expected at least one fold"
        for train, test in folds:
            latest_train = max(keys[i] for i in train)
            earliest_test = min(keys[i] for i in test)
            assert latest_train <= earliest_test

    def test_unsorted_input_is_ordered_internally(self):
        # Reverse-chronological input must still produce chronological folds.
        keys = [_dt(d) for d in range(10, 0, -1)]
        folds = expanding_window_splits(keys, n_splits=3)
        first_train, first_test = folds[0]
        assert max(keys[i] for i in first_train) <= min(keys[i] for i in first_test)

    def test_windows_expand_monotonically(self):
        keys = list(range(20))
        folds = expanding_window_splits(keys, n_splits=4)
        train_sizes = [len(train) for train, _ in folds]
        assert train_sizes == sorted(train_sizes)
        assert all(b > a for a, b in zip(train_sizes, train_sizes[1:]))

    def test_every_index_used_at_most_once_per_role(self):
        keys = list(range(12))
        folds = expanding_window_splits(keys, n_splits=3)
        for train, test in folds:
            assert not set(train) & set(test)

    def test_returns_empty_when_too_few_rows(self):
        assert expanding_window_splits([_dt(1)], n_splits=5, min_train_size=2) == []

    def test_none_keys_sort_last(self):
        keys = [_dt(3), None, _dt(1), _dt(2)]
        folds = expanding_window_splits(keys, n_splits=2, min_train_size=1)
        # The undated row (index 1) must never appear in the earliest training block.
        first_train = folds[0][0]
        assert 1 not in first_train

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            expanding_window_splits([1, 2, 3], n_splits=0)
        with pytest.raises(ValueError):
            expanding_window_splits([1, 2, 3], min_train_size=0)

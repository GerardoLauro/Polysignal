"""Tests for ``polysignal.evaluation.metrics``."""

from __future__ import annotations

import math

import pytest

from polysignal.evaluation.metrics import (
    accuracy,
    brier_score,
    calibration_bins,
    expected_calibration_error,
    f1_per_class,
    log_loss,
    macro_f1,
)


class TestBrierScore:
    def test_perfect_calls(self):
        assert brier_score([1.0, 0.0, 1.0, 0.0], [1, 0, 1, 0]) == 0.0

    def test_completely_wrong(self):
        assert brier_score([0.0, 1.0], [1, 0]) == 1.0

    def test_uniform_half(self):
        # Constant prediction 0.5 → Brier = 0.25
        assert brier_score([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.25

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            brier_score([0.5], [1, 0])

    def test_invalid_probability(self):
        with pytest.raises(ValueError):
            brier_score([1.5], [1])


class TestLogLoss:
    def test_perfect_predictions(self):
        # Clipped at 1-eps so it's near zero, not exactly zero.
        loss = log_loss([1.0, 0.0], [1, 0])
        assert loss == pytest.approx(0.0, abs=1e-10)

    def test_completely_wrong(self):
        loss = log_loss([0.0, 1.0], [1, 0])
        # log(eps) is large negative; final value is large positive.
        assert loss > 30.0

    def test_uniform_half(self):
        loss = log_loss([0.5, 0.5], [1, 0])
        assert loss == pytest.approx(math.log(2.0), rel=1e-6)


class TestAccuracy:
    def test_threshold_default(self):
        assert accuracy([0.6, 0.4, 0.9, 0.1], [1, 0, 1, 0]) == 1.0

    def test_threshold_custom(self):
        # With threshold 0.8, only 0.9 is predicted 1
        assert accuracy([0.6, 0.4, 0.9, 0.1], [1, 0, 1, 0], threshold=0.8) == 0.75


class TestExpectedCalibrationError:
    def test_perfectly_calibrated_constant(self):
        # If probability matches frequency exactly per bin, ECE = 0.
        probs = [0.0] * 5 + [1.0] * 5
        outs = [0] * 5 + [1] * 5
        assert expected_calibration_error(probs, outs, n_bins=2) == pytest.approx(0.0)

    def test_known_miscalibration(self):
        # All predictions 0.8 but actual frequency is 0.5 → ECE = 0.3
        probs = [0.8] * 10
        outs = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        assert expected_calibration_error(probs, outs, n_bins=5) == pytest.approx(0.3, rel=1e-6)


class TestMacroF1:
    def test_perfect_predictions_score_one(self):
        probs = [0.9, 0.1, 0.8, 0.2]
        outs = [1, 0, 1, 0]
        assert macro_f1(probs, outs) == pytest.approx(1.0)
        per_class = f1_per_class(probs, outs)
        assert per_class[0] == pytest.approx(1.0)
        assert per_class[1] == pytest.approx(1.0)

    def test_all_one_class_predicted_zero_f1_for_absent_class(self):
        # Predict positive for everything; the negative class never gets a TP.
        probs = [0.9, 0.9, 0.9, 0.9]
        outs = [1, 1, 0, 0]
        per_class = f1_per_class(probs, outs)
        # Class 1: tp=2, fp=2, fn=0 -> F1 = 2*2/(2*2+2+0) = 4/6
        assert per_class[1] == pytest.approx(4 / 6)
        # Class 0: never predicted -> F1 = 0
        assert per_class[0] == pytest.approx(0.0)
        assert macro_f1(probs, outs) == pytest.approx((4 / 6) / 2)

    def test_matches_sklearn_on_mixed_case(self):
        sklearn_metrics = pytest.importorskip("sklearn.metrics")
        probs = [0.7, 0.3, 0.6, 0.4, 0.8, 0.2, 0.55, 0.45]
        outs = [1, 0, 0, 1, 1, 0, 1, 0]
        preds = [1 if p >= 0.5 else 0 for p in probs]
        expected = sklearn_metrics.f1_score(outs, preds, average="macro", zero_division=0)
        assert macro_f1(probs, outs) == pytest.approx(expected)

    def test_threshold_is_respected(self):
        probs = [0.6, 0.6, 0.4, 0.4]
        outs = [1, 0, 1, 0]
        # At threshold 0.7 nothing is predicted positive.
        assert f1_per_class(probs, outs, threshold=0.7)[1] == pytest.approx(0.0)


class TestCalibrationBins:
    def test_returns_populated_bins_with_observed_rates(self):
        bins = calibration_bins([0.1, 0.2, 0.8, 0.9], [0, 1, 1, 1], n_bins=2)
        assert [row["count"] for row in bins] == [2, 2]
        assert bins[0]["mean_probability"] == pytest.approx(0.15)
        assert bins[0]["observed_rate"] == pytest.approx(0.5)
        assert bins[1]["mean_probability"] == pytest.approx(0.85)
        assert bins[1]["observed_rate"] == pytest.approx(1.0)

"""Audit-regression tests for the dashboard data contract (``build_ps_data``).

These lock in the numerical correctness of every figure the PolySignal dashboard shows:

* aggregation subsets / denominators (Avg divergence, Avg confidence) exclude the
  no-signal coverage layer's structural zeros — the same exclusion the backend scorecard
  already applies;
* the decision-time snapshot (``polyProb``/``signalProb``) and the per-market ``delta``
  are taken at the **same** market and the **same** ``snap`` timestamp, and
  ``delta == signalProb - polyProb``;
* probabilities stay on the [0,1] scale and pressure/confidence on [0,100] (no
  double-scaling between the payload and the ``pct``/x100 frontend helpers);
* every contract loads (unique ids, no slug-collision drops → all 300);
* the forecast scorecard scores only evidence-bearing resolved markets;
* no default/placeholder zero from the no-signal layer fabricates divergence.

The build reads the real on-disk artifacts, so the whole module is skipped when they are
absent (e.g. a fresh checkout without the data tree).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_ps_data as bps  # noqa: E402

# Decision-time band — must match build_ps_data's LO/HI exactly.
LO, HI = 0.03, 0.97

_REQUIRED = [bps.TIMELINES, bps.PRESSURE, bps.STANCE, bps.ACCEPTED, bps.MARKETS]
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in _REQUIRED),
    reason="dashboard artifacts not present in this checkout",
)


@pytest.fixture(scope="module")
def payload() -> dict:
    """Build the dashboard payload once and reuse it across the module."""
    return bps.build_ps_data(verbose=False)


@pytest.fixture(scope="module")
def markets(payload: dict) -> list[dict]:
    return payload["markets"]


def _snap(poly: list[float]) -> int:
    """Re-derive the decision-time index: last day the order book was liquid."""
    snap = len(poly) - 1
    for i in range(len(poly) - 1, -1, -1):
        if LO <= poly[i] <= HI:
            snap = i
            break
    return snap


# --------------------------------------------------------------------------- coverage
class TestMarketCoverage:
    def test_every_contract_loads_with_unique_id(self, markets: list[dict]) -> None:
        # Slug collisions used to silently drop 5 finance contracts (295 not 300).
        import pandas as pd

        n_contracts = len(pd.read_parquet(bps.PRESSURE))
        assert len(markets) == n_contracts
        ids = [m["id"] for m in markets]
        assert len(set(ids)) == len(ids), "duplicate market ids (slug collision)"

    def test_tiers_partition_all_markets(self, markets: list[dict]) -> None:
        tiers = {"demo", "sparse", "none"}
        assert all(m["coverageTier"] in tiers for m in markets)


# ------------------------------------------------------------------ aggregation subsets
class TestAggregationSubsets:
    def test_no_signal_layer_has_structural_zeros(self, markets: list[dict]) -> None:
        none = [m for m in markets if m["coverageTier"] == "none"]
        assert none, "expected a no-signal coverage layer"
        for m in none:
            assert m["delta"] == 0.0  # signal line is drawn equal to the price
            assert m["conf"] == 0
            assert m["posts"] == [] and m["platforms"] == []

    def test_avg_divergence_denominator_excludes_no_signal(self, markets: list[dict]) -> None:
        signal = [m for m in markets if m["coverageTier"] != "none"]
        # Frontend formula, corrected denominator.
        avg_div_signal = round(sum(abs(m["delta"]) for m in signal) / len(signal) * 100)
        avg_div_all = round(sum(abs(m["delta"]) for m in markets) / len(markets) * 100)
        # The structural zeros must not be averaged in — they dilute toward zero.
        assert avg_div_signal > avg_div_all
        assert len(signal) < len(markets)

    def test_avg_confidence_denominator_is_valid_confidence_only(
        self, markets: list[dict]
    ) -> None:
        conf_markets = [m for m in markets if m["conf"] > 0]
        assert conf_markets, "expected markets with a real confidence value"
        # Every market with conf>0 carries a social signal (no no-signal market qualifies).
        assert all(m["coverageTier"] != "none" for m in conf_markets)
        avg_conf_valid = round(sum(m["conf"] for m in conf_markets) / len(conf_markets))
        avg_conf_all = round(sum(m["conf"] for m in markets) / len(markets))
        assert avg_conf_valid > avg_conf_all

    def test_high_divergence_count_is_signal_only_anyway(self, markets: list[dict]) -> None:
        # |delta|>=0.1 can only be hit by signal-bearing markets (none-tier delta is 0),
        # so the count is identical over all markets vs the signal subset.
        over_all = sum(1 for m in markets if abs(m["delta"]) >= 0.1)
        over_signal = sum(
            1 for m in markets if m["coverageTier"] != "none" and abs(m["delta"]) >= 0.1
        )
        assert over_all == over_signal


# ----------------------------------------------------------- decision-time + delta calc
class TestDecisionTimeAlignment:
    def test_poly_and_signal_taken_at_same_snap(self, markets: list[dict]) -> None:
        for m in markets:
            poly = m["series"]["poly"]
            sig = m["series"]["signal"]
            snap = _snap(poly)
            assert m["polyProb"] == poly[snap], m["id"]
            assert m["signalProb"] == sig[snap], m["id"]

    def test_delta_is_signal_minus_poly(self, markets: list[dict]) -> None:
        for m in markets:
            assert m["delta"] == round(m["signalProb"] - m["polyProb"], 4), m["id"]


# --------------------------------------------------------------------- probability scale
class TestProbabilityScaling:
    def test_probabilities_on_unit_interval(self, markets: list[dict]) -> None:
        for m in markets:
            for v in m["series"]["poly"] + m["series"]["signal"]:
                assert 0.0 <= v <= 1.0, m["id"]
            assert 0.0 <= m["polyProb"] <= 1.0
            assert 0.0 <= m["signalProb"] <= 1.0
            assert -1.0 <= m["delta"] <= 1.0

    def test_pressure_and_confidence_on_0_100(self, markets: list[dict]) -> None:
        for m in markets:
            for v in m["series"]["pressure"]:
                assert 0.0 <= v <= 100.0, m["id"]
            assert 0 <= m["conf"] <= 100, m["id"]
            assert 0.0 <= m["pressureNow"] <= 100.0, m["id"]
            assert 0 <= m["evidenceQuality"] <= 100, m["id"]

    def test_no_nan_or_inf_in_numeric_fields(self, markets: list[dict]) -> None:
        for m in markets:
            for key in ("polyProb", "signalProb", "delta", "pressureNow", "netPressure"):
                v = m[key]
                assert v is None or math.isfinite(v), f"{m['id']}.{key}={v}"


# ------------------------------------------------------------------------- scorecard
class TestScorecardSubset:
    def test_scorecard_counts_only_evidence_bearing_resolved(self, payload: dict) -> None:
        markets = payload["markets"]
        sc = payload["system"]["scorecard"]
        scored = [
            m for m in markets if m.get("scorecard") and m.get("coverageTier") != "none"
        ]
        assert sc["n"] == len(scored)
        # No no-signal market may contribute to the scorecard totals.
        assert all(m["coverageTier"] != "none" for m in scored)
        assert sc["marketCorrect"] == sum(m["scorecard"]["marketCorrect"] for m in scored)
        assert sc["signalCorrect"] == sum(m["scorecard"]["signalCorrect"] for m in scored)

    def test_brier_aggregates_match_per_market(self, payload: dict) -> None:
        markets = payload["markets"]
        sc = payload["system"]["scorecard"]
        scored = [
            m for m in markets if m.get("scorecard") and m.get("coverageTier") != "none"
        ]
        m_brier = sum(m["scorecard"]["marketBrier"] for m in scored) / len(scored)
        s_brier = sum(m["scorecard"]["signalBrier"] for m in scored) / len(scored)
        assert sc["marketBrier"] == round(m_brier, 3)
        assert sc["signalBrier"] == round(s_brier, 3)


# ----------------------------------------------------------------- stale / default data
class TestNoFabricatedDivergence:
    def test_no_signal_markets_track_price_exactly(self, markets: list[dict]) -> None:
        # The no-signal/coverage layer must never fabricate divergence: its social line is
        # the market price, so signal == poly pointwise and the scored delta is 0.
        for m in markets:
            if m["coverageTier"] != "none":
                continue
            assert m["series"]["signal"] == m["series"]["poly"], m["id"]
            assert m["signalProb"] == m["polyProb"]

    def test_system_market_count_is_consistent(self, payload: dict) -> None:
        assert payload["system"]["marketsLoaded"] == len(payload["markets"])


# ----------------------------------------------------- predictive vs descriptive split
class TestPredictiveVsDescriptiveSeparation:
    """The dashboard must keep three quantities distinguishable: the raw social
    pressure, the in-sample market-anchored *descriptive* estimate (signalProb /
    scorecard), and the leakage-free *out-of-fold predictive* benchmark. These lock
    that the in-sample scorecard is flagged descriptive and the predictive headline is
    a separate, out-of-fold artifact — so neither can be silently read as the other.
    """

    def test_in_sample_scorecard_is_flagged_descriptive(self, payload: dict) -> None:
        sc = payload["system"]["scorecard"]
        assert sc["kind"] == "descriptive_in_sample"
        # Every per-market scorecard carries the same honest flag.
        for m in payload["markets"]:
            if m.get("scorecard"):
                assert m["scorecard"]["kind"] == "descriptive_in_sample", m["id"]

    def test_predictive_benchmark_is_out_of_fold_and_separate(self, payload: dict) -> None:
        pb = payload["system"].get("predictiveBenchmark")
        assert pb is not None, "predictive benchmark must be surfaced for the headline claim"
        if not pb.get("available"):
            pytest.skip("Task-3 backtest artifact absent in this checkout")
        # It is the temporal, out-of-fold backtest — not the in-sample blend.
        assert pb["scheme"] == "expanding_window_temporal"
        assert pb["nFolds"] and pb["nFolds"] >= 2
        # The three ablations are reported so social-vs-market lift is explicit.
        assert set(pb["ablations"]) <= {"market_only", "social_only", "market_social"}
        assert "market_only" in pb["ablations"]
        for row in pb["ablations"].values():
            assert 0.0 <= row["brier"] <= 1.0
            assert 0.0 <= row["auc"] <= 1.0

    def test_headline_predictor_is_market_not_social(self, payload: dict) -> None:
        # The honest finding: the order book predicts; social adds no out-of-fold lift.
        # This guards against a regression that re-points the headline to social.
        pb = payload["system"].get("predictiveBenchmark") or {}
        if not pb.get("available"):
            pytest.skip("Task-3 backtest artifact absent in this checkout")
        ab = pb["ablations"]
        if "social_only" in ab and "market_only" in ab:
            assert ab["market_only"]["brier"] <= ab["social_only"]["brier"]


# ----------------------------------------------------- market-anchored bounded nudge
class TestSocialBlendCap:
    """``signalProb`` is a market-anchored estimate: social may only *nudge* the order
    book price, never override it. The blend weight is clamped to ``SOCIAL_BLEND_CAP``,
    so the displayed probability can never deviate from the market by more than that
    fraction of the social-vs-market gap. The raw social pressure (``pressureNow``) is
    intentionally NOT capped — it is a distinct quantity and must keep its true strength.
    """

    def test_blend_cap_is_a_conservative_nudge(self) -> None:
        assert 0.0 < bps.SOCIAL_BLEND_CAP <= 0.5, "cap should keep the market as the anchor"

    def test_signal_never_deviates_from_market_beyond_cap(self, markets: list[dict]) -> None:
        eps = 1e-6
        for m in markets:
            poly = m["series"]["poly"]
            sig = m["series"]["signal"]
            for pi, si in zip(poly, sig):
                # |w*social + (1-w)*poly - poly| = w*|social-poly| <= cap*|social-poly| <= cap
                assert abs(si - pi) <= bps.SOCIAL_BLEND_CAP + eps, m["id"]
            assert abs(m["delta"]) <= bps.SOCIAL_BLEND_CAP + eps, m["id"]

    def test_raw_pressure_strength_is_not_capped(self, markets: list[dict]) -> None:
        # The cap is on the probability blend only; raw pressure keeps its true scale, so
        # at least one market should show strength above the cap (uncapped raw signal).
        max_pressure = max((m["pressureNow"] for m in markets), default=0.0)
        assert max_pressure > bps.SOCIAL_BLEND_CAP * 100.0

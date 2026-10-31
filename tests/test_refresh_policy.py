"""Tests for ``polysignal.scheduling.refresh``."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from polysignal.scheduling import (
    RefreshMode,
    RefreshPolicy,
    RefreshState,
    RefreshStep,
)


def _now() -> datetime:
    return datetime(2025, 3, 1, 12, tzinfo=timezone.utc)


def test_fast_mode_runs_only_light_steps_on_fresh_state():
    policy = RefreshPolicy()
    plan = policy.plan(mode=RefreshMode.FAST, state=RefreshState(), now=_now())
    assert set(plan.steps) == {
        RefreshStep.FETCH_LATEST_PRICES,
        RefreshStep.COMPUTE_SOCIAL_PRESSURE,
        RefreshStep.COMPUTE_PROBABILITY,
        RefreshStep.REFRESH_DASHBOARD,
    }


def test_heavy_mode_runs_every_step_on_fresh_state():
    policy = RefreshPolicy()
    plan = policy.plan(mode=RefreshMode.HEAVY, state=RefreshState(), now=_now())
    assert len(plan.steps) == len(list(RefreshStep))


def test_manual_mode_respects_requested_subset():
    policy = RefreshPolicy()
    plan = policy.plan(
        mode=RefreshMode.MANUAL,
        state=RefreshState(),
        now=_now(),
        requested_steps={RefreshStep.GRAPH_UPDATE, RefreshStep.COMPUTE_PROBABILITY},
    )
    assert set(plan.steps) == {RefreshStep.GRAPH_UPDATE, RefreshStep.COMPUTE_PROBABILITY}


def test_cooldown_skips_recent_step():
    policy = RefreshPolicy()
    state = RefreshState()
    now = _now()
    state.mark_done(RefreshStep.GRAPH_UPDATE, now - timedelta(seconds=10))
    plan = policy.plan(mode=RefreshMode.HEAVY, state=state, now=now)
    assert RefreshStep.GRAPH_UPDATE not in plan.steps
    assert any(step is RefreshStep.GRAPH_UPDATE for step, _ in plan.skipped)


def test_force_bypasses_cooldowns():
    policy = RefreshPolicy()
    state = RefreshState()
    now = _now()
    state.mark_done(RefreshStep.GRAPH_UPDATE, now - timedelta(seconds=1))
    plan = policy.plan(mode=RefreshMode.HEAVY, state=state, now=now, force=True)
    assert RefreshStep.GRAPH_UPDATE in plan.steps


def test_determinism():
    policy = RefreshPolicy()
    state = RefreshState()
    now = _now()
    plan_a = policy.plan(mode=RefreshMode.HEAVY, state=state, now=now)
    plan_b = policy.plan(mode=RefreshMode.HEAVY, state=state, now=now)
    assert plan_a == plan_b

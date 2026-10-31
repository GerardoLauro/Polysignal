"""Refresh-policy state machine.

See ``docs/REFRESH_POLICY.md`` for the conceptual model.

The policy is **side-effect free**: it takes a state plus a request,
and returns the ordered list of steps to execute. The caller (the
dashboard, the CLI) is responsible for running the steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum


class RefreshStep(str, Enum):
    FETCH_LATEST_PRICES = "fetch_latest_prices"
    INGEST_SOCIAL = "ingest_social_incremental"
    LINK_POSTS = "link_posts"
    NLP_ENRICH = "nlp_enrich"
    STANCE_CLASSIFY = "stance_classify"
    GRAPH_UPDATE = "graph_update"
    COMPUTE_SOCIAL_PRESSURE = "compute_social_pressure"
    COMPUTE_PROBABILITY = "compute_probability"
    REFRESH_DASHBOARD = "refresh_dashboard_views"


class RefreshMode(str, Enum):
    FAST = "fast"
    HEAVY = "heavy"
    MANUAL = "manual"


# Order matters: dependencies flow top-to-bottom.
_STEP_ORDER: tuple[RefreshStep, ...] = (
    RefreshStep.FETCH_LATEST_PRICES,
    RefreshStep.INGEST_SOCIAL,
    RefreshStep.LINK_POSTS,
    RefreshStep.NLP_ENRICH,
    RefreshStep.STANCE_CLASSIFY,
    RefreshStep.GRAPH_UPDATE,
    RefreshStep.COMPUTE_SOCIAL_PRESSURE,
    RefreshStep.COMPUTE_PROBABILITY,
    RefreshStep.REFRESH_DASHBOARD,
)

_DEFAULT_COOLDOWNS: dict[RefreshStep, timedelta] = {
    RefreshStep.FETCH_LATEST_PRICES: timedelta(seconds=30),
    RefreshStep.INGEST_SOCIAL: timedelta(minutes=10),
    RefreshStep.LINK_POSTS: timedelta(minutes=10),
    RefreshStep.NLP_ENRICH: timedelta(minutes=10),
    RefreshStep.STANCE_CLASSIFY: timedelta(minutes=10),
    RefreshStep.GRAPH_UPDATE: timedelta(minutes=15),
    RefreshStep.COMPUTE_SOCIAL_PRESSURE: timedelta(seconds=60),
    RefreshStep.COMPUTE_PROBABILITY: timedelta(seconds=30),
    RefreshStep.REFRESH_DASHBOARD: timedelta(seconds=5),
}

_FAST_STEPS: frozenset[RefreshStep] = frozenset(
    {
        RefreshStep.FETCH_LATEST_PRICES,
        RefreshStep.COMPUTE_SOCIAL_PRESSURE,
        RefreshStep.COMPUTE_PROBABILITY,
        RefreshStep.REFRESH_DASHBOARD,
    }
)


@dataclass
class RefreshState:
    """Per-step last successful execution time."""

    last_run: dict[RefreshStep, datetime] = field(default_factory=dict)

    def mark_done(self, step: RefreshStep, when: datetime) -> None:
        self.last_run[step] = when

    def age_of(self, step: RefreshStep, now: datetime) -> timedelta | None:
        ts = self.last_run.get(step)
        if ts is None:
            return None
        return now - ts


@dataclass(frozen=True)
class RefreshPlan:
    mode: RefreshMode
    steps: tuple[RefreshStep, ...]
    skipped: tuple[tuple[RefreshStep, str], ...] = ()


class RefreshPolicy:
    """Decide which steps to run for a given request."""

    def __init__(
        self,
        cooldowns: dict[RefreshStep, timedelta] | None = None,
    ) -> None:
        self.cooldowns = {**_DEFAULT_COOLDOWNS, **(cooldowns or {})}

    # ----------------------------------------------------------- planning
    def plan(
        self,
        *,
        mode: RefreshMode,
        state: RefreshState,
        now: datetime | None = None,
        requested_steps: set[RefreshStep] | None = None,
        force: bool = False,
    ) -> RefreshPlan:
        now = now or datetime.now(tz=timezone.utc)

        if mode is RefreshMode.FAST:
            candidate = [s for s in _STEP_ORDER if s in _FAST_STEPS]
        elif mode is RefreshMode.HEAVY:
            candidate = list(_STEP_ORDER)
        else:  # MANUAL
            if not requested_steps:
                return RefreshPlan(mode=mode, steps=())
            candidate = [s for s in _STEP_ORDER if s in requested_steps]

        runnable: list[RefreshStep] = []
        skipped: list[tuple[RefreshStep, str]] = []
        for step in candidate:
            if force:
                runnable.append(step)
                continue
            age = state.age_of(step, now)
            cooldown = self.cooldowns[step]
            if age is None or age >= cooldown:
                runnable.append(step)
            else:
                remaining = cooldown - age
                skipped.append((step, f"cooldown: {int(remaining.total_seconds())}s remaining"))
        return RefreshPlan(mode=mode, steps=tuple(runnable), skipped=tuple(skipped))


__all__ = [
    "RefreshMode",
    "RefreshPlan",
    "RefreshPolicy",
    "RefreshState",
    "RefreshStep",
]

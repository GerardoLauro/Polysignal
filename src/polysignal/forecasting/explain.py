"""Build human-readable explanations for forecast outputs."""

from __future__ import annotations

from polysignal.domain import SocialPressureRecord


def build_explanation(
    *,
    p_m: float,
    p_s: float,
    weight: float,
    n_posts: int,
    pressure: SocialPressureRecord | None,
) -> list[str]:
    """Return a list of short, ordered explanation lines.

    The output is meant to be rendered in the dashboard as bullet
    points. Each line is self-contained and contains the numbers it
    refers to, so callers do not have to re-derive context.
    """
    lines: list[str] = []
    lines.append(f"Market-implied probability: {p_m:.1%}")
    lines.append(f"Social-derived probability: {p_s:.1%} (from {n_posts} supporting posts)")
    lines.append(f"Blend weight on social signal: {weight:.0%}")

    delta = p_s - p_m
    if abs(delta) < 0.01:
        lines.append("Social and market signals are aligned within 1pp.")
    elif delta > 0:
        lines.append(f"Social signal leans toward YES by {delta:+.1%} relative to the market.")
    else:
        lines.append(f"Social signal leans toward NO by {delta:+.1%} relative to the market.")

    if pressure is not None and pressure.drivers:
        top = sorted(pressure.drivers, key=lambda d: -abs(d.contribution))[:3]
        drivers_summary = ", ".join(f"{d.name}={d.contribution:.2f}" for d in top)
        lines.append(f"Main social drivers: {drivers_summary}.")

    return lines


__all__ = ["build_explanation"]

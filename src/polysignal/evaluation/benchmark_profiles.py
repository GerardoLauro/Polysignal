"""Branch 09 benchmark profiles.

Only **real-data** evaluation profiles remain. The earlier controlled
synthetic-signal / synthetic-null engineering benchmarks (and the
``build_controlled_synthetic_scenario`` overlay) were removed: the project
reports honest results on the real corpus only, so no target-informed
synthetic pathway exists anywhere in the pipeline.

Two real profiles are kept:

* ``real_optional`` — the default. Real resolved contracts joined to the
  canonical Branch 08 social-pressure features and latest valid pre-close
  YES prices.
* ``real_social_enriched_main`` — identical evaluation policy, run against
  the Branch 12 public-web enriched social root (wider real evidence base).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

REAL_OPTIONAL = "real_optional"
REAL_SOCIAL_ENRICHED_MAIN = "real_social_enriched_main"
DEFAULT_BENCHMARK_PROFILE = REAL_OPTIONAL


@dataclass(frozen=True)
class BenchmarkProfile:
    """One explicit Branch 09 evaluation mode."""

    name: str
    benchmark_type: str
    dataset_mode: str
    default_root: str
    feature_source: str
    market_probability_source: str
    controlled_social_signal: bool
    supports_dashboard_demo: bool
    supports_real_empirical_claims: bool
    leakage_boundary: str
    warnings: tuple[str, ...]

    def to_metadata(self) -> dict[str, Any]:
        metadata = asdict(self)
        metadata["warnings"] = list(self.warnings)
        return metadata


_PROFILES = {
    REAL_OPTIONAL: BenchmarkProfile(
        name=REAL_OPTIONAL,
        benchmark_type="optional_real_data",
        dataset_mode="real",
        default_root="data_resolved_eval",
        feature_source="branch08_social_pressure_features",
        market_probability_source="latest_valid_preclose_yes_price",
        controlled_social_signal=False,
        supports_dashboard_demo=False,
        supports_real_empirical_claims=True,
        leakage_boundary=(
            "Resolved labels are joined only after Branch 08 input validation. Market priors "
            "come only from canonical YES prices timestamped at or before close."
        ),
        warnings=(
            "Current real Telegram evidence and canonical pre-close price coverage are sparse.",
            "Treat real-data results as preliminary until coverage improves.",
        ),
    ),
    REAL_SOCIAL_ENRICHED_MAIN: BenchmarkProfile(
        name=REAL_SOCIAL_ENRICHED_MAIN,
        benchmark_type="optional_real_data_enriched",
        dataset_mode="real",
        default_root="data_social_enriched_eval",
        feature_source="branch08_social_pressure_features",
        market_probability_source="latest_valid_preclose_yes_price",
        controlled_social_signal=False,
        supports_dashboard_demo=False,
        supports_real_empirical_claims=True,
        leakage_boundary=(
            "Identical to real_optional: resolved labels are joined only after Branch 08 input "
            "validation, and market priors come only from canonical YES prices timestamped at or "
            "before close. The only difference is the real social evidence base, which is widened "
            "with Branch 12 public-web Reddit + Telegram collection."
        ),
        warnings=(
            "Branch 12 enriches real social evidence from public old.reddit.com and t.me/s/<channel> "
            "pages; it is NOT official Reddit API access and is not a complete crawl.",
            "Canonical pre-close price and resolved-market coverage may still be sparse; treat "
            "results as preliminary until linked/stance coverage improves.",
        ),
    ),
}


def benchmark_profiles() -> tuple[BenchmarkProfile, ...]:
    """Return all profiles in stable CLI-display order."""
    return tuple(
        _PROFILES[name]
        for name in (
            REAL_OPTIONAL,
            REAL_SOCIAL_ENRICHED_MAIN,
        )
    )


def get_benchmark_profile(name: str) -> BenchmarkProfile:
    """Resolve one profile name or raise an actionable error."""
    try:
        return _PROFILES[name]
    except KeyError as exc:
        supported = ", ".join(_PROFILES)
        raise ValueError(f"benchmark_profile must be one of: {supported}") from exc


def resolve_benchmark_profile(
    *,
    benchmark_profile: str | None = None,
    legacy_dataset_mode: str | None = None,
) -> BenchmarkProfile:
    """Resolve new profile selection and legacy ``dataset_mode`` compatibility.

    The only supported ``dataset_mode`` values are ``real`` and ``auto`` —
    the synthetic pathway was removed. ``auto`` resolves (upstream) to the
    same real profile as the default; an explicit ``synthetic`` request is
    rejected.
    """
    if legacy_dataset_mode not in {None, "real"}:
        raise ValueError("dataset_mode must be one of: real, auto")
    legacy_profile = {
        None: None,
        "real": REAL_OPTIONAL,
    }[legacy_dataset_mode]
    selected = benchmark_profile or legacy_profile or DEFAULT_BENCHMARK_PROFILE
    profile = get_benchmark_profile(selected)
    if legacy_profile is not None and profile.name != legacy_profile:
        raise ValueError(
            f"benchmark_profile={profile.name!r} conflicts with "
            f"dataset_mode={legacy_dataset_mode!r}; use {legacy_profile!r}"
        )
    return profile


__all__ = [
    "DEFAULT_BENCHMARK_PROFILE",
    "REAL_OPTIONAL",
    "REAL_SOCIAL_ENRICHED_MAIN",
    "BenchmarkProfile",
    "benchmark_profiles",
    "get_benchmark_profile",
    "resolve_benchmark_profile",
]

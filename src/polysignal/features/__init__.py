"""Feature engineering — Social Pressure Index, graph features, time features."""

from polysignal.features.models import (
    PRESSURE_DIRECTIONS,
    SocialPressureFeatures,
    read_social_pressure_features,
    write_social_pressure_features,
)
from polysignal.features.social_pressure import (
    SocialPressureConfig,
    SocialPressureScorer,
    StanceObservation,
)
from polysignal.features.spi import (
    SPIConfig,
    SPIResult,
    build_features,
    classify_direction,
    compare_with_expected,
    compute_social_pressure,
    detect_dataset_mode,
)
from polysignal.features.timeline import (
    TimelineConfig,
    TimelineResult,
    build_market_timelines,
    read_market_timelines,
)

__all__ = [
    "PRESSURE_DIRECTIONS",
    "SPIConfig",
    "SPIResult",
    "SocialPressureConfig",
    "SocialPressureFeatures",
    "SocialPressureScorer",
    "StanceObservation",
    "TimelineConfig",
    "TimelineResult",
    "build_features",
    "build_market_timelines",
    "classify_direction",
    "compare_with_expected",
    "compute_social_pressure",
    "detect_dataset_mode",
    "read_market_timelines",
    "read_social_pressure_features",
    "write_social_pressure_features",
]

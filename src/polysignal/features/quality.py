"""Evidence-quality and source-diversity helpers for the SPI (Branch 08).

All functions are pure, deterministic, and bounded to ``[0, 1]`` so the
resulting quality columns are directly comparable across markets.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


def diversity_ratio(unique: int, total: int) -> float:
    """Fraction of distinct sources among ``total`` evidence rows (``[0, 1]``)."""
    if total <= 0:
        return 0.0
    return min(1.0, unique / total)


def shannon_entropy_normalized(counts: Iterable[int]) -> float:
    """Normalized Shannon entropy of a count distribution (``[0, 1]``).

    Returns ``0`` when the evidence is concentrated in a single bucket (or
    empty) and ``1`` when it is spread perfectly evenly across all buckets.
    Normalization is by ``log(n_buckets)`` so the value is comparable across
    markets with different channel counts.
    """
    positive = [c for c in counts if c > 0]
    n = len(positive)
    if n <= 1:
        return 0.0
    total = sum(positive)
    entropy = -sum((c / total) * math.log(c / total) for c in positive)
    return round(entropy / math.log(n), 6)


def source_diversity_score(unique_authors: int, unique_channels: int, accepted: int) -> float:
    """Blend author and channel diversity into a single score (``[0, 1]``)."""
    if accepted <= 0:
        return 0.0
    author_div = diversity_ratio(unique_authors, accepted)
    channel_div = diversity_ratio(unique_channels, accepted)
    return round((author_div + channel_div) / 2.0, 6)


def evidence_quality_score(
    confidence_mean: float,
    accepted_evidence_ratio: float,
    source_diversity: float,
) -> float:
    """Composite evidence quality in ``[0, 1]``.

    Weighted blend favouring stance confidence, then the share of linked
    rows that survived the accepted-evidence policy, then source diversity.
    """
    score = 0.5 * confidence_mean + 0.3 * accepted_evidence_ratio + 0.2 * source_diversity
    return round(min(1.0, max(0.0, score)), 6)


__all__ = [
    "diversity_ratio",
    "evidence_quality_score",
    "shannon_entropy_normalized",
    "source_diversity_score",
]

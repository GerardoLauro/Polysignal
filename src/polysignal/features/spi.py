"""Social Pressure Index (SPI) feature pipeline — Branch 08.

This module turns the canonical Branch 05/06/07 artefacts (linked evidence,
stance labels, posts, markets) into a deterministic, interpretable per-market
feature table (:class:`~polysignal.features.models.SocialPressureFeatures`).

Design principles
-----------------
* **Reuse, don't re-implement, the acceptance policy.** Accepted evidence and
  directional (SUPPORTS/OPPOSES) stances come straight from
  :func:`polysignal.graph.loaders.build_load_plan`, so the SPI consumes
  exactly the rows that the Branch 07 graph would, with the same temporal
  admissibility and weak-link confidence floor. The policy is never loosened.
* **No label leakage.** Only the market's close/resolution *timestamp* is used
  (as a temporal anchor). The resolved YES/NO outcome is never a feature.
* **Deterministic.** Given the same inputs the output is bit-for-bit stable;
  there is no randomness, no live API, and no LLM call.

The core directional signal is a confidence-weighted YES/NO balance::

    yes_w = sum(confidence_i for accepted directional SUPPORTS_YES)
    no_w  = sum(confidence_i for accepted directional SUPPORTS_NO)
    stance_balance = (yes_w - no_w) / (yes_w + no_w)   in [-1, 1]
    confidence_weighted_score = yes_w / (yes_w + no_w)  in [0, 1]  (the SPI)

``pressure_direction`` is then a 5-way deterministic classification (YES / NO /
MIXED / NEUTRAL / INSUFFICIENT_EVIDENCE) gated by a minimum amount of
directional evidence.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import log1p
from pathlib import Path
from typing import Any

import pandas as pd

from polysignal.domain import Market
from polysignal.features import quality, temporal
from polysignal.features.models import (
    PRESSURE_INSUFFICIENT,
    PRESSURE_MIXED,
    PRESSURE_NEUTRAL,
    PRESSURE_NO,
    PRESSURE_YES,
    SocialPressureFeatures,
)
from polysignal.features.temporal import TimedPressure
from polysignal.storage.parquet_store import read_dataframe, read_parquet
from polysignal.storage.paths import DataPaths, get_data_paths

# Relevance labels that survive into "accepted" linked evidence are RELEVANT
# and (high-confidence) WEAKLY_RELEVANT — but the SPI never re-derives that
# filter; it consumes the accepted rows produced by build_load_plan and counts
# the raw relevance breakdown only for reporting.
_DIRECTIONAL_LABELS = {"supports_yes", "supports_no"}


@dataclass(frozen=True)
class SPIConfig:
    """Tunable, documented knobs for the SPI computation."""

    window_days: tuple[int, ...] = (7, 30, 90)
    horizon_days: int = 180
    decay_half_life_days: float = 30.0
    min_evidence: int = 3          # min directional stances to call a direction
    direction_band: float = 0.12   # |stance_balance| threshold for YES/NO
    minor_min: int = 2             # min on each side for MIXED vs NEUTRAL
    confidence_floor: float = 0.6  # confidence threshold for confidence_coverage
    laplace_alpha: float = 1.0     # pseudo-counts shrinking the SPI toward 0.5


@dataclass(frozen=True)
class EvidenceStance:
    """One accepted directional stance feeding the pressure math."""

    is_yes: bool
    confidence: float
    created_at: datetime | None
    engagement: float = 0.0  # raw engagement of the parent post (>= 0)


@dataclass
class MarketEvidence:
    """All SPI inputs for a single market, assembled by the pipeline."""

    market_id: str
    condition_id: str | None = None
    question: str = ""
    domain: str | None = None
    close_time: datetime | None = None
    linked_post_count: int = 0
    accepted_link_count: int = 0
    relevant_count: int = 0
    weakly_relevant_count: int = 0
    ambiguous_count: int = 0
    unique_post_ids: set[str] = field(default_factory=set)
    authors: set[str] = field(default_factory=set)
    channel_counter: Counter[str] = field(default_factory=Counter)
    directional: list[EvidenceStance] = field(default_factory=list)
    raw_supports_yes: int = 0
    raw_supports_no: int = 0
    raw_neutral: int = 0
    raw_ambiguous: int = 0


@dataclass
class SPIResult:
    """Outcome of an SPI run."""

    dataset_mode: str
    rows: list[SocialPressureFeatures]
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Pure feature computation
# ---------------------------------------------------------------------------
def classify_direction(
    yes_count: int,
    no_count: int,
    stance_balance: float,
    cfg: SPIConfig,
) -> str:
    """Deterministic 5-way pressure classification."""
    directional = yes_count + no_count
    if directional < cfg.min_evidence:
        return PRESSURE_INSUFFICIENT
    if stance_balance >= cfg.direction_band:
        return PRESSURE_YES
    if stance_balance <= -cfg.direction_band:
        return PRESSURE_NO
    if yes_count >= cfg.minor_min and no_count >= cfg.minor_min:
        return PRESSURE_MIXED
    return PRESSURE_NEUTRAL


def build_features(
    ev: MarketEvidence,
    cfg: SPIConfig,
    dataset_mode: str,
) -> SocialPressureFeatures:
    """Compute one :class:`SocialPressureFeatures` row from market evidence."""
    yes_stances = [s for s in ev.directional if s.is_yes]
    no_stances = [s for s in ev.directional if not s.is_yes]
    yes_count = len(yes_stances)
    no_count = len(no_stances)
    directional_count = yes_count + no_count

    yes_w = sum(s.confidence for s in yes_stances)
    no_w = sum(s.confidence for s in no_stances)
    total_w = yes_w + no_w
    stance_balance = ((yes_w - no_w) / total_w) if total_w > 0 else 0.0
    confidence_weighted_score = (yes_w / total_w) if total_w > 0 else 0.5
    # Laplace/Beta shrinkage toward 0.5 so a market with only 1-2 directional
    # posts is not reported as a 0/1 certainty. Converges to the raw score as
    # directional weight grows; this is the per-market stabiliser for sparse
    # evidence (it never changes the pinned raw ``confidence_weighted_score``).
    alpha = cfg.laplace_alpha
    spi_score_shrunk = (yes_w + alpha) / (total_w + 2 * alpha) if (total_w + 2 * alpha) > 0 else 0.5
    confidence_mean = (
        sum(s.confidence for s in ev.directional) / directional_count
        if directional_count
        else 0.0
    )
    direction = classify_direction(yes_count, no_count, stance_balance, cfg)

    # --- temporal features ---------------------------------------------------
    anchor = _temporal_anchor(ev)
    timed = _timed(ev.directional, anchor)
    windows = temporal.standard_windows(cfg.window_days, cfg.horizon_days)
    window_vals = [temporal.window_pressure(timed, lo, hi) for lo, hi in windows]
    window_vals = [*window_vals, 0.0, 0.0, 0.0, 0.0][:4]
    decay = temporal.decay_pressure(timed, cfg.decay_half_life_days)
    early, late = temporal.early_late_pressure(timed)

    # --- quality features ----------------------------------------------------
    accepted = ev.accepted_link_count
    accepted_evidence_ratio = (
        round(min(1.0, accepted / ev.linked_post_count), 6) if ev.linked_post_count else 0.0
    )
    source_div = quality.source_diversity_score(len(ev.authors), len(ev.channel_counter), accepted)
    channel_entropy = quality.shannon_entropy_normalized(ev.channel_counter.values())
    confident_dir = sum(1 for s in ev.directional if s.confidence >= cfg.confidence_floor)
    confidence_coverage = (
        round(confident_dir / directional_count, 6) if directional_count else 0.0
    )
    quality_score = quality.evidence_quality_score(
        confidence_mean, accepted_evidence_ratio, source_div
    )

    # --- engagement features (secondary, log-scaled) -------------------------
    eng = [max(0.0, s.engagement) for s in ev.directional]
    engagement_total = sum(eng)
    engagement_mean = (engagement_total / len(eng)) if eng else 0.0
    engagement_max = max(eng) if eng else 0.0
    engagement_log_total = sum(log1p(e) for e in eng)
    yes_eng = sum(log1p(max(0.0, s.engagement)) for s in yes_stances)
    no_eng = sum(log1p(max(0.0, s.engagement)) for s in no_stances)

    return SocialPressureFeatures(
        market_id=ev.market_id,
        condition_id=ev.condition_id,
        question=ev.question,
        domain=ev.domain,
        close_time=ev.close_time,
        dataset_mode=dataset_mode,
        # evidence counts
        accepted_link_count=accepted,
        linked_post_count=ev.linked_post_count,
        unique_post_count=len(ev.unique_post_ids),
        unique_author_count=len(ev.authors),
        unique_channel_count=len(ev.channel_counter),
        relevant_count=ev.relevant_count,
        weakly_relevant_count=ev.weakly_relevant_count,
        ambiguous_count=ev.ambiguous_count,
        skipped_count=max(0, ev.linked_post_count - accepted),
        # stance counts (raw)
        supports_yes_count=ev.raw_supports_yes,
        supports_no_count=ev.raw_supports_no,
        neutral_stance_count=ev.raw_neutral,
        ambiguous_stance_count=ev.raw_ambiguous,
        directional_stance_count=ev.raw_supports_yes + ev.raw_supports_no,
        # pressure
        yes_pressure_raw=float(yes_count),
        no_pressure_raw=float(no_count),
        net_pressure_raw=float(yes_count - no_count),
        yes_pressure_weighted=round(yes_w, 6),
        no_pressure_weighted=round(no_w, 6),
        net_pressure_weighted=round(yes_w - no_w, 6),
        pressure_direction=direction,
        pressure_strength=round(abs(stance_balance), 6),
        stance_balance=round(stance_balance, 6),
        confidence_mean=round(confidence_mean, 6),
        confidence_weighted_score=round(confidence_weighted_score, 6),
        spi_score_shrunk=round(spi_score_shrunk, 6),
        # temporal
        pressure_0_7d=window_vals[0],
        pressure_8_30d=window_vals[1],
        pressure_31_90d=window_vals[2],
        pressure_91_180d=window_vals[3],
        temporal_decay_pressure=decay,
        early_pressure=early,
        late_pressure=late,
        pressure_acceleration=temporal.pressure_acceleration(early, late),
        # quality
        evidence_quality_score=quality_score,
        source_diversity_score=source_div,
        channel_entropy=channel_entropy,
        confidence_coverage=confidence_coverage,
        accepted_evidence_ratio=accepted_evidence_ratio,
        # graph-derived
        graph_linked_post_count=accepted,
        graph_supports_count=yes_count,
        graph_opposes_count=no_count,
        graph_author_count=len(ev.authors),
        graph_channel_count=len(ev.channel_counter),
        graph_neighborhood_size=accepted + yes_count + no_count,
        # engagement (secondary)
        engagement_total=round(engagement_total, 6),
        engagement_mean=round(engagement_mean, 6),
        engagement_max=round(engagement_max, 6),
        engagement_log_total=round(engagement_log_total, 6),
        yes_pressure_engagement_weighted=round(yes_eng, 6),
        no_pressure_engagement_weighted=round(no_eng, 6),
        net_pressure_engagement_weighted=round(yes_eng - no_eng, 6),
    )


def _temporal_anchor(ev: MarketEvidence) -> datetime | None:
    """Pick the time the pressure is measured *before* (never the outcome)."""
    if ev.close_time is not None:
        return ev.close_time
    timestamps = [s.created_at for s in ev.directional if s.created_at is not None]
    return max(timestamps) if timestamps else None


def _timed(directional: list[EvidenceStance], anchor: datetime | None) -> list[TimedPressure]:
    if anchor is None:
        return []
    items: list[TimedPressure] = []
    for stance in directional:
        if stance.created_at is None:
            continue
        items.append(
            TimedPressure(
                days_before=temporal.days_before(anchor, stance.created_at),
                is_yes=stance.is_yes,
                weight=stance.confidence,
            )
        )
    return items


# ---------------------------------------------------------------------------
# Dataset orchestration
# ---------------------------------------------------------------------------
def detect_dataset_mode(paths: DataPaths) -> str:
    """Return the dataset mode for the given root.

    The synthetic benchmark pathway was removed from the project, so the
    canonical corpus is always real data. This helper is retained for the
    ``--dataset-mode auto`` CLI selector and downstream metadata, and now
    always resolves to ``"real"``.
    """
    return "real"


def compute_social_pressure(
    *,
    paths: DataPaths | None = None,
    dataset_mode: str = "auto",
    config: SPIConfig | None = None,
) -> SPIResult:
    """Read the canonical artefacts and compute per-market SPI features."""
    from polysignal.graph.loaders import build_load_plan

    paths = paths or get_data_paths()
    cfg = config or SPIConfig()
    mode = detect_dataset_mode(paths) if dataset_mode == "auto" else dataset_mode

    markets = read_parquet(paths.polymarket_markets_parquet, Market)
    posts_meta = _posts_meta(read_dataframe(paths.social_posts_parquet))
    relevance = _relevance_counts(read_dataframe(paths.linked_posts_parquet))
    raw_stance = _raw_stance_counts(read_dataframe(paths.stance_labels_parquet))

    plan = build_load_plan(paths=paths)
    accepted_links = _operation_rows(plan, "link_post_contracts")
    directional_rows = _operation_rows(plan, "link_post_supports_outcome") + _operation_rows(
        plan, "link_post_opposes_outcome"
    )

    evidence = _assemble_evidence(
        markets=markets,
        posts_meta=posts_meta,
        relevance=relevance,
        raw_stance=raw_stance,
        accepted_links=accepted_links,
        directional_rows=directional_rows,
    )

    rows = [build_features(evidence[m.market_id], cfg, mode) for m in markets]
    rows.sort(key=lambda r: r.market_id)
    summary = summarize(rows, mode, cfg, plan_counts=plan.planned_relationship_counts)
    return SPIResult(dataset_mode=mode, rows=rows, summary=summary)


def _assemble_evidence(
    *,
    markets: Sequence[Market],
    posts_meta: dict[str, tuple[str | None, str | None, float]],
    relevance: dict[str, dict[str, int]],
    raw_stance: dict[str, dict[str, int]],
    accepted_links: list[dict[str, Any]],
    directional_rows: list[dict[str, Any]],
) -> dict[str, MarketEvidence]:
    evidence: dict[str, MarketEvidence] = {}
    for market in markets:
        rel = relevance.get(market.market_id, {})
        stance = raw_stance.get(market.market_id, {})
        evidence[market.market_id] = MarketEvidence(
            market_id=market.market_id,
            condition_id=market.condition_id,
            question=market.question,
            domain=market.domain.value if market.domain else None,
            close_time=market.close_time,
            linked_post_count=rel.get("_total", 0),
            relevant_count=rel.get("relevant", 0),
            weakly_relevant_count=rel.get("weakly_relevant", 0),
            ambiguous_count=rel.get("ambiguous", 0),
            raw_supports_yes=stance.get("supports_yes", 0),
            raw_supports_no=stance.get("supports_no", 0),
            raw_neutral=stance.get("neutral", 0),
            raw_ambiguous=stance.get("ambiguous", 0),
        )

    for row in accepted_links:
        ev = evidence.get(row.get("market_id"))
        if ev is None:
            continue
        ev.accepted_link_count += 1
        parent = row.get("parent_post_id")
        if parent:
            ev.unique_post_ids.add(parent)
            author, channel, _engagement = posts_meta.get(parent, (None, None, 0.0))
            if author:
                ev.authors.add(author)
            if channel:
                ev.channel_counter[channel] += 1

    for row in directional_rows:
        ev = evidence.get(row.get("market_id"))
        if ev is None:
            continue
        label = str(row.get("stance_label", "")).lower()
        if label not in _DIRECTIONAL_LABELS:
            continue
        _author, _channel, engagement = posts_meta.get(
            row.get("parent_post_id"), (None, None, 0.0)
        )
        ev.directional.append(
            EvidenceStance(
                is_yes=label == "supports_yes",
                confidence=float(row.get("confidence") or 0.0),
                created_at=_as_datetime(row.get("source_post_created_at")),
                engagement=engagement,
            )
        )
    return evidence


def _operation_rows(plan: Any, name: str) -> list[dict[str, Any]]:
    for op in plan.operations:
        if op.name == name:
            return list(op.rows)
    return []


def _posts_meta(df: pd.DataFrame) -> dict[str, tuple[str | None, str | None, float]]:
    """Map post_id -> (author_id, channel, engagement_score)."""
    if df.empty or "post_id" not in df.columns:
        return {}
    meta: dict[str, tuple[str | None, str | None, float]] = {}
    cols = df.columns
    for row in df.itertuples(index=False):
        record = dict(zip(cols, row, strict=False))
        meta[record["post_id"]] = (
            _clean(record.get("author_id")),
            _clean(record.get("subreddit_or_channel")),
            _as_float(record.get("engagement_score")),
        )
    return meta


def _relevance_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    if df.empty or "market_id" not in df.columns:
        return {}
    label_col = "final_relevance_label"
    has_final = label_col in df.columns
    base_col = "relevance_label"
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    cols = df.columns
    for row in df.itertuples(index=False):
        record = dict(zip(cols, row, strict=False))
        market_id = record.get("market_id")
        if not market_id:
            continue
        label = None
        if has_final:
            label = _clean(record.get(label_col))
        if label is None:
            label = _clean(record.get(base_col))
        bucket = counts[market_id]
        bucket["_total"] += 1
        if label:
            bucket[str(label).lower()] += 1
    return {k: dict(v) for k, v in counts.items()}


def _raw_stance_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    if df.empty or "market_id" not in df.columns or "stance_label" not in df.columns:
        return {}
    grouped = (
        df.groupby(["market_id", "stance_label"]).size().reset_index(name="n")
    )
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    for row in grouped.itertuples(index=False):
        counts[row.market_id][str(row.stance_label).lower()] = int(row.n)
    return dict(counts)


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _as_float(value: Any) -> float:
    """Coerce a possibly-missing engagement value to a non-negative float."""
    if value is None:
        return 0.0
    try:
        if pd.isna(value):
            return 0.0
    except (TypeError, ValueError):
        pass
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _as_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        ts = value.to_pydatetime()
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Summary and expected-results comparison
# ---------------------------------------------------------------------------
def summarize(
    rows: Sequence[SocialPressureFeatures],
    dataset_mode: str,
    cfg: SPIConfig,
    *,
    plan_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    direction_counts = Counter(r.pressure_direction for r in rows)
    summary: dict[str, Any] = {
        "dataset_mode": dataset_mode,
        "markets_evaluated": len(rows),
        "config": {
            "window_days": list(cfg.window_days),
            "horizon_days": cfg.horizon_days,
            "decay_half_life_days": cfg.decay_half_life_days,
            "min_evidence": cfg.min_evidence,
            "direction_band": cfg.direction_band,
            "minor_min": cfg.minor_min,
            "confidence_floor": cfg.confidence_floor,
        },
        "pressure_direction_counts": {d: direction_counts.get(d, 0) for d in _DIRECTION_ORDER},
        "totals": {
            "accepted_links": sum(r.accepted_link_count for r in rows),
            "linked_posts": sum(r.linked_post_count for r in rows),
            "supports_yes": sum(r.supports_yes_count for r in rows),
            "supports_no": sum(r.supports_no_count for r in rows),
            "neutral": sum(r.neutral_stance_count for r in rows),
            "ambiguous": sum(r.ambiguous_stance_count for r in rows),
            "directional": sum(r.directional_stance_count for r in rows),
        },
    }
    if plan_counts is not None:
        summary["graph_relationship_counts"] = dict(plan_counts)
    return summary


_DIRECTION_ORDER = (
    PRESSURE_YES,
    PRESSURE_NO,
    PRESSURE_MIXED,
    PRESSURE_NEUTRAL,
    PRESSURE_INSUFFICIENT,
)

# Expected → predicted direction agreement is "lenient" for MIXED: an
# expected MIXED market is satisfied by either a MIXED or a NEUTRAL call,
# since both describe balanced/low-signal evidence.
_LENIENT_MATCHES = {"MIXED": {PRESSURE_MIXED, PRESSURE_NEUTRAL}}


def compare_with_expected(
    rows: Sequence[SocialPressureFeatures],
    expected_path: Path,
) -> dict[str, Any]:
    """Compare predicted directions against synthetic expected-results."""
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    expected_markets = {m["market_id"]: m for m in expected.get("markets", [])}
    by_id = {r.market_id: r for r in rows}

    expected_dir_counts: Counter[str] = Counter()
    agree = 0
    exact = 0
    evaluated = 0
    correct_examples: list[dict[str, Any]] = []
    incorrect_examples: list[dict[str, Any]] = []
    supports_yes_match = 0
    supports_no_match = 0

    for market_id, exp in expected_markets.items():
        exp_dir = str(exp.get("expected_net_pressure_direction", "")).upper()
        expected_dir_counts[exp_dir] += 1
        row = by_id.get(market_id)
        if row is None:
            continue
        evaluated += 1
        if row.supports_yes_count == exp.get("supports_yes"):
            supports_yes_match += 1
        if row.supports_no_count == exp.get("supports_no"):
            supports_no_match += 1
        is_exact = row.pressure_direction == exp_dir
        is_lenient = is_exact or row.pressure_direction in _LENIENT_MATCHES.get(exp_dir, set())
        exact += is_exact
        agree += is_lenient
        sample = {
            "market_id": market_id,
            "question": exp.get("question", "")[:80],
            "expected": exp_dir,
            "predicted": row.pressure_direction,
            "supports_yes": row.supports_yes_count,
            "supports_no": row.supports_no_count,
        }
        if is_lenient and len(correct_examples) < 5:
            correct_examples.append(sample)
        elif not is_lenient and len(incorrect_examples) < 5:
            incorrect_examples.append(sample)

    return {
        "markets_in_expected": len(expected_markets),
        "markets_evaluated": evaluated,
        "expected_direction_counts": dict(expected_dir_counts),
        "agreement_lenient": agree,
        "agreement_exact": exact,
        "agreement_rate": round(agree / evaluated, 4) if evaluated else 0.0,
        "supports_yes_count_match": supports_yes_match,
        "supports_no_count_match": supports_no_match,
        "expected_totals": expected.get("totals", {}),
        "correct_examples": correct_examples,
        "incorrect_examples": incorrect_examples,
    }


__all__ = [
    "EvidenceStance",
    "MarketEvidence",
    "SPIConfig",
    "SPIResult",
    "build_features",
    "classify_direction",
    "compare_with_expected",
    "compute_social_pressure",
    "detect_dataset_mode",
    "summarize",
]

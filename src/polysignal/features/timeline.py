"""Per-market timelines: real Polymarket probability vs a smoothed social signal.

This module produces the data behind the core PolySignal graph: for each
market, a daily series that pairs

* ``market_prob`` — the real Polymarket YES price (implied probability),
  taken from the ingested CLOB timeline and forward-filled across
  no-trade days; with
* ``social_signal`` — a smoothed, accepted-evidence-weighted social
  estimate of P(YES) in ``[0, 1]``; and
* ``signal_strength`` — how much directional evidence backs the signal
  that day, in ``[0, 1)``.

Why not a raw daily YES-share? A single post on a quiet day would slam
the ratio to 0.0 or 1.0. Instead:

1. **Accepted evidence only.** The inputs are the rows that survived the
   evidence policy (``analytical/evidence/accepted_evidence.parquet``),
   joined with directional stance labels.
2. **Per-source daily cap.** One author/channel contributes at most
   ``source_daily_cap`` stance-weight per market per day, so a single
   spamming account cannot dominate the signal (source diversity).
3. **Confidence & engagement weighting.** Each stance contributes
   ``confidence * (1 + log1p(engagement))`` capped at ``max_post_weight``.
4. **Exponential smoothing.** Daily YES/NO weights are smoothed with an
   EWMA (half-life ``half_life_days``), so the signal carries memory and
   decays smoothly instead of jumping.
5. **Laplace anchoring.** ``social_signal = (yes + a) / (yes + no + 2a)``
   keeps the signal near 0.5 when evidence is thin; ``signal_strength``
   tells the reader exactly how thin.

No fabrication: days without evidence show ``signal_strength ~ 0`` and a
signal anchored at 0.5 — they are never invented to look informative.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from polysignal.storage.parquet_store import read_dataframe
from polysignal.storage.paths import DataPaths, get_data_paths

__all__ = [
    "TimelineConfig",
    "TimelineResult",
    "build_market_timelines",
    "read_market_timelines",
]

TIMELINE_COLUMNS = (
    "market_id",
    "question",
    "domain",
    "resolution",
    "date",
    "market_prob",
    "market_prob_observed",
    "social_signal",
    "signal_strength",
    "evidence_count",
    "yes_weight_raw",
    "no_weight_raw",
    "yes_weight_smooth",
    "no_weight_smooth",
    "source_count",
)


@dataclass(frozen=True)
class TimelineConfig:
    """Documented knobs for the timeline computation."""

    half_life_days: float = 3.0     # EWMA half-life for the social signal
    laplace_alpha: float = 0.75     # pseudo-counts anchoring the signal at 0.5
    strength_k: float = 3.0         # strength = n_eff / (n_eff + k)
    source_daily_cap: float = 3.0   # max stance-weight one source adds per market-day
    max_post_weight: float = 4.0    # cap on a single post's stance weight


@dataclass
class TimelineResult:
    frame: pd.DataFrame
    summary: dict[str, Any]
    parquet_path: Path | None = None
    summary_path: Path | None = None


def _utc_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, format="ISO8601").dt.floor("D")


def _yes_outcome_ids(outcomes: pd.DataFrame) -> dict[str, str]:
    """Map market_id -> YES outcome (CLOB token) id."""
    if outcomes.empty:
        return {}
    frame = outcomes.copy()
    frame["_is_yes"] = frame["label"].astype(str).str.strip().str.lower().isin(["yes", "true"])
    yes = frame[frame["_is_yes"]]
    # Fall back to index 0 for markets whose labels are not literal yes/no.
    fallback = frame[frame["index"] == 0]
    mapping: dict[str, str] = {}
    for row in fallback.itertuples(index=False):
        mapping[str(row.market_id)] = str(row.outcome_id)
    for row in yes.itertuples(index=False):
        mapping[str(row.market_id)] = str(row.outcome_id)
    return mapping


def _daily_market_prob(prices: pd.DataFrame, yes_ids: dict[str, str]) -> pd.DataFrame:
    """Daily last YES price per market (observed days only)."""
    if prices.empty:
        return pd.DataFrame(columns=["market_id", "date", "market_prob"])
    frame = prices.copy()
    frame["market_id"] = frame["market_id"].astype(str)
    frame["outcome_id"] = frame["outcome_id"].astype(str)
    frame = frame[
        frame.apply(lambda r: yes_ids.get(r["market_id"]) == r["outcome_id"], axis=1)
    ]
    if frame.empty:
        return pd.DataFrame(columns=["market_id", "date", "market_prob"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    frame = frame.sort_values("timestamp")
    frame["date"] = frame["timestamp"].dt.floor("D")
    daily = (
        frame.groupby(["market_id", "date"], as_index=False)
        .agg(market_prob=("price", "last"))
    )
    return daily


def _stance_direction(label: Any) -> int | None:
    text = str(label or "").strip().lower()
    if text == "supports_yes":
        return 1
    if text == "supports_no":
        return -1
    return None


def _post_weight(confidence: float, engagement: float, cfg: TimelineConfig) -> float:
    base = max(0.0, float(confidence)) * (1.0 + math.log1p(max(0.0, float(engagement))))
    return min(cfg.max_post_weight, base)


def _daily_social_weights(
    accepted: pd.DataFrame,
    stances: pd.DataFrame,
    cfg: TimelineConfig,
) -> pd.DataFrame:
    """Per (market, day): capped YES/NO stance weight + evidence/source counts."""
    empty = pd.DataFrame(
        columns=["market_id", "date", "yes_w", "no_w", "evidence_count", "source_count"]
    )
    if accepted.empty:
        return empty

    frame = accepted[accepted["accepted"] == True].copy()  # noqa: E712 — parquet bool column
    if frame.empty or "post_created_at" not in frame.columns:
        return empty
    frame = frame[frame["post_created_at"].notna()]
    if frame.empty:
        return empty
    frame["date"] = _utc_date(frame["post_created_at"])
    frame["source"] = (
        frame["post_channel"].fillna("")
        .where(frame["post_channel"].notna() & (frame["post_channel"] != ""), frame["post_author"])
        .fillna("unknown")
        .astype(str)
    )

    direction_map: dict[tuple[str, str], tuple[int, float]] = {}
    if not stances.empty and {"market_id", "post_id", "stance_label"}.issubset(stances.columns):
        for row in stances.itertuples(index=False):
            direction = _stance_direction(getattr(row, "stance_label", None))
            if direction is None:
                continue
            confidence = float(getattr(row, "confidence", 0.0) or 0.0)
            direction_map[(str(row.market_id), str(row.post_id))] = (direction, confidence)

    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        market_id = str(row.market_id)
        # Stance may be keyed by the scored segment id or the parent post.
        stance = direction_map.get((market_id, str(getattr(row, "segment_id", "") or "")))
        if stance is None:
            stance = direction_map.get((market_id, str(row.parent_post_id)))
        engagement = float(getattr(row, "post_engagement_score", 0.0) or 0.0)
        if stance is None:
            records.append(
                {
                    "market_id": market_id,
                    "date": row.date,
                    "source": row.source,
                    "yes_w": 0.0,
                    "no_w": 0.0,
                }
            )
            continue
        direction, stance_confidence = stance
        weight = _post_weight(stance_confidence, engagement, cfg)
        records.append(
            {
                "market_id": market_id,
                "date": row.date,
                "source": row.source,
                "yes_w": weight if direction > 0 else 0.0,
                "no_w": weight if direction < 0 else 0.0,
            }
        )
    if not records:
        return empty
    raw = pd.DataFrame(records)

    # Source-diversity cap: one source contributes at most source_daily_cap
    # of stance weight per market-day (applied per direction sum).
    per_source = raw.groupby(["market_id", "date", "source"], as_index=False).agg(
        yes_w=("yes_w", "sum"),
        no_w=("no_w", "sum"),
        evidence_count=("source", "size"),
    )
    per_source["total_w"] = per_source["yes_w"] + per_source["no_w"]
    over = per_source["total_w"] > cfg.source_daily_cap
    scale = (cfg.source_daily_cap / per_source["total_w"]).where(over, 1.0)
    per_source["yes_w"] *= scale
    per_source["no_w"] *= scale

    daily = per_source.groupby(["market_id", "date"], as_index=False).agg(
        yes_w=("yes_w", "sum"),
        no_w=("no_w", "sum"),
        evidence_count=("evidence_count", "sum"),
        source_count=("source", "nunique"),
    )
    return daily


def _smooth_market(
    daily: pd.DataFrame,
    date_index: pd.DatetimeIndex,
    cfg: TimelineConfig,
) -> pd.DataFrame:
    """EWMA-smooth daily YES/NO weights over a continuous date index."""
    reindexed = (
        daily.set_index("date")
        .reindex(date_index)
        .fillna({"yes_w": 0.0, "no_w": 0.0, "evidence_count": 0, "source_count": 0})
    )
    decay = math.log(2.0) / max(cfg.half_life_days, 0.25)
    alpha = 1.0 - math.exp(-decay)
    yes_s = reindexed["yes_w"].ewm(alpha=alpha, adjust=False).mean()
    no_s = reindexed["no_w"].ewm(alpha=alpha, adjust=False).mean()
    # Scale the EWMA mean back to an "effective evidence mass" so strength
    # reflects roughly the half-life window's worth of evidence.
    horizon = 1.0 / alpha
    n_eff = (yes_s + no_s) * horizon
    a = cfg.laplace_alpha
    out = pd.DataFrame(index=date_index)
    out["yes_weight_raw"] = reindexed["yes_w"].to_numpy()
    out["no_weight_raw"] = reindexed["no_w"].to_numpy()
    out["yes_weight_smooth"] = yes_s.to_numpy()
    out["no_weight_smooth"] = no_s.to_numpy()
    out["evidence_count"] = reindexed["evidence_count"].astype(int).to_numpy()
    out["source_count"] = reindexed["source_count"].astype(int).to_numpy()
    out["social_signal"] = ((yes_s * horizon + a) / (n_eff + 2 * a)).to_numpy()
    out["signal_strength"] = (n_eff / (n_eff + cfg.strength_k)).to_numpy()
    return out


def build_market_timelines(
    *,
    paths: DataPaths | None = None,
    config: TimelineConfig | None = None,
    write: bool = True,
) -> TimelineResult:
    """Build the per-market daily timeline table from canonical artifacts."""
    paths = paths or get_data_paths()
    cfg = config or TimelineConfig()

    markets = read_dataframe(paths.polymarket_markets_parquet)
    outcomes = read_dataframe(paths.polymarket_outcomes_parquet)
    prices = read_dataframe(paths.polymarket_price_points_parquet)
    accepted = read_dataframe(paths.analytical / "evidence" / "accepted_evidence.parquet")
    stances = read_dataframe(paths.stance_labels_parquet)

    yes_ids = _yes_outcome_ids(outcomes)
    daily_prob = _daily_market_prob(prices, yes_ids)
    daily_social = _daily_social_weights(accepted, stances, cfg)

    market_meta: dict[str, dict[str, Any]] = {}
    if not markets.empty:
        for row in markets.itertuples(index=False):
            market_meta[str(row.market_id)] = {
                "question": getattr(row, "question", ""),
                "domain": getattr(row, "domain", None),
                "resolution": getattr(row, "resolution", None),
                "close_time": getattr(row, "close_time", None),
            }

    frames: list[pd.DataFrame] = []
    per_market_summary: list[dict[str, Any]] = []
    for market_id, market_prices in daily_prob.groupby("market_id"):
        meta = market_meta.get(str(market_id), {})
        start = market_prices["date"].min()
        end = market_prices["date"].max()
        close_time = meta.get("close_time")
        if close_time:
            try:
                close_day = pd.to_datetime(close_time, utc=True).floor("D")
                end = max(end, min(close_day, end))
            except (ValueError, TypeError):
                pass
        date_index = pd.date_range(start, end, freq="D", tz="UTC")

        prob = (
            market_prices.set_index("date")["market_prob"]
            .reindex(date_index)
        )
        observed = prob.notna()
        prob = prob.ffill()

        social_daily = daily_social[daily_social["market_id"] == market_id][
            ["date", "yes_w", "no_w", "evidence_count", "source_count"]
        ]
        smooth = _smooth_market(social_daily, date_index, cfg)

        frame = smooth.copy()
        frame.insert(0, "market_id", str(market_id))
        frame.insert(1, "question", meta.get("question", ""))
        frame.insert(2, "domain", meta.get("domain"))
        frame.insert(3, "resolution", meta.get("resolution"))
        frame.insert(4, "date", date_index)
        frame.insert(5, "market_prob", prob.to_numpy())
        frame.insert(6, "market_prob_observed", observed.to_numpy())
        frame = frame.reset_index(drop=True)
        frames.append(frame)

        per_market_summary.append(
            {
                "market_id": str(market_id),
                "question": meta.get("question", ""),
                "days": len(date_index),
                "price_days_observed": int(observed.sum()),
                "evidence_total": int(frame["evidence_count"].sum()),
                "max_signal_strength": round(float(frame["signal_strength"].max()), 4),
                "directional_weight_total": round(
                    float((frame["yes_weight_raw"] + frame["no_weight_raw"]).sum()), 4
                ),
            }
        )

    result_frame = (
        pd.concat(frames, ignore_index=True)[list(TIMELINE_COLUMNS)]
        if frames
        else pd.DataFrame(columns=list(TIMELINE_COLUMNS))
    )
    summary = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "config": {
            "half_life_days": cfg.half_life_days,
            "laplace_alpha": cfg.laplace_alpha,
            "strength_k": cfg.strength_k,
            "source_daily_cap": cfg.source_daily_cap,
            "max_post_weight": cfg.max_post_weight,
        },
        "markets": per_market_summary,
        "markets_with_timeline": len(per_market_summary),
        "rows": len(result_frame),
        "interpretation": (
            "market_prob is the real Polymarket YES price (forward-filled on "
            "no-trade days). social_signal is a smoothed accepted-evidence "
            "estimate of P(YES); it stays near 0.5 and signal_strength near 0 "
            "when directional evidence is thin. Compare the two lines only "
            "where signal_strength is meaningfully above 0."
        ),
    }

    result = TimelineResult(frame=result_frame, summary=summary)
    if write:
        out_dir = paths.analytical / "timelines"
        out_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = out_dir / "market_timelines.parquet"
        write_frame = result_frame.copy()
        write_frame["date"] = write_frame["date"].astype(str)
        write_frame.to_parquet(parquet_path, index=False)
        write_frame.to_csv(out_dir / "market_timelines.csv", index=False)
        summary_path = out_dir / "timeline_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        result.parquet_path = parquet_path
        result.summary_path = summary_path
    return result


def read_market_timelines(paths: DataPaths | None = None) -> pd.DataFrame:
    """Read the persisted timeline table (empty frame when missing)."""
    paths = paths or get_data_paths()
    frame = read_dataframe(paths.analytical / "timelines" / "market_timelines.parquet")
    if not frame.empty and "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], utc=True, format="ISO8601")
    return frame

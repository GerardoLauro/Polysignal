"""Tests for the evidence export and market-timeline modules.

Offline + deterministic: a small canonical dataset is written to a tmp
root, then the evidence policy and timeline math are exercised end to end.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from polysignal.domain import (
    LinkedPost,
    Market,
    MarketDomain,
    Outcome,
    Platform,
    PricePoint,
    SocialPostRecord,
    StanceLabel,
)
from polysignal.domain.enums import ConstraintMatch, LinkMethod, RelevanceLabel, StanceMethod
from polysignal.features.timeline import TimelineConfig, build_market_timelines
from polysignal.linking.evidence import export_evidence
from polysignal.stance.models import StanceRecord, write_stance_records
from polysignal.storage.parquet_store import write_parquet
from polysignal.storage.paths import get_data_paths

CLOSE = datetime(2025, 6, 1, tzinfo=timezone.utc)
START = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _seed(root: Path) -> None:
    paths = get_data_paths(root=root)
    market = Market(
        market_id="m1",
        slug="will-x-happen",
        question="Will X happen?",
        domain=MarketDomain.CRYPTO,
        created_at=START,
        close_time=CLOSE,
        is_resolved=True,
        is_closed=True,
        outcomes=[
            Outcome(outcome_id="yes", label="Yes", index=0),
            Outcome(outcome_id="no", label="No", index=1),
        ],
    )
    write_parquet([market], paths.polymarket_markets_parquet)
    write_parquet(
        [
            Outcome(outcome_id="yes", label="Yes", index=0, market_id="m1"),
            Outcome(outcome_id="no", label="No", index=1, market_id="m1"),
        ],
        paths.polymarket_outcomes_parquet,
    )
    # A short YES price timeline inside the post window.
    prices = [
        PricePoint(
            market_id="m1",
            outcome_id="yes",
            timestamp=START + timedelta(days=d),
            price=0.4 + 0.01 * d,
        )
        for d in range(0, 40, 2)
    ]
    write_parquet(prices, paths.polymarket_price_points_parquet)

    posts, links, stances = [], [], []
    # 3 accepted (RELEVANT, pre-close) + 1 post-close (must be rejected) +
    # 1 duplicate post (must be rejected) + 1 unrelated (must be rejected).
    plan = [
        ("p0", "supports_yes", 0.85, RelevanceLabel.RELEVANT, START + timedelta(days=2), False),
        ("p1", "supports_yes", 0.80, RelevanceLabel.RELEVANT, START + timedelta(days=10), False),
        ("p2", "supports_no", 0.75, RelevanceLabel.RELEVANT, START + timedelta(days=20), False),
        ("p3", "supports_yes", 0.90, RelevanceLabel.RELEVANT, CLOSE + timedelta(days=5), False),
        ("p4", "supports_yes", 0.88, RelevanceLabel.RELEVANT, START + timedelta(days=3), True),
        ("p5", "neutral", 0.50, RelevanceLabel.UNRELATED, START + timedelta(days=4), False),
    ]
    for pid, label, conf, rel, created, is_dup in plan:
        posts.append(
            SocialPostRecord(
                post_id=pid,
                platform=Platform.TELEGRAM,
                source_id="chan_a",
                created_at=created,
                text=f"post {pid}",
                author_id="author_1",
                author_name="author_1",
                subreddit_or_channel="chan_a",
                engagement_score=50.0,
                collected_at=CLOSE,
                is_duplicate=is_dup,
            )
        )
        links.append(
            LinkedPost(
                link_id=f"l_{pid}",
                market_id="m1",
                post_id=pid,
                platform=Platform.TELEGRAM,
                relevance_score=0.8,
                relevance_label=rel,
                confidence=conf,
                method=LinkMethod.DETERMINISTIC_BASELINE,
                constraint_match=ConstraintMatch.UNKNOWN,
                created_at=CLOSE,
                post_created_at=created,
                market_question="Will X happen?",
                query="x happen",
                query_type="core",
            )
        )
        stances.append(
            StanceRecord(
                stance_id=f"s_{pid}",
                market_id="m1",
                post_id=pid,
                outcome_id=None,
                stance_label=StanceLabel(label),
                stance_score=0.5 if label == "supports_yes" else (-0.5 if label == "supports_no" else 0.0),
                confidence=conf,
                method=StanceMethod.HEURISTIC,
                created_at=CLOSE,
                post_created_at=created,
                source_platform=Platform.TELEGRAM,
                source_channel="chan_a",
            )
        )
    write_parquet(posts, paths.social_posts_parquet)
    write_parquet(links, paths.linked_posts_parquet)
    write_stance_records(stances, paths.stance_labels_parquet)


def test_export_evidence_applies_policy(tmp_path: Path) -> None:
    _seed(tmp_path)
    paths = get_data_paths(root=tmp_path)
    result = export_evidence(paths=paths)

    # p0, p1, p2 are accepted; p3 (post-close), p4 (duplicate), p5 (unrelated)
    # are rejected with distinct reasons.
    assert result.accepted_rows == 3
    accepted = pd.read_parquet(result.accepted_path)
    assert set(accepted["parent_post_id"]) == {"p0", "p1", "p2"}
    assert accepted["accepted"].all()
    # Every accepted row carries provenance.
    assert accepted["retrieval_query"].notna().all()

    rejected = pd.read_parquet(result.rejected_path)
    reasons = {row["parent_post_id"]: row["rejection_stage"] for _, row in rejected.iterrows()}
    assert reasons["p3"] == "temporal"
    assert reasons["p4"] == "dedup"
    assert reasons["p5"] == "relevance"
    assert result.summary["markets_with_accepted_evidence"] == 1


def test_build_timelines_real_prob_and_bounded_signal(tmp_path: Path) -> None:
    _seed(tmp_path)
    paths = get_data_paths(root=tmp_path)
    export_evidence(paths=paths)
    result = build_market_timelines(paths=paths, config=TimelineConfig())

    frame = result.frame
    assert not frame.empty
    assert set(frame["market_id"]) == {"m1"}
    # Market probability is the real (forward-filled) price, bounded to [0, 1].
    assert frame["market_prob"].between(0.0, 1.0).all()
    # Social signal is always a probability and stays at 0.5 when no evidence.
    assert frame["social_signal"].between(0.0, 1.0).all()
    assert frame["signal_strength"].between(0.0, 1.0).all()
    # Days with directional accepted evidence carry positive strength;
    # the YES-leaning evidence pushes the signal above 0.5 somewhere.
    assert frame["signal_strength"].max() > 0.0
    assert frame["social_signal"].max() > 0.5


def test_timeline_signal_flat_without_evidence(tmp_path: Path) -> None:
    _seed(tmp_path)
    paths = get_data_paths(root=tmp_path)
    # No evidence export -> accepted_evidence.parquet absent -> signal anchored.
    result = build_market_timelines(paths=paths, config=TimelineConfig())
    frame = result.frame
    assert (frame["signal_strength"] == 0.0).all()
    assert (frame["social_signal"] == 0.5).all()

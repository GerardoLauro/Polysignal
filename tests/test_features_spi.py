"""Tests for the Branch 08 Social Pressure Index (SPI) feature pipeline.

All tests are offline and deterministic: no network, no LLM, no live API.
They cover the pure feature layer, the temporal/quality helpers, the
expected-results comparison, sparse real-data behaviour, resolved-label
non-leakage, and an end-to-end dataset run plus CLI smoke checks.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from typer.testing import CliRunner

from polysignal.cli.main import app
from polysignal.domain import (
    LinkedPost,
    Market,
    MarketDomain,
    Outcome,
    Platform,
    SocialPostRecord,
    StanceLabel,
)
from polysignal.domain.enums import ConstraintMatch, LinkMethod, RelevanceLabel, StanceMethod
from polysignal.features import quality, temporal
from polysignal.features.models import (
    PRESSURE_INSUFFICIENT,
    PRESSURE_MIXED,
    PRESSURE_NEUTRAL,
    PRESSURE_NO,
    PRESSURE_YES,
    SocialPressureFeatures,
    read_social_pressure_features,
    write_social_pressure_features,
)
from polysignal.features.spi import (
    EvidenceStance,
    MarketEvidence,
    SPIConfig,
    build_features,
    classify_direction,
    compare_with_expected,
    compute_social_pressure,
    detect_dataset_mode,
)
from polysignal.features.temporal import TimedPressure
from polysignal.stance.models import StanceRecord, read_stance_records, write_stance_records
from polysignal.storage.parquet_store import read_parquet, write_parquet
from polysignal.storage.paths import get_data_paths

ANCHOR = datetime(2025, 6, 1, tzinfo=timezone.utc)
runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stance(
    is_yes: bool, conf: float, days_before: float = 1.0, engagement: float = 0.0
) -> EvidenceStance:
    return EvidenceStance(
        is_yes=is_yes,
        confidence=conf,
        created_at=ANCHOR - timedelta(days=days_before),
        engagement=engagement,
    )


def _evidence(directional: list[EvidenceStance], **kwargs) -> MarketEvidence:
    yes = sum(1 for s in directional if s.is_yes)
    no = sum(1 for s in directional if not s.is_yes)
    defaults = dict(
        market_id="m1",
        question="Will it happen?",
        domain="crypto",
        close_time=ANCHOR,
        linked_post_count=max(len(directional), 1) * 2,
        accepted_link_count=max(len(directional), 1),
        relevant_count=len(directional),
        unique_post_ids={f"p{i}" for i in range(len(directional))} or {"p0"},
        authors={"a1", "a2"},
        directional=directional,
        raw_supports_yes=yes,
        raw_supports_no=no,
    )
    defaults.update(kwargs)
    ev = MarketEvidence(**{k: v for k, v in defaults.items() if k != "channel_counter"})
    ev.channel_counter = kwargs.get("channel_counter", Counter({"chan_a": yes, "chan_b": no}))
    return ev


# ---------------------------------------------------------------------------
# Pure pressure math
# ---------------------------------------------------------------------------
def test_yes_no_pressure_counts_are_correct():
    ev = _evidence([_stance(True, 0.8), _stance(True, 0.8), _stance(False, 0.7)])
    row = build_features(ev, SPIConfig(), "synthetic")
    assert row.yes_pressure_raw == 2.0
    assert row.no_pressure_raw == 1.0
    assert row.net_pressure_raw == 1.0
    assert row.graph_supports_count == 2
    assert row.graph_opposes_count == 1


def test_confidence_weighted_pressure():
    ev = _evidence([_stance(True, 0.9), _stance(False, 0.3)])
    row = build_features(ev, SPIConfig(), "synthetic")
    assert row.yes_pressure_weighted == 0.9
    assert row.no_pressure_weighted == 0.3
    assert row.net_pressure_weighted == 0.6
    # balance = (0.9 - 0.3) / (0.9 + 0.3) = 0.5
    assert row.stance_balance == 0.5
    assert row.confidence_weighted_score == 0.75
    assert row.pressure_strength == 0.5


def test_all_yes_gives_full_yes_pressure():
    ev = _evidence([_stance(True, 0.8) for _ in range(4)])
    row = build_features(ev, SPIConfig(), "synthetic")
    assert row.stance_balance == 1.0
    assert row.confidence_weighted_score == 1.0
    assert row.pressure_direction == PRESSURE_YES


# ---------------------------------------------------------------------------
# Direction classification (all five outcomes reachable)
# ---------------------------------------------------------------------------
def test_classify_direction_all_cases():
    cfg = SPIConfig(min_evidence=3, direction_band=0.12, minor_min=2)
    assert classify_direction(1, 1, 0.0, cfg) == PRESSURE_INSUFFICIENT  # < min_evidence
    assert classify_direction(5, 1, 0.6, cfg) == PRESSURE_YES
    assert classify_direction(1, 5, -0.6, cfg) == PRESSURE_NO
    assert classify_direction(4, 4, 0.0, cfg) == PRESSURE_MIXED  # balanced, both sides
    assert classify_direction(3, 1, 0.05, cfg) == PRESSURE_NEUTRAL  # balanced, thin minor side


def test_insufficient_evidence_below_threshold():
    ev = _evidence([_stance(True, 0.8), _stance(True, 0.8)])  # only 2 directional
    row = build_features(ev, SPIConfig(min_evidence=3), "synthetic")
    assert row.pressure_direction == PRESSURE_INSUFFICIENT


def test_empty_evidence_is_insufficient_and_neutral_index():
    ev = MarketEvidence(market_id="m0", close_time=ANCHOR)
    row = build_features(ev, SPIConfig(), "real")
    assert row.pressure_direction == PRESSURE_INSUFFICIENT
    assert row.confidence_weighted_score == 0.5
    assert row.stance_balance == 0.0
    assert row.directional_stance_count == 0
    assert row.accepted_link_count == 0


# ---------------------------------------------------------------------------
# Temporal aggregation and decay
# ---------------------------------------------------------------------------
def test_window_pressure_buckets_by_days_before():
    items = [
        TimedPressure(days_before=2.0, is_yes=True, weight=1.0),   # 0-7d
        TimedPressure(days_before=20.0, is_yes=False, weight=1.0),  # 8-30d
    ]
    assert temporal.window_pressure(items, -1.0, 7.0) == 1.0
    assert temporal.window_pressure(items, 7.0, 30.0) == -1.0
    assert temporal.window_pressure(items, 30.0, 90.0) == 0.0  # empty window


def test_standard_windows_shape():
    windows = temporal.standard_windows((7, 30, 90), 180)
    assert len(windows) == 4
    assert windows[0][1] == 7
    assert windows[-1] == (90, 180)


def test_temporal_decay_weights_recent_more():
    # Recent YES (1 day) vs older NO (60 days): with a 30-day half-life the
    # YES evidence dominates, so the decayed balance leans positive.
    items = [
        TimedPressure(days_before=1.0, is_yes=True, weight=1.0),
        TimedPressure(days_before=60.0, is_yes=False, weight=1.0),
    ]
    decayed = temporal.decay_pressure(items, half_life_days=30.0)
    plain = temporal.net_balance(items)
    assert plain == 0.0
    assert decayed > 0.5


def test_window_aggregation_through_build_features():
    # 3 recent YES (within 7d) + 3 older NO (around 60d) → opposite windows.
    directional = [_stance(True, 0.8, days_before=2.0) for _ in range(3)]
    directional += [_stance(False, 0.8, days_before=60.0) for _ in range(3)]
    row = build_features(_evidence(directional), SPIConfig(), "synthetic")
    assert row.pressure_0_7d == 1.0   # only YES in 0-7d window
    assert row.pressure_31_90d == -1.0  # only NO in 31-90d window
    # later (recent) evidence is YES, earlier is NO → positive acceleration
    assert row.late_pressure > row.early_pressure
    assert row.pressure_acceleration > 0


def test_early_late_split_single_item_zero_acceleration():
    items = [TimedPressure(days_before=3.0, is_yes=True, weight=1.0)]
    early, late = temporal.early_late_pressure(items)
    assert early == late == 1.0


# ---------------------------------------------------------------------------
# Quality helpers
# ---------------------------------------------------------------------------
def test_channel_entropy_bounds():
    assert quality.shannon_entropy_normalized([5]) == 0.0          # single channel
    assert quality.shannon_entropy_normalized([]) == 0.0           # empty
    assert quality.shannon_entropy_normalized([5, 5]) == 1.0       # perfectly even
    assert 0.0 < quality.shannon_entropy_normalized([9, 1]) < 1.0  # skewed


def test_diversity_and_quality_scores_bounded():
    assert quality.diversity_ratio(0, 0) == 0.0
    assert quality.diversity_ratio(10, 5) == 1.0  # capped
    score = quality.evidence_quality_score(0.8, 0.7, 0.5)
    assert 0.0 <= score <= 1.0


def test_accepted_evidence_ratio_and_skipped_count():
    ev = _evidence([_stance(True, 0.8) for _ in range(3)], linked_post_count=10, accepted_link_count=4)
    row = build_features(ev, SPIConfig(), "synthetic")
    assert row.accepted_evidence_ratio == 0.4
    assert row.skipped_count == 6


# ---------------------------------------------------------------------------
# Engagement features (secondary, log-scaled)
# ---------------------------------------------------------------------------
def test_engagement_features_aggregate_correctly():
    from math import log1p

    ev = _evidence(
        [
            _stance(True, 0.8, engagement=100.0),
            _stance(True, 0.8, engagement=300.0),
            _stance(False, 0.7, engagement=200.0),
        ]
    )
    row = build_features(ev, SPIConfig(), "synthetic")
    assert row.engagement_total == 600.0
    assert row.engagement_mean == 200.0
    assert row.engagement_max == 300.0
    assert row.engagement_log_total == round(log1p(100) + log1p(300) + log1p(200), 6)
    assert row.yes_pressure_engagement_weighted == round(log1p(100) + log1p(300), 6)
    assert row.no_pressure_engagement_weighted == round(log1p(200), 6)
    assert row.net_pressure_engagement_weighted == round(
        (log1p(100) + log1p(300)) - log1p(200), 6
    )


def test_engagement_weighting_is_log_scaled_and_conservative():
    # A single viral post must NOT dominate: the log-weighted contribution of
    # a 1,000,000-engagement post is ~13.8, not 1,000,000.
    from math import log1p

    ev = _evidence([_stance(True, 0.8, engagement=1_000_000.0)])
    row = build_features(ev, SPIConfig(min_evidence=1), "synthetic")
    assert row.engagement_total == 1_000_000.0  # raw total is honest
    assert row.yes_pressure_engagement_weighted == round(log1p(1_000_000.0), 6)
    assert row.yes_pressure_engagement_weighted < 14.0


def test_engagement_does_not_change_primary_spi():
    # The confidence-weighted SPI and direction must be identical whether or
    # not engagement metadata is present.
    directional_plain = [_stance(True, 0.8), _stance(True, 0.8), _stance(False, 0.7)]
    directional_eng = [
        _stance(True, 0.8, engagement=999.0),
        _stance(True, 0.8, engagement=5.0),
        _stance(False, 0.7, engagement=12345.0),
    ]
    plain = build_features(_evidence(directional_plain), SPIConfig(), "synthetic")
    eng = build_features(_evidence(directional_eng), SPIConfig(), "synthetic")
    assert plain.stance_balance == eng.stance_balance
    assert plain.confidence_weighted_score == eng.confidence_weighted_score
    assert plain.pressure_direction == eng.pressure_direction
    assert plain.net_pressure_weighted == eng.net_pressure_weighted


def test_missing_engagement_is_zero():
    ev = _evidence([_stance(True, 0.8) for _ in range(3)])  # engagement defaults 0.0
    row = build_features(ev, SPIConfig(), "real")
    assert row.engagement_total == 0.0
    assert row.engagement_log_total == 0.0
    assert row.net_pressure_engagement_weighted == 0.0


# ---------------------------------------------------------------------------
# Determinism and no-leakage
# ---------------------------------------------------------------------------
def test_build_features_is_deterministic():
    directional = [_stance(True, 0.8), _stance(False, 0.7), _stance(True, 0.6)]
    a = build_features(_evidence(directional), SPIConfig(), "synthetic")
    b = build_features(_evidence(directional), SPIConfig(), "synthetic")
    assert a.model_dump() == b.model_dump()


def test_no_resolved_label_leakage():
    # The feature model and the evidence input must not expose the resolved
    # YES/NO outcome — only the close timestamp may be consumed.
    forbidden = {"resolution", "is_resolved", "winning_outcome_id", "real_resolution"}
    assert forbidden.isdisjoint(SocialPressureFeatures.model_fields)
    assert forbidden.isdisjoint(MarketEvidence.__dataclass_fields__)


# ---------------------------------------------------------------------------
# Expected-results comparison
# ---------------------------------------------------------------------------
def test_compare_with_expected(tmp_path: Path):
    rows = [
        build_features(
            _evidence([_stance(True, 0.8) for _ in range(5)], market_id="mk-yes"),
            SPIConfig(),
            "synthetic",
        ),
        build_features(
            _evidence([_stance(False, 0.8) for _ in range(5)], market_id="mk-no"),
            SPIConfig(),
            "synthetic",
        ),
    ]
    expected = {
        "totals": {"supports_yes": 5, "supports_no": 5},
        "markets": [
            {
                "market_id": "mk-yes",
                "question": "yes market",
                "expected_net_pressure_direction": "YES",
                "supports_yes": 5,
                "supports_no": 0,
            },
            {
                "market_id": "mk-no",
                "question": "no market",
                "expected_net_pressure_direction": "NO",
                "supports_yes": 0,
                "supports_no": 5,
            },
        ],
    }
    expected_path = tmp_path / "synthetic_expected_results.json"
    expected_path.write_text(json.dumps(expected), encoding="utf-8")
    report = compare_with_expected(rows, expected_path)
    assert report["markets_evaluated"] == 2
    assert report["agreement_exact"] == 2
    assert report["agreement_rate"] == 1.0
    assert report["supports_yes_count_match"] == 2
    assert report["supports_no_count_match"] == 2


def test_compare_lenient_mixed_accepts_neutral(tmp_path: Path):
    # An expected MIXED market is satisfied by a NEUTRAL prediction.
    row = build_features(
        _evidence([_stance(True, 0.8), _stance(False, 0.8), _stance(True, 0.79)], market_id="mk"),
        SPIConfig(direction_band=0.5, minor_min=5),  # forces NEUTRAL
        "synthetic",
    )
    assert row.pressure_direction == PRESSURE_NEUTRAL
    expected = {
        "totals": {},
        "markets": [
            {
                "market_id": "mk",
                "expected_net_pressure_direction": "MIXED",
                "supports_yes": 2,
                "supports_no": 1,
            }
        ],
    }
    path = tmp_path / "exp.json"
    path.write_text(json.dumps(expected), encoding="utf-8")
    report = compare_with_expected([row], path)
    assert report["agreement_lenient"] == 1
    assert report["agreement_exact"] == 0


# ---------------------------------------------------------------------------
# Dataset-level integration (small fixture) + CLI smoke
# ---------------------------------------------------------------------------
def _seed_dataset(root: Path, *, with_stance: bool = True) -> None:
    paths = get_data_paths(root=root)
    close = datetime(2025, 6, 1, tzinfo=timezone.utc)
    post_time = datetime(2025, 3, 1, tzinfo=timezone.utc)
    market = Market(
        market_id="m1",
        slug="will-x-happen",
        question="Will X happen?",
        domain=MarketDomain.CRYPTO,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        close_time=close,
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

    posts, links, stances = [], [], []
    # 4 YES + 1 NO directional, plus 1 neutral; all on accepted RELEVANT links.
    plan = [("supports_yes", 0.8)] * 4 + [("supports_no", 0.7)] + [("neutral", 0.5)]
    for i, (label, conf) in enumerate(plan):
        pid = f"p{i}"
        posts.append(
            SocialPostRecord(
                post_id=pid,
                platform=Platform.TELEGRAM,
                source_id="synthetic_chan_a" if i % 2 == 0 else "synthetic_chan_b",
                created_at=post_time + timedelta(days=i),
                text=f"post {i}",
                author_id=f"author_{i % 3}",
                author_name=f"author_{i % 3}",
                subreddit_or_channel="synthetic_chan_a" if i % 2 == 0 else "synthetic_chan_b",
                engagement_score=float(100 * (i + 1)),
                collected_at=close,
            )
        )
        links.append(
            LinkedPost(
                link_id=f"l{i}",
                market_id="m1",
                post_id=pid,
                platform=Platform.TELEGRAM,
                relevance_score=0.8,
                relevance_label=RelevanceLabel.RELEVANT,
                confidence=0.8,
                method=LinkMethod.DETERMINISTIC_BASELINE,
                constraint_match=ConstraintMatch.UNKNOWN,
                created_at=close,
                post_created_at=post_time + timedelta(days=i),
                market_question="Will X happen?",
            )
        )
        stances.append(
            StanceRecord(
                stance_id=f"s{i}",
                market_id="m1",
                post_id=pid,
                outcome_id=None,
                stance_label=StanceLabel(label),
                stance_score=0.5 if label == "supports_yes" else (-0.5 if label == "supports_no" else 0.0),
                confidence=conf,
                method=StanceMethod.HEURISTIC,
                created_at=close,
                post_created_at=post_time + timedelta(days=i),
                source_platform=Platform.TELEGRAM,
                source_channel="synthetic_chan_a" if i % 2 == 0 else "synthetic_chan_b",
            )
        )
    write_parquet(posts, paths.social_posts_parquet)
    write_parquet(links, paths.linked_posts_parquet)
    if with_stance:
        write_stance_records(stances, paths.stance_labels_parquet)


def test_compute_social_pressure_end_to_end(tmp_path: Path):
    _seed_dataset(tmp_path)
    paths = get_data_paths(root=tmp_path)
    result = compute_social_pressure(paths=paths, dataset_mode="real")
    assert result.dataset_mode == "real"
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.supports_yes_count == 4
    assert row.supports_no_count == 1
    assert row.neutral_stance_count == 1
    assert row.directional_stance_count == 5
    assert row.accepted_link_count == 6  # all 6 RELEVANT links accepted
    assert row.pressure_direction == PRESSURE_YES
    assert row.unique_author_count == 3
    assert row.unique_channel_count == 2
    assert result.summary["totals"]["supports_yes"] == 4
    # engagement metadata flows end-to-end (directional posts p0..p4 -> 100..500)
    assert row.engagement_total == 100 + 200 + 300 + 400 + 500
    assert row.engagement_max == 500.0
    assert row.net_pressure_engagement_weighted != 0.0


def test_compute_social_pressure_excludes_missing_timestamp_evidence(tmp_path: Path):
    _seed_dataset(tmp_path)
    paths = get_data_paths(root=tmp_path)
    posts = read_parquet(paths.social_posts_parquet, SocialPostRecord)
    links = read_parquet(paths.linked_posts_parquet, LinkedPost)
    stances = read_stance_records(paths.stance_labels_parquet)
    posts[0].created_at = None
    posts[0].created_at_parse_status = "missing"
    links[0].post_created_at = None
    stances[0].post_created_at = None
    write_parquet(posts, paths.social_posts_parquet)
    write_parquet(links, paths.linked_posts_parquet)
    write_stance_records(stances, paths.stance_labels_parquet)
    row = compute_social_pressure(paths=paths, dataset_mode="real").rows[0]
    assert row.accepted_link_count == 5
    assert row.graph_supports_count == 3
    assert row.yes_pressure_raw == 3.0


def test_compute_sparse_dataset_is_insufficient(tmp_path: Path):
    # No stance labels at all → directional evidence is empty → INSUFFICIENT.
    _seed_dataset(tmp_path, with_stance=False)
    paths = get_data_paths(root=tmp_path)
    result = compute_social_pressure(paths=paths, dataset_mode="real")
    row = result.rows[0]
    assert row.directional_stance_count == 0
    assert row.pressure_direction == PRESSURE_INSUFFICIENT
    # the linked evidence is still surfaced even without stance
    assert row.accepted_link_count == 6


def test_detect_dataset_mode(tmp_path: Path):
    # The synthetic pathway was removed; the canonical corpus is always real,
    # so detect_dataset_mode always resolves to "real" regardless of on-disk
    # markers.
    paths = get_data_paths(root=tmp_path)
    assert detect_dataset_mode(paths) == "real"
    meta = tmp_path / "metadata"
    meta.mkdir(parents=True)
    (meta / "synthetic_generation_manifest.json").write_text("{}", encoding="utf-8")
    assert detect_dataset_mode(paths) == "real"


def test_cli_features_help():
    res = runner.invoke(app, ["features", "--help"])
    assert res.exit_code == 0
    assert "compute-social-pressure" in res.stdout
    assert "inspect-social-pressure" in res.stdout


def test_cli_compute_and_inspect(tmp_path: Path):
    _seed_dataset(tmp_path)
    compute = runner.invoke(
        app,
        [
            "features",
            "compute-social-pressure",
            "--output-dir",
            str(tmp_path),
            "--dataset-mode",
            "real",
            "--overwrite",
        ],
    )
    assert compute.exit_code == 0, compute.stdout
    paths = get_data_paths(root=tmp_path)
    assert paths.social_pressure_features_parquet.exists()
    rows = read_social_pressure_features(paths.social_pressure_features_parquet)
    assert len(rows) == 1
    assert rows[0].pressure_direction == PRESSURE_YES

    inspect = runner.invoke(
        app,
        ["features", "inspect-social-pressure", "--output-dir", str(tmp_path)],
    )
    assert inspect.exit_code == 0, inspect.stdout
    assert "social_pressure.parquet" in inspect.stdout


def test_cli_compute_dry_run_writes_nothing(tmp_path: Path):
    _seed_dataset(tmp_path)
    res = runner.invoke(
        app,
        [
            "features",
            "compute-social-pressure",
            "--output-dir",
            str(tmp_path),
            "--dataset-mode",
            "real",
            "--dry-run",
        ],
    )
    assert res.exit_code == 0, res.stdout
    paths = get_data_paths(root=tmp_path)
    assert not paths.social_pressure_features_parquet.exists()


def test_feature_roundtrip_parquet(tmp_path: Path):
    row = build_features(_evidence([_stance(True, 0.8) for _ in range(3)]), SPIConfig(), "synthetic")
    target = tmp_path / "spi.parquet"
    write_social_pressure_features([row], target)
    back = read_social_pressure_features(target)
    assert len(back) == 1
    assert back[0].pressure_direction == PRESSURE_YES
    assert back[0].market_id == row.market_id

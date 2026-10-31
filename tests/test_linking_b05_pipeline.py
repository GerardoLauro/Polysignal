"""End-to-end tests for the Branch 05 pipeline + IO + evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from polysignal.domain import (
    LinkedPost,
    LinkMethod,
    Platform,
    RelevanceLabel,
    SocialPostRecord,
)
from polysignal.linking import (
    CandidateMode,
    build_pairs_from_records,
    build_pairs_from_records_with_diagnostics,
    evaluate_predictions,
    fixture_labels,
    fixture_markets,
    fixture_posts,
    read_linked_posts,
    run_linking_cross_product,
    run_linking_from_records,
    summarize_by_label,
    write_linked_posts,
)
from polysignal.linking.relevance import RelevanceJudgement

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_candidate_pair_builder_matches_records_to_markets() -> None:
    markets = fixture_markets()
    btc = next(m for m in markets if m.market_id == "m-btc-150k")
    record = SocialPostRecord(
        post_id="r-1",
        platform=Platform.REDDIT,
        source_id="src-1",
        url="https://reddit.com/r/test/comments/1",
        author_id="a-1",
        author_name="bob",
        created_at=_NOW,
        text="Bitcoin could hit $150k by December 2026 if ETF inflows continue.",
        title="BTC outlook",
        subreddit_or_channel="bitcoin",
        query="Bitcoin hit $150k by December 2026",
        market_id_candidate="m-btc-150k",
        engagement_score=10.0,
        upvotes=20,
        comments_count=5,
        collected_at=_NOW,
        text_length=64,
    )
    pairs = build_pairs_from_records([record], markets)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.market.market_id == btc.market_id
    assert pair.post.post_id == "r-1"
    assert pair.query is not None


def test_prefilter_bounds_provenance_free_candidates_and_full_scan_is_explicit() -> None:
    markets = fixture_markets()
    record = SocialPostRecord(
        post_id="telegram:1",
        platform=Platform.TELEGRAM,
        source_id="channel",
        created_at=_NOW,
        text="Bitcoin adoption is increasing among institutions.",
        collected_at=_NOW,
        text_length=50,
    )
    generated = build_pairs_from_records_with_diagnostics([record], markets)
    assert 0 < len(generated.pairs) < len(markets)
    assert generated.diagnostics.candidate_mode == CandidateMode.AUTO
    assert generated.diagnostics.fallback_mode_used == "prefilter"
    assert generated.diagnostics.average_candidate_markets_per_post < len(markets)

    full_scan = build_pairs_from_records([record], markets, candidate_mode=CandidateMode.FULL_SCAN)
    assert len(full_scan) == len(markets)


def test_prefilter_ignores_year_only_candidate_anchors() -> None:
    record = SocialPostRecord(
        post_id="telegram:year-only",
        platform=Platform.TELEGRAM,
        source_id="channel",
        created_at=_NOW,
        text="AI-related capex could reach $905 billion in 2027.",
        collected_at=_NOW,
        text_length=49,
    )
    generated = build_pairs_from_records_with_diagnostics([record], fixture_markets())
    assert generated.pairs == []
    assert generated.diagnostics.skipped_posts_no_candidate_anchors == 1


def test_candidate_builder_skips_missing_timestamp_audit_rows() -> None:
    record = SocialPostRecord(
        post_id="telegram:missing-time",
        platform=Platform.TELEGRAM,
        source_id="channel",
        created_at=None,
        created_at_parse_status="missing",
        text="Bitcoin adoption is increasing among institutions.",
        collected_at=_NOW,
    )
    generated = build_pairs_from_records_with_diagnostics([record], fixture_markets())
    assert generated.pairs == []
    assert generated.diagnostics.posts == 0


def test_run_linking_from_records_produces_linked_posts() -> None:
    markets = fixture_markets()
    record = SocialPostRecord(
        post_id="r-1",
        platform=Platform.REDDIT,
        source_id="src-1",
        url=None,
        author_id="a-1",
        author_name="bob",
        created_at=_NOW,
        text="Bitcoin could hit $150k by December 2026 if ETF inflows continue.",
        title=None,
        subreddit_or_channel="bitcoin",
        query="Bitcoin $150k December",
        market_id_candidate="m-btc-150k",
        engagement_score=0.0,
        collected_at=_NOW,
        text_length=64,
    )
    rows = run_linking_from_records([record], markets)
    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, LinkedPost)
    assert row.method == LinkMethod.DETERMINISTIC_BASELINE
    assert row.relevance_label == RelevanceLabel.RELEVANT


def test_linked_posts_parquet_round_trip(tmp_path: Path) -> None:
    markets = fixture_markets()
    posts = fixture_posts()
    rows = run_linking_cross_product(posts, markets)
    target = tmp_path / "linked_posts.parquet"
    written = write_linked_posts(rows, target)
    assert written.exists()
    reloaded = read_linked_posts(written)
    assert len(reloaded) == len(rows)
    # Schema validation — every required field round-trips.
    for original, restored in zip(rows, reloaded, strict=True):
        assert original.link_id == restored.link_id
        assert original.market_id == restored.market_id
        assert original.post_id == restored.post_id
        assert original.relevance_label == restored.relevance_label
        assert original.constraint_match == restored.constraint_match
        assert abs(original.relevance_score - restored.relevance_score) < 1e-9


def test_linked_post_schema_has_branch05_fields() -> None:
    """Schema sanity: every documented column exists on the model."""
    expected = {
        "link_id",
        "market_id",
        "post_id",
        "platform",
        "query",
        "query_type",
        "relevance_score",
        "relevance_label",
        "confidence",
        "method",
        "explanation",
        "matched_entities",
        "matched_terms",
        "constraint_match",
        "created_at",
        "post_created_at",
        "market_question",
        "post_text_preview",
        "raw_post_path",
        "anchor_score",
        "constraint_score",
        "domain_mismatch",
        "rejection_reason",
    }
    actual = set(LinkedPost.model_fields.keys())
    missing = expected - actual
    assert not missing, f"LinkedPost is missing schema fields: {missing}"


def test_summarize_by_label_groups_counts() -> None:
    markets = fixture_markets()
    posts = fixture_posts()
    rows = run_linking_cross_product(posts, markets)
    counts = summarize_by_label(rows)
    assert sum(counts.values()) == len(rows)
    # The fixture set should produce at least one row in each non-trivial bucket.
    assert counts.get("relevant", 0) >= 1
    assert counts.get("unrelated", 0) >= 1


def test_evaluation_reports_metrics_on_fixture() -> None:
    """Fixture predictions must hit > 0.8 accuracy and macro_f1."""
    markets = fixture_markets()
    posts = fixture_posts()
    rows = run_linking_cross_product(posts, markets)
    predictions = [
        RelevanceJudgement(
            market_id=r.market_id,
            post_id=r.post_id,
            relevance_score=r.relevance_score,
            relevance_label=r.relevance_label,
            confidence=r.confidence,
            constraint_match=r.constraint_match,
            matched_entities=list(r.matched_entities),
            matched_terms=list(r.matched_terms),
            explanation=r.explanation,
        )
        for r in rows
    ]
    report = evaluate_predictions(predictions, fixture_labels())
    assert report.matched_pairs == len(fixture_labels())
    assert report.accuracy >= 0.80
    assert report.macro_f1 >= 0.60
    assert report.constraint_accuracy >= 0.80
    # Confusion matrix is square over RelevanceLabel.
    for row_label in report.confusion_matrix:
        assert set(report.confusion_matrix[row_label].keys()) == set(RelevanceLabel)

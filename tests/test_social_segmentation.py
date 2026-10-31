"""Tests for the Branch 05G deterministic digest segmenter."""

from __future__ import annotations

from datetime import datetime, timezone

from polysignal.domain import Platform, SocialPostRecord
from polysignal.ingestion.social.segmentation import (
    segment_models_to_records,
    segment_post,
    segment_records,
    to_segment_models,
    write_segments,
)

_NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _record(post_id: str, text: str) -> SocialPostRecord:
    return SocialPostRecord(
        post_id=post_id,
        platform=Platform.TELEGRAM,
        source_id=post_id,
        url=None,
        author_id=None,
        author_name=None,
        created_at=_NOW,
        text=text,
        title=None,
        subreddit_or_channel="cointelegraph",
        query=None,
        query_type="telegram_channel",
        market_id_candidate=None,
        raw_path="/tmp/x.json",
        collected_at=_NOW,
        is_duplicate=False,
        text_length=len(text),
    )


_DIGEST = (
    "Catch up on the news over the last 24 hours!\n\n"
    "* Bitcoin could hit $150k by December 2026 if ETF inflows continue.\n\n"
    "* Donald Trump said his administration will protect America's crypto position.\n\n"
    "* China announced new economic stimulus targeting 2027 growth.\n\n"
    "* Apple may reveal a foldable iPhone before WWDC.\n\n"
    "* Kraken IPO is expected before December 31, 2026, according to insiders.\n\n"
    "* Fed signals it may cut rates after the next meeting.\n"
)


def test_short_post_not_split() -> None:
    rec = _record("telegram:1:1", "Bitcoin hit $150k today.")
    segments = segment_post(rec)
    assert len(segments) == 1
    assert segments[0].segment_text == "Bitcoin hit $150k today."
    assert segments[0].segment_index == 0
    assert segments[0].segmentation_method == "single"
    assert segments[0].parent_post_id == "telegram:1:1"


def test_empty_post_returns_no_segments() -> None:
    rec = _record("telegram:1:2", "")
    assert segment_post(rec) == []


def test_multi_topic_digest_is_split() -> None:
    rec = _record("telegram:1:3", _DIGEST)
    segments = segment_post(rec)
    # At least 5 distinct news items (the header is dropped or short).
    assert len(segments) >= 5
    joined = "\n".join(seg.segment_text for seg in segments)
    assert "Bitcoin could hit $150k" in joined
    assert "Kraken IPO" in joined
    assert "foldable iPhone" in joined


def test_segment_id_is_stable_and_indexed() -> None:
    rec = _record("telegram:1:3", _DIGEST)
    segments = segment_post(rec)
    ids = [seg.segment_id for seg in segments]
    assert ids == sorted(ids)
    assert all(seg.segment_id.startswith("telegram:1:3#seg") for seg in segments)
    assert all(seg.parent_post_id == "telegram:1:3" for seg in segments)
    # Re-running produces identical ids (deterministic).
    again = segment_post(rec)
    assert [s.segment_id for s in again] == ids


def test_segments_to_records_preserve_metadata() -> None:
    rec = _record("telegram:1:3", _DIGEST)
    segments = segment_post(rec)
    base_lookup = {rec.post_id: rec}
    derived = segment_models_to_records(to_segment_models(segments), base_lookup)
    assert all(r.source_id == "telegram:1:3" for r in derived)
    assert all(r.platform == Platform.TELEGRAM for r in derived)
    assert all(r.raw_path == "/tmp/x.json" for r in derived)
    assert all(r.subreddit_or_channel == "cointelegraph" for r in derived)
    # post_id on the scorer-facing record is the segment id.
    assert {r.post_id for r in derived} == {s.segment_id for s in segments}


def test_segment_records_round_trips_through_parquet(tmp_path) -> None:
    records = [
        _record("telegram:1:1", "Short post about Bitcoin."),
        _record("telegram:1:2", _DIGEST),
    ]
    segments = segment_records(records)
    target = tmp_path / "social_post_segments.parquet"
    write_segments(segments, target)
    assert target.exists()


def test_segmenter_separates_anchors_across_sections() -> None:
    """A multi-topic digest must place Trump and 'capital gains tax' in
    separate segments so the linker cannot combine them."""
    text = (
        "Catch up on the news over the last 24 hours!\n\n"
        "* Donald Trump said his administration will protect America's crypto position.\n\n"
        "* Some unrelated section mentions capital gains tax on equity income.\n\n"
        "* Another section talks about 2027 growth in Asia.\n"
    )
    rec = _record("telegram:1:99", text)
    segments = segment_post(rec)
    bodies = [s.segment_text.lower() for s in segments]
    # No single segment combines all three anchors.
    combined = [
        body for body in bodies if "trump" in body and "capital gains" in body and "2027" in body
    ]
    assert combined == []

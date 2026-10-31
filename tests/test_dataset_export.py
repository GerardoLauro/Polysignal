"""Tests for the canonical Traccia_2 §2 dataset export (Prompt 1 Task C)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from polysignal.cli.main import app
from polysignal.dataset.export import (
    COMMENT_COLUMNS,
    CONTRACT_COLUMNS,
    LINK_COLUMNS,
    PRICE_POINT_COLUMNS,
    SOCIAL_POST_COLUMNS,
    export_dataset,
    resolve_best_links,
)
from polysignal.domain import (
    Market,
    MarketDomain,
    Outcome,
    PostComment,
    PricePoint,
    SocialPostRecord,
)
from polysignal.storage.parquet_store import write_parquet
from polysignal.storage.paths import get_data_paths

runner = CliRunner()
ANCHOR = datetime(2026, 1, 1, tzinfo=timezone.utc)

# Section-2 required minimum columns per entity (from Traccia_2 §2).
SECTION2_MIN = {
    "contracts": {"market_id", "question", "category", "outcomes", "resolution_outcome",
                  "resolution_date", "creation_date", "close_date", "url"},
    "price_points": {"market_id", "outcome", "timestamp", "price", "volume"},
    "social_posts": {"post_id", "platform", "market_id", "category", "search_query",
                     "collected_at", "url", "author_id", "author_url", "author_followers",
                     "published_at", "text", "hashtags", "mentioned_entities", "language",
                     "like_count", "reply_count", "repost_count", "view_count"},
    "links": {"market_id", "post_id", "method", "confidence"},
    "comments": {"post_id", "platform", "comment_id", "author_id", "text",
                 "published_at", "like_count", "reply_count"},
}


def _seed(root: Path) -> None:
    paths = get_data_paths(root=root)
    yes = Outcome(market_id="m1", outcome_id="m1-yes", label="Yes", index=0)
    no = Outcome(market_id="m1", outcome_id="m1-no", label="No", index=1)
    market = Market(
        market_id="m1", condition_id="m1", slug="m1", question="Will X happen?",
        domain=MarketDomain.POLITICS, thematic_category="politics",
        created_at=ANCHOR, close_time=ANCHOR, is_resolved=True, outcomes=[yes, no],
        source_url="https://polymarket.com/m1",
    )
    write_parquet([market], paths.polymarket_markets_parquet)
    write_parquet([yes, no], paths.polymarket_outcomes_parquet)
    write_parquet(
        [PricePoint(market_id="m1", outcome_id="m1-yes", timestamp=ANCHOR, price=0.6)],
        paths.polymarket_price_points_parquet,
    )
    write_parquet(
        [SocialPostRecord(post_id="p1", platform="reddit", source_id="p1", text="hello",
                          collected_at=ANCHOR, query="x")],
        paths.social_posts_parquet,
    )
    # post_enrichment + linked_posts via raw DataFrames (mixed schemas)
    pd.DataFrame([{"post_id": "p1", "platform": "reddit", "language": "en",
                   "hashtags": ["#x"], "mentioned_entities": [], "sentiment_label": "neutral",
                   "sentiment_compound": 0.0}]).to_parquet(
        paths.normalized_social / "post_enrichment.parquet", index=False
    )
    pd.DataFrame([{"link_id": "l1", "market_id": "m1", "post_id": "p1", "platform": "reddit",
                   "relevance_score": 0.9, "relevance_label": "RELEVANT", "confidence": 0.8,
                   "method": "deterministic_baseline", "constraint_match": "MATCH",
                   "final_relevance_label": None, "final_relevance_score": None}]).to_parquet(
        paths.linked_posts_parquet, index=False
    )
    write_parquet(
        [PostComment(post_id="p1", platform="reddit", comment_id="c1", text="nice",
                     collected_at=ANCHOR)],
        paths.post_comments_parquet,
    )


def _read_frames(root: Path):
    paths = get_data_paths(root=root)
    from polysignal.storage.parquet_store import read_dataframe
    return dict(
        markets=read_dataframe(paths.polymarket_markets_parquet),
        outcomes=read_dataframe(paths.polymarket_outcomes_parquet),
        prices=read_dataframe(paths.polymarket_price_points_parquet),
        posts=read_dataframe(paths.social_posts_parquet),
        enrichment=read_dataframe(paths.normalized_social / "post_enrichment.parquet"),
        linked=read_dataframe(paths.linked_posts_parquet),
        comments=read_dataframe(paths.post_comments_parquet),
    )


def test_export_columns_superset_of_section2_schema(tmp_path: Path):
    _seed(tmp_path)
    frames = _read_frames(tmp_path)
    out = tmp_path / "export"
    result = export_dataset(output_dir=out, **frames)
    for entity, required in SECTION2_MIN.items():
        df = pd.read_parquet(out / f"{entity}.parquet")
        missing = required - set(df.columns)
        assert not missing, f"{entity} missing §2 columns: {missing}"
    assert result.row_counts["contracts"] == 1
    assert result.row_counts["social_posts"] == 1


def test_export_resolves_linked_market_and_category(tmp_path: Path):
    _seed(tmp_path)
    frames = _read_frames(tmp_path)
    out = tmp_path / "export"
    export_dataset(output_dir=out, **frames)
    sp = pd.read_parquet(out / "social_posts.parquet")
    row = sp.iloc[0]
    assert row["market_id"] == "m1"
    assert row["category"] == "politics"
    assert row["comment_sample_count"] == 1


def test_export_writes_csv_samples_and_dictionary(tmp_path: Path):
    _seed(tmp_path)
    frames = _read_frames(tmp_path)
    out = tmp_path / "export"
    export_dataset(output_dir=out, sample_rows=5, **frames)
    for entity in ("contracts", "price_points", "social_posts", "links", "comments"):
        assert (out / f"{entity}_sample.csv").exists()
    dictionary = (out / "DATA_DICTIONARY.md").read_text(encoding="utf-8")
    assert "data dictionary" in dictionary.lower()
    for col in CONTRACT_COLUMNS:
        assert f"`{col}`" in dictionary


def test_resolve_best_links_prefers_accepted_high_score():
    linked = pd.DataFrame([
        {"post_id": "p", "market_id": "m-low", "relevance_label": "RELEVANT",
         "relevance_score": 0.4, "final_relevance_label": None, "final_relevance_score": None},
        {"post_id": "p", "market_id": "m-high", "relevance_label": "RELEVANT",
         "relevance_score": 0.95, "final_relevance_label": None, "final_relevance_score": None},
        {"post_id": "p", "market_id": "m-irrel", "relevance_label": "NOT_RELEVANT",
         "relevance_score": 0.99, "final_relevance_label": None, "final_relevance_score": None},
    ])
    best = resolve_best_links(linked)
    assert len(best) == 1
    assert best.iloc[0]["market_id"] == "m-high"


def test_cli_export_dry_run_writes_nothing(tmp_path: Path):
    _seed(tmp_path)
    result = runner.invoke(
        app, ["dataset", "export", "--output-dir", str(tmp_path), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output.lower()
    assert not (tmp_path / "dataset_export").exists()


def test_cli_export_writes_bundle(tmp_path: Path):
    _seed(tmp_path)
    result = runner.invoke(
        app, ["dataset", "export", "--output-dir", str(tmp_path), "--sample-rows", "10"]
    )
    assert result.exit_code == 0, result.output
    export_dir = tmp_path / "dataset_export"
    for entity in ("contracts", "price_points", "social_posts", "links", "comments"):
        assert (export_dir / f"{entity}.parquet").exists()
    assert (export_dir / "DATA_DICTIONARY.md").exists()


def test_export_partitioned_layout(tmp_path: Path):
    _seed(tmp_path)
    frames = _read_frames(tmp_path)
    out = tmp_path / "export"
    result = export_dataset(output_dir=out, partition=True, **frames)
    # social_posts partitioned by platform -> a directory with platform=… subdirs
    assert "social_posts" in result.partitioned
    social_dir = out / "social_posts"
    assert social_dir.is_dir()
    assert any("platform=" in p.name for p in social_dir.iterdir())


# Keep the canonical column tuples referenced so they are import-checked.
assert SOCIAL_POST_COLUMNS and PRICE_POINT_COLUMNS and LINK_COLUMNS and COMMENT_COLUMNS

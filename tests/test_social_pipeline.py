"""Integration tests for the social ingestion pipeline (Branch 04).

The HTTP layer is faked through a stub :class:`SocialIngestor`; the
pipeline runs the real query generator, raw store, normalizer, dedup,
and Parquet writer against ``tmp_path`` data root. No network.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from polysignal.domain import (
    Engagement,
    Market,
    MarketDomain,
    Outcome,
    Platform,
    SocialPost,
    SocialPostRecord,
)
from polysignal.ingestion.social.base import SearchQuery, SocialIngestor
from polysignal.ingestion.social.pipeline import (
    load_markets_from_parquet,
    run_social_ingestion,
)
from polysignal.storage.parquet_store import read_parquet, write_parquet
from polysignal.storage.paths import get_data_paths

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeIngestor(SocialIngestor):
    platform = Platform.REDDIT

    def __init__(
        self,
        posts_per_query: dict[str, list[SocialPost]] | None = None,
        errors: dict[str, str] | None = None,
        available: bool = True,
    ) -> None:
        self._posts = posts_per_query or {}
        self._errors = errors or {}
        self._available = available
        self.calls: list[SearchQuery] = []

    def is_available(self) -> bool:
        return self._available

    def search(self, query: SearchQuery) -> Iterable[SocialPost]:
        self.calls.append(query)
        if query.text in self._errors:
            raise RuntimeError(self._errors[query.text])
        return list(self._posts.get(query.text, []))

    def search_raw(self, query: SearchQuery):
        self.calls.append(query)
        if query.text in self._errors:
            raise RuntimeError(self._errors[query.text])
        posts = list(self._posts.get(query.text, []))
        raws = [
            {"id": post.post_id, "title": post.extra.get("title", ""), "query": query.text}
            for post in posts
        ]
        return posts, raws


def _post(
    post_id: str, title: str = "T", text: str = "Body", subreddit: str = "test"
) -> SocialPost:
    return SocialPost(
        post_id=post_id,
        platform=Platform.REDDIT,
        author_id="someone",
        author_handle="someone",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        text=f"{title}\n\n{text}",
        url=f"https://www.reddit.com/r/{subreddit}/comments/{post_id}/",
        engagement=Engagement(score=5, replies=2),
        extra={
            "title": title,
            "selftext": text,
            "subreddit": subreddit,
            "num_comments": 2,
            "score": 5,
            "over_18": False,
        },
    )


def _market(market_id: str, question: str, domain: MarketDomain = MarketDomain.POLITICS) -> Market:
    return Market(
        market_id=market_id,
        slug=f"slug-{market_id}",
        question=question,
        domain=domain,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        outcomes=[
            Outcome(outcome_id=f"{market_id}-yes", label="Yes", index=0),
            Outcome(outcome_id=f"{market_id}-no", label="No", index=1),
        ],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_writes_parquet_and_raw_layout(tmp_path: Path) -> None:
    market = _market("cond-1", "Will Trump be out as President before GTA VI?")
    posts_by_query = {
        # The generator emits these queries (verified manually above);
        # only the ones we want to mock need entries here.
        "Trump be out as President": [_post("p1"), _post("p2")],
        "Trump out President": [_post("p3")],
    }
    ingestor = _FakeIngestor(posts_per_query=posts_by_query)
    paths = get_data_paths(root=tmp_path)
    result = run_social_ingestion(
        ingestor=ingestor,
        markets=[market],
        limit_markets=5,
        queries_per_market=6,
        posts_per_query=10,
        save_raw=True,
        paths=paths,
        now=datetime(2026, 5, 31, 12, tzinfo=timezone.utc),
    )

    assert result.markets_seen == 1
    assert result.queries_generated > 0
    assert result.queries_executed == result.queries_generated
    assert result.posts_collected == 3  # 2 + 1
    assert result.unique_posts == 3
    assert result.duplicates == 0

    # Parquet outputs exist and round-trip back to SocialPostRecord.
    assert result.social_posts_parquet is not None
    assert result.social_posts_parquet.exists()
    records = read_parquet(result.social_posts_parquet, SocialPostRecord)
    assert {r.post_id for r in records} == {"p1", "p2", "p3"}
    for r in records:
        assert r.market_id_candidate == "cond-1"
        assert r.platform is Platform.REDDIT
        assert r.collected_at.tzinfo is not None

    # Queries parquet exists and reflects all generated queries.
    assert result.queries_parquet is not None and result.queries_parquet.exists()
    queries_df = pd.read_parquet(result.queries_parquet)
    assert len(queries_df) == result.queries_generated

    # Raw payload directory is created with a manifest.
    assert result.raw_dir is not None and result.raw_dir.exists()
    assert (result.raw_dir / "_manifest.json").exists()
    manifest = json.loads((result.raw_dir / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["platform"] == "reddit"
    assert manifest["object_count"] == 3


def test_pipeline_dedups_when_same_post_comes_from_multiple_queries(tmp_path: Path) -> None:
    market = _market("cond-1", "Will Trump be out as President before GTA VI?")
    same_post = _post("dup")
    posts_by_query = {
        "Trump be out as President": [same_post],
        "Trump out President": [same_post],
        "Trump out President GTA VI": [same_post],
    }
    ingestor = _FakeIngestor(posts_per_query=posts_by_query)
    paths = get_data_paths(root=tmp_path)
    result = run_social_ingestion(
        ingestor=ingestor,
        markets=[market],
        limit_markets=5,
        queries_per_market=6,
        posts_per_query=10,
        save_raw=False,
        paths=paths,
        now=datetime(2026, 5, 31, 12, tzinfo=timezone.utc),
    )
    assert result.posts_collected == 3
    # One unique post, two are flagged as duplicates.
    assert result.unique_posts == 1
    assert result.duplicates == 2
    # The unique row references the *first* query that surfaced it.
    records = read_parquet(result.social_posts_parquet, SocialPostRecord)
    unique = [r for r in records if not r.is_duplicate]
    duplicates = [r for r in records if r.is_duplicate]
    assert len(unique) == 1
    assert len(duplicates) == 2
    # Duplicates retain their own retrieval context.
    assert {r.query for r in duplicates} != {unique[0].query}


def test_pipeline_idempotent_repeat_run(tmp_path: Path) -> None:
    """Re-running the same ingestion must NOT grow social_posts.parquet."""
    market = _market("cond-1", "Will Trump be out as President before GTA VI?")
    same_post = _post("dup")
    paths = get_data_paths(root=tmp_path)
    fake_posts = {"Trump be out as President": [same_post]}

    # Run once → seed the parquet file.
    first = run_social_ingestion(
        ingestor=_FakeIngestor(posts_per_query=fake_posts),
        markets=[market],
        limit_markets=1,
        queries_per_market=2,
        posts_per_query=2,
        save_raw=False,
        paths=paths,
        now=datetime(2026, 5, 31, 12, tzinfo=timezone.utc),
    )
    rows_after_first = read_parquet(first.social_posts_parquet, SocialPostRecord)

    # Run again with the *exact same* retrieval tuple. The exact tuple
    # must be skipped, not appended as a duplicate row.
    second = run_social_ingestion(
        ingestor=_FakeIngestor(posts_per_query=fake_posts),
        markets=[market],
        limit_markets=1,
        queries_per_market=2,
        posts_per_query=2,
        save_raw=False,
        paths=paths,
        now=datetime(2026, 5, 31, 13, tzinfo=timezone.utc),
    )
    rows_after_second = read_parquet(second.social_posts_parquet, SocialPostRecord)
    assert len(rows_after_second) == len(rows_after_first), (
        "exact (post, query, market) repeat must be skipped, not appended"
    )


def test_pipeline_appends_when_same_post_seen_via_new_query(tmp_path: Path) -> None:
    """Same post via a *different* query must still be appended (traceability)."""
    market = _market("cond-1", "Will Trump be out as President before GTA VI?")
    same_post = _post("dup")
    paths = get_data_paths(root=tmp_path)

    # First run: post surfaced by query A.
    run_social_ingestion(
        ingestor=_FakeIngestor(posts_per_query={"Trump be out as President": [same_post]}),
        markets=[market],
        limit_markets=1,
        queries_per_market=2,
        posts_per_query=2,
        save_raw=False,
        paths=paths,
        now=datetime(2026, 5, 31, 12, tzinfo=timezone.utc),
    )
    # Second run: same post surfaced by query B (the entity variant).
    second = run_social_ingestion(
        ingestor=_FakeIngestor(posts_per_query={"Trump out President": [same_post]}),
        markets=[market],
        limit_markets=1,
        queries_per_market=2,
        posts_per_query=2,
        save_raw=False,
        paths=paths,
        now=datetime(2026, 5, 31, 13, tzinfo=timezone.utc),
    )
    records = read_parquet(second.social_posts_parquet, SocialPostRecord)
    unique = [r for r in records if not r.is_duplicate]
    duplicates = [r for r in records if r.is_duplicate]
    # One unique post (first encounter), plus one duplicate row for the
    # second retrieval. Both rows carry their own query.
    assert len(unique) == 1
    assert len(duplicates) == 1
    assert {r.query for r in records} == {
        "Trump be out as President",
        "Trump out President",
    }


def test_pipeline_dry_run_persists_queries_but_skips_api(tmp_path: Path) -> None:
    """Dry-run must not call the ingestor — only generate + persist queries."""
    market = _market("cond-1", "Will the Lakers win the NBA Finals?")
    ingestor = _FakeIngestor(
        posts_per_query={
            "the Lakers win the NBA Finals": [_post("p1")],
            "Lakers win NBA Finals": [_post("p2")],
        }
    )
    paths = get_data_paths(root=tmp_path)
    result = run_social_ingestion(
        ingestor=ingestor,
        markets=[market],
        limit_markets=1,
        queries_per_market=3,
        posts_per_query=5,
        save_raw=False,
        dry_run=True,
        paths=paths,
        now=datetime(2026, 5, 31, 12, tzinfo=timezone.utc),
    )
    # Dry-run must NOT have called the ingestor.
    assert ingestor.calls == []
    assert result.posts_collected == 0
    assert result.social_posts_parquet is None
    assert not paths.social_posts_parquet.exists()
    # …but queries are persisted regardless, so the plan is auditable.
    assert result.queries_parquet is not None
    assert paths.social_queries_parquet.exists()
    assert result.queries_generated > 0


def test_pipeline_dry_run_does_not_require_credentials(tmp_path: Path) -> None:
    """A dry-run with an unavailable ingestor must succeed (no API contact)."""
    market = _market("cond-1", "Will the Lakers win the NBA Finals?")
    paths = get_data_paths(root=tmp_path)
    unavailable = _FakeIngestor(available=False)
    result = run_social_ingestion(
        ingestor=unavailable,
        markets=[market],
        limit_markets=1,
        queries_per_market=3,
        posts_per_query=5,
        save_raw=False,
        dry_run=True,
        paths=paths,
        now=datetime(2026, 5, 31, 12, tzinfo=timezone.utc),
    )
    assert unavailable.calls == []
    assert result.queries_generated > 0
    assert paths.social_queries_parquet.exists()


def test_pipeline_records_per_query_errors_without_failing_run(tmp_path: Path) -> None:
    market = _market("cond-1", "Will Trump be out as President before GTA VI?")
    ingestor = _FakeIngestor(
        posts_per_query={"Trump out President": [_post("p1")]},
        errors={"Trump be out as President": "HTTP 503: server error"},
    )
    paths = get_data_paths(root=tmp_path)
    result = run_social_ingestion(
        ingestor=ingestor,
        markets=[market],
        limit_markets=1,
        queries_per_market=3,
        posts_per_query=5,
        save_raw=True,
        paths=paths,
        now=datetime(2026, 5, 31, 12, tzinfo=timezone.utc),
    )
    failing = [a for a in result.attempts if a.error]
    assert failing, "the failing query must be recorded as an attempt"
    assert any("503" in (a.error or "") for a in failing)
    # Successful queries still produced posts.
    assert result.posts_collected == 1


def test_pipeline_refuses_when_ingestor_is_unavailable(tmp_path: Path) -> None:
    market = _market("cond-1", "Will Trump be out as President?")
    ingestor = _FakeIngestor(available=False)
    paths = get_data_paths(root=tmp_path)
    with pytest.raises(RuntimeError) as exc_info:
        run_social_ingestion(
            ingestor=ingestor,
            markets=[market],
            limit_markets=1,
            queries_per_market=2,
            posts_per_query=2,
            paths=paths,
        )
    assert "not available" in str(exc_info.value).lower()


def test_load_markets_from_parquet_reads_branch03_output(tmp_path: Path) -> None:
    market = _market("cond-1", "Will Trump be out as President?", MarketDomain.POLITICS)
    paths = get_data_paths(root=tmp_path)
    paths.ensure()
    write_parquet([market], paths.polymarket_markets_parquet)
    out = load_markets_from_parquet(paths.polymarket_markets_parquet)
    assert len(out) == 1
    assert out[0].market_id == "cond-1"


def test_load_markets_raises_when_file_missing(tmp_path: Path) -> None:
    paths = get_data_paths(root=tmp_path)
    with pytest.raises(FileNotFoundError):
        load_markets_from_parquet(paths.polymarket_markets_parquet)


# ---------------------------------------------------------------------------
# Fix 6 — social_queries.parquet must be persisted even when zero posts were found.
# ---------------------------------------------------------------------------


def test_pipeline_persists_queries_even_when_search_returns_zero_posts(
    tmp_path: Path,
) -> None:
    market = _market("cond-1", "Will Trump be out as President before GTA VI?")
    paths = get_data_paths(root=tmp_path)
    # No mocked posts at all — every search returns an empty list.
    empty_ingestor = _FakeIngestor(posts_per_query={})
    result = run_social_ingestion(
        ingestor=empty_ingestor,
        markets=[market],
        limit_markets=1,
        queries_per_market=4,
        posts_per_query=10,
        save_raw=False,
        paths=paths,
        now=datetime(2026, 5, 31, 12, tzinfo=timezone.utc),
    )
    assert result.posts_collected == 0
    assert result.queries_generated > 0
    # Queries are still written to disk — they are the audit log of "what
    # was the system going to ask Reddit on this run".
    assert result.queries_parquet is not None
    assert paths.social_queries_parquet.exists()
    queries_df = pd.read_parquet(paths.social_queries_parquet)
    assert len(queries_df) == result.queries_generated
    expected_columns = {
        "market_id",
        "query",
        "query_type",
        "rank",
        "source_fields",
        "generated_at",
    }
    assert expected_columns.issubset(set(queries_df.columns))


def test_pipeline_cli_dry_run(tmp_path: Path, monkeypatch) -> None:
    """The CLI ``social ingest --dry-run`` must work end-to-end with a fake."""
    from typer.testing import CliRunner

    from polysignal.cli.main import _SOCIAL_INGESTORS, app

    market = _market("cond-1", "Will the Lakers win the NBA Finals?", MarketDomain.SPORTS)
    paths = get_data_paths(root=tmp_path)
    paths.ensure()
    write_parquet([market], paths.polymarket_markets_parquet)

    fake = _FakeIngestor(
        posts_per_query={"the Lakers win the NBA Finals": [_post("p1")]},
    )

    class _Factory:
        def __call__(self, *args, **kwargs):
            return fake

    monkeypatch.setitem(_SOCIAL_INGESTORS, "reddit", _Factory())

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "social",
            "ingest",
            "--platform",
            "reddit",
            "--limit-markets",
            "1",
            "--queries-per-market",
            "2",
            "--posts-per-query",
            "5",
            "--no-save-raw",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "queries generated" in result.output
    # Dry-run does not create the parquet file.
    assert not paths.social_posts_parquet.exists()

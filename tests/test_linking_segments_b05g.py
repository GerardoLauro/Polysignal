"""Branch 05G end-to-end tests: segment-aware linking + phrase gate."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from polysignal.cli.main import app
from polysignal.config import get_settings
from polysignal.domain import (
    Market,
    MarketDomain,
    Outcome,
    Platform,
    RelevanceLabel,
    SocialPost,
    SocialPostRecord,
)
from polysignal.ingestion.social.segmentation import (
    segment_models_to_records,
    segment_post,
    to_segment_models,
)
from polysignal.linking import (
    RelevanceScorer,
    run_linking_cross_product,
    score_pairs,
)
from polysignal.linking.candidates import CandidatePair
from polysignal.linking.linking_pipeline import write_linked_posts

_NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _mk_post(post_id: str, text: str) -> SocialPost:
    return SocialPost(
        post_id=post_id,
        platform=Platform.TELEGRAM,
        author_id="u",
        author_handle="h",
        created_at=_NOW,
        text=text,
    )


def _mk_market(market_id: str, slug: str, question: str, domain: MarketDomain) -> Market:
    return Market(
        market_id=market_id,
        slug=slug,
        question=question,
        domain=domain,
        created_at=_NOW,
        outcomes=[
            Outcome(outcome_id=f"{market_id}-yes", label="Yes", index=0),
            Outcome(outcome_id=f"{market_id}-no", label="No", index=1),
        ],
    )


def test_digest_trump_cryptotax_false_positive_is_not_relevant() -> None:
    """A multi-topic digest that mentions Trump in one section, crypto in
    another, and 2027 in a third must NOT score as ``relevant`` against
    the 'Trump eliminates capital gains tax on crypto before 2027' market
    once segmentation + phrase gate are in place."""
    market = _mk_market(
        "m-trump-tax",
        "will-trump-eliminate-capital-gains-tax-on-crypto-before-2027",
        "Will Trump eliminate capital gains tax on crypto before 2027?",
        MarketDomain.POLITICS,
    )
    digest = (
        "Catch up on the news over the last 24 hours!\n\n"
        "* Donald Trump said his administration will protect America's crypto position.\n\n"
        "* Unrelated section about housing market and capital gains tax on equity income.\n\n"
        "* Another section about 2027 growth in Asia.\n\n"
        "* Aave restored WETH LTVs after the rsETH technical recovery.\n"
    )
    post = _mk_post("telegram:c:1", digest)
    record = SocialPostRecord(
        post_id=post.post_id,
        platform=Platform.TELEGRAM,
        source_id=post.post_id,
        url=None,
        author_id="u",
        author_name="h",
        created_at=_NOW,
        text=digest,
        title=None,
        subreddit_or_channel=None,
        query=None,
        query_type=None,
        market_id_candidate=None,
        raw_path=None,
        collected_at=_NOW,
        is_duplicate=False,
        text_length=len(digest),
    )
    segments = segment_post(record)
    # Each segment should be scored independently.
    derived = segment_models_to_records(to_segment_models(segments), {record.post_id: record})
    scorer = RelevanceScorer()
    for derived_record in derived:
        seg_post = _mk_post(derived_record.post_id, derived_record.text)
        verdict = scorer.score(seg_post, market)
        assert verdict.relevance_label != RelevanceLabel.RELEVANT, (
            f"segment {derived_record.post_id} promoted to relevant: {verdict.explanation}"
        )


def test_kraken_ipo_requires_phrase_in_segment() -> None:
    market = _mk_market(
        "m-kraken-ipo",
        "will-kraken-ipo-by-december-31-2026",
        "Will Kraken IPO by December 31, 2026?",
        MarketDomain.ECONOMY,
    )
    # SpaceX IPO segment — no Kraken anywhere.
    post = _mk_post(
        "telegram:c:2",
        "SpaceX has officially filed for an IPO with the SEC under the ticker $SPCX.",
    )
    verdict = RelevanceScorer().score(post, market)
    assert verdict.relevance_label != RelevanceLabel.RELEVANT


def test_stanley_cup_phrase_required() -> None:
    market = _mk_market(
        "m-stanley",
        "will-the-edmonton-oilers-win-the-stanley-cup",
        "Will the Edmonton Oilers win the Stanley Cup?",
        MarketDomain.SPORTS,
    )
    post = _mk_post(
        "p-tucci",
        "Stanley Tucci stars in a charming new film about cooking and family.",
    )
    verdict = RelevanceScorer().score(post, market)
    assert verdict.relevance_label != RelevanceLabel.RELEVANT


def test_gta_vi_alias_still_works() -> None:
    from polysignal.linking import fixture_markets, fixture_posts

    posts = {p.post_id: p for p in fixture_posts()}
    markets = {m.market_id: m for m in fixture_markets()}
    rows = {
        (r.market_id, r.post_id): r
        for r in run_linking_cross_product(list(posts.values()), list(markets.values()))
    }
    # gold: gta-2026 + p-gta-relevant ("Rockstar may delay GTA VI until 2026.")
    verdict = rows[("m-gta-2026", "p-gta-relevant")]
    assert verdict.relevance_label == RelevanceLabel.RELEVANT


def test_year_only_post_remains_unrelated() -> None:
    market = _mk_market(
        "m-china",
        "will-china-invade-taiwan-before-2027",
        "Will China invade Taiwan before 2027?",
        MarketDomain.POLITICS,
    )
    post = _mk_post("p-year", "Big things expected in 2027 across global markets.")
    verdict = RelevanceScorer().score(post, market)
    assert verdict.relevance_label == RelevanceLabel.UNRELATED


def test_score_pairs_passes_segment_metadata() -> None:
    market = _mk_market("m1", "will-x-do-y", "Will X do Y?", MarketDomain.OTHER)
    seg_post = _mk_post("telegram:c:1#seg002", "Bitcoin price action looks strong in late 2026.")
    pair = CandidatePair(market=market, post=seg_post)
    rows = score_pairs(
        [pair],
        segment_lookup={"telegram:c:1#seg002": ("telegram:c:1", 2, "preview text")},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.segment_id == "telegram:c:1#seg002"
    assert row.parent_post_id == "telegram:c:1"
    assert row.segment_index == 2
    assert row.segment_text_preview == "preview text"


def test_cli_export_manual_review(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("POLYSIGNAL_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        runner = CliRunner()
        # Seed a tiny linked_posts.parquet via run_linking_cross_product +
        # write_linked_posts.
        from polysignal.linking import fixture_markets, fixture_posts

        rows = run_linking_cross_product(fixture_posts(), fixture_markets())
        links_path = tmp_path / "linked.parquet"
        write_linked_posts(rows, links_path)
        out_csv = tmp_path / "manual_review.csv"
        result = runner.invoke(
            app,
            [
                "linking",
                "export-manual-review",
                "--links-path",
                str(links_path),
                "--output-path",
                str(out_csv),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_csv.exists()
        header = out_csv.read_text(encoding="utf-8-sig").splitlines()[0]
        for col in (
            "market_id",
            "segment_id",
            "parent_post_id",
            "human_relevance_label",
            "review_status",
        ):
            assert col in header
    finally:
        get_settings.cache_clear()


def test_cli_segment_posts_writes_segments_parquet(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("POLYSIGNAL_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        from polysignal.storage.parquet_store import write_parquet

        records = [
            SocialPostRecord(
                post_id=f"telegram:c:{i}",
                platform=Platform.TELEGRAM,
                source_id=f"telegram:c:{i}",
                url=None,
                author_id="u",
                author_name="h",
                created_at=_NOW,
                text="Bitcoin hit $150k today."
                if i == 0
                else (
                    "Catch up on the news!\n\n"
                    "* Trump in news today.\n\n"
                    "* Bitcoin volatile.\n\n"
                    "* Fed signals rate cut.\n\n"
                    "* Kraken IPO speculation grows.\n"
                ),
                title=None,
                subreddit_or_channel="cointelegraph",
                query=None,
                query_type="telegram_channel",
                market_id_candidate=None,
                raw_path=None,
                collected_at=_NOW,
                is_duplicate=False,
                text_length=80,
            )
            for i in range(2)
        ]
        posts_path = tmp_path / "social_posts.parquet"
        write_parquet(records, posts_path)
        out_path = tmp_path / "segments.parquet"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "social",
                "segment-posts",
                "--posts-path",
                str(posts_path),
                "--output-path",
                str(out_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_path.exists()
    finally:
        get_settings.cache_clear()


def test_full_scan_mode_still_available() -> None:
    """Regression: full-scan mode must remain accessible for debugging."""
    from polysignal.linking.candidates import (
        CandidateMode,
        build_pairs_from_records_with_diagnostics,
    )

    record = SocialPostRecord(
        post_id="p1",
        platform=Platform.TELEGRAM,
        source_id="p1",
        url=None,
        author_id="u",
        author_name="h",
        created_at=_NOW,
        text="Bitcoin update.",
        title=None,
        subreddit_or_channel="c",
        query=None,
        query_type="t",
        market_id_candidate=None,
        raw_path=None,
        collected_at=_NOW,
        is_duplicate=False,
        text_length=15,
    )
    markets = [
        _mk_market("m1", "will-bitcoin-hit-150k", "Will Bitcoin hit $150k?", MarketDomain.CRYPTO),
        _mk_market("m2", "will-apple-launch", "Will Apple launch?", MarketDomain.CULTURE),
    ]
    result = build_pairs_from_records_with_diagnostics(
        [record], markets, candidate_mode=CandidateMode.FULL_SCAN
    )
    # full-scan: every post is paired with every market.
    assert result.diagnostics.candidate_mode == CandidateMode.FULL_SCAN
    assert len(result.pairs) == 2

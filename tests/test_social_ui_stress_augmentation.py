from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from polysignal.cli.main import app
from polysignal.domain import Market, Outcome, SocialPostRecord
from polysignal.domain.enums import MarketDomain, Platform
from polysignal.ingestion.social.ui_stress_augmentation import (
    DEFAULT_AUGMENTATION_VERSION,
    SELECTED_MARKETS,
    augment_main_social_dataset,
    augmentation_marker,
    generate_augmented_rows,
    load_selected_markets,
)
from polysignal.linking import CandidateMode, run_linking_from_records
from polysignal.stance import run_stance_pipeline
from polysignal.storage.parquet_store import models_to_dataframe, read_parquet, write_parquet
from polysignal.storage.paths import get_data_paths

UTC = timezone.utc

_QUESTION_BY_ID = {
    "0x6874f3bbffd0ff156b09768d1e1a5bdfaac48a7aaee9fbe1e883dccdc5f3d68d": (
        "Bitcoin all time high in Q4?"
    ),
    "0x2dae146105569fc37e2c7365f540fd7028b3c1a42fcc1fca7f0dbd9902624e15": (
        "Will another party win the popular vote in Canada's next election?"
    ),
    "0x0851b31ba4f1c35e13c1ca100be710a6cc68bb5c958c37a967e65f03dd828beb": (
        "Will the Milwaukee Bucks make the NBA Playoffs?"
    ),
    "0x77825ad3698fb528b217de4da17a61660be6a1678425679f017a9cf4d7a4ddb2": (
        "Will Elon Musk post 40-59 tweets from December 16 to December 23, 2025?"
    ),
    "0x0f8f5962a2f9b62abe4d6d88de65a6327a44b749fbecd89a780fcd463824b7f1": (
        "Will 'Bring Her Back' gross between $3.5-4.5m opening weekend?"
    ),
    "0xb943884a1226dad793d08f5143b8c13db715a1e40f846ce80e06499654dfcb1e": (
        "Will the National Congress for Liberty win the most seats in the 2025 "
        "Burundian parliamentary election?"
    ),
}


def _seed_selected_markets(root: Path) -> list[Market]:
    paths = get_data_paths(root=root)
    paths.ensure()
    markets: list[Market] = []
    outcomes: list[Outcome] = []
    for idx, (domain_label, market_id) in enumerate(SELECTED_MARKETS):
        yes = Outcome(market_id=market_id, outcome_id=f"{market_id}-yes", label="Yes", index=0)
        no = Outcome(market_id=market_id, outcome_id=f"{market_id}-no", label="No", index=1)
        outcomes.extend([yes, no])
        close = datetime(2025, 6, 1 + idx, 12, tzinfo=UTC)
        markets.append(
            Market(
                market_id=market_id,
                slug=f"selected-{idx}",
                question=_QUESTION_BY_ID[market_id],
                domain=_domain(domain_label),
                created_at=datetime(2025, 1, 1, tzinfo=UTC),
                close_time=close,
                outcomes=[yes, no],
            )
        )
    write_parquet(markets, paths.polymarket_markets_parquet)
    write_parquet(outcomes, paths.polymarket_outcomes_parquet)
    return markets


def _domain(label: str) -> MarketDomain:
    if "crypto" in label:
        return MarketDomain.CRYPTO
    if "politics" in label:
        return MarketDomain.POLITICS
    if label == "sports":
        return MarketDomain.SPORTS
    if "culture" in label:
        return MarketDomain.CULTURE
    return MarketDomain.OTHER


def test_market_selection_uses_exact_saved_ids(tmp_path: Path) -> None:
    _seed_selected_markets(tmp_path)
    selected = load_selected_markets(get_data_paths(root=tmp_path))
    assert [market.market_id for _, market in selected] == [market_id for _, market_id in SELECTED_MARKETS]


def test_generated_rows_use_native_ids_and_realistic_distribution(tmp_path: Path) -> None:
    _seed_selected_markets(tmp_path)
    selected = load_selected_markets(get_data_paths(root=tmp_path))
    rows = generate_augmented_rows(selected, messages_per_market=150)

    assert len(rows) == 900
    assert all(not row.post_id.startswith("ui_test_") for row in rows)
    assert all(row.source_id == row.post_id for row in rows)
    assert all(
        re.fullmatch(r"t3_[0-9a-z]+", row.post_id)
        or re.fullmatch(r"telegram:\d+:\d+", row.post_id)
        for row in rows
    )
    assert {row.platform for row in rows} == {Platform.REDDIT, Platform.TELEGRAM}
    assert all(augmentation_marker(DEFAULT_AUGMENTATION_VERSION) in (row.query_provenance or "") for row in rows)

    for _, market in selected:
        market_rows = [row for row in rows if row.market_id_candidate == market.market_id]
        counts = {label: sum(row.retrieval_window_label == label for row in market_rows) for label in {
            "supports_yes",
            "supports_no",
            "neutral",
            "mixed",
        }}
        assert len(market_rows) == 150
        assert counts["supports_yes"] + counts["supports_no"] > counts["neutral"] + counts["mixed"]
        assert counts["supports_yes"] != 60 or counts["supports_no"] != 60
        assert all(row.created_at is not None and row.created_at < market.close_time for row in market_rows)


def test_append_replace_preserves_real_rows_and_writes_csv(tmp_path: Path) -> None:
    _seed_selected_markets(tmp_path)
    paths = get_data_paths(root=tmp_path)
    real = SocialPostRecord(
        post_id="t3_realrow",
        platform=Platform.REDDIT,
        source_id="t3_realrow",
        created_at=datetime(2025, 1, 5, tzinfo=UTC),
        text="Real row about unrelated news",
        subreddit_or_channel="worldnews",
        collected_at=datetime(2025, 1, 5, tzinfo=UTC),
        text_length=29,
    )
    csv_only = real.model_copy(update={"post_id": "telegram:123456:987", "platform": Platform.TELEGRAM, "source_id": "telegram:123456:987"})
    write_parquet([real], paths.social_posts_parquet)
    models_to_dataframe([real, csv_only]).to_csv(paths.social_posts_parquet.with_suffix(".csv"), index=False)

    first = augment_main_social_dataset(root=tmp_path, messages_per_market=25)
    second = augment_main_social_dataset(root=tmp_path, messages_per_market=25)
    rows = read_parquet(paths.social_posts_parquet, SocialPostRecord)

    assert first.final_rows == second.final_rows == 152
    assert sum(row.post_id == "t3_realrow" for row in rows) == 1
    assert sum(row.post_id == "telegram:123456:987" for row in rows) == 1
    assert sum(augmentation_marker(DEFAULT_AUGMENTATION_VERSION) in (row.query_provenance or "") for row in rows) == 150
    assert paths.social_posts_parquet.with_suffix(".csv").exists()
    assert not (tmp_path / "data_demo_fixture_eval").exists()


def test_dry_run_cli_does_not_write(tmp_path: Path) -> None:
    _seed_selected_markets(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "social",
            "augment-main-for-ui-test",
            "--root",
            str(tmp_path),
            "--messages-per-market",
            "25",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "generated rows" in result.output
    assert not get_data_paths(root=tmp_path).social_posts_parquet.exists()


def test_generated_rows_flow_through_linking_and_stance(tmp_path: Path) -> None:
    markets = _seed_selected_markets(tmp_path)
    paths = get_data_paths(root=tmp_path)
    selected = load_selected_markets(paths)
    rows = generate_augmented_rows(selected, messages_per_market=25)
    links = run_linking_from_records(
        rows,
        markets,
        candidate_mode=CandidateMode.PROVENANCE,
        include_duplicates=True,
    )
    assert links
    write_parquet(rows, paths.social_posts_parquet)
    write_parquet(links, paths.linked_posts_parquet)
    result = run_stance_pipeline(paths=paths, min_link_confidence=0.45, dry_run=True)
    assert result.rows_scored > 0

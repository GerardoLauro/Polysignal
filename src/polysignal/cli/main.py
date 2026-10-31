"""Typer entry point that orchestrates the PolySignal modules.

The CLI is organized as nested Typer apps, mirroring the conceptual
modules:

    polysignal polymarket fetch-markets
    polysignal polymarket inspect-market
    polysignal polymarket fetch-prices
    polysignal polymarket ingest
    polysignal social ingest
    polysignal nlp enrich
    polysignal graph init
    polysignal graph load
    polysignal evaluate backtest
    polysignal refresh status

Most subcommands are wired to the existing module interfaces. Where
the underlying implementation is still a stub, the CLI prints an
informative TODO message and exits with **code 2**, so that a script
consumer cannot mistake a placeholder for a real successful run. The
Polymarket commands above are **real** (no stub) — see
``docs/MODULE_STATUS.md`` for which modules are still skeletons.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections.abc import Sequence
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from polysignal import __version__
from polysignal.config import get_settings
from polysignal.dataset import (
    DATASET_TARGETS,
    HighSignalMarket,
    TelegramSearchPlanEntry,
    build_coverage_report,
    build_manifest,
    build_telegram_search_plan,
    export_dataset,
    select_high_signal_markets,
    write_coverage_report,
    write_high_signal_markets,
    write_manifest,
    write_telegram_search_plan,
)
from polysignal.domain import (
    LinkedPost,
    Market,
    Outcome,
    SocialPostRecord,
    SocialPostSegment,
    StanceLabel,
)
from polysignal.evaluation import (
    BacktestConfig,
    assess_post_admissibility,
    benchmark_profiles,
    build_dashboard_prediction_view,
    build_evaluation_dataset,
    build_resolved_labels,
    build_usefulness_report,
    resolve_backtest_profile_paths,
    run_backtest,
    write_backtest_artifacts,
    write_dashboard_prediction_view,
    write_evaluation_dataset_artifacts,
    write_resolved_labels,
)
from polysignal.features import (
    SPIConfig,
    SPIResult,
    compare_with_expected,
    compute_social_pressure,
    read_social_pressure_features,
    write_social_pressure_features,
)
from polysignal.graph import queries as graph_queries
from polysignal.graph.client import Neo4jClient
from polysignal.graph.loaders import (
    ACCEPTED_WEAK_LINK_MIN_CONFIDENCE,
    GraphLoadPlan,
    build_load_plan,
    execute_load_plan,
)
from polysignal.ingestion.polymarket import (
    IngestionResult,
    PolymarketAPIError,
    PolymarketClient,
    classify_market,
    fetch_prices_for_market,
    ingest_curated_markets,
    normalize_market,
    run_market_ingestion,
)
from polysignal.ingestion.social import (
    LargeEnrichmentOptions,
    RedditIngestor,
    SocialIngestionResult,
    SocialIngestor,
    TelegramChannel,
    TelegramIngestor,
    TelegramLargeOptions,
    assemble_artifact_index,
    augment_main_social_dataset,
    collect_large_enriched,
    collect_reddit_scrapling,
    collect_telegram_large,
    collect_telegram_public_web,
    estimate_market_token_overlap,
    fetch_telegram_channels,
    generate_queries,
    latest_telegram_raw_directory,
    load_enrichment_config,
    load_markets_from_parquet,
    load_telegram_channels,
    merge_enriched_datasets,
    normalize_telegram_raw_directory,
    plan_augmentation,
    promote_staging_social,
    read_segments,
    recover_telegram_dataset,
    run_social_ingestion,
    search_telegram_market_plan,
    segment_models_to_records,
    segment_records,
    write_segments,
)
from polysignal.ingestion.social.enrichment_config import EnrichmentOverrides
from polysignal.ingestion.social.enrichment_pipeline import (
    compare_with_real_optional,
    read_collection_summary,
    read_downstream_coverage,
    write_collection_summary,
    write_report_and_index,
)
from polysignal.linking import (
    CandidateMode,
    LLMJudgePolicy,
    LLMRelevanceJudge,
    RelevanceConfig,
    build_default_provider,
    build_linking_coverage_report,
    build_pairs_from_records_with_diagnostics,
    evaluate_predictions,
    export_evidence,
    fixture_labels,
    fixture_markets,
    fixture_posts,
    read_linked_posts,
    run_linking_cross_product,
    score_pairs,
    summarize_by_label,
    write_evaluation_report,
    write_linked_posts,
    write_linking_coverage_report,
)
from polysignal.llm.client import NullLLMClient, build_llm_client
from polysignal.scheduling import RefreshMode, RefreshPolicy, RefreshState
from polysignal.stance import (
    HumanStanceLabel,
    LLMStanceJudge,
    LLMStanceJudgePolicy,
    evaluate_stance,
    read_stance_records,
    run_stance_pipeline,
    write_manual_stance_review,
)
from polysignal.stance import (
    write_evaluation_report as write_stance_evaluation_report,
)
from polysignal.storage.cache import JSONCache
from polysignal.storage.parquet_store import read_dataframe, read_parquet
from polysignal.storage.paths import get_data_paths
from polysignal.storage.run_lock import RunLockError, run_lock

console = Console()

app = typer.Typer(help="PolySignal — a social intelligence layer for Polymarket.")
polymarket_app = typer.Typer(help="Polymarket ingestion commands.")
social_app = typer.Typer(help="Social-platform ingestion commands.")
telegram_app = typer.Typer(help="Telegram public-channel ingestion commands.")
linking_app = typer.Typer(
    help="Branch 05 — semantic-relevance linking commands (score, inspect, evaluate)."
)
nlp_app = typer.Typer(help="NLP enrichment commands.")
graph_app = typer.Typer(help="Knowledge-graph commands.")
graph_queries_app = typer.Typer(help="Parameterized read-query builders (dry-run by default).")
evaluate_app = typer.Typer(help="Evaluation / backtesting commands.")
refresh_app = typer.Typer(help="Refresh-policy commands.")
dataset_app = typer.Typer(help="High-signal dataset preparation commands.")
stance_app = typer.Typer(help="Branch 06 — outcome-aware stance commands.")
features_app = typer.Typer(
    help="Branch 08 — feature engineering and the Social Pressure Index (SPI)."
)
processing_app = typer.Typer(
    help="Big-data processing jobs (PySpark with a pandas fallback)."
)
analytics_app = typer.Typer(
    help="Task-2 analytics (descriptive / discourse / correlation / cross-platform)."
)

app.add_typer(polymarket_app, name="polymarket")
app.add_typer(social_app, name="social")
social_app.add_typer(telegram_app, name="telegram")
app.add_typer(linking_app, name="linking")
app.add_typer(nlp_app, name="nlp")
app.add_typer(graph_app, name="graph")
graph_app.add_typer(graph_queries_app, name="queries")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(refresh_app, name="refresh")
app.add_typer(dataset_app, name="dataset")
app.add_typer(stance_app, name="stance")
app.add_typer(features_app, name="features")
app.add_typer(processing_app, name="process")
app.add_typer(analytics_app, name="analytics")


# ---------------------------------------------------------------------------
# Root commands
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print PolySignal version."""
    rprint(f"[bold]PolySignal[/bold] v{__version__}")


@app.command()
def init_data_dirs() -> None:
    """Create the data directory layout under the configured data root."""
    paths = get_data_paths()
    paths.ensure()
    rprint(f"[green]OK[/green] Data layout ready under [bold]{paths.root}[/bold]")


@app.command()
def config_show() -> None:
    """Show the resolved configuration (secrets masked)."""
    settings = get_settings()
    table = Table(title="PolySignal — resolved configuration")
    table.add_column("Key", style="bold")
    table.add_column("Value")
    table.add_row("env", settings.env)
    table.add_row("data_dir", str(settings.data_dir))
    table.add_row("polymarket.gamma_api_base", settings.polymarket.gamma_api_base)
    table.add_row("polymarket.clob_api_base", settings.polymarket.clob_api_base)
    table.add_row("reddit.configured", str(settings.reddit.configured))
    table.add_row("twitter.configured", str(settings.twitter.configured))
    table.add_row("telegram.configured", str(settings.telegram.configured))
    table.add_row("neo4j.uri", settings.neo4j.uri)
    table.add_row("neo4j.database", settings.neo4j.database)
    table.add_row("llm.provider", settings.llm.provider)
    table.add_row("llm.enabled", str(settings.llm.enabled))
    console.print(table)


# ---------------------------------------------------------------------------
# Polymarket
# ---------------------------------------------------------------------------


@polymarket_app.command("fetch-markets")
def polymarket_fetch_markets(
    limit: int = typer.Option(50, help="Maximum markets to fetch."),
    binary_only: bool = typer.Option(
        True,
        "--binary-only/--include-non-binary",
        help=(
            "Keep only binary YES/NO markets (default). With "
            "--include-non-binary, non-binary markets are logged but "
            "still excluded from parquet outputs — this milestone "
            "supports binary markets only."
        ),
    ),
    save_raw: bool = typer.Option(
        True, "--save-raw/--no-save-raw", help="Persist raw API pages under data/raw/polymarket/."
    ),
    dry_run: bool = typer.Option(False, help="Run filter + normalize but skip parquet writes."),
    output_dir: Path | None = typer.Option(
        None, help="Override data root (defaults to POLYSIGNAL_DATA_DIR)."
    ),
    page_size: int = typer.Option(100, help="Upstream API page size."),
    order: str = typer.Option("volume_24hr", help="Gamma events ordering field."),
    ascending: bool = typer.Option(
        False,
        "--ascending/--descending",
        help="Sort Gamma discovery results ascending or descending.",
    ),
    market_status: str = typer.Option(
        "active",
        "--market-status",
        help=(
            "Which Polymarket markets to discover: active (default), closed, "
            "or all. Use 'closed' to collect already-resolved markets for the "
            "Branch 07 resolved-market evaluation set."
        ),
    ),
    discovery: str = typer.Option(
        "events",
        "--discovery",
        help=(
            "Discovery strategy: 'events' (default) walks /events families; "
            "'markets' walks /markets directly. Use 'markets' with "
            "--market-status closed to net far more binary YES/NO resolved "
            "markets (events discovery surfaces mostly multi-outcome sports)."
        ),
    ),
    end_date_min: str = typer.Option(
        None,
        "--end-date-min",
        help=(
            "Only discover markets whose end date is on/after this ISO date "
            "(e.g. 2024-06-01). Useful with --market-status closed to favour "
            "recent resolved markets that still expose CLOB price history."
        ),
    ),
    max_markets_per_event: int = typer.Option(
        2,
        help="Cap markets selected from one event family. Use 0 for no cap.",
    ),
) -> None:
    """Fetch markets from the Polymarket Gamma API and save metadata."""
    paths = get_data_paths(root=output_dir) if output_dir else None
    status = _validate_market_status(market_status)
    disc = _validate_discovery(discovery)
    extra = _end_date_extra_params(end_date_min)
    try:
        result = run_market_ingestion(
            limit=limit,
            binary_only=binary_only,
            save_raw=save_raw,
            dry_run=dry_run,
            discovery=disc,
            extra_params=extra,
            page_size=page_size,
            order=order,
            ascending=ascending,
            market_status=status,
            max_markets_per_event=max_markets_per_event,
            paths=paths,
        )
    except PolymarketAPIError as exc:
        rprint(f"[red]ERROR[/red] Polymarket API: {exc}")
        raise typer.Exit(code=1) from exc
    _render_ingestion_summary(result, with_prices=False)


@polymarket_app.command("inspect-market")
def polymarket_inspect_market(
    market_id: str = typer.Option(
        ..., "--market-id", help="condition_id, numeric id, or slug of the market to inspect."
    ),
    save_raw: bool = typer.Option(False, help="Also save the raw payload under data/raw."),
) -> None:
    """Show classification + normalized view of a single market."""
    settings = get_settings()
    try:
        with PolymarketClient(settings=settings) as client:
            payload = client.fetch_market(market_id)
    except PolymarketAPIError as exc:
        rprint(f"[red]ERROR[/red] Polymarket API: {exc}")
        raise typer.Exit(code=1) from exc

    if payload is None:
        rprint(f"[red]Not found[/red] no Polymarket market matched {market_id!r}.")
        raise typer.Exit(code=1)

    classification = classify_market(payload)
    rprint(
        f"[bold]Market[/bold] "
        f"{payload.get('conditionId') or payload.get('id')} — "
        f"{payload.get('question')!r}"
    )
    rprint(f"  slug:       {payload.get('slug')}")
    rprint(f"  closed:     {payload.get('closed')}")
    rprint(f"  outcomes:   {classification.raw_outcomes}")
    if classification.is_binary and classification.market is not None:
        binary = classification.market
        rprint("  [green]binary YES/NO[/green]")
        rprint(f"  yes_token:  {binary.yes_token_id}")
        rprint(f"  no_token:   {binary.no_token_id}")
        rprint(f"  yes_price:  {binary.yes_price}")
        rprint(f"  no_price:   {binary.no_price}")
        market = normalize_market(payload, binary=binary)
        rprint(f"  domain:     {market.domain.value}")
        rprint(f"  created_at: {market.created_at.isoformat()}")
        rprint(f"  close_time: {market.close_time.isoformat() if market.close_time else None}")
        rprint(f"  volume:     {market.volume}")
        rprint(f"  liquidity:  {market.liquidity}")
        rprint(f"  resolution: {market.resolution.value if market.resolution else None}")
    else:
        rprint(f"  [yellow]not binary[/yellow] — {classification.reason}")

    if save_raw:
        paths = get_data_paths()
        paths.ensure()
        target = paths.raw_polymarket / "inspect" / f"{_sanitize(market_id)}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        rprint(f"  raw saved to {target}")


@polymarket_app.command("fetch-prices")
def polymarket_fetch_prices(
    market_id: str = typer.Option(
        ..., "--market-id", help="Market identifier (condition_id, numeric id, or slug)."
    ),
    save_raw: bool = typer.Option(True, "--save-raw/--no-save-raw"),
    output_dir: Path | None = typer.Option(None, help="Override data root."),
) -> None:
    """Fetch CLOB price history for one binary market."""
    paths = get_data_paths(root=output_dir) if output_dir else None
    try:
        result = fetch_prices_for_market(market_id, save_raw=save_raw, paths=paths)
    except PolymarketAPIError as exc:
        rprint(f"[red]ERROR[/red] Polymarket API: {exc}")
        raise typer.Exit(code=1) from exc

    if not result.markets:
        if result.skipped:
            skip = result.skipped[0]
            rprint(f"[yellow]Skipped[/yellow] {skip.market_ref}: {skip.reason}")
        else:
            rprint(f"[red]No market normalized for {market_id}[/red]")
        raise typer.Exit(code=1)
    _render_ingestion_summary(result, with_prices=True)


@polymarket_app.command("ingest")
def polymarket_ingest(
    limit: int = typer.Option(50, help="Maximum markets to fetch."),
    binary_only: bool = typer.Option(
        True,
        "--binary-only/--include-non-binary",
        help="Restrict to binary YES/NO markets (default).",
    ),
    save_raw: bool = typer.Option(True, "--save-raw/--no-save-raw"),
    dry_run: bool = typer.Option(False, help="Skip parquet writes."),
    page_size: int = typer.Option(100, help="Upstream API page size."),
    output_dir: Path | None = typer.Option(None, help="Override data root."),
    order: str = typer.Option("volume_24hr", help="Gamma events ordering field."),
    ascending: bool = typer.Option(
        False,
        "--ascending/--descending",
        help="Sort Gamma discovery results ascending or descending.",
    ),
    market_status: str = typer.Option(
        "active",
        "--market-status",
        help=(
            "Which Polymarket markets to discover: active (default), closed, "
            "or all. Use 'closed' to collect already-resolved markets (with "
            "price history) for the Branch 07 resolved-market evaluation set."
        ),
    ),
    discovery: str = typer.Option(
        "events",
        "--discovery",
        help=(
            "Discovery strategy: 'events' (default) or 'markets'. Use "
            "'markets' with --market-status closed to net far more binary "
            "YES/NO resolved markets."
        ),
    ),
    end_date_min: str = typer.Option(
        None,
        "--end-date-min",
        help=(
            "Only discover markets ending on/after this ISO date (e.g. "
            "2024-06-01). Favours recent resolved markets that still expose "
            "CLOB price history."
        ),
    ),
    max_markets_per_event: int = typer.Option(
        2,
        help="Cap markets selected from one event family. Use 0 for no cap.",
    ),
) -> None:
    """End-to-end ingestion: fetch markets, filter, fetch prices, save."""
    paths = get_data_paths(root=output_dir) if output_dir else None
    status = _validate_market_status(market_status)
    disc = _validate_discovery(discovery)
    extra = _end_date_extra_params(end_date_min)
    try:
        result = run_market_ingestion(
            limit=limit,
            binary_only=binary_only,
            save_raw=save_raw,
            fetch_prices=True,
            dry_run=dry_run,
            discovery=disc,
            extra_params=extra,
            page_size=page_size,
            order=order,
            ascending=ascending,
            market_status=status,
            max_markets_per_event=max_markets_per_event,
            paths=paths,
        )
    except PolymarketAPIError as exc:
        rprint(f"[red]ERROR[/red] Polymarket API: {exc}")
        raise typer.Exit(code=1) from exc
    _render_ingestion_summary(result, with_prices=True)


@polymarket_app.command("ingest-curated")
def polymarket_ingest_curated(
    market_id: list[str] = typer.Option(
        None,
        "--market-id",
        help="Market reference (condition id, numeric id, or slug). Repeatable.",
    ),
    ids_file: Path = typer.Option(
        None,
        "--ids-file",
        help="Optional text file with one market reference per line (# comments allowed).",
    ),
    min_price_points: int = typer.Option(
        100,
        "--min-price-points",
        help="Reject a market whose YES timeline has fewer points than this.",
    ),
    min_price_span_days: float = typer.Option(
        7.0,
        "--min-price-span-days",
        help="Reject a market whose YES timeline spans fewer days than this.",
    ),
    fidelity: int = typer.Option(60, "--fidelity", help="CLOB candle fidelity in minutes."),
    save_raw: bool = typer.Option(True, "--save-raw/--no-save-raw"),
    output_dir: Path | None = typer.Option(None, help="Override data root."),
) -> None:
    """Ingest an explicit market list with full-lifetime CLOB price timelines.

    Resolved markets expose at most ~3 weeks of history through
    ``interval=max``; this command walks bounded startTs/endTs windows so
    the complete real timeline is recovered. Markets without a usable
    timeline are reported and NOT persisted (replace them with a better
    market), so every curated market the frontend sees has a real
    probability timeline.
    """
    refs: list[str] = list(market_id or [])
    if ids_file is not None:
        try:
            for line in ids_file.read_text(encoding="utf-8").splitlines():
                cleaned = line.strip()
                if cleaned and not cleaned.startswith("#"):
                    refs.append(cleaned)
        except OSError as exc:
            rprint(f"[red]ERROR[/red] cannot read --ids-file: {exc}")
            raise typer.Exit(code=1) from exc
    if not refs:
        rprint("[red]ERROR[/red] provide at least one --market-id or an --ids-file.")
        raise typer.Exit(code=2)

    paths = get_data_paths(root=output_dir) if output_dir else None
    try:
        result, reports = ingest_curated_markets(
            refs,
            min_price_points=min_price_points,
            min_price_span_days=min_price_span_days,
            fidelity=fidelity,
            save_raw=save_raw,
            paths=paths,
        )
    except PolymarketAPIError as exc:
        rprint(f"[red]ERROR[/red] Polymarket API: {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Curated Polymarket ingestion")
    table.add_column("Market", style="bold")
    table.add_column("Resolution")
    table.add_column("Points")
    table.add_column("Span (d)")
    table.add_column("Status")
    for report in reports:
        table.add_row(
            (report.question or report.market_ref)[:60],
            report.resolution or "—",
            str(report.price_points),
            f"{report.price_span_days:.1f}",
            "[green]ingested[/green]" if report.ingested else f"[red]{report.reason}[/red]",
        )
    console.print(table)
    if result.markets_parquet:
        rprint(f"[green]markets[/green] -> {result.markets_parquet}")
        rprint(f"[green]outcomes[/green] -> {result.outcomes_parquet}")
        rprint(f"[green]price points[/green] -> {result.price_points_parquet}")
    failed = [r for r in reports if not r.ingested]
    if failed:
        rprint(
            f"[yellow]{len(failed)} market(s) were not ingested — see reasons above. "
            "Replace markets without a usable timeline.[/yellow]"
        )
        raise typer.Exit(code=1)


def _validate_market_status(value: str) -> str:
    """Validate the --market-status flag against the pipeline's allowed values."""
    status = (value or "").strip().lower()
    if status not in {"active", "closed", "all"}:
        rprint(
            f"[red]Invalid --market-status:[/red] {value!r} "
            "(choose from: active, closed, all)"
        )
        raise typer.Exit(code=2)
    return status


def _validate_discovery(value: str) -> str:
    """Validate the --discovery flag (events vs markets pagination)."""
    discovery = (value or "").strip().lower()
    if discovery not in {"events", "markets"}:
        rprint(
            f"[red]Invalid --discovery:[/red] {value!r} (choose from: events, markets)"
        )
        raise typer.Exit(code=2)
    return discovery


def _end_date_extra_params(end_date_min: str | None) -> dict[str, str] | None:
    """Build the Gamma ``end_date_min`` filter from an ISO date, if provided."""
    if not end_date_min:
        return None
    value = end_date_min.strip()
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        rprint(
            f"[red]Invalid --end-date-min:[/red] {end_date_min!r} "
            "(use an ISO date such as 2024-06-01)"
        )
        raise typer.Exit(code=2) from exc
    return {"end_date_min": value}


def _is_market_resolved_row(market: Market) -> bool:
    """A normalized market counts as resolved when is_resolved or resolution is set."""
    return bool(market.is_resolved) or market.resolution is not None


@polymarket_app.command("inspect")
def polymarket_inspect(
    markets_path: Path = typer.Option(None, "--markets-path"),
    outcomes_path: Path = typer.Option(None, "--outcomes-path"),
    prices_path: Path = typer.Option(None, "--prices-path"),
    resolved_only: bool = typer.Option(
        False,
        "--resolved-only/--all-markets",
        help="Restrict the summary to already-resolved markets (evaluation focus).",
    ),
    output_dir: Path = typer.Option(None, "--output-dir", help="Override the data root."),
) -> None:
    """Offline summary of the normalized Polymarket datasets (no API calls).

    Reports market / outcome / price-point counts, resolution status,
    binary vs multi-outcome shape, domain coverage, orphan checks, and
    whether price timestamps parse as UTC. This command never contacts an
    API, so it never needs credentials.
    """
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    m_target = markets_path or paths.polymarket_markets_parquet
    if not m_target.exists():
        rprint(
            f"[yellow]No markets parquet at {m_target}.[/yellow] "
            "Run `polysignal polymarket ingest` first."
        )
        raise typer.Exit(code=2)
    raw_markets = read_parquet(m_target, Market)
    if not raw_markets:
        rprint(f"[yellow]markets parquet at {m_target} is empty.[/yellow]")
        raise typer.Exit(code=2)

    o_target = outcomes_path or paths.polymarket_outcomes_parquet
    raw_outcomes = read_parquet(o_target, Outcome) if o_target.exists() else []
    markets = list({market.market_id: market for market in raw_markets}.values())
    outcomes = list(
        {
            (outcome.market_id, outcome.outcome_id): outcome
            for outcome in raw_outcomes
        }.values()
    )
    p_target = prices_path or paths.polymarket_price_points_parquet
    prices_df = read_dataframe(p_target)

    resolved = [mk for mk in markets if _is_market_resolved_row(mk)]
    closed = [mk for mk in markets if mk.is_closed]
    scope = resolved if resolved_only else markets
    scope_ids = {mk.market_id for mk in scope}

    market_ids = {mk.market_id for mk in markets}
    outcomes_by_market: dict[str, int] = {}
    for outcome in outcomes:
        if outcome.market_id:
            outcomes_by_market[outcome.market_id] = outcomes_by_market.get(outcome.market_id, 0) + 1
    orphan_outcomes = sum(1 for o in outcomes if o.market_id and o.market_id not in market_ids)
    markets_without_outcomes = sum(1 for mk in scope if outcomes_by_market.get(mk.market_id, 0) == 0)
    binary = sum(1 for mk in scope if outcomes_by_market.get(mk.market_id, len(mk.outcomes)) == 2)
    multi = len(scope) - binary

    scope_outcomes = [o for o in outcomes if o.market_id in scope_ids]
    domains: dict[str, int] = {}
    for mk in scope:
        domains[mk.domain.value] = domains.get(mk.domain.value, 0) + 1

    ts_ok = "n/a"
    ts_min = ts_max = "n/a"
    price_rows_scope = 0
    markets_with_prices = 0
    if not prices_df.empty and "timestamp" in prices_df.columns:
        scoped_prices = prices_df[prices_df["market_id"].isin(scope_ids)] if "market_id" in prices_df else prices_df
        price_rows_scope = len(scoped_prices)
        if "market_id" in scoped_prices:
            markets_with_prices = scoped_prices["market_id"].nunique()
        try:
            parsed = pd.to_datetime(scoped_prices["timestamp"], utc=True, errors="raise")
            ts_ok = "yes"
            if len(parsed):
                ts_min, ts_max = str(parsed.min()), str(parsed.max())
        except Exception:  # pragma: no cover - reporting only
            ts_ok = "no"

    title = "Polymarket dataset — resolved-only" if resolved_only else "Polymarket dataset — all markets"
    table = Table(title=title)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("markets (total, deduplicated)", str(len(markets)))
    table.add_row("duplicate market rows removed", str(len(raw_markets) - len(markets)))
    table.add_row("resolved markets", str(len(resolved)))
    table.add_row("unresolved markets", str(len(markets) - len(resolved)))
    table.add_row("closed markets", str(len(closed)))
    table.add_row("markets in scope", str(len(scope)))
    table.add_row("outcomes in scope", str(len(scope_outcomes) if outcomes else "n/a"))
    table.add_row("duplicate outcome rows removed", str(len(raw_outcomes) - len(outcomes)))
    table.add_row("binary / multi in scope", f"{binary} / {multi}")
    table.add_row("price points in scope", str(price_rows_scope))
    table.add_row("markets w/ price history", str(markets_with_prices))
    table.add_row("price ts parse UTC", ts_ok)
    table.add_row("price ts range", f"{ts_min} .. {ts_max}")
    table.add_row("orphan outcomes (no market)", str(orphan_outcomes))
    table.add_row("scope markets w/o outcomes", str(markets_without_outcomes))
    table.add_row("domains", ", ".join(f"{k}={v}" for k, v in sorted(domains.items())))
    console.print(table)

    if resolved_only:
        if not resolved:
            rprint(
                "[yellow]No resolved markets in this dataset.[/yellow] "
                "The current normalized markets were discovered with "
                "--market-status active. Re-run "
                "`polysignal polymarket ingest --market-status closed` to "
                "collect already-resolved markets."
            )
        else:
            res_counts: dict[str, int] = {}
            for mk in resolved:
                key = mk.resolution.value if mk.resolution else "unknown"
                res_counts[key] = res_counts.get(key, 0) + 1
            rprint(
                "[green]resolution breakdown:[/green] "
                + ", ".join(f"{k}={v}" for k, v in sorted(res_counts.items()))
            )


@polymarket_app.command("export-resolved-labels")
def polymarket_export_resolved_labels(
    source_dir: Path = typer.Option(
        None,
        "--source-dir",
        help=(
            "Data root to read normalized markets/outcomes from. Defaults to "
            "the configured data root. Point this at a resolved-market "
            "ingestion root to build labels from already-resolved markets."
        ),
    ),
    markets_path: Path = typer.Option(None, "--markets-path"),
    outcomes_path: Path = typer.Option(None, "--outcomes-path"),
    output_path: Path = typer.Option(
        None,
        "--output-path",
        help=(
            "Where to write resolved_market_labels.parquet. Defaults to "
            "data/analytical/evaluation/resolved_market_labels.parquet."
        ),
    ),
) -> None:
    """Build the resolved-only evaluation labels (evaluation targets only).

    Resolved labels are **never** used for retrieval, query generation,
    linking, stance, the Social Pressure Index, or prediction inputs. They
    are written so a later evaluation step can score forecasts against
    ground truth. When the source dataset has no resolved markets, a
    column-stable empty Parquet is written and the blocker is reported.
    """
    read_paths = get_data_paths(root=source_dir) if source_dir else get_data_paths()
    m_target = markets_path or read_paths.polymarket_markets_parquet
    if not m_target.exists():
        rprint(
            f"[yellow]No markets parquet at {m_target}.[/yellow] "
            "Run `polysignal polymarket ingest --market-status closed` first."
        )
        raise typer.Exit(code=2)
    markets = read_parquet(m_target, Market)
    o_target = outcomes_path or read_paths.polymarket_outcomes_parquet
    outcomes = read_parquet(o_target, Outcome) if o_target.exists() else []

    labels = build_resolved_labels(markets, outcomes)
    target = output_path or get_data_paths().resolved_market_labels_parquet
    write_resolved_labels(labels, target)

    table = Table(title="Resolved-market labels (evaluation targets only)")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("markets read", str(len(markets)))
    table.add_row("resolved labels", str(len(labels)))
    with_winner = sum(1 for label in labels if label.winning_outcome_id is not None)
    table.add_row("labels with winning outcome", str(with_winner))
    res_counts: dict[str, int] = {}
    for label in labels:
        key = label.resolution or "unknown"
        res_counts[key] = res_counts.get(key, 0) + 1
    if res_counts:
        table.add_row("resolution breakdown", ", ".join(f"{k}={v}" for k, v in sorted(res_counts.items())))
    console.print(table)
    rprint(f"[green]resolved labels[/green] -> {target} ({len(labels)} rows)")
    if not labels:
        rprint(
            "[yellow]No resolved markets in the source dataset.[/yellow] "
            "Collect already-resolved markets with "
            "`polysignal polymarket ingest --market-status closed --limit 60` "
            "then re-run this command with --source-dir pointing at that root."
        )


def _render_ingestion_summary(result: IngestionResult, *, with_prices: bool) -> None:
    table = Table(title=f"Polymarket ingestion — run {result.run_id}")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("fetched", str(result.fetched))
    if result.discovery_method == "events":
        table.add_row("events fetched", str(result.events_fetched))
        table.add_row("raw markets discovered", str(result.markets_discovered))
        distinct_events = len({m.event_id for m in result.markets if m.event_id})
        table.add_row("distinct event_id", str(distinct_events))
        if result.markets:
            counts: dict[str, int] = {}
            for market in result.markets:
                key = market.event_id or "<none>"
                counts[key] = counts.get(key, 0) + 1
            top_count = max(counts.values())
            table.add_row("top event concentration", f"{top_count}/{len(result.markets)}")
    table.add_row("kept (binary YES/NO)", str(result.kept))
    table.add_row("skipped (non-binary / error)", str(len(result.skipped)))
    table.add_row("outcomes", str(len(result.outcomes)))
    if with_prices:
        table.add_row("price points", str(len(result.price_points)))
        ok = sum(1 for a in result.price_attempts if a.fetched)
        markets_with_prices = len(
            {a.market_id for a in result.price_attempts if a.fetched and a.point_count > 0}
        )
        table.add_row("price histories fetched", f"{ok}/{len(result.price_attempts)}")
        table.add_row("markets w/ price history", f"{markets_with_prices}/{len(result.markets)}")
    console.print(table)

    if result.markets_parquet:
        rprint(f"[green]markets[/green]      -> {result.markets_parquet}")
    if result.outcomes_parquet:
        rprint(f"[green]outcomes[/green]     -> {result.outcomes_parquet}")
    if result.price_points_parquet:
        rprint(f"[green]price points[/green] -> {result.price_points_parquet}")
    if result.raw_markets_dir:
        rprint(f"[green]raw markets[/green]  -> {result.raw_markets_dir}")
    if result.raw_events_dir:
        rprint(f"[green]raw events[/green]   -> {result.raw_events_dir}")
    if result.raw_prices_dir:
        rprint(f"[green]raw prices[/green]   -> {result.raw_prices_dir}")

    if result.skipped:
        rprint("\n[bold]Skipped markets (first 10):[/bold]")
        for entry in result.skipped[:10]:
            rprint(f"  - {entry.market_ref}: {entry.reason}")
        if len(result.skipped) > 10:
            rprint(f"  ... and {len(result.skipped) - 10} more")


def _sanitize(value: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in value)[:80]


def _terminal_text(value: str | None, *, limit: int | None = None) -> str:
    """Return a compact preview safe for legacy Windows consoles."""
    text = " ".join((value or "").split())
    if not text:
        return "<empty>"
    encoding = getattr(console.file, "encoding", None) or "utf-8"
    safe = text.encode(encoding, errors="replace").decode(encoding)
    return safe[:limit] if limit is not None else safe


def _compact_id(value: str, *, limit: int = 18) -> str:
    """Keep Telegram message suffixes visible while shortening long ids."""
    if value.startswith("telegram:"):
        parts = value.split(":")
        if len(parts) >= 3:
            message = parts[-1]
            return f"tg:{message}"
    if len(value) <= limit:
        return value
    return f"{value[:8]}...{value[-6:]}"


# ---------------------------------------------------------------------------
# Social  (Branch 04)
# ---------------------------------------------------------------------------


_SOCIAL_INGESTORS: dict[str, type[SocialIngestor]] = {
    "reddit": RedditIngestor,
    "telegram": TelegramIngestor,
}


def _resolve_platforms(name: str) -> list[str]:
    name = (name or "").strip().lower()
    if name in ("all", "*"):
        return ["reddit", "telegram"]
    if name not in _SOCIAL_INGESTORS:
        rprint(
            f"[red]Unknown platform:[/red] {name!r} (choose from: reddit, telegram, all)"
        )
        raise typer.Exit(code=2)
    return [name]


@social_app.command("generate-queries")
def social_generate_queries(
    markets_path: Path = typer.Option(
        None,
        "--markets-path",
        help=("Path to markets.parquet. Defaults to data/normalized/polymarket/markets.parquet."),
    ),
    limit_markets: int = typer.Option(
        20, "--limit-markets", help="Maximum markets to expand into queries."
    ),
    queries_per_market: int = typer.Option(
        6, "--queries-per-market", help="Maximum queries to emit per market."
    ),
    market_id: str = typer.Option(
        None, "--market-id", help="If set, only expand the matching market."
    ),
    output_path: Path = typer.Option(
        None,
        "--output-path",
        help=(
            "Write the queries to a Parquet file at this path. "
            "Default location: data/normalized/social/social_queries.parquet."
        ),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run/--no-dry-run", help="Print queries without writing parquet."
    ),
) -> None:
    """Generate Reddit-friendly search queries from Polymarket markets."""
    paths = get_data_paths()
    source_path = markets_path or paths.polymarket_markets_parquet
    try:
        markets = load_markets_from_parquet(source_path, limit=None)
    except FileNotFoundError as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if market_id:
        markets = [m for m in markets if m.market_id == market_id or m.slug == market_id]
        if not markets:
            rprint(f"[red]No market matched --market-id={market_id!r}[/red]")
            raise typer.Exit(code=1)

    queries = generate_queries(
        markets,
        queries_per_market=queries_per_market,
        limit_markets=limit_markets,
    )
    if not queries:
        rprint("[yellow]No queries generated[/yellow] — markets file may be empty.")
        raise typer.Exit(code=2)

    table = Table(title=f"Social query generation ({len(queries)} queries)")
    table.add_column("market_id", style="bold")
    table.add_column("rank")
    table.add_column("type")
    table.add_column("query")
    for q in queries[:60]:
        short_id = q.market_id if len(q.market_id) <= 18 else q.market_id[:15] + "..."
        table.add_row(short_id, str(q.rank), q.query_type, q.query)
    console.print(table)
    if len(queries) > 60:
        rprint(f"  ... and {len(queries) - 60} more queries")

    if dry_run:
        rprint("[yellow]--dry-run set, no parquet written.[/yellow]")
        return

    target = output_path or paths.social_queries_parquet
    target.parent.mkdir(parents=True, exist_ok=True)
    from polysignal.storage.parquet_store import write_parquet  # local import for clarity

    write_parquet(queries, target)
    rprint(f"[green]queries[/green] -> {target}")


@social_app.command("ingest")
def social_ingest(
    platform: str = typer.Option(
        "reddit", "--platform", help="One of: reddit, twitter, telegram, all."
    ),
    markets_path: Path = typer.Option(
        None,
        "--markets-path",
        help="Path to markets.parquet (defaults to the Polymarket output).",
    ),
    limit_markets: int = typer.Option(
        10, "--limit-markets", help="Maximum markets to use for query generation."
    ),
    queries_per_market: int = typer.Option(
        6, "--queries-per-market", help="Maximum queries emitted per market."
    ),
    posts_per_query: int = typer.Option(
        20, "--posts-per-query", help="Maximum posts requested per query."
    ),
    save_raw: bool = typer.Option(
        True,
        "--save-raw/--no-save-raw",
        help="Persist raw per-post snapshots under data/raw/social/<platform>/.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--no-dry-run",
        help=(
            "Plan and persist generated queries without platform API calls, "
            "raw snapshots, or social-post writes."
        ),
    ),
    output_dir: Path = typer.Option(None, help="Override the data root."),
) -> None:
    """Run market-query social ingestion. Use ``social telegram`` for channels."""
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    platforms = _resolve_platforms(platform)
    any_executed = False
    for name in platforms:
        if name == "telegram" and not dry_run:
            rprint(
                "[yellow]Skipping telegram[/yellow] - use "
                "`polysignal social telegram ingest --config config/telegram_channels.yml`."
            )
            continue
        ingestor_cls = _SOCIAL_INGESTORS[name]
        ingestor = ingestor_cls()
        # Dry-run does not contact the platform, so it does NOT need
        # credentials. Live ingestion still requires them.
        if not dry_run and not ingestor.is_available():
            rprint(
                f"[yellow]Skipping {name}[/yellow] — credentials not configured. "
                "See .env.example for the required environment variables, "
                "or re-run with --dry-run to preview queries."
            )
            continue
        try:
            result = run_social_ingestion(
                ingestor=ingestor,
                markets_path=markets_path,
                limit_markets=limit_markets,
                queries_per_market=queries_per_market,
                posts_per_query=posts_per_query,
                save_raw=save_raw,
                dry_run=dry_run,
                paths=paths,
            )
        except FileNotFoundError as exc:
            rprint(f"[red]ERROR[/red] {exc}")
            raise typer.Exit(code=1) from exc
        except RuntimeError as exc:
            rprint(f"[red]ERROR[/red] {name}: {exc}")
            raise typer.Exit(code=1) from exc
        _render_social_summary(result, dry_run=dry_run)
        any_executed = True

    if not any_executed:
        rprint(
            "[yellow]No platform was executed.[/yellow] "
            "Configure credentials in .env (see .env.example), "
            "or use --dry-run to preview queries without contacting the API."
        )
        raise typer.Exit(code=2)


@social_app.command("reddit-search")
def social_reddit_search(
    limit_markets: int = typer.Option(
        10, "--limit-markets", help="Maximum markets to expand into queries."
    ),
    posts_per_query: int = typer.Option(
        20, "--posts-per-query", help="Maximum posts requested per query."
    ),
    queries_per_market: int = typer.Option(
        6, "--queries-per-market", help="Maximum queries per market."
    ),
    markets_path: Path = typer.Option(None, "--markets-path"),
    save_raw: bool = typer.Option(True, "--save-raw/--no-save-raw"),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run"),
) -> None:
    """Convenience alias for ``polysignal social ingest --platform reddit``."""
    social_ingest(
        platform="reddit",
        markets_path=markets_path,
        limit_markets=limit_markets,
        queries_per_market=queries_per_market,
        posts_per_query=posts_per_query,
        save_raw=save_raw,
        dry_run=dry_run,
        output_dir=None,
    )


def _parse_cli_datetime(value: str | None, option: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise typer.BadParameter(
            "use ISO-8601, for example 2026-05-01T00:00:00Z", param_hint=option
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _render_telegram_fetch(result, *, dry_run: bool) -> None:
    mode = "dry-run" if dry_run else "live"
    succeeded = getattr(result, "succeeded", [])
    skipped = getattr(result, "skipped", [])
    rprint(
        f"[green]Telegram {mode}[/green]: channels={len(result.channels)}, "
        f"succeeded={len(succeeded)}, skipped={len(skipped)}, "
        f"messages={result.posts_collected}"
    )
    if succeeded:
        rprint("[bold]Channels ingested:[/bold]")
        for username, count in succeeded:
            rprint(f"  [green]OK[/green] @{username}: {count} messages")
    if skipped:
        rprint("[bold]Channels skipped (reason):[/bold]")
        for username, reason in skipped[:20]:
            console.print(f"  skip @{username}: {reason}", markup=False)
        if len(skipped) > 20:
            rprint(f"  ... and {len(skipped) - 20} more")
    if result.raw_dir is not None:
        rprint(f"[green]raw snapshots[/green] -> {result.raw_dir}")


@telegram_app.command("auth")
def social_telegram_auth() -> None:
    """Authenticate the local Telethon session interactively."""
    ingestor = TelegramIngestor()
    try:
        ingestor.authenticate()
    except RuntimeError as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        ingestor.close()
    rprint(f"[green]Telegram session ready[/green] -> {ingestor.session_path}")


@telegram_app.command("fetch-channel")
def social_telegram_fetch_channel(
    channel: str = typer.Option(..., "--channel", help="Explicit public channel/group username."),
    query: str = typer.Option(None, "--query", help="Optional Telegram message search query."),
    limit: int = typer.Option(200, "--limit", min=1, max=1500),
    start_date: str = typer.Option(None, "--start-date", help="Optional ISO-8601 lower bound."),
    end_date: str = typer.Option(None, "--end-date", help="Optional ISO-8601 upper bound."),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run"),
    output_dir: Path = typer.Option(None, "--output-dir", help="Override the data root."),
) -> None:
    """Fetch raw snapshots from one explicit Telegram target."""
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    try:
        result = fetch_telegram_channels(
            [TelegramChannel(username=channel.strip().removeprefix("@"))],
            query=query,
            limit=limit,
            since=_parse_cli_datetime(start_date, "--start-date"),
            until=_parse_cli_datetime(end_date, "--end-date"),
            dry_run=dry_run,
            paths=paths,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc
    _render_telegram_fetch(result, dry_run=dry_run)


@telegram_app.command("fetch-config")
def social_telegram_fetch_config(
    config: Path = typer.Option(
        Path("config/telegram_channels.yml"),
        "--config",
        help="YAML file listing enabled public channels/groups.",
    ),
    query: str = typer.Option(None, "--query", help="Optional Telegram message search query."),
    limit: int = typer.Option(200, "--limit-per-channel", min=1, max=1500),
    start_date: str = typer.Option(None, "--start-date", help="Optional ISO-8601 lower bound."),
    end_date: str = typer.Option(None, "--end-date", help="Optional ISO-8601 upper bound."),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run"),
    output_dir: Path = typer.Option(None, "--output-dir", help="Override the data root."),
) -> None:
    """Fetch raw snapshots from enabled YAML-configured Telegram targets."""
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    try:
        channels = load_telegram_channels(config)
        result = fetch_telegram_channels(
            channels,
            query=query,
            limit=limit,
            since=_parse_cli_datetime(start_date, "--start-date"),
            until=_parse_cli_datetime(end_date, "--end-date"),
            dry_run=dry_run,
            paths=paths,
            config_path=config,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc
    _render_telegram_fetch(result, dry_run=dry_run)


@telegram_app.command("normalize")
def social_telegram_normalize(
    raw_dir: Path = typer.Option(None, "--raw-dir", help="Raw Telegram run directory."),
    posts_path: Path = typer.Option(None, "--posts-path", help="Override social_posts.parquet."),
    output_dir: Path = typer.Option(None, "--output-dir", help="Override the data root."),
    config: Path = typer.Option(
        None,
        "--config",
        help="Optional enabled-channel config used to filter and enrich normalized rows.",
    ),
    require_text: bool = typer.Option(
        True,
        "--require-text/--allow-empty-text",
        help="Keep only normalized messages with usable text.",
    ),
) -> None:
    """Append one raw Telegram run to the unified social parquet dataset."""
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    try:
        source = raw_dir or latest_telegram_raw_directory(paths)
        channels = load_telegram_channels(config) if config is not None else None
        records = normalize_telegram_raw_directory(
            source,
            posts_path=posts_path,
            paths=paths,
            channels=channels,
            require_text=require_text,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc
    rprint(f"[green]social posts[/green] -> {posts_path or paths.social_posts_parquet}")
    rprint(f"[green]dataset rows[/green] -> {len(records)}")


@telegram_app.command("collect-large")
def social_telegram_collect_large(
    config: Path = typer.Option(
        Path("config/telegram_channels.yml"),
        "--config",
        help="YAML file listing enabled public channels/groups.",
    ),
    target_total: int = typer.Option(
        1_000_000,
        "--target-total",
        help=(
            "Cumulative dataset target: stop once the saved dataset holds this many "
            "messages in total. With --resume, messages already collected count toward "
            "it (e.g. resume a ~600k dataset with --target-total 1000000 to add ~400k more)."
        ),
    ),
    batch_per_channel: int = typer.Option(
        1000,
        "--batch-per-channel",
        help="Messages fetched per channel per round (paged backwards by id).",
    ),
    max_rounds: int = typer.Option(
        1000, "--max-rounds", help="Safety cap on collection rounds."
    ),
    pause_seconds: float = typer.Option(
        1.0, "--pause-seconds", help="Extra spacing between channels (Telethon also honours FloodWait)."
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Resume from data/state/telegram_collection_state.json (recommended).",
    ),
    require_text: bool = typer.Option(
        True, "--require-text/--allow-empty-text", help="Keep only messages with usable text."
    ),
    output_dir: Path = typer.Option(None, "--output-dir", help="Override the data root."),
) -> None:
    """Staged, resumable, logged Telegram API collection for large datasets.

    Walks the full history of the configured channels via the Telegram API
    (Telethon), paging backwards by message id across rounds and runs. State
    is checkpointed after every round so an overnight run that is killed
    resumes exactly where it stopped; messages already saved are never
    re-collected. Designed to accumulate up to ~1M messages over staged
    runs. Authenticate once first with `polysignal social telegram auth`.
    """
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    # Pre-flight: fail loudly and early on a bad config path so the command never
    # exits silently. ('config/' is singular — 'configs/' holds the Scrapling config.)
    if not config.exists():
        rprint(
            f"[red]ERROR[/red] Telegram channels config not found at {config}. "
            "Expected 'config/telegram_channels.yml' (note: 'config/', not 'configs/')."
        )
        raise typer.Exit(code=1)
    paths.social_enrichment_reports.mkdir(parents=True, exist_ok=True)
    lock_path = paths.root / "telegram_api.lock"
    try:
        with run_lock(lock_path, label="telegram-collect-large"):
            rprint(
                "[cyan]Starting Telegram API collection "
                "(resumable — Ctrl-C is safe, state is checkpointed each round)...[/cyan]"
            )
            channels = load_telegram_channels(config)
            result = collect_telegram_large(
                channels,
                options=TelegramLargeOptions(
                    target_total=target_total,
                    batch_per_channel=batch_per_channel,
                    max_rounds=max_rounds,
                    pause_seconds=pause_seconds,
                    require_text=require_text,
                    resume=resume,
                ),
                paths=paths,
            )
    except RunLockError as exc:
        rprint(f"[red]ALREADY RUNNING[/red] {exc}")
        raise typer.Exit(code=3) from exc
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Telegram staged collection")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("run_id", result.run_id)
    table.add_row("rounds executed", str(result.rounds_executed))
    table.add_row("new messages (this run)", str(result.new_messages))
    table.add_row("total messages (state)", str(result.total_messages_state))
    table.add_row("channels active / exhausted", f"{result.channels_active} / {result.channels_exhausted}")
    table.add_row("target reached", str(result.target_reached))
    table.add_row("stop reason", result.stop_reason)
    console.print(table)
    rprint(f"[green]social posts[/green] -> {result.posts_path or paths.social_posts_parquet}")
    rprint(f"[green]state[/green]        -> {result.state_path}")
    rprint(f"[green]log[/green]          -> {result.log_path}")


@telegram_app.command("recover")
def social_telegram_recover(
    config: Path = typer.Option(
        Path("config/telegram_channels.yml"),
        "--config",
        help="YAML file listing enabled public channels/groups.",
    ),
    output_dir: Path = typer.Option(
        Path("data/staging/telegram_api"),
        "--output-dir",
        help="Staging data root to recover (where the raw chunks live).",
    ),
    require_text: bool = typer.Option(
        True, "--require-text/--allow-empty-text", help="Keep only messages with usable text."
    ),
    rebuild_state: bool = typer.Option(
        True,
        "--rebuild-state/--no-rebuild-state",
        help="Also rebuild telegram_collection_state.json so resume continues from deep history.",
    ),
) -> None:
    """Rebuild social_posts.parquet (and resume state) from the raw chunks.

    Recovery path for the 0-byte / overwritten parquet incident: the per-round
    raw message JSONs under <output-dir>/raw/social/telegram survive even when
    the parquet is destroyed, so this re-normalizes every preserved chunk into
    a fresh, de-duplicated dataset. The prior parquet is backed up first and a
    catastrophic shrink is refused. Safe to run repeatedly (idempotent).
    """
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    if not config.exists():
        rprint(
            f"[red]ERROR[/red] Telegram channels config not found at {config}. "
            "Expected 'config/telegram_channels.yml' (note: 'config/', not 'configs/')."
        )
        raise typer.Exit(code=1)
    # The lock also guards recovery: never recover while a collector is writing.
    lock_path = paths.root / "telegram_api.lock"
    try:
        with run_lock(lock_path, label="telegram-recover"):
            channels = load_telegram_channels(config)
            rprint(
                "[cyan]Recovering social_posts.parquet from raw chunks "
                f"under {paths.raw_social('telegram')} ...[/cyan]"
            )
            summary = recover_telegram_dataset(
                channels,
                paths=paths,
                require_text=require_text,
                rebuild_state=rebuild_state,
            )
    except RunLockError as exc:
        rprint(f"[red]ALREADY RUNNING[/red] {exc} (stop the collector before recovering).")
        raise typer.Exit(code=3) from exc
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Telegram dataset recovery")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("raw dirs scanned", str(summary.get("raw_dirs_scanned")))
    table.add_row("records parsed", str(summary.get("records_parsed")))
    table.add_row("existing rows merged", str(summary.get("existing_rows_merged")))
    table.add_row("unique rows written", str(summary.get("unique_rows_written")))
    if rebuild_state:
        table.add_row("channels in state", str(summary.get("channels_in_state")))
        table.add_row("state total", str(summary.get("state_total")))
    console.print(table)
    rprint(f"[green]social posts[/green] -> {summary.get('output')}")
    if summary.get("backup"):
        rprint(f"[green]prior parquet backed up[/green] -> {summary.get('backup')}")
    if summary.get("state_path"):
        rprint(f"[green]state rebuilt[/green] -> {summary.get('state_path')}")


@telegram_app.command("ingest")
def social_telegram_ingest(
    config: Path = typer.Option(
        Path("config/telegram_channels.yml"),
        "--config",
        help="YAML file listing enabled public channels/groups.",
    ),
    query: str = typer.Option(None, "--query", help="Optional Telegram message search query."),
    limit: int = typer.Option(200, "--limit-per-channel", min=1, max=1500),
    max_total: int = typer.Option(
        0,
        "--max-total",
        min=0,
        help="Global cap on total messages across all channels (0 = no cap).",
    ),
    start_date: str = typer.Option(None, "--start-date", help="Optional ISO-8601 lower bound."),
    end_date: str = typer.Option(None, "--end-date", help="Optional ISO-8601 upper bound."),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run"),
    output_dir: Path = typer.Option(None, "--output-dir", help="Override the data root."),
) -> None:
    """Fetch YAML-configured Telegram channels and normalize them for linking.

    Channels are fetched independently: an inaccessible/private/empty channel
    is skipped with a recorded reason and the run continues. Per-channel
    ``limit:`` in the config overrides ``--limit-per-channel``.
    """
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    try:
        channels = load_telegram_channels(config)
        result = fetch_telegram_channels(
            channels,
            query=query,
            limit=limit,
            max_total=max_total or None,
            since=_parse_cli_datetime(start_date, "--start-date"),
            until=_parse_cli_datetime(end_date, "--end-date"),
            dry_run=dry_run,
            paths=paths,
            config_path=config,
        )
        _render_telegram_fetch(result, dry_run=dry_run)
        if not dry_run and result.raw_dir is not None:
            records = normalize_telegram_raw_directory(
                result.raw_dir,
                paths=paths,
                channels=channels,
                require_text=True,
            )
            rprint(f"[green]social posts[/green] -> {paths.social_posts_parquet}")
            rprint(f"[green]dataset rows[/green] -> {len(records)}")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# Branch 12 — public-web social dataset enrichment (Reddit + Telegram)
# ---------------------------------------------------------------------------


def _parse_only(only: str | None) -> tuple[str, ...]:
    if not only:
        return ()
    return tuple(name.strip() for name in only.split(",") if name.strip())


def _enrichment_overrides(
    *,
    max_pages: int | None,
    limit: int | None,
    rate_limit: float | None,
    only: str | None,
    include_needs_validation: bool,
    dry_run: bool,
) -> EnrichmentOverrides:
    return EnrichmentOverrides(
        max_pages=max_pages,
        max_items=limit,
        rate_limit_seconds=rate_limit,
        include_needs_validation=include_needs_validation,
        only_sources=_parse_only(only),
        dry_run=dry_run,
    )


def _render_collection_result(result, *, platform_label: str) -> None:
    table = Table(title=f"Branch 12 — {platform_label} public-web collection")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("run_id", result.run_id)
    table.add_row("sources total", str(len(result.outcomes)))
    table.add_row("sources succeeded", str(len(result.succeeded)))
    table.add_row("sources failed/empty", str(len(result.failed)))
    table.add_row("raw HTML snapshots", str(result.raw_html_snapshots))
    table.add_row("raw extracted", str(result.raw_extracted))
    table.add_row("new records built", str(result.new_records_built))
    table.add_row("rows added", str(result.rows_added))
    table.add_row("exact duplicates dropped", str(result.exact_duplicates_dropped))
    table.add_row("duplicate rows flagged", str(result.duplicate_rows_flagged))
    table.add_row("normalized total", str(result.normalized_total))
    table.add_row("posts parquet", str(result.posts_path))
    console.print(table)
    if result.outcomes:
        per_source = Table(title="Per-source outcomes")
        per_source.add_column("source")
        per_source.add_column("status")
        per_source.add_column("pages")
        per_source.add_column("normalized")
        per_source.add_column("reason")
        for outcome in result.outcomes:
            per_source.add_row(
                outcome.name,
                outcome.status,
                str(outcome.pages_fetched),
                str(outcome.normalized),
                _terminal_text(outcome.reason or "-", limit=48),
            )
        console.print(per_source)


@social_app.command("collect-reddit-scrapling")
def social_collect_reddit_scrapling(
    config: Path = typer.Option(
        Path("configs/social_enrichment_sources.example.yaml"),
        "--config",
        help="Enrichment source config (YAML).",
    ),
    output_root: Path = typer.Option(
        Path("data_reddit_scrapling_eval"), "--output-root", help="Dataset root for Reddit output."
    ),
    max_pages: int = typer.Option(None, "--max-pages", help="Override max pages per source."),
    limit: int = typer.Option(None, "--limit", help="Override max items per source."),
    rate_limit: float = typer.Option(None, "--rate-limit", help="Override rate limit seconds."),
    only: str = typer.Option(None, "--only", help="Comma-separated source names to include."),
    include_needs_validation: bool = typer.Option(
        True,
        "--include-needs-validation/--skip-needs-validation",
        help="Include sources tagged status: needs_validation.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="List the planned sources only."),
) -> None:
    """Collect a Reddit public-web dataset from old.reddit.com with Scrapling."""
    overrides = _enrichment_overrides(
        max_pages=max_pages,
        limit=limit,
        rate_limit=rate_limit,
        only=only,
        include_needs_validation=include_needs_validation,
        dry_run=dry_run,
    )
    try:
        cfg = load_enrichment_config(config, overrides=overrides)
    except (FileNotFoundError, ValueError) as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc
    if cfg.reddit is None or not cfg.reddit.enabled or not cfg.reddit.sources:
        rprint("[yellow]No enabled Reddit sources in config.[/yellow]")
        raise typer.Exit(code=2)
    if dry_run:
        rprint(f"[cyan]Planned Reddit sources ({len(cfg.reddit.sources)}):[/cyan]")
        for source in cfg.reddit.sources:
            rprint(f"  - {source.name} -> {source.url}")
        return
    paths = get_data_paths(root=output_root)
    result = collect_reddit_scrapling(cfg.reddit, paths=paths)
    write_collection_summary(result.summary(), paths)
    _render_collection_result(result, platform_label="Reddit")
    rprint(f"[green]reddit social_posts[/green] -> {result.posts_path}")


@social_app.command("collect-telegram-public")
def social_collect_telegram_public(
    config: Path = typer.Option(
        Path("configs/social_enrichment_sources.example.yaml"),
        "--config",
        help="Enrichment source config (YAML).",
    ),
    output_root: Path = typer.Option(
        Path("data_telegram_enriched_eval"),
        "--output-root",
        help="Dataset root for Telegram output.",
    ),
    max_pages: int = typer.Option(None, "--max-pages", help="Override max pages per source."),
    limit: int = typer.Option(None, "--limit", help="Override max items per source."),
    rate_limit: float = typer.Option(None, "--rate-limit", help="Override rate limit seconds."),
    only: str = typer.Option(None, "--only", help="Comma-separated source names to include."),
    include_needs_validation: bool = typer.Option(
        True,
        "--include-needs-validation/--skip-needs-validation",
        help="Include sources tagged status: needs_validation.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="List the planned sources only."),
) -> None:
    """Collect a Telegram public-web dataset from t.me/s/<channel> with Scrapling."""
    overrides = _enrichment_overrides(
        max_pages=max_pages,
        limit=limit,
        rate_limit=rate_limit,
        only=only,
        include_needs_validation=include_needs_validation,
        dry_run=dry_run,
    )
    try:
        cfg = load_enrichment_config(config, overrides=overrides)
    except (FileNotFoundError, ValueError) as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc
    if cfg.telegram is None or not cfg.telegram.enabled or not cfg.telegram.sources:
        rprint("[yellow]No enabled Telegram sources in config.[/yellow]")
        raise typer.Exit(code=2)
    if dry_run:
        rprint(f"[cyan]Planned Telegram sources ({len(cfg.telegram.sources)}):[/cyan]")
        for source in cfg.telegram.sources:
            rprint(f"  - {source.name} -> {source.url}")
        return
    paths = get_data_paths(root=output_root)
    result = collect_telegram_public_web(cfg.telegram, paths=paths)
    write_collection_summary(result.summary(), paths)
    _render_collection_result(result, platform_label="Telegram")
    rprint(f"[green]telegram social_posts[/green] -> {result.posts_path}")


@social_app.command("collect-large-enriched")
def social_collect_large_enriched(
    config: Path = typer.Option(
        Path("configs/social_enrichment_sources.example.yaml"),
        "--config",
        help="Branch 12 public-web source config.",
    ),
    markets_path: Path = typer.Option(
        Path("data/normalized/polymarket/markets.parquet"),
        "--markets-path",
        help="Saved Polymarket markets parquet used for timeline-aware query generation.",
    ),
    reddit_root: Path = typer.Option(
        Path("data/staging/reddit"), "--reddit-root", help="Reddit staging dataset root."
    ),
    telegram_root: Path = typer.Option(
        Path("data/staging/telegram_web"), "--telegram-root", help="Telegram public-web staging root."
    ),
    output_root: Path = typer.Option(
        Path("data/staging/enriched"), "--output-root", help="Combined enriched staging root."
    ),
    target_total_records: int = typer.Option(1_000_000, "--target-total-records"),
    target_reddit_records: int = typer.Option(700_000, "--target-reddit-records"),
    target_telegram_records: int = typer.Option(300_000, "--target-telegram-records"),
    max_pages_per_source: int = typer.Option(50, "--max-pages-per-source"),
    max_pages_per_query: int = typer.Option(5, "--max-pages-per-query"),
    max_results_per_query: int = typer.Option(100, "--max-results-per-query"),
    max_markets: int = typer.Option(None, "--max-markets"),
    max_queries_per_market: int = typer.Option(10, "--max-queries-per-market"),
    fallback_window_days: int = typer.Option(180, "--fallback-window-days"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    stop_when_target_reached: bool = typer.Option(
        True, "--stop-when-target-reached/--no-stop-when-target-reached"
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    rate_limit_seconds: float = typer.Option(4.0, "--rate-limit-seconds"),
) -> None:
    """Collect the resumable Branch 12B public-web enrichment dataset."""
    # A live-PID lock under the output root stops a duplicate collector from
    # racing the same checkpoints. --dry-run is read-only, so it skips the lock.
    lock_ctx = (
        nullcontext()
        if dry_run
        else run_lock(output_root / "scrapling_web.lock", label="scrapling-collect-large-enriched")
    )
    try:
        with lock_ctx:
            if not dry_run:
                rprint(
                    "[cyan]Starting Scrapling public-web collection "
                    "(resumable — every fetched page is checkpointed)...[/cyan]"
                )
            cfg = load_enrichment_config(config)
            markets = load_markets_from_parquet(markets_path, limit=max_markets)
            # Honor the per-source ``base_fetcher`` from the config. Reddit needs the
            # StealthyFetcher (real browser) — the plain Fetcher is hard-blocked with
            # HTTP 403; Telegram's t.me preview pages serve the plain Fetcher fine.
            from polysignal.ingestion.social.enrichment import build_html_fetcher

            reddit_fetcher = None
            if cfg.reddit is not None:
                reddit_fetcher = build_html_fetcher(
                    cfg.reddit.base_fetcher,
                    user_agent=cfg.reddit.user_agent,
                    timeout=cfg.reddit.request_timeout_seconds,
                )
            telegram_fetcher = None
            telegram_base = getattr(cfg.telegram, "base_fetcher", None) if cfg.telegram else None
            if telegram_base:
                telegram_fetcher = build_html_fetcher(
                    telegram_base,
                    user_agent=cfg.telegram.user_agent,
                    timeout=cfg.telegram.request_timeout_seconds,
                )
            result = collect_large_enriched(
                config=cfg,
                markets=markets,
                reddit_paths=get_data_paths(root=reddit_root),
                telegram_paths=get_data_paths(root=telegram_root),
                output_paths=get_data_paths(root=output_root),
                reddit_fetcher=reddit_fetcher,
                telegram_fetcher=telegram_fetcher,
                options=LargeEnrichmentOptions(
                    target_total_records=target_total_records,
                    target_reddit_records=target_reddit_records,
                    target_telegram_records=target_telegram_records,
                    max_pages_per_source=max_pages_per_source,
                    max_pages_per_query=max_pages_per_query,
                    max_results_per_query=max_results_per_query,
                    max_markets=max_markets,
                    max_queries_per_market=max_queries_per_market,
                    fallback_window_days=fallback_window_days,
                    resume=resume,
                    stop_when_target_reached=stop_when_target_reached,
                    dry_run=dry_run,
                    rate_limit_seconds=rate_limit_seconds,
                ),
            )
    except RunLockError as exc:
        rprint(f"[red]ALREADY RUNNING[/red] {exc}")
        raise typer.Exit(code=3) from exc
    except (FileNotFoundError, ValueError) as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc
    table = Table(title="Branch 12B — large market-aware enrichment")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    for key in (
        "dry_run",
        "markets_input",
        "market_queries",
        "reddit_sources",
        "telegram_sources",
        "fetched_pages",
        "persisted_retrieval_rows",
    ):
        if key in result.summary:
            table.add_row(key, str(result.summary[key]))
    final_unique = result.summary.get("final_unique")
    if final_unique:
        table.add_row("final unique", json.dumps(final_unique, sort_keys=True))
    completion = result.summary.get("completion") or {}
    if completion.get("reason"):
        table.add_row("completion", str(completion["reason"]))
    reddit_circuit = (result.summary.get("circuit_breakers") or {}).get("reddit")
    if reddit_circuit:
        table.add_row(
            "reddit blocked",
            f"{reddit_circuit.get('reason', 'blocked')} "
            f"(cooldown until {completion.get('resume_after', 'n/a')})",
        )
    table.add_row("query plan rows", str(len(result.plan)))
    console.print(table)
    if dry_run:
        rprint("[cyan]Dry run only: no network calls or filesystem writes were performed.[/cyan]")
        return
    # Be explicit when Reddit yielded nothing: old.reddit.com returns HTTP 403 to
    # automated fetches. Branch 12 policy reports the block rather than evading it.
    if isinstance(final_unique, dict) and final_unique.get("reddit", 0) == 0:
        reddit_progress = (result.summary.get("collection_progress") or {}).get("reddit") or {}
        failed_sources = reddit_progress.get("failed_sources") or []
        detail = (
            reddit_circuit.get("reason")
            if reddit_circuit
            else (f"blocked sources: {', '.join(failed_sources)}" if failed_sources else None)
        )
        rprint(
            "[yellow]Reddit collected 0 rows[/yellow] — old.reddit.com blocks automated "
            "fetches with HTTP 403/429; this is honestly reported and respected, not bypassed."
            + (f" ({detail})" if detail else "")
        )
        rprint(
            "[yellow]Telegram public-web collection is unaffected; "
            "rerun later to let the Reddit cooldown lapse.[/yellow]"
        )
    rprint(f"[green]collection summary[/green] -> {output_root / 'reports/collection_summary.json'}")


@social_app.command("promote-staging")
def social_promote_staging(
    sources: str = typer.Option(
        "telegram_api,telegram_web,reddit,enriched",
        "--sources",
        help="Comma-separated staging sources to promote, in order.",
    ),
    output_dir: Path = typer.Option(
        None, "--output-dir", help="Canonical data root (defaults to data/)."
    ),
    include_synthetic: bool = typer.Option(
        False,
        "--include-synthetic/--exclude-synthetic",
        help="Include UI-stress/synthetic rows. Off by default — keep demo data out of the product.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run/--no-dry-run", help="Preview the merge without writing."
    ),
) -> None:
    """Promote staged social collections into canonical social_posts.parquet.

    Backs up the canonical file, merges + de-duplicates the chosen staging
    sources into it (keeping provenance), and refuses a catastrophic shrink.
    Synthetic/UI-stress rows are excluded by default. Run with --dry-run first
    to preview what would change.
    """
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    source_tuple = tuple(s.strip() for s in sources.split(",") if s.strip())
    try:
        summary = promote_staging_social(
            canonical_paths=paths,
            sources=source_tuple,
            exclude_synthetic=not include_synthetic,
            dry_run=dry_run,
        )
    except (RuntimeError, ValueError) as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Promote staging -> canonical social posts")
    table.add_column("Source", style="bold")
    table.add_column("Staged rows")
    table.add_column("Promoted")
    for name, info in summary.get("per_source", {}).items():
        table.add_row(
            name,
            str(info.get("rows", 0)),
            str(info.get("promoted", 0)) if info.get("present") else "(absent)",
        )
    console.print(table)
    rprint(
        f"canonical rows: [bold]{summary.get('before_rows')}[/bold] -> "
        f"[bold]{summary.get('after_rows')}[/bold] "
        f"(+{summary.get('rows_added', 0)}, "
        f"retrieval-dupes dropped {summary.get('retrieval_duplicates_dropped', 0)}, "
        f"synthetic excluded {summary.get('synthetic_excluded', 0)})"
    )
    if summary.get("backup"):
        rprint(f"[green]canonical backed up[/green] -> {summary['backup']}")
    if dry_run:
        rprint("[cyan]Dry run: nothing was written.[/cyan]")
    else:
        rprint(f"[green]canonical social posts[/green] -> {summary.get('output')}")


@social_app.command("usefulness-report")
def social_usefulness_report(
    output_dir: Path = typer.Option(
        None, "--output-dir", help="Canonical data root to read from (defaults to data/)."
    ),
    report_path: Path = typer.Option(
        None,
        "--report",
        help="Where to write the JSON report (default: <root>/reports/usefulness_report.json).",
    ),
    top_n: int = typer.Option(15, "--top-n", help="How many channels/markets to list."),
) -> None:
    """Report how many collected posts are actually useful for PolySignal.

    Reads the staging counts, canonical posts, linking verdicts, and stance
    labels and prints (plus saves) the headline funnel: collected -> unique ->
    matched-to-market -> accepted evidence, the stance distribution, per-market
    coverage, per-source coverage, and the remaining data gaps. Read-only.
    """
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    report = build_usefulness_report(paths=paths, top_n=top_n)

    posts = report["posts"]
    headline = report["headline"]
    table = Table(title="PolySignal usefulness funnel")
    table.add_column("Stage", style="bold")
    table.add_column("Count")
    table.add_row("collected (raw)", str(posts["collected_total"]))
    table.add_row("unique (after dedup)", str(posts["unique_after_dedup"]))
    table.add_row("with usable text", str(posts["with_usable_text"]))
    linking = report.get("linking", {})
    if linking.get("available"):
        table.add_row("matched to markets", str(linking["posts_matched_to_markets"]))
        table.add_row("social signal posts (>=weak)", str(linking["social_signal_unique_posts"]))
        table.add_row("accepted evidence (posts)", str(linking["accepted_evidence_unique_posts"]))
        verdict = linking["verdict_split"]
        table.add_row("  accepted pairs", str(verdict["accepted_evidence_pairs"]))
        table.add_row("  weakly-relevant pairs", str(verdict["weakly_relevant_pairs"]))
        table.add_row("  ambiguous pairs", str(verdict["ambiguous_pairs"]))
        table.add_row("  unrelated pairs", str(verdict["unrelated_pairs"]))
        table.add_row("  LLM judge applied", str(linking.get("llm_judge_applied", 0)))
    table.add_row("SOCIAL SIGNAL %", f"{headline.get('social_signal_percentage', 0)}%")
    table.add_row("ACCEPTED EVIDENCE %", f"{headline.get('accepted_evidence_percentage', 0)}%")
    console.print(table)

    staging = Table(title="Per-source staging coverage")
    staging.add_column("Source", style="bold")
    staging.add_column("Rows")
    staging.add_column("Present")
    for name, info in report["staging"].items():
        staging.add_row(name, str(info["rows"]), "yes" if info["present"] else "no")
    console.print(staging)

    stance = report.get("stance", {})
    if stance.get("available"):
        rprint(f"[bold]stance distribution[/bold]: {json.dumps(stance['distribution'])}")
    if report["data_gaps"]:
        rprint("[yellow]Data gaps:[/yellow]")
        for gap in report["data_gaps"]:
            rprint(f"  - {gap}")

    out = report_path or (paths.root / "reports" / "usefulness_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    rprint(f"[green]usefulness report[/green] -> {out}")


@social_app.command("merge-enriched")
def social_merge_enriched(
    reddit_root: Path = typer.Option(
        Path("data_reddit_scrapling_eval"), "--reddit-root", help="Reddit dataset root."
    ),
    telegram_root: Path = typer.Option(
        Path("data_telegram_enriched_eval"), "--telegram-root", help="Telegram dataset root."
    ),
    output_root: Path = typer.Option(
        Path("data_social_enriched_eval"), "--output-root", help="Combined dataset root."
    ),
) -> None:
    """Merge the Reddit and Telegram datasets into the combined enriched dataset."""
    output_paths = get_data_paths(root=output_root)
    result = merge_enriched_datasets(
        reddit_paths=get_data_paths(root=reddit_root),
        telegram_paths=get_data_paths(root=telegram_root),
        output_paths=output_paths,
    )
    summary = result.summary()
    merge_path = output_paths.analytical / "enrichment" / "merge_summary.json"
    merge_path.parent.mkdir(parents=True, exist_ok=True)
    merge_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    table = Table(title="Branch 12 — combined enriched dataset")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("reddit rows", str(result.reddit_rows))
    table.add_row("telegram rows", str(result.telegram_rows))
    table.add_row("combined rows (incl. dup-flagged)", str(result.combined_rows))
    table.add_row("unique rows", str(result.unique_rows))
    table.add_row("duplicate rows flagged", str(result.duplicate_rows_flagged))
    table.add_row("platform breakdown", ", ".join(f"{k}={v}" for k, v in sorted(result.platform_counts.items())))
    table.add_row("combined parquet", str(result.posts_path))
    console.print(table)
    rprint(f"[green]combined social_posts[/green] -> {result.posts_path}")


def _enriched_verdict(
    *,
    reddit_summary: dict | None,
    telegram_summary: dict | None,
    merge_summary: dict | None,
    downstream: dict | None = None,
) -> str:
    """Honest verdict.

    NOT READY when no dataset was produced. READY only when both platform
    datasets are substantial *and* the enriched profile carries usable
    leakage-free downstream evidence (at least one evidence-sufficient market).
    Otherwise CONDITIONALLY READY: the datasets and pipeline are delivered, but
    the enriched profile does not yet demonstrate improved pre-close evidence.
    """
    breakdown = (merge_summary or {}).get("platform_breakdown", {})
    reddit_rows = breakdown.get("reddit", 0)
    telegram_rows = breakdown.get("telegram", 0)
    combined = reddit_rows + telegram_rows
    if combined == 0:
        return "BRANCH 12 NOT READY"
    both_platforms = reddit_rows > 0 and telegram_rows > 0
    evidence_sufficient = int((downstream or {}).get("evidence_sufficient_markets", 0) or 0)
    if both_platforms and combined >= 200 and evidence_sufficient > 0:
        return "BRANCH 12 READY"
    return "BRANCH 12 CONDITIONALLY READY"


@social_app.command("build-enriched-profile")
def social_build_enriched_profile(
    input_root: Path = typer.Option(
        Path("data_social_enriched_eval"), "--input-root", help="Combined enriched dataset root."
    ),
    reddit_root: Path = typer.Option(
        Path("data_reddit_scrapling_eval"), "--reddit-root", help="Reddit dataset root."
    ),
    telegram_root: Path = typer.Option(
        Path("data_telegram_enriched_eval"), "--telegram-root", help="Telegram dataset root."
    ),
    profile: str = typer.Option(
        "real_social_enriched_main", "--profile", help="Enriched Branch 09 profile name."
    ),
    config: Path = typer.Option(
        Path("configs/social_enrichment_sources.example.yaml"), "--config", help="Source config."
    ),
    reference_root: Path = typer.Option(
        Path("data_resolved_eval"),
        "--reference-root",
        help="Baseline real_optional dataset root for the coverage comparison.",
    ),
    markets_path: Path = typer.Option(
        None,
        "--markets-path",
        help="Optional markets.parquet for a deterministic market token-overlap estimate.",
    ),
    report_path: Path = typer.Option(
        Path("BRANCH_12_DATASET_ENRICHMENT.md"), "--report-path", help="Coverage report path."
    ),
    index_path: Path = typer.Option(
        Path("BRANCH_12_ENRICHMENT_ARTIFACT_INDEX.json"),
        "--index-path",
        help="Artifact index path.",
    ),
) -> None:
    """Assemble the Branch 12 coverage report + artifact index and final verdict.

    Reads the on-disk collection/merge summaries and whatever downstream
    linking/stance/SPI/Branch 09 artifacts exist for the enriched root. It never
    fabricates downstream results — run linking/stance/features/evaluate against
    ``--output-dir <input_root>`` first if you want those covered.
    """
    output_paths = get_data_paths(root=input_root)
    reddit_summary = read_collection_summary(get_data_paths(root=reddit_root))
    telegram_summary = read_collection_summary(get_data_paths(root=telegram_root))
    merge_path = output_paths.analytical / "enrichment" / "merge_summary.json"
    merge_summary = (
        json.loads(merge_path.read_text(encoding="utf-8")) if merge_path.exists() else None
    )
    downstream = read_downstream_coverage(output_paths, profile=profile)
    if reference_root and reference_root.exists():
        comparison = compare_with_real_optional(get_data_paths(root=reference_root))
        if comparison:
            downstream["comparison_real_optional"] = comparison
    if markets_path and markets_path.exists():
        try:
            markets = load_markets_from_parquet(markets_path)
            records = read_parquet(output_paths.social_posts_parquet, SocialPostRecord)
            downstream["market_token_overlap"] = estimate_market_token_overlap(records, markets)
        except (FileNotFoundError, ValueError) as exc:
            rprint(f"[yellow]market token-overlap skipped: {exc}[/yellow]")
    verdict = _enriched_verdict(
        reddit_summary=reddit_summary,
        telegram_summary=telegram_summary,
        merge_summary=merge_summary,
        downstream=downstream,
    )
    index = assemble_artifact_index(
        config_path=config,
        profile=profile,
        reddit_summary=reddit_summary,
        telegram_summary=telegram_summary,
        merge_summary=merge_summary,
        downstream=downstream,
        verdict=verdict,
    )
    write_report_and_index(index, report_path=report_path, index_path=index_path)
    rprint(f"[green]coverage report[/green] -> {report_path}")
    rprint(f"[green]artifact index[/green] -> {index_path}")
    rprint(f"[bold]{verdict}[/bold]")


@social_app.command("augment-main-for-ui-test")
def social_augment_main_for_ui_test(
    root: Path = typer.Option(
        Path("data_social_enriched_eval"),
        "--root",
        help="Existing enriched dataset root to augment in place.",
    ),
    markets: int = typer.Option(6, "--markets", min=1, help="Must be 6 for this pinned run."),
    messages_per_market: int = typer.Option(
        150, "--messages-per-market", min=20, help="Generated social rows per selected market."
    ),
    augmentation_version: str = typer.Option(
        "ui_stress_test_12E",
        "--augmentation-version",
        help="Rerun marker stored in query_provenance.",
    ),
    seed: int = typer.Option(1205, "--seed", help="Deterministic generation seed."),
    rerun_pipeline: bool = typer.Option(
        False,
        "--rerun-pipeline/--no-rerun-pipeline",
        help="After writing social_posts, rerun linking, stance, SPI, Branch 09, and dashboard artifacts.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview selected markets and generated rows without writing files or rerunning pipelines.",
    ),
) -> None:
    """Append deterministic Reddit/Telegram-like UI stress rows to the main dataset."""
    try:
        plan = (
            plan_augmentation(
                root=root,
                markets=markets,
                messages_per_market=messages_per_market,
                augmentation_version=augmentation_version,
                seed=seed,
                dry_run=True,
            )
            if dry_run
            else augment_main_social_dataset(
                root=root,
                markets=markets,
                messages_per_market=messages_per_market,
                augmentation_version=augmentation_version,
                seed=seed,
                dry_run=False,
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _render_ui_stress_plan(plan)
    if dry_run:
        rprint("[yellow]--dry-run set, no files written and no pipeline rerun performed.[/yellow]")
        return

    rprint(f"[green]social_posts parquet[/green] -> {root / 'normalized' / 'social' / 'social_posts.parquet'}")
    rprint(f"[green]social_posts csv[/green]     -> {root / 'normalized' / 'social' / 'social_posts.csv'}")

    if rerun_pipeline:
        _run_ui_stress_pipeline(root)


def _render_ui_stress_plan(plan) -> None:
    summary = Table(title="Main UI stress-test augmentation")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value")
    summary.add_row("selected markets", str(len(plan.markets)))
    summary.add_row("generated rows", str(len(plan.generated_rows)))
    summary.add_row("existing rows", str(plan.existing_rows))
    summary.add_row("previous augmented rows removed", str(plan.previous_augmented_rows))
    summary.add_row("final rows after write", str(plan.final_rows))
    summary.add_row(
        "platform breakdown",
        ", ".join(f"{k}={v}" for k, v in sorted(plan.platform_breakdown.items())),
    )
    summary.add_row(
        "stance-hint breakdown",
        ", ".join(f"{k}={v}" for k, v in sorted(plan.stance_breakdown.items())),
    )
    summary.add_row("files", ", ".join(str(path) for path in plan.files_to_update))
    console.print(summary)

    markets_table = Table(title="Selected markets")
    markets_table.add_column("domain")
    markets_table.add_column("market_id")
    markets_table.add_column("close_time")
    markets_table.add_column("question")
    for market in plan.markets:
        markets_table.add_row(
            market.domain_label,
            _compact_id(market.market_id),
            market.close_time.isoformat(),
            _terminal_text(market.question, limit=70),
        )
    console.print(markets_table)


def _run_ui_stress_pipeline(root: Path) -> None:
    commands = [
        [
            sys.executable,
            "-m",
            "polysignal",
            "linking",
            "score-candidates",
            "--markets-path",
            str(root / "normalized" / "polymarket" / "markets.parquet"),
            "--posts-path",
            str(root / "normalized" / "social" / "social_posts.parquet"),
            "--output-path",
            str(root / "normalized" / "social" / "linked_posts.parquet"),
            "--no-segments",
        ],
        [
            sys.executable,
            "-m",
            "polysignal",
            "stance",
            "score",
            "--output-dir",
            str(root),
            "--min-link-confidence",
            "0.45",
            "--force",
        ],
        [
            sys.executable,
            "-m",
            "polysignal",
            "features",
            "compute-social-pressure",
            "--output-dir",
            str(root),
            "--dataset-mode",
            "real",
            "--overwrite",
            "--export-csv",
        ],
        [
            sys.executable,
            "-m",
            "polysignal",
            "evaluate",
            "backtest",
            "--output-dir",
            str(root),
            "--benchmark-profile",
            "real_social_enriched_main",
            "--overwrite",
            "--export-csv",
        ],
        [
            sys.executable,
            "-m",
            "polysignal",
            "evaluate",
            "build-dashboard-view",
            "--output-dir",
            str(root),
            "--benchmark-profile",
            "real_social_enriched_main",
            "--overwrite",
            "--export-csv",
        ],
    ]
    for command in commands:
        rprint(f"[bold]running[/bold] {' '.join(command)}")
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            rprint(f"[red]pipeline command failed[/red] exit={completed.returncode}")
            raise typer.Exit(code=completed.returncode)


@social_app.command("inspect-posts")
def social_inspect_posts(
    posts_path: Path = typer.Option(
        None,
        "--posts-path",
        help=(
            "Path to social_posts.parquet. Defaults to data/normalized/social/social_posts.parquet."
        ),
    ),
    limit: int = typer.Option(20, "--limit", help="Maximum rows to display."),
    include_duplicates: bool = typer.Option(
        False, "--include-duplicates/--unique-only", help="Show duplicate rows."
    ),
    output_dir: Path = typer.Option(None, "--output-dir", help="Override the data root."),
) -> None:
    """Show a summary of the collected social_posts.parquet file."""
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    target = posts_path or paths.social_posts_parquet
    if not target.exists():
        rprint(
            f"[yellow]No social posts file at {target}.[/yellow] "
            "Run Reddit or Telegram social ingestion first."
        )
        raise typer.Exit(code=2)

    records = read_parquet(target, SocialPostRecord)
    if not records:
        rprint(f"[yellow]File at {target} is empty.[/yellow]")
        raise typer.Exit(code=2)

    if not include_duplicates:
        records = [r for r in records if not r.is_duplicate]

    by_platform: dict[str, int] = {}
    candidate_markets: set[str] = set()
    without_candidate = 0
    by_language: dict[str, int] = {}
    for r in records:
        by_platform[r.platform.value] = by_platform.get(r.platform.value, 0) + 1
        if r.market_id_candidate:
            candidate_markets.add(r.market_id_candidate)
        else:
            without_candidate += 1
        language = r.language or "unknown"
        by_language[language] = by_language.get(language, 0) + 1

    summary = Table(title=f"social_posts.parquet — {target}")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value")
    summary.add_row("total rows", str(len(records)))
    summary.add_row("distinct post_ids", str(len({r.post_id for r in records})))
    summary.add_row("distinct candidate markets", str(len(candidate_markets)))
    summary.add_row("rows without market candidate", str(without_candidate))
    summary.add_row("channels/sources", str(len({r.subreddit_or_channel for r in records if r.subreddit_or_channel})))
    summary.add_row("authors/senders", str(len({r.author_id for r in records if r.author_id})))
    summary.add_row("empty text rows", str(sum(1 for r in records if not r.text.strip())))
    summary.add_row("duplicate rows", str(sum(1 for r in records if r.is_duplicate)))
    summary.add_row("date range", f"{min(r.created_at for r in records).isoformat()} -> {max(r.created_at for r in records).isoformat()}")
    summary.add_row("language", ", ".join(f"{k}={v}" for k, v in sorted(by_language.items())))
    summary.add_row("with views", str(sum(1 for r in records if r.views_count is not None)))
    summary.add_row("with shares/forwards", str(sum(1 for r in records if r.shares_count is not None)))
    summary.add_row("with comments/replies", str(sum(1 for r in records if r.comments_count is not None)))
    summary.add_row("with reactions", str(sum(1 for r in records if r.reactions_count is not None)))
    summary.add_row(
        "by platform",
        ", ".join(f"{k}={v}" for k, v in sorted(by_platform.items())),
    )
    console.print(summary)

    table = Table(title=f"First {min(limit, len(records))} rows")
    table.add_column("platform")
    table.add_column("post_id")
    table.add_column("subreddit/channel")
    table.add_column("title (truncated)")
    table.add_column("market_id_candidate")
    table.add_column("dup?")
    for r in records[:limit]:
        table.add_row(
            r.platform.value,
            _compact_id(r.post_id),
            (r.subreddit_or_channel or "-")[:24],
            _terminal_text(r.title or r.text, limit=60),
            _compact_id(r.market_id_candidate or "-", limit=16),
            "Y" if r.is_duplicate else "N",
        )
    console.print(table)


@social_app.command("segment-posts")
def social_segment_posts(
    posts_path: Path = typer.Option(
        None,
        "--posts-path",
        help=(
            "Source social_posts.parquet (defaults to data/normalized/social/social_posts.parquet)."
        ),
    ),
    output_path: Path = typer.Option(
        None,
        "--output-path",
        help=(
            "Where to write the derived segments parquet (defaults to "
            "data/normalized/social/social_post_segments.parquet)."
        ),
    ),
    include_duplicates: bool = typer.Option(
        False,
        "--include-duplicates/--unique-only",
        help="Include duplicate retrieval rows when segmenting.",
    ),
) -> None:
    """Branch 05G — split long multi-topic digest posts into segments.

    The original social_posts.parquet is preserved untouched. Segments are
    written to a separate parquet keyed by parent_post_id so the scorer
    can prefer segments without losing the original message.
    """
    paths = get_data_paths()
    source = posts_path or paths.social_posts_parquet
    target = output_path or paths.social_post_segments_parquet
    if not source.exists():
        rprint(
            f"[red]No social_posts.parquet at {source}.[/red] "
            "Ingest social posts before segmenting."
        )
        raise typer.Exit(code=2)
    records = read_parquet(source, SocialPostRecord)
    if not include_duplicates:
        records = [r for r in records if not r.is_duplicate]
    segments = segment_records(records)
    write_segments(segments, target)
    parent_ids = {seg.parent_post_id for seg in segments}
    multi = sum(
        1
        for seg in segments
        if seg.segmentation_method not in ("single", "empty") and seg.segment_index == 0
    )
    table = Table(title="Branch 05G — social post segmentation")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("input posts (after dedup)", str(len(records)))
    table.add_row("non-empty parents", str(len(parent_ids)))
    table.add_row("total segments", str(len(segments)))
    table.add_row(
        "avg segments / non-empty post",
        f"{(len(segments) / len(parent_ids)) if parent_ids else 0:.2f}",
    )
    table.add_row("multi-topic posts split", str(multi))
    console.print(table)
    rprint(f"[green]segments[/green] -> {target}")


def _render_social_summary(result: SocialIngestionResult, *, dry_run: bool = False) -> None:
    title_suffix = " (dry-run, no API calls)" if dry_run else ""
    table = Table(
        title=f"Social ingestion — {result.platform.value} run {result.run_id}{title_suffix}"
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("markets seen", str(result.markets_seen))
    table.add_row("queries generated", str(result.queries_generated))
    if not dry_run:
        table.add_row("queries executed", str(result.queries_executed))
        failed = sum(1 for a in result.attempts if a.error)
        table.add_row("queries failed", str(failed))
        table.add_row("posts collected", str(result.posts_collected))
        table.add_row("unique posts (dataset)", str(result.unique_posts))
        table.add_row("duplicate rows (dataset)", str(result.duplicates))
    console.print(table)

    if result.raw_dir is not None:
        rprint(f"[green]raw snapshots[/green]   -> {result.raw_dir}")
    if result.social_posts_parquet is not None:
        rprint(f"[green]social posts[/green]    -> {result.social_posts_parquet}")
    if result.queries_parquet is not None:
        rprint(f"[green]queries parquet[/green] -> {result.queries_parquet}")
    if dry_run:
        rprint(
            "[yellow]--dry-run was set; no platform API calls were made "
            "and no social_posts.parquet was written.[/yellow]"
        )

    failing = [a for a in result.attempts if a.error]
    if failing:
        rprint("\n[bold]Failed queries (first 5):[/bold]")
        for attempt in failing[:5]:
            rprint(f"  - {attempt.query!r} ({attempt.market_id}): {attempt.error}")


# ---------------------------------------------------------------------------
# Linking (Branch 05) — semantic relevance scorer + linked_posts.parquet
# ---------------------------------------------------------------------------


def _load_markets_or_fixtures(
    markets_path: Path | None,
    *,
    use_fixtures: bool,
) -> tuple[list, bool]:
    """Return ``(markets, used_fixtures)`` honouring CLI flags.

    Falls back to offline fixtures when ``use_fixtures`` is set or when
    the markets parquet does not exist. Never invents data silently —
    the boolean return value lets the CLI report which source was used.
    """
    paths = get_data_paths()
    target = markets_path or paths.polymarket_markets_parquet
    if use_fixtures:
        return list(fixture_markets()), True
    if not target.exists():
        rprint(
            f"[yellow]Markets parquet not found at {target}.[/yellow] "
            "Falling back to offline fixtures."
        )
        return list(fixture_markets()), True
    markets = load_markets_from_parquet(target, limit=None)
    return markets, False


def _load_records_or_none(posts_path: Path | None) -> tuple[list[SocialPostRecord], Path | None]:
    """Read ``social_posts.parquet`` if available; otherwise return ``[]``."""
    paths = get_data_paths()
    target = posts_path or paths.social_posts_parquet
    if not target.exists():
        return [], target
    return read_parquet(target, SocialPostRecord), target


def _build_cli_llm_judge(
    *,
    enabled: bool,
    provider: str | None,
    model: str | None,
    borderline_only: bool,
    min_score: float,
    max_score: float,
    cache_dir: Path | None,
    dry_run: bool,
) -> LLMRelevanceJudge | None:
    """Build the optional offline-safe or explicitly configured live 05C judge."""
    if not enabled:
        return None
    paths = get_data_paths()
    settings = get_settings()
    resolved_provider = provider or settings.llm.provider
    resolved_model = model or settings.llm.model or "unconfigured"
    client = NullLLMClient()
    if not dry_run:
        if resolved_provider == "openai":
            api_key = settings.llm.openai_api_key or settings.llm.api_key
        elif resolved_provider == "anthropic":
            api_key = settings.llm.anthropic_api_key or settings.llm.api_key
        else:
            api_key = ""
        try:
            client = build_llm_client(
                provider=resolved_provider,
                api_key=api_key,
                model_id=resolved_model,
                temperature=settings.llm.temperature,
                timeout_seconds=settings.llm.timeout_seconds,
                max_tokens=settings.llm.max_tokens,
            )
        except RuntimeError as exc:
            rprint(f"[red]ERROR[/red] {exc}")
            raise typer.Exit(code=2) from exc
    return LLMRelevanceJudge(
        client,
        JSONCache(cache_dir or paths.linking_llm_cache_dir),
        provider=resolved_provider,
        model_id=resolved_model,
        policy=LLMJudgePolicy(
            borderline_only=borderline_only,
            min_score=min_score,
            max_score=max_score,
        ),
        dry_run=dry_run,
    )


@linking_app.command("score-candidates")
def linking_score_candidates(
    markets_path: Path = typer.Option(
        None,
        "--markets-path",
        help="Path to markets.parquet (defaults to the Polymarket output).",
    ),
    posts_path: Path = typer.Option(
        None,
        "--posts-path",
        help=(
            "Path to social_posts.parquet (defaults to "
            "data/normalized/social/social_posts.parquet)."
        ),
    ),
    output_path: Path = typer.Option(
        None,
        "--output-path",
        help=(
            "Where to write linked_posts.parquet (defaults to "
            "data/normalized/social/linked_posts.parquet)."
        ),
    ),
    use_fixtures: bool = typer.Option(
        False,
        "--use-fixtures/--no-use-fixtures",
        help=(
            "Score the offline fixture markets/posts instead of the parquet "
            "datasets. Useful for a clean-checkout smoke run."
        ),
    ),
    include_duplicates: bool = typer.Option(
        False,
        "--include-duplicates/--unique-only",
        help="Include duplicate retrieval rows from social_posts.parquet.",
    ),
    candidate_mode: CandidateMode = typer.Option(
        CandidateMode.AUTO,
        "--candidate-mode",
        help="Candidate policy: auto, provenance, prefilter, or full-scan.",
    ),
    max_candidate_markets_per_post: int = typer.Option(
        10,
        "--max-candidate-markets-per-post",
        min=1,
        help="Maximum prefilter candidates per provenance-free post.",
    ),
    min_anchor_score: float = typer.Option(
        0.35,
        "--min-anchor-score",
        min=0.0,
        max=1.0,
        help="Minimum meaningful-anchor score for label promotion.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--no-dry-run",
        help="Score pairs and print the summary, but skip the parquet write.",
    ),
    use_embeddings: bool = typer.Option(
        False,
        "--use-embeddings/--no-embeddings",
        help=(
            "Branch 05B: enable the embedding-similarity overlay. Requires "
            "sentence-transformers installed; falls back to the null "
            "provider with a warning otherwise."
        ),
    ),
    embedding_model: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2",
        "--embedding-model",
        help="HuggingFace / sentence-transformers model id to use.",
    ),
    embedding_cache_dir: Path = typer.Option(
        None,
        "--embedding-cache-dir",
        help="Directory for the JSON embedding cache (default: data/cache/embeddings).",
    ),
    use_llm_judge: bool = typer.Option(
        False,
        "--use-llm-judge/--no-llm-judge",
        help="Branch 05C: opt into the cached structured LLM relevance verifier.",
    ),
    llm_provider: str = typer.Option(
        None, "--llm-provider", help="LLM provider id for audit/cache."
    ),
    llm_model: str = typer.Option(None, "--llm-model", help="LLM model id for audit/cache."),
    llm_borderline_only: bool = typer.Option(
        True,
        "--llm-borderline-only/--llm-all",
        help="Select only borderline/low-confidence rows for LLM verification (default).",
    ),
    llm_min_score: float = typer.Option(0.35, "--llm-min-score"),
    llm_max_score: float = typer.Option(0.75, "--llm-max-score"),
    llm_cache_dir: Path = typer.Option(None, "--llm-cache-dir"),
    llm_dry_run: bool = typer.Option(
        False,
        "--llm-dry-run/--no-llm-dry-run",
        help="Report 05C row selection without making LLM calls.",
    ),
    use_segments: bool = typer.Option(
        True,
        "--use-segments/--no-segments",
        help=(
            "Branch 05G: when social_post_segments.parquet exists, score "
            "individual segments instead of whole multi-topic digest posts. "
            "Falls back to whole posts silently when no segments file is "
            "present."
        ),
    ),
    segments_path: Path = typer.Option(
        None,
        "--segments-path",
        help="Override path to social_post_segments.parquet.",
    ),
) -> None:
    """Score Branch 04 retrieval candidates -> ``linked_posts.parquet``.

    By default this command reads:

    * ``data/normalized/polymarket/markets.parquet`` for the markets;
    * ``data/normalized/social/social_posts.parquet`` for the candidate posts.

    In ``auto`` mode pairs use ``SocialPostRecord.market_id_candidate``
    provenance when available and bounded prefiltering otherwise. When
    ``social_posts.parquet`` is missing the command falls back to the offline
    fixture set so the operator can still verify the pipeline end-to-end.

    When ``social_post_segments.parquet`` exists and ``--use-segments`` is
    on (default), scoring runs on segments instead of whole posts so that
    anchors from unrelated sections of a multi-topic digest cannot combine
    into a false positive.
    """
    markets, used_fixtures = _load_markets_or_fixtures(markets_path, use_fixtures=use_fixtures)
    if not markets:
        rprint("[red]ERROR[/red] no markets available to score against.")
        raise typer.Exit(code=1)

    paths = get_data_paths()
    target_output = output_path or paths.linked_posts_parquet
    relevance_config = RelevanceConfig(min_anchor_score=min_anchor_score)
    candidate_diagnostics = None

    provider = build_default_provider(
        enabled=use_embeddings,
        model_id=embedding_model,
        cache_dir=embedding_cache_dir or paths.embedding_cache_dir,
    )
    if use_embeddings and not provider.is_available():
        rprint(
            "[yellow]embeddings requested but unavailable[/yellow] "
            "(install sentence-transformers). Falling back to deterministic-only."
        )
    llm_judge = _build_cli_llm_judge(
        enabled=use_llm_judge,
        provider=llm_provider,
        model=llm_model,
        borderline_only=llm_borderline_only,
        min_score=llm_min_score,
        max_score=llm_max_score,
        cache_dir=llm_cache_dir,
        dry_run=llm_dry_run,
    )

    segments_used = False
    segment_lookup: dict[str, tuple[str, int, str]] | None = None

    if used_fixtures:
        posts = fixture_posts()
        rows = run_linking_cross_product(
            posts,
            markets,
            config=relevance_config,
            embedding_provider=provider,
            llm_judge=llm_judge,
        )
        source = "fixtures"
    else:
        records, posts_target = _load_records_or_none(posts_path)
        if not records:
            rprint(
                f"[yellow]No social_posts.parquet at {posts_target}.[/yellow] "
                "Falling back to offline fixtures for the demo.\n"
                "Run `polysignal social ingest --platform reddit` to collect "
                "real posts."
            )
            posts = fixture_posts()
            markets = list(fixture_markets())
            rows = run_linking_cross_product(
                posts,
                markets,
                config=relevance_config,
                embedding_provider=provider,
                llm_judge=llm_judge,
            )
            used_fixtures = True
            source = "fixtures (posts missing)"
        else:
            seg_target = segments_path or paths.social_post_segments_parquet
            scoring_records = records
            if use_segments and seg_target.exists():
                segment_models = read_segments(seg_target)
                if segment_models:
                    base_lookup = {r.post_id: r for r in records}
                    scoring_records = segment_models_to_records(segment_models, base_lookup)
                    segment_lookup = {
                        seg.segment_id: (
                            seg.parent_post_id,
                            seg.segment_index,
                            seg.segment_text[:240].replace("\n", " "),
                        )
                        for seg in segment_models
                    }
                    segments_used = True
            generated = build_pairs_from_records_with_diagnostics(
                scoring_records,
                markets,
                include_duplicates=include_duplicates,
                candidate_mode=candidate_mode,
                max_candidate_markets_per_post=max_candidate_markets_per_post,
            )
            candidate_diagnostics = generated.diagnostics
            rows = score_pairs(
                generated.pairs,
                config=relevance_config,
                embedding_provider=provider,
                llm_judge=llm_judge,
                segment_lookup=segment_lookup,
            )
            source = (
                f"segments={len(scoring_records)} (from {len(records)} posts)"
                if segments_used
                else f"records={len(records)}"
            )

    by_label = summarize_by_label(rows)
    table = Table(title="Branch 05 — semantic-relevance linking")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("markets", str(len(markets)))
    table.add_row("source", source)
    table.add_row("pairs scored", str(len(rows)))
    if candidate_diagnostics is not None:
        table.add_row("candidate mode", candidate_diagnostics.candidate_mode.value)
        table.add_row("posts", str(candidate_diagnostics.posts))
        table.add_row(
            "avg candidates/post",
            f"{candidate_diagnostics.average_candidate_markets_per_post:.2f}",
        )
        table.add_row(
            "skipped posts (no anchors)",
            str(candidate_diagnostics.skipped_posts_no_candidate_anchors),
        )
        table.add_row("fallback mode used", candidate_diagnostics.fallback_mode_used)
        table.add_row("top inclusion reason", candidate_diagnostics.top_inclusion_reason)
    table.add_row("segments used", "yes" if segments_used else "no")
    table.add_row(
        "embeddings",
        f"{provider.model_id}" if provider.is_available() else "disabled",
    )
    if llm_judge is not None:
        table.add_row(
            "llm judge",
            f"dry-run selected={llm_judge.stats.selected}/{llm_judge.stats.considered}",
        )
    elif use_llm_judge:
        table.add_row("llm judge", "unavailable; baseline preserved")
    else:
        table.add_row("llm judge", "disabled")
    for label in (
        "relevant",
        "weakly_relevant",
        "ambiguous",
        "unrelated",
    ):
        table.add_row(f"  {label}", str(by_label.get(label, 0)))
    console.print(table)

    if dry_run:
        rprint("[yellow]--dry-run set, no parquet written.[/yellow]")
        return

    write_linked_posts(rows, target_output)
    rprint(f"[green]linked posts[/green] -> {target_output}")
    if used_fixtures:
        rprint(
            "[yellow]Note:[/yellow] output was generated from offline fixtures; "
            "rerun after live Reddit ingestion for real data."
        )


def _effective_link_score(row: LinkedPost) -> float:
    return (
        row.final_relevance_score if row.final_relevance_score is not None else row.relevance_score
    )


def _effective_link_confidence(row: LinkedPost) -> float:
    return row.final_confidence if row.final_confidence is not None else row.confidence


def _write_linked_posts_csv(rows: Sequence[LinkedPost], target: Path) -> Path:
    """Export analysis-friendly linked-post columns in the displayed sort order."""
    columns = [
        "market_id",
        "market_question",
        "post_id",
        "platform",
        "relevance_label",
        "relevance_score",
        "confidence",
        "constraint_match",
        "anchor_score",
        "constraint_score",
        "domain_mismatch",
        "rejection_reason",
        "post_text_preview",
        "explanation",
        "final_relevance_label",
        "final_relevance_score",
        "final_confidence",
        "llm_judged",
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "market_id": row.market_id,
                    "market_question": row.market_question,
                    "post_id": row.post_id,
                    "platform": row.platform.value,
                    "relevance_label": row.relevance_label.value,
                    "relevance_score": row.relevance_score,
                    "confidence": row.confidence,
                    "constraint_match": row.constraint_match.value,
                    "anchor_score": row.anchor_score,
                    "constraint_score": row.constraint_score,
                    "domain_mismatch": row.domain_mismatch,
                    "rejection_reason": row.rejection_reason,
                    "post_text_preview": row.post_text_preview,
                    "explanation": row.explanation,
                    "final_relevance_label": (
                        row.final_relevance_label.value if row.final_relevance_label else None
                    ),
                    "final_relevance_score": row.final_relevance_score,
                    "final_confidence": row.final_confidence,
                    "llm_judged": row.llm_judged,
                }
            )
    return target


@linking_app.command("inspect-links")
def linking_inspect_links(
    links_path: Path = typer.Option(
        None,
        "--links-path",
        help="Path to linked_posts.parquet.",
    ),
    limit: int = typer.Option(20, "--limit", help="Maximum rows to display."),
    label: str = typer.Option(
        None,
        "--label",
        help="Filter to one of: relevant, weakly_relevant, ambiguous, unrelated.",
    ),
    min_score: float = typer.Option(
        0.0, "--min-score", help="Only show rows whose relevance_score >= min."
    ),
    export_csv: Path = typer.Option(
        None,
        "--export-csv",
        help="Export all filtered rows, sorted by relevance_score, for manual review.",
    ),
) -> None:
    """Inspect and optionally export the contents of linked_posts.parquet."""
    paths = get_data_paths()
    target = links_path or paths.linked_posts_parquet
    if not target.exists():
        rprint(
            f"[yellow]No linked_posts file at {target}.[/yellow] "
            "Run `polysignal linking score-candidates` first."
        )
        raise typer.Exit(code=2)
    rows = read_linked_posts(target)
    if not rows:
        rprint(f"[yellow]File at {target} is empty.[/yellow]")
        raise typer.Exit(code=2)

    if label:
        normalized_label = label.strip().lower()
        rows = [
            r
            for r in rows
            if (r.final_relevance_label or r.relevance_label).value == normalized_label
        ]
    if min_score > 0.0:
        rows = [r for r in rows if r.relevance_score >= min_score]

    rows = sorted(rows, key=lambda r: r.relevance_score, reverse=True)
    if export_csv is not None:
        _write_linked_posts_csv(rows, export_csv)
        rprint(f"[green]CSV export[/green] -> {export_csv} ({len(rows)} rows)")

    summary = Table(title=f"linked_posts.parquet — {target}")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value")
    summary.add_row("rows shown", str(min(limit, len(rows))))
    summary.add_row("rows after filter", str(len(rows)))
    by_label = summarize_by_label(rows)
    summary.add_row("by label", ", ".join(f"{k}={v}" for k, v in sorted(by_label.items())))
    console.print(summary)

    shown = rows[:limit]
    table = Table(title=f"Top {len(shown)} links by relevance_score")
    table.add_column("#")
    table.add_column("market")
    table.add_column("post")
    table.add_column("label")
    table.add_column("score")
    table.add_column("conf.")
    table.add_column("constraint")
    for index, r in enumerate(shown, start=1):
        table.add_row(
            str(index),
            _compact_id(r.market_id),
            _compact_id(r.post_id),
            (r.final_relevance_label or r.relevance_label).value,
            f"{_effective_link_score(r):.3f}",
            f"{_effective_link_confidence(r):.2f}",
            (r.final_constraint_match or r.constraint_match).value,
        )
    console.print(table)
    for index, row in enumerate(shown, start=1):
        console.print(
            f"{index}. market: {_compact_id(row.market_id)} | "
            f"{_terminal_text(row.market_question, limit=100)}",
            markup=False,
        )
        console.print(
            f"   post:   {_compact_id(row.post_id)} | "
            f"{_terminal_text(row.post_text_preview, limit=140)}",
            markup=False,
        )
        console.print(f"   why:    {_terminal_text(row.explanation, limit=160)}", markup=False)


def _write_manual_review_csv(
    rows: Sequence[LinkedPost], target: Path, *, top_unrelated: int = 50
) -> tuple[Path, int]:
    """Export a curated manual-review CSV (Branch 05G).

    Always includes all rows labelled ``relevant``/``weakly_relevant``/
    ``ambiguous``, plus the top ``top_unrelated`` by score among
    ``unrelated`` rows. Human reviewers can fill the ``human_*`` columns.
    """
    columns = [
        "market_id",
        "market_question",
        "post_id",
        "segment_id",
        "parent_post_id",
        "segment_index",
        "platform",
        "post_created_at",
        "relevance_label",
        "relevance_score",
        "confidence",
        "constraint_match",
        "anchor_score",
        "constraint_score",
        "domain_mismatch",
        "rejection_reason",
        "explanation",
        "post_text_preview",
        "segment_text_preview",
        "url",
        "human_relevance_label",
        "human_notes",
        "review_status",
    ]
    review_rows: list[LinkedPost] = []
    unrelated: list[LinkedPost] = []
    for row in rows:
        label = (row.final_relevance_label or row.relevance_label).value
        if label == "unrelated":
            unrelated.append(row)
        else:
            review_rows.append(row)
    unrelated.sort(key=_effective_link_score, reverse=True)
    review_rows.extend(unrelated[:top_unrelated])
    review_rows.sort(key=_effective_link_score, reverse=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in review_rows:
            writer.writerow(
                {
                    "market_id": row.market_id,
                    "market_question": row.market_question,
                    "post_id": row.post_id,
                    "segment_id": row.segment_id,
                    "parent_post_id": row.parent_post_id,
                    "segment_index": row.segment_index,
                    "platform": row.platform.value,
                    "post_created_at": (
                        row.post_created_at.isoformat() if row.post_created_at else None
                    ),
                    "relevance_label": (row.final_relevance_label or row.relevance_label).value,
                    "relevance_score": row.relevance_score,
                    "confidence": row.confidence,
                    "constraint_match": (row.final_constraint_match or row.constraint_match).value,
                    "anchor_score": row.anchor_score,
                    "constraint_score": row.constraint_score,
                    "domain_mismatch": row.domain_mismatch,
                    "rejection_reason": row.rejection_reason,
                    "explanation": row.explanation,
                    "post_text_preview": row.post_text_preview,
                    "segment_text_preview": row.segment_text_preview,
                    "url": None,
                    "human_relevance_label": "",
                    "human_notes": "",
                    "review_status": "pending",
                }
            )
    return target, len(review_rows)


@linking_app.command("export-manual-review")
def linking_export_manual_review(
    links_path: Path = typer.Option(
        None,
        "--links-path",
        help="Path to linked_posts.parquet (defaults to data/normalized/social/linked_posts.parquet).",
    ),
    output_path: Path = typer.Option(
        None,
        "--output-path",
        help="Where to write the manual-review CSV (defaults to data/analytical/linking/manual_review_candidates.csv).",
    ),
    top_unrelated: int = typer.Option(
        50,
        "--top-unrelated",
        min=0,
        help="How many highest-scoring unrelated rows to include for spot-check.",
    ),
) -> None:
    """Curate a manual-review CSV from ``linked_posts.parquet`` (Branch 05G).

    The CSV is intended for human spot-checking before Branch 06 stance
    detection consumes the linked rows. Human labels are not auto-filled.
    """
    paths = get_data_paths()
    target_links = links_path or paths.linked_posts_parquet
    target_csv = output_path or paths.manual_review_candidates_csv
    if not target_links.exists():
        rprint(
            f"[yellow]No linked_posts file at {target_links}.[/yellow] "
            "Run `polysignal linking score-candidates` first."
        )
        raise typer.Exit(code=2)
    rows = read_linked_posts(target_links)
    if not rows:
        rprint(f"[yellow]linked_posts at {target_links} is empty.[/yellow]")
        raise typer.Exit(code=2)
    _, exported = _write_manual_review_csv(rows, target_csv, top_unrelated=top_unrelated)
    rprint(f"[green]manual review[/green] -> {target_csv} ({exported} rows)")


_REVIEW_COLUMNS = [
    "market_id",
    "market_question",
    "domain",
    "platform",
    "post_id",
    "message_id",
    "parent_post_id",
    "segment_id",
    "segment_index",
    "author_name",
    "channel_source",
    "created_at",
    "market_temporal_anchor",
    "temporal_anchor_kind",
    "temporal_valid",
    "market_close_time",
    "market_resolved_at",
    "retrieval_query",
    "retrieval_run_id",
    "retrieval_work_id",
    "retrieval_window_start",
    "retrieval_window_end",
    "lookback_days",
    "window_label",
    "retrieval_mode",
    "temporal_valid_candidate",
    "accepted_by_graph_policy",
    "relevance_label",
    "relevance_score",
    "link_confidence",
    "constraint_match",
    "link_method",
    "text",
    "stance_label",
    "stance_confidence",
    "stance_method",
]


def _build_top_linked_evidence(
    rows: Sequence[LinkedPost],
    *,
    markets: Sequence[Market],
    posts: Sequence[SocialPostRecord],
    stances: Sequence,
    top_per_market: int,
    include_unrelated: bool,
    weak_min_confidence: float = ACCEPTED_WEAK_LINK_MIN_CONFIDENCE,
) -> list[dict[str, object]]:
    """Assemble the per-market top linked-evidence review rows.

    Joins each linked row to its market domain, the parent post's author /
    timestamp, and the best available stance verdict. Stance fields stay
    blank when ``stance_labels.parquet`` was not generated yet — the pack
    is for human inspection, never a training signal.
    """
    market_by_id = {m.market_id: m for m in markets}
    posts_by_key: dict[tuple[str, str], list[SocialPostRecord]] = {}
    for post in posts:
        posts_by_key.setdefault((post.platform.value, post.post_id), []).append(post)

    # Best stance per (market_id, linked post_id): prefer the highest-confidence
    # directional verdict so the reviewer sees the strongest available signal.
    stance_by_key: dict[tuple[str, str], object] = {}
    for record in stances:
        key = (record.market_id, record.post_id)
        current = stance_by_key.get(key)
        if current is None or record.confidence > current.confidence:
            stance_by_key[key] = record

    grouped: dict[str, list[LinkedPost]] = {}
    for row in rows:
        label = (row.final_relevance_label or row.relevance_label).value
        market = market_by_id.get(row.market_id)
        if market is None:
            continue
        temporal = assess_post_admissibility(market, row.post_created_at)
        if not temporal.admissible:
            continue
        if not include_unrelated:
            if label == "weakly_relevant" and _effective_link_confidence(row) < weak_min_confidence:
                continue
            if label not in {"relevant", "weakly_relevant"}:
                continue
        grouped.setdefault(row.market_id, []).append(row)

    out: list[dict[str, object]] = []
    for market_id, market_rows in grouped.items():
        market_rows.sort(key=_effective_link_score, reverse=True)
        for row in market_rows[: top_per_market if top_per_market > 0 else len(market_rows)]:
            parent = row.parent_post_id or row.post_id
            candidates = posts_by_key.get((row.platform.value, parent), [])
            author = next(
                (
                    post
                    for post in candidates
                    if post.market_id_candidate == market_id and post.query == row.query
                ),
                candidates[0] if candidates else None,
            )
            market = market_by_id[market_id]
            temporal = assess_post_admissibility(market, row.post_created_at)
            stance = stance_by_key.get((market_id, row.post_id))
            text = row.segment_text_preview or row.post_text_preview
            out.append(
                {
                    "market_id": market_id,
                    "market_question": row.market_question,
                    "domain": market.domain.value,
                    "platform": row.platform.value,
                    "post_id": row.post_id,
                    "message_id": (
                        parent.rsplit(":", 1)[-1]
                        if row.platform.value == "telegram" and ":" in parent
                        else parent
                    ),
                    "parent_post_id": row.parent_post_id,
                    "segment_id": row.segment_id,
                    "segment_index": row.segment_index,
                    "author_name": author.author_name if author else None,
                    "channel_source": author.subreddit_or_channel if author else None,
                    "created_at": (
                        row.post_created_at.isoformat() if row.post_created_at else None
                    ),
                    "market_temporal_anchor": (
                        temporal.anchor.isoformat() if temporal.anchor is not None else None
                    ),
                    "temporal_anchor_kind": temporal.anchor_kind,
                    "temporal_valid": temporal.temporal_valid,
                    "market_close_time": (
                        market.close_time.isoformat() if market.close_time is not None else None
                    ),
                    "market_resolved_at": (
                        market.resolved_at.isoformat() if market.resolved_at is not None else None
                    ),
                    "retrieval_query": row.query,
                    "retrieval_run_id": author.retrieval_run_id if author else None,
                    "retrieval_work_id": author.retrieval_work_id if author else None,
                    "retrieval_window_start": (
                        author.retrieval_window_start.isoformat()
                        if author and author.retrieval_window_start
                        else None
                    ),
                    "retrieval_window_end": (
                        author.retrieval_window_end.isoformat()
                        if author and author.retrieval_window_end
                        else None
                    ),
                    "lookback_days": author.retrieval_lookback_days if author else None,
                    "window_label": author.retrieval_window_label if author else None,
                    "retrieval_mode": author.retrieval_mode if author else None,
                    "temporal_valid_candidate": (
                        author.temporal_valid_candidate if author else None
                    ),
                    "accepted_by_graph_policy": (
                        temporal.admissible
                        and (
                            (row.final_relevance_label or row.relevance_label).value == "relevant"
                            or (
                                (row.final_relevance_label or row.relevance_label).value
                                == "weakly_relevant"
                                and _effective_link_confidence(row) >= weak_min_confidence
                            )
                        )
                    ),
                    "relevance_label": (row.final_relevance_label or row.relevance_label).value,
                    "relevance_score": _effective_link_score(row),
                    "link_confidence": _effective_link_confidence(row),
                    "constraint_match": (row.final_constraint_match or row.constraint_match).value,
                    "link_method": (row.final_method or row.method).value,
                    "text": text,
                    "stance_label": stance.stance_label.value if stance else None,
                    "stance_confidence": stance.confidence if stance else None,
                    "stance_method": stance.method.value if stance else None,
                }
            )
    out.sort(key=lambda r: (r["market_id"], -(r["relevance_score"] or 0.0)))
    return out


@linking_app.command("export-review")
def linking_export_review(
    links_path: Path = typer.Option(
        None, "--links-path", help="Path to linked_posts.parquet."
    ),
    stance_path: Path = typer.Option(
        None,
        "--stance-path",
        help="Optional stance_labels.parquet to join stance labels onto the pack.",
    ),
    markets_path: Path = typer.Option(None, "--markets-path"),
    posts_path: Path = typer.Option(None, "--posts-path"),
    csv_output: Path = typer.Option(
        None,
        "--csv-output",
        help="CSV output (default: data/analytical/review/top_linked_social_evidence.csv).",
    ),
    parquet_output: Path = typer.Option(
        None,
        "--parquet-output",
        help="Parquet output (default: data/analytical/review/top_linked_social_evidence.parquet).",
    ),
    top_per_market: int = typer.Option(
        20, "--top-per-market", min=0, help="Max evidence rows per market (0 = no cap)."
    ),
    include_unrelated: bool = typer.Option(
        False,
        "--include-unrelated/--meaningful-only",
        help="Include 'unrelated' rows too (default: only relevant/weakly/ambiguous).",
    ),
    weak_min_confidence: float = typer.Option(
        ACCEPTED_WEAK_LINK_MIN_CONFIDENCE,
        "--weak-min-confidence",
        min=0.0,
        max=1.0,
        help="Minimum confidence for weakly_relevant review evidence.",
    ),
    output_dir: Path = typer.Option(None, "--output-dir", help="Override the data root."),
) -> None:
    """Export a human review pack of top linked social evidence per market.

    The pack joins the linked rows to market domain, parent-post author,
    retrieval query, and (when present) the stance verdict. It is for
    manual quality control, report confidence, and demo preparation — not
    for training and never a leakage path for resolved labels.
    """
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    target_links = links_path or paths.linked_posts_parquet
    if not target_links.exists():
        rprint(
            f"[yellow]No linked_posts file at {target_links}.[/yellow] "
            "Run `polysignal linking score-candidates` first."
        )
        raise typer.Exit(code=2)
    rows = read_linked_posts(target_links)
    if not rows:
        rprint(f"[yellow]linked_posts at {target_links} is empty.[/yellow]")
        raise typer.Exit(code=2)

    m_target = markets_path or paths.polymarket_markets_parquet
    markets = read_parquet(m_target, Market) if m_target.exists() else []
    p_target = posts_path or paths.social_posts_parquet
    posts = read_parquet(p_target, SocialPostRecord) if p_target.exists() else []
    s_target = stance_path or paths.stance_labels_parquet
    stances = read_stance_records(s_target) if s_target.exists() else []

    evidence = _build_top_linked_evidence(
        rows,
        markets=markets,
        posts=posts,
        stances=stances,
        top_per_market=top_per_market,
        include_unrelated=include_unrelated,
        weak_min_confidence=weak_min_confidence,
    )

    csv_target = csv_output or paths.top_linked_social_evidence_csv
    parquet_target = parquet_output or paths.top_linked_social_evidence_parquet
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    with csv_target.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(evidence)
    frame = (
        pd.DataFrame(evidence, columns=_REVIEW_COLUMNS)
        if evidence
        else pd.DataFrame({name: pd.Series(dtype="object") for name in _REVIEW_COLUMNS})
    )
    parquet_target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_target, index=False)

    table = Table(title="Top linked social evidence (review pack)")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("linked rows read", str(len(rows)))
    table.add_row("evidence rows exported", str(len(evidence)))
    table.add_row("markets covered", str(len({r["market_id"] for r in evidence})))
    table.add_row("rows with stance", str(sum(1 for r in evidence if r["stance_label"])))
    console.print(table)
    rprint(f"[green]review CSV[/green]     -> {csv_target} ({len(evidence)} rows)")
    rprint(f"[green]review parquet[/green] -> {parquet_target}")


@linking_app.command("coverage")
def linking_coverage(
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Dataset root to read normalized parquets from."
    ),
    report_output: Path | None = typer.Option(
        None, "--report-output", help="Where to write coverage_report.json."
    ),
    thresholds: str = typer.Option(
        "1,5,10,50", "--thresholds", help="Comma-separated k values for >=k coverage."
    ),
) -> None:
    """Persist linking coverage stats (fraction of contracts with >=k linked posts).

    Broken down per thematic domain and per platform, as Traccia_2 requires.
    """
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    if not paths.polymarket_markets_parquet.exists():
        rprint("[red]ERROR[/red] normalized markets parquet missing.")
        raise typer.Exit(code=1)
    markets = read_dataframe(paths.polymarket_markets_parquet)
    linked = (
        read_dataframe(paths.linked_posts_parquet)
        if paths.linked_posts_parquet.exists()
        else pd.DataFrame()
    )
    try:
        ks = tuple(int(x.strip()) for x in thresholds.split(",") if x.strip())
    except ValueError:
        rprint(f"[red]Invalid --thresholds:[/red] {thresholds!r}")
        raise typer.Exit(code=1) from None
    report = build_linking_coverage_report(markets, linked, thresholds=ks or (1, 5, 10, 50))
    target = report_output or paths.linking_coverage_report_json
    write_linking_coverage_report(report, target)

    table = Table(title="Linking coverage — fraction of contracts with >=k linked posts")
    table.add_column("Group", style="bold")
    table.add_column("Contracts", justify="right")
    for k in report.thresholds:
        table.add_column(f">={k}", justify="right")
    rows = [("overall", report.overall)]
    rows += [(f"domain:{k}", v) for k, v in report.by_domain.items()]
    rows += [(f"platform:{k}", v) for k, v in report.by_platform.items()]
    for label, stats in rows:
        table.add_row(
            label,
            str(stats.total_contracts),
            *[f"{stats.fraction_at_least_k.get(str(k), 0.0):.2%}" for k in report.thresholds],
        )
    console.print(table)
    rprint(f"[green]coverage report[/green] -> {target}")


@linking_app.command("evaluate")
def linking_evaluate(
    use_fixtures: bool = typer.Option(
        True,
        "--use-fixtures/--no-use-fixtures",
        help=(
            "Evaluate against the bundled offline fixture labels (default). "
            "This is the deterministic regression check for the baseline."
        ),
    ),
    use_embeddings: bool = typer.Option(
        False,
        "--use-embeddings/--no-embeddings",
        help="Enable the Branch 05B embedding-similarity overlay (off by default).",
    ),
    embedding_model: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2",
        "--embedding-model",
        help="HuggingFace model id for the embedding overlay.",
    ),
    report_output: Path = typer.Option(
        None,
        "--report-output",
        help=(
            "Where to write the JSON report (defaults to "
            "data/analytical/linking/evaluation_report.json)."
        ),
    ),
    no_report: bool = typer.Option(
        False,
        "--no-report/--write-report",
        help="Skip writing the JSON evaluation_report.json.",
    ),
    use_llm_judge: bool = typer.Option(
        False,
        "--use-llm-judge/--no-llm-judge",
        help="Branch 05C: opt into the cached structured LLM relevance verifier.",
    ),
    llm_provider: str = typer.Option(
        None, "--llm-provider", help="LLM provider id for audit/cache."
    ),
    llm_model: str = typer.Option(None, "--llm-model", help="LLM model id for audit/cache."),
    llm_borderline_only: bool = typer.Option(
        True,
        "--llm-borderline-only/--llm-all",
        help="Select only borderline/low-confidence rows for LLM verification (default).",
    ),
    llm_min_score: float = typer.Option(0.35, "--llm-min-score"),
    llm_max_score: float = typer.Option(0.75, "--llm-max-score"),
    llm_cache_dir: Path = typer.Option(None, "--llm-cache-dir"),
    llm_dry_run: bool = typer.Option(
        False,
        "--llm-dry-run/--no-llm-dry-run",
        help="Report 05C row selection without making LLM calls.",
    ),
) -> None:
    """Run the scorer on the fixture set and report metrics + diagnostics.

    Prints accuracy, per-label precision/recall/F1, macro-F1, confusion
    matrix, score distribution per label, and top errors. Also writes
    a JSON report under ``data/analytical/linking/evaluation_report.json``
    unless ``--no-report`` is set.
    """
    if not use_fixtures:
        rprint(
            "[yellow]Only the offline fixture evaluation is wired up today.[/yellow] "
            "Run with --use-fixtures."
        )
        raise typer.Exit(code=2)

    paths = get_data_paths()
    provider = build_default_provider(
        enabled=use_embeddings,
        model_id=embedding_model,
        cache_dir=paths.embedding_cache_dir,
    )
    llm_judge = _build_cli_llm_judge(
        enabled=use_llm_judge,
        provider=llm_provider,
        model=llm_model,
        borderline_only=llm_borderline_only,
        min_score=llm_min_score,
        max_score=llm_max_score,
        cache_dir=llm_cache_dir,
        dry_run=llm_dry_run,
    )

    markets = fixture_markets()
    posts = fixture_posts()
    labels = fixture_labels()
    rows = run_linking_cross_product(
        posts,
        markets,
        embedding_provider=provider,
        llm_judge=llm_judge,
    )
    report = evaluate_predictions(rows, labels)

    summary = Table(title="Branch 05 — fixture evaluation")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value")
    summary.add_row("labels evaluated", str(report.matched_pairs))
    summary.add_row("accuracy", f"{report.accuracy:.3f}")
    summary.add_row("macro_f1", f"{report.macro_f1:.3f}")
    summary.add_row("constraint accuracy", f"{report.constraint_accuracy:.3f}")
    summary.add_row("llm rows sent", str(report.llm_rows_sent))
    summary.add_row("llm rows judged", str(report.llm_rows_judged))
    summary.add_row("llm cached judgments", str(report.llm_cached_judgments))
    summary.add_row("llm invalid outputs", str(report.llm_invalid_outputs))
    summary.add_row("llm label changes", str(report.llm_label_changes))
    console.print(summary)

    per_label = Table(title="Per-label precision / recall / F1")
    per_label.add_column("label", style="bold")
    per_label.add_column("precision")
    per_label.add_column("recall")
    per_label.add_column("f1")
    per_label.add_column("support")
    for label, metric in report.per_label.items():
        per_label.add_row(
            label.value,
            f"{metric.precision:.3f}",
            f"{metric.recall:.3f}",
            f"{metric.f1:.3f}",
            str(metric.support),
        )
    console.print(per_label)

    cm_table = Table(title="Confusion matrix (rows=gold, cols=predicted)")
    cm_table.add_column("gold \\ pred", style="bold")
    label_order = list(report.confusion_matrix.keys())
    for col in label_order:
        cm_table.add_column(col.value)
    for row_label in label_order:
        cm_table.add_row(
            row_label.value,
            *[str(report.confusion_matrix[row_label][col]) for col in label_order],
        )
    console.print(cm_table)

    dist_table = Table(title="Score distribution by predicted label")
    dist_table.add_column("label", style="bold")
    dist_table.add_column("n")
    dist_table.add_column("mean")
    dist_table.add_column("min")
    dist_table.add_column("max")
    for label, dist in report.score_distribution.items():
        dist_table.add_row(
            label.value,
            str(dist.n),
            f"{dist.mean:.3f}",
            f"{dist.minimum:.3f}",
            f"{dist.maximum:.3f}",
        )
    console.print(dist_table)

    if report.top_errors:
        rprint(f"[bold]Top errors ({len(report.top_errors)}):[/bold]")
        for err in report.top_errors:
            rprint(
                f"  - {err.market_id}/{err.post_id}: "
                f"gold={err.gold.value} pred={err.predicted.value} "
                f"score={err.relevance_score:.3f} ({err.explanation})"
            )

    if report.notes:
        rprint("[bold]Notes:[/bold]")
        for note in report.notes:
            rprint(f"  - {note}")

    if not no_report:
        target = report_output or paths.linking_evaluation_report
        write_evaluation_report(report, target)
        rprint(f"[green]evaluation report[/green] -> {target}")


@linking_app.command("export-evidence")
def linking_export_evidence(
    output_dir: Path = typer.Option(
        None, "--output-dir", help="Dataset root to read from and write evidence under."
    ),
) -> None:
    """Export accepted/rejected evidence with per-row reasons and provenance.

    Applies exactly the graph-loader acceptance policy (label, weak-link
    confidence floor, temporal admissibility, duplicate handling) row by
    row, and writes ``analytical/evidence/accepted_evidence.parquet``,
    ``rejected_evidence.parquet`` and ``evidence_summary.json``.
    """
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    if not paths.linked_posts_parquet.exists():
        rprint(
            "[red]ERROR[/red] linked_posts.parquet not found — run "
            "`polysignal linking score-candidates` first."
        )
        raise typer.Exit(code=1)
    result = export_evidence(paths=paths)
    table = Table(title="Evidence export")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("linked rows scored", str(result.summary["linked_rows_scored"]))
    table.add_row("accepted rows", str(result.accepted_rows))
    table.add_row("rejected rows", str(result.rejected_rows))
    table.add_row(
        "markets with accepted evidence",
        str(result.summary["markets_with_accepted_evidence"]),
    )
    console.print(table)
    reasons = result.summary.get("rejection_reasons", {})
    if reasons:
        reason_table = Table(title="Top rejection reasons")
        reason_table.add_column("Reason")
        reason_table.add_column("Rows")
        for reason, count in list(reasons.items())[:12]:
            reason_table.add_row(_terminal_text(reason, limit=80), str(count))
        console.print(reason_table)
    rprint(f"[green]accepted[/green] -> {result.accepted_path}")
    rprint(f"[green]rejected[/green] -> {result.rejected_path}")
    rprint(f"[green]summary[/green]  -> {result.summary_path}")


# ---------------------------------------------------------------------------
# Link (legacy) / NLP / Graph / Forecast / Evaluate
# ---------------------------------------------------------------------------


@nlp_app.command("enrich")
def nlp_enrich(
    limit: int | None = typer.Option(
        None, help="Max posts to enrich (default: all). Use for quick runs."
    ),
    sentiment_backend: str = typer.Option(
        "auto", help="auto | vader | transformer | lexicon"
    ),
    entity_backend: str = typer.Option("auto", help="auto | spacy | heuristic"),
    posts_path: Path | None = typer.Option(
        None, help="Override the social posts parquet path."
    ),
    output_dir: Path | None = typer.Option(None, help="Override the data root."),
) -> None:
    """Enrich posts with sentiment + NER + hashtags + language.

    Writes the derived ``post_enrichment.parquet`` (one row per post) without
    rewriting the raw corpus. Satisfies Traccia_2 §2 (hashtags,
    mentioned_entities, language) and the sentiment requirement (Task 2/3).
    """
    from polysignal.nlp.enrichment_runner import run_enrichment

    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    src = posts_path or paths.social_posts_parquet
    out = paths.normalized_social / "post_enrichment.parquet"
    if not src.exists():
        rprint(f"[red]ERROR[/red] social posts parquet not found: {src}")
        raise typer.Exit(code=1)
    rprint(f"[bold]NLP enrichment[/bold] reading {src}")
    summary = run_enrichment(
        src,
        out,
        sentiment_prefer=sentiment_backend,
        entity_prefer=entity_backend,
        limit=limit,
    )
    table = Table(title="NLP enrichment")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("posts enriched", f"{summary.posts_enriched:,}")
    table.add_row("sentiment backend", summary.sentiment_method)
    table.add_row("entity backend", summary.entity_method)
    for label, n in sorted(summary.label_counts.items()):
        table.add_row(f"  sentiment {label}", f"{n:,}")
    table.add_row("total hashtags", f"{summary.total_hashtags:,}")
    table.add_row("total entities", f"{summary.total_entities:,}")
    Console().print(table)
    rprint(f"[green]OK[/green] wrote {out}")


@nlp_app.command("topics")
def nlp_topics(
    n_topics: int = typer.Option(10, help="Number of LDA topics."),
    source: str = typer.Option("linked", help="linked | posts"),
    limit: int | None = typer.Option(20000, help="Max docs to model."),
    relevant_only: bool = typer.Option(
        True, help="When source=linked, keep only relevant/weakly-relevant rows."
    ),
    output_dir: Path | None = typer.Option(None, help="Override the data root."),
) -> None:
    """Fit an LDA topic model over linked (or raw) post text → topics.parquet."""
    from polysignal.nlp.topics import fit_topic_model
    from polysignal.storage.parquet_store import write_dataframe

    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    if source == "linked":
        src = paths.linked_posts_parquet
        text_cols = ["post_text_preview", "segment_text_preview"]
    else:
        src = paths.social_posts_parquet
        text_cols = ["text", "title"]
    if not src.exists():
        rprint(f"[red]ERROR[/red] source parquet not found: {src}")
        raise typer.Exit(code=1)
    df = read_dataframe(src)
    if source == "linked" and relevant_only and "final_relevance_label" in df.columns:
        keep = {"relevant", "weakly_relevant"}
        label_col = df["final_relevance_label"].fillna(df.get("relevance_label"))
        df = df[label_col.isin(keep)]
    # Coalesce available text columns row-wise (preview columns are sparsely
    # populated, so picking a single column can yield all-nulls).
    present = [c for c in text_cols if c in df.columns]
    if not present or df.empty:
        rprint("[yellow]No usable text column / rows for topic modeling.[/yellow]")
        raise typer.Exit(code=1)
    text_series = df[present[0]].astype("object")
    for c in present[1:]:
        text_series = text_series.fillna(df[c])
    texts = text_series.dropna().astype(str).tolist()
    texts = [t for t in texts if t.strip()]
    if limit:
        texts = texts[:limit]
    rprint(f"[bold]Topic modeling[/bold] over {len(texts):,} docs from {src.name}")
    result = fit_topic_model(texts, n_topics=n_topics)
    if not result.topics:
        rprint("[yellow]Topic model produced no topics (vocabulary too small).[/yellow]")
        raise typer.Exit(code=1)
    out = paths.normalized_social / "topics.parquet"
    write_dataframe(pd.DataFrame(result.topics_as_rows()), out)
    table = Table(title=f"LDA topics ({result.n_docs:,} docs)")
    table.add_column("#", justify="right")
    table.add_column("weight", justify="right")
    table.add_column("top terms")
    for t in result.topics:
        table.add_row(str(t.topic_id), f"{t.weight:.3f}", ", ".join(t.terms[:8]))
    Console().print(table)
    rprint(f"[green]OK[/green] wrote {out}")


@graph_app.command("init")
def graph_init(
    dry_run: bool = typer.Option(True, help="Print the cypher statements without executing them."),
) -> None:
    """Apply the Neo4j schema (constraints + indexes)."""
    client = Neo4jClient()
    try:
        statements = client.apply_schema(dry_run=dry_run)
    except RuntimeError as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        client.close()
    verb = "planned" if dry_run else "applied"
    rprint(f"[green]OK[/green] {len(statements)} schema statements {verb}")
    for stmt in statements:
        rprint(f"  - {stmt}")


@graph_app.command("load")
def graph_load(
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="Plan the load without contacting Neo4j (default).",
    ),
    output_dir: Path | None = typer.Option(None, help="Override the data root."),
    segments_path: Path | None = typer.Option(
        None, help="Override social_post_segments.parquet."
    ),
    stance_path: Path | None = typer.Option(
        None,
        help="Override stance_labels.parquet, including an explicit smoke fixture for planning.",
    ),
    batch_size: int = typer.Option(500, min=1, help="Rows per Neo4j UNWIND batch."),
) -> None:
    """Plan or execute idempotent Parquet-to-Neo4j loading."""
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    try:
        plan = build_load_plan(paths=paths, segments_path=segments_path, stance_path=stance_path)
    except (FileNotFoundError, ValueError) as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc
    _render_graph_plan(plan)
    if dry_run:
        rprint("[yellow]--dry-run set; Neo4j was not contacted.[/yellow]")
        return
    client = Neo4jClient()
    try:
        result = execute_load_plan(client, plan, batch_size=batch_size)
    except RuntimeError as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        client.close()
    rprint(
        f"[green]OK[/green] graph load complete: {result.rows_processed} planned rows "
        f"across {result.batches_executed} batches."
    )


@graph_app.command("inspect")
def graph_inspect(
    live: bool = typer.Option(
        False,
        "--live/--no-live",
        help="Also query a running Neo4j database for current label and relationship counts.",
    ),
    output_dir: Path | None = typer.Option(None, help="Override the data root."),
    segments_path: Path | None = typer.Option(None, help="Override social_post_segments.parquet."),
    stance_path: Path | None = typer.Option(None, help="Override stance_labels.parquet."),
) -> None:
    """Inspect graph inputs and planned counts without loading data."""
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    try:
        plan = build_load_plan(paths=paths, segments_path=segments_path, stance_path=stance_path)
    except (FileNotFoundError, ValueError) as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc
    _render_graph_plan(plan)
    if not live:
        return
    client = Neo4jClient()
    try:
        _render_live_graph_counts(client)
    except RuntimeError as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        client.close()


def _render_graph_plan(plan: GraphLoadPlan) -> None:
    datasets = Table(title="Graph Loader v1 - datasets")
    datasets.add_column("Dataset", style="bold")
    datasets.add_column("Required")
    datasets.add_column("Exists")
    datasets.add_column("Rows", justify="right")
    datasets.add_column("Path")
    for status in plan.datasets.values():
        datasets.add_row(
            status.name,
            str(status.required),
            str(status.exists),
            str(status.row_count),
            str(status.path),
        )
    console.print(datasets)

    counts = Table(title="Graph Loader v1 - planned graph")
    counts.add_column("Kind", style="bold")
    counts.add_column("Type")
    counts.add_column("Count", justify="right")
    for label, count in plan.planned_node_counts.items():
        counts.add_row("node", label, str(count))
    for relationship, count in plan.planned_relationship_counts.items():
        counts.add_row("relationship", relationship, str(count))
    console.print(counts)

    if plan.missing_optional_datasets:
        rprint(
            "[yellow]Missing optional datasets:[/yellow] "
            + ", ".join(plan.missing_optional_datasets)
        )
    for reason in plan.skipped_components:
        rprint(f"[yellow]Skipped:[/yellow] {reason}")
    for reason, count in sorted(plan.skipped_rows.items()):
        rprint(f"[yellow]Skipped rows:[/yellow] {reason}={count}")


def _render_live_graph_counts(client: Neo4jClient) -> None:
    table = Table(title="Neo4j live counts")
    table.add_column("Kind", style="bold")
    table.add_column("Type")
    table.add_column("Count", justify="right")
    for row in client.run_query(graph_queries.LIVE_NODE_COUNTS):
        table.add_row("node", str(row["label"]), str(row["count"]))
    for row in client.run_query(graph_queries.LIVE_RELATIONSHIP_COUNTS):
        table.add_row("relationship", str(row["relationship"]), str(row["count"]))
    console.print(table)


def _run_graph_query(gq: graph_queries.GraphQuery, *, dry_run: bool) -> None:
    """Print a parameterized query (dry-run) or execute it against live Neo4j."""
    rprint(f"[bold]{gq.name}[/bold] — {gq.description}")
    rprint("[bold]params:[/bold] " + json.dumps(gq.params))
    console.print(gq.cypher, markup=False)
    if dry_run:
        rprint("[yellow]--dry-run set; Neo4j was not contacted.[/yellow]")
        return
    client = Neo4jClient()
    try:
        rows = client.run_query(gq.cypher, gq.params)
    except RuntimeError as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        client.close()
    table = Table(title=f"{gq.name} — {len(rows)} rows")
    if rows:
        for column in rows[0]:
            table.add_column(str(column))
        for row in rows[:50]:
            table.add_row(*[_terminal_text(str(row.get(c)), limit=48) for c in rows[0]])
    console.print(table)


@graph_queries_app.command("list")
def graph_queries_list() -> None:
    """List the available parameterized graph read queries."""
    table = Table(title="Graph read queries")
    table.add_column("name", style="bold")
    table.add_column("description")
    for name, description in graph_queries.list_queries():
        table.add_row(name, description)
    console.print(table)


@graph_queries_app.command("contract-coverage")
def graph_queries_contract_coverage(
    limit: int = typer.Option(50, "--limit", min=1),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run"),
) -> None:
    """Per-contract linked-post coverage, author/channel breadth, and stance."""
    _run_graph_query(graph_queries.contract_coverage(limit=limit), dry_run=dry_run)


@graph_queries_app.command("top-contracts")
def graph_queries_top_contracts(
    limit: int = typer.Option(20, "--limit", min=1),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run"),
) -> None:
    """Top contracts by linked-post count * average link confidence."""
    _run_graph_query(graph_queries.top_contracts(limit=limit), dry_run=dry_run)


@graph_queries_app.command("author-activity")
def graph_queries_author_activity(
    limit: int = typer.Option(50, "--limit", min=1),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run"),
) -> None:
    """Per-author/source posts, linked contracts, and avg link confidence."""
    _run_graph_query(graph_queries.author_activity(limit=limit), dry_run=dry_run)


@graph_queries_app.command("platform-coverage")
def graph_queries_platform_coverage(
    limit: int = typer.Option(100, "--limit", min=1),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run"),
) -> None:
    """Per-platform/channel messages, linked messages, and covered contracts."""
    _run_graph_query(graph_queries.platform_coverage(limit=limit), dry_run=dry_run)


@graph_queries_app.command("contract-neighborhood")
def graph_queries_contract_neighborhood(
    market_id: str = typer.Option(..., "--market-id", help="Contract market_id to expand."),
    limit: int = typer.Option(50, "--limit", min=1),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run"),
) -> None:
    """Outcomes, linked posts, authors, channels, and stance edges for one contract."""
    _run_graph_query(
        graph_queries.contract_neighborhood(market_id=market_id, limit=limit), dry_run=dry_run
    )


@graph_queries_app.command("stance-summary")
def graph_queries_stance_summary(
    limit: int = typer.Option(100, "--limit", min=1),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run"),
) -> None:
    """SUPPORTS/OPPOSES counts per contract outcome (empty-safe)."""
    _run_graph_query(graph_queries.stance_summary(limit=limit), dry_run=dry_run)


@graph_queries_app.command("health")
def graph_queries_health(
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run"),
) -> None:
    """Graph integrity counters (orphan posts, contracts without outcomes/links, ...)."""
    _run_graph_query(graph_queries.graph_health(), dry_run=dry_run)


@evaluate_app.command("build-dataset")
def evaluate_build_dataset(
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Dataset root to read from and write backtest artefacts under."
    ),
    benchmark_profile: str | None = typer.Option(
        None,
        "--benchmark-profile",
        help="real_optional | real_social_enriched_main.",
    ),
    dataset_mode: str | None = typer.Option(
        None,
        "--dataset-mode",
        help="Legacy compatibility selector: synthetic | real | auto.",
    ),
    target: str = typer.Option(
        "yes", "--target", help="yes | resolved (both score the canonical YES resolution target)."
    ),
    min_evidence: int = typer.Option(
        3, "--min-evidence", min=1, help="Directional stance count required for sufficient evidence."
    ),
    exclude_insufficient: bool = typer.Option(
        False,
        "--exclude-insufficient/--include-insufficient",
        help="Drop insufficient-evidence rows instead of retaining and labelling them.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    export_csv: bool = typer.Option(False, "--export-csv"),
) -> None:
    """Build the evaluation-only target join without running any model."""
    try:
        profile, paths = resolve_backtest_profile_paths(
            paths=get_data_paths(root=output_dir) if output_dir else None,
            benchmark_profile=benchmark_profile,
            dataset_mode=dataset_mode,
        )
        dataset = build_evaluation_dataset(
            paths=paths,
            benchmark_profile=profile.name,
            target=target,
            min_evidence=min_evidence,
            exclude_insufficient=exclude_insufficient,
        )
    except (FileNotFoundError, ValueError) as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc
    _render_backtest_dataset_summary(dataset.metadata)
    if dry_run:
        rprint("[yellow]--dry-run set; no files written.[/yellow]")
        return
    try:
        written = write_evaluation_dataset_artifacts(
            dataset,
            paths=paths,
            overwrite=overwrite,
            export_csv=export_csv,
        )
    except FileExistsError as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=2) from exc
    for name, target_path in written.items():
        rprint(f"[green]{name}[/green] -> {target_path}")


@evaluate_app.command("backtest")
def evaluate_backtest(
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Dataset root to read from and write backtest artefacts under."
    ),
    benchmark_profile: str | None = typer.Option(
        None,
        "--benchmark-profile",
        help="real_optional | real_social_enriched_main.",
    ),
    dataset_mode: str | None = typer.Option(
        None,
        "--dataset-mode",
        help="Legacy compatibility selector: synthetic | real | auto.",
    ),
    model: str = typer.Option(
        "all", "--model", help="market-only | social-only | market-social | all."
    ),
    target: str = typer.Option(
        "yes", "--target", help="yes | resolved (both score the canonical YES resolution target)."
    ),
    min_evidence: int = typer.Option(
        3, "--min-evidence", min=1, help="Directional stance count required for sufficient evidence."
    ),
    exclude_insufficient: bool = typer.Option(
        False,
        "--exclude-insufficient/--include-insufficient",
        help="Drop insufficient-evidence rows instead of retaining and labelling them.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    export_csv: bool = typer.Option(False, "--export-csv"),
) -> None:
    """Run the leakage-free final-pre-close Branch 09 backtest."""
    try:
        profile, paths = resolve_backtest_profile_paths(
            paths=get_data_paths(root=output_dir) if output_dir else None,
            benchmark_profile=benchmark_profile,
            dataset_mode=dataset_mode,
        )
        config = BacktestConfig(
            benchmark_profile=profile.name,
            target=target,
            min_evidence=min_evidence,
            exclude_insufficient=exclude_insufficient,
        )
        run = run_backtest(paths=paths, config=config, model=model)
    except (FileNotFoundError, ValueError) as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc
    _render_backtest_dataset_summary(run.dataset.metadata)
    _render_backtest_metrics(run.metrics_summary)
    rprint(f"[bold]Interpretation:[/bold] {run.report['interpretation']}")
    if dry_run:
        rprint("[yellow]--dry-run set; no files written.[/yellow]")
        return
    try:
        written = write_backtest_artifacts(
            run, paths=paths, overwrite=overwrite, export_csv=export_csv
        )
    except FileExistsError as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=2) from exc
    for name, target_path in written.items():
        rprint(f"[green]{name}[/green] -> {target_path}")


@evaluate_app.command("task3")
def evaluate_task3(
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Dataset root to read from and write Task-3 artefacts under."
    ),
    n_splits: int = typer.Option(
        5, "--n-splits", min=2, help="Number of expanding-window temporal CV folds."
    ),
    profile: str = typer.Option(
        "real_optional", "--profile", help="Backtest profile directory for the results JSON/CSV."
    ),
    overwrite: bool = typer.Option(False, "--overwrite"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Task 3 — outcome prediction: feature matrix, temporal CV, model ablation.

    Builds ``task3_feature_matrix.parquet`` (volume/engagement/sentiment/linguistic),
    trains four model families under expanding-window temporal cross-validation, runs
    the market/social/combined ablation, and persists ``task3_results.{json,csv}``.
    """
    from polysignal.evaluation.task3 import run_task3, write_task3_artifacts

    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    if not paths.social_pressure_features_parquet.exists():
        rprint(
            "[red]ERROR[/red] missing social-pressure features. "
            "Run `polysignal features compute-social-pressure` first."
        )
        raise typer.Exit(code=1)
    try:
        run = run_task3(paths=paths, n_splits=n_splits)
    except (FileNotFoundError, ValueError) as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc

    cv = run.results["cross_validation"]
    rprint(
        f"[cyan]Task-3[/cyan] contracts={run.results['n_contracts']} "
        f"(social_evidence={run.results['n_with_social_evidence']}) "
        f"target={run.results['target_distribution']} folds={cv['n_folds']}"
    )
    table = Table(title="Task-3 temporal-CV ablation (out-of-fold)")
    for col in ("model", "ablation", "n_eval", "accuracy", "macro_f1", "auc", "brier"):
        table.add_column(col, justify="right" if col not in {"model", "ablation"} else "left")
    for _, row in run.results_table.iterrows():
        table.add_row(*[str(row.get(c)) for c in
                        ("model", "ablation", "n_eval", "accuracy", "macro_f1", "auc", "brier")])
    console.print(table)
    best = run.results["best_by_macro_f1"]
    rprint(f"[bold]Best by macro-F1:[/bold] {best.get('model')} / {best.get('ablation')} "
           f"(macro_f1={best.get('macro_f1')}, auc={best.get('auc')})")
    rprint(f"[bold]Conclusion:[/bold] {run.results['conclusion']}")
    if dry_run:
        rprint("[yellow]--dry-run set; no files written.[/yellow]")
        return
    try:
        written = write_task3_artifacts(run, paths=paths, profile=profile, overwrite=overwrite)
    except FileExistsError as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=2) from exc
    for name, target_path in written.items():
        if target_path is not None:
            rprint(f"[green]{name}[/green] -> {target_path}")


@evaluate_app.command("build-dashboard-view")
def evaluate_build_dashboard_view(
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Dataset root whose persisted backtest artifacts should be read."
    ),
    benchmark_profile: str | None = typer.Option(
        None,
        "--benchmark-profile",
        help="real_optional | real_social_enriched_main.",
    ),
    dataset_mode: str | None = typer.Option(
        None,
        "--dataset-mode",
        help="Legacy compatibility selector: synthetic | real | auto.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    export_csv: bool = typer.Option(False, "--export-csv"),
) -> None:
    """Build the Branch 10 dashboard read layer from persisted Branch 09 artifacts."""
    try:
        profile, paths = resolve_backtest_profile_paths(
            paths=get_data_paths(root=output_dir) if output_dir else None,
            benchmark_profile=benchmark_profile,
            dataset_mode=dataset_mode,
        )
        view = build_dashboard_prediction_view(
            paths=paths,
            benchmark_profile=profile.name,
        )
    except (FileNotFoundError, ValueError) as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc
    _render_dashboard_view_summary(view.manifest)
    if dry_run:
        rprint("[yellow]--dry-run set; no files written.[/yellow]")
        return
    try:
        written = write_dashboard_prediction_view(
            view,
            paths=paths,
            overwrite=overwrite,
            export_csv=export_csv,
        )
    except FileExistsError as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=2) from exc
    for name, target_path in written.items():
        rprint(f"[green]{name}[/green] -> {target_path}")


@evaluate_app.command("inspect-backtest")
def evaluate_inspect_backtest(
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Dataset root whose backtest report should be inspected."
    ),
    report_path: Path | None = typer.Option(
        None, "--report", help="Explicit backtest_report.json path (overrides --output-dir)."
    ),
    benchmark_profile: str | None = typer.Option(
        None,
        "--benchmark-profile",
        help="real_optional | real_social_enriched_main.",
    ),
    dataset_mode: str | None = typer.Option(
        None,
        "--dataset-mode",
        help="Legacy compatibility selector: synthetic | real | auto.",
    ),
) -> None:
    """Inspect a persisted Branch 09 backtest report."""
    try:
        profile, paths = resolve_backtest_profile_paths(
            paths=get_data_paths(root=output_dir) if output_dir else None,
            benchmark_profile=benchmark_profile,
            dataset_mode=dataset_mode,
        )
    except ValueError as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc
    target_path = report_path or paths.backtest_profile(profile.name).backtest_report_json
    if not target_path.exists():
        rprint(
            f"[yellow]No backtest report at {target_path}.[/yellow] "
            "Run `polysignal evaluate backtest` first."
        )
        raise typer.Exit(code=2)
    report = json.loads(target_path.read_text(encoding="utf-8"))
    _render_backtest_dataset_summary(report["dataset_construction"])
    _render_backtest_metrics(report["metrics"])
    rprint(f"[bold]Interpretation:[/bold] {report['interpretation']}")


@evaluate_app.command("inspect-profile")
def evaluate_inspect_profile(
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Optional dataset-root override for profile paths."
    ),
    benchmark_profile: str | None = typer.Option(
        None,
        "--benchmark-profile",
        help="real_optional | real_social_enriched_main.",
    ),
    dataset_mode: str | None = typer.Option(
        None,
        "--dataset-mode",
        help="Legacy compatibility selector: synthetic | real | auto.",
    ),
) -> None:
    """Inspect benchmark-profile policy, claim boundaries, and output path."""
    try:
        profile, paths = resolve_backtest_profile_paths(
            paths=get_data_paths(root=output_dir) if output_dir else None,
            benchmark_profile=benchmark_profile,
            dataset_mode=dataset_mode,
        )
    except ValueError as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc
    _render_benchmark_profile(profile.to_metadata(), paths.backtest_profile(profile.name).root)


@evaluate_app.command("list-profiles")
def evaluate_list_profiles() -> None:
    """List explicit Branch 09 benchmark profiles."""
    table = Table(title="Branch 09 benchmark profiles")
    table.add_column("Profile", style="bold")
    table.add_column("Dataset")
    table.add_column("Benchmark type")
    table.add_column("Default root")
    for profile in benchmark_profiles():
        table.add_row(
            profile.name,
            profile.dataset_mode,
            profile.benchmark_type,
            profile.default_root,
        )
    console.print(table)


def _render_backtest_dataset_summary(metadata: dict) -> None:
    counts = metadata["row_counts"]
    coverage = metadata["evidence_coverage"]
    table = Table(
        title=(
            f"Branch 09 evaluation dataset - {metadata['dataset_mode']} "
            f"({metadata['benchmark_profile']})"
        )
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("target source", metadata["target_source"])
    table.add_row("benchmark profile", metadata["benchmark_profile"])
    table.add_row("benchmark type", metadata["benchmark_type"])
    table.add_row("market probability source", metadata["market_probability_source"])
    table.add_row("join keys", ", ".join(metadata["join_keys"]))
    table.add_row("feature rows (unique)", str(counts["feature_rows_unique"]))
    table.add_row("binary target rows", str(counts["binary_resolved_targets"]))
    table.add_row("evaluation rows", str(counts["evaluation_rows"]))
    table.add_row("rows with valid pre-close price", str(counts["rows_with_preclose_market_probability"]))
    table.add_row("rows with sufficient evidence", str(coverage["rows_with_sufficient_evidence"]))
    console.print(table)


def _render_benchmark_profile(metadata: dict, output_path: Path) -> None:
    table = Table(title=f"Branch 09 benchmark profile - {metadata['name']}")
    table.add_column("Policy", style="bold")
    table.add_column("Value")
    table.add_row("benchmark type", metadata["benchmark_type"])
    table.add_row("dataset mode", metadata["dataset_mode"])
    table.add_row("feature source", metadata["feature_source"])
    table.add_row("market probability source", metadata["market_probability_source"])
    table.add_row("controlled social signal", str(metadata["controlled_social_signal"]))
    table.add_row("dashboard demo allowed", str(metadata["supports_dashboard_demo"]))
    table.add_row("real empirical claims allowed", str(metadata["supports_real_empirical_claims"]))
    table.add_row("output path", str(output_path))
    console.print(table)
    rprint(f"[bold]Leakage boundary:[/bold] {metadata['leakage_boundary']}")
    for warning in metadata["warnings"]:
        rprint(f"[yellow]WARNING[/yellow] {warning}")


def _render_dashboard_view_summary(manifest: dict) -> None:
    table = Table(title=f"Branch 10 dashboard view - {manifest['profile']}")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Rows", str(manifest["row_count"]))
    table.add_row("Columns", str(len(manifest["key_columns"])))
    table.add_row("Panels", str(len(manifest["available_panels"])))
    console.print(table)
    for warning in manifest["warnings"]:
        rprint(f"[yellow]WARNING[/yellow] {warning}")


def _render_backtest_metrics(summary: dict) -> None:
    table = Table(title="Branch 09 backtest metrics")
    table.add_column("Model", style="bold")
    table.add_column("Scope")
    table.add_column("Rows", justify="right")
    table.add_column("Brier", justify="right")
    table.add_column("Log loss", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("AUC", justify="right")
    table.add_column("ECE", justify="right")
    for name, metrics in summary["models"].items():
        table.add_row(
            name,
            metrics["scope"],
            str(metrics["rows_evaluated"]),
            _metric_text(metrics["brier_score"]),
            _metric_text(metrics["log_loss"]),
            _metric_text(metrics["accuracy"]),
            _metric_text(metrics["auc"]),
            _metric_text(metrics["expected_calibration_error"]),
        )
    console.print(table)
    lift = summary["lift_over_market_only"]
    rprint(
        "[bold]market+social lift over market-only:[/bold] "
        f"delta_brier={_metric_text(lift['delta_brier'])}, "
        f"delta_log_loss={_metric_text(lift['delta_log_loss'])}"
    )


def _metric_text(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------


@refresh_app.command("status")
def refresh_status(mode: str = typer.Option("fast", help="fast | heavy | manual")) -> None:
    """Show which steps a refresh of the requested mode would run right now."""
    try:
        chosen = RefreshMode(mode)
    except ValueError as exc:
        rprint(f"[red]Unknown mode:[/red] {mode}")
        raise typer.Exit(code=2) from exc
    policy = RefreshPolicy()
    state = RefreshState()
    plan = policy.plan(mode=chosen, state=state, now=datetime.now(tz=timezone.utc))
    rprint(f"[bold]Mode:[/bold] {plan.mode.value}")
    rprint(f"[bold]Would run ({len(plan.steps)} steps):[/bold]")
    for step in plan.steps:
        rprint(f"  - {step.value}")
    if plan.skipped:
        rprint("[bold]Skipped:[/bold]")
        for step, reason in plan.skipped:
            rprint(f"  - {step.value} -- {reason}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# High-signal Telegram dataset (selection + plan + coverage + manifest)
# ---------------------------------------------------------------------------


def _load_high_signal_markets(path: Path) -> list[HighSignalMarket]:
    if not path.exists():
        raise FileNotFoundError(
            f"high-signal markets parquet not found at {path}. "
            "Run `polysignal dataset select-high-signal-markets` first."
        )
    return read_parquet(path, HighSignalMarket)


def _load_telegram_search_plan(path: Path) -> list[TelegramSearchPlanEntry]:
    if not path.exists():
        raise FileNotFoundError(
            f"telegram search plan parquet not found at {path}. "
            "Run `polysignal social telegram build-search-plan` first."
        )
    return read_parquet(path, TelegramSearchPlanEntry)


@dataset_app.command("select-high-signal-markets")
def dataset_select_high_signal_markets(
    markets_path: Path = typer.Option(
        None,
        "--markets-path",
        help="Path to markets.parquet (defaults to the Polymarket output).",
    ),
    target_markets: int = typer.Option(
        DATASET_TARGETS.selected_market_target,
        "--target-markets",
        min=1,
        help="How many markets to select (default: 50).",
    ),
    window_days: int = typer.Option(
        180,
        "--window-days",
        min=7,
        help="Maximum search-window length per market, in days.",
    ),
    resolved_only: bool = typer.Option(
        True,
        "--resolved-only/--all-markets",
        help="Select resolved markets only (default).",
    ),
    exclude_micro_markets: bool = typer.Option(
        True,
        "--exclude-micro-markets/--include-micro-markets",
        help="Exclude obvious daily weather and exact-score micro-markets.",
    ),
    include_other: bool = typer.Option(
        False,
        "--include-other/--exclude-other",
        help="Include unclassified other-domain markets (excluded by default).",
    ),
    output_path: Path = typer.Option(
        None,
        "--output-path",
        help=(
            "Where to write the high-signal markets parquet. Defaults to "
            "data/analytical/dataset/high_signal_markets.parquet."
        ),
    ),
    output_dir: Path = typer.Option(None, "--output-dir", help="Override the data root."),
    dry_run: bool = typer.Option(
        False, "--dry-run/--no-dry-run", help="Print the summary without writing parquet."
    ),
) -> None:
    """Select up to ``--target-markets`` high-signal Polymarket markets."""
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    source = markets_path or paths.polymarket_markets_parquet
    try:
        markets = load_markets_from_parquet(source, limit=None)
    except FileNotFoundError as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc
    if not markets:
        rprint(f"[red]ERROR[/red] no markets loaded from {source}.")
        raise typer.Exit(code=1)

    has_history: dict[str, bool] = {}
    price_points_path = paths.polymarket_price_points_parquet
    if price_points_path.exists():
        from polysignal.storage.parquet_store import read_dataframe

        prices_df = read_dataframe(price_points_path)
        if not prices_df.empty and "market_id" in prices_df.columns:
            has_history = {str(mid): True for mid in prices_df["market_id"].dropna().unique()}

    result = select_high_signal_markets(
        markets,
        target=target_markets,
        has_price_history=has_history,
        window_days=window_days,
        resolved_only=resolved_only,
        exclude_micro_markets=exclude_micro_markets,
        include_other=include_other,
    )
    if not result.markets:
        rprint("[red]ERROR[/red] selection produced zero markets.")
        raise typer.Exit(code=1)

    target_path = output_path or paths.high_signal_markets_parquet

    table = Table(title="High-signal market selection")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("input markets", str(len(markets)))
    table.add_row("selected", str(len(result.markets)))
    table.add_row("duplicate market rows removed", str(result.duplicates_removed))
    table.add_row("skipped unresolved", str(result.skipped_unresolved))
    table.add_row("skipped missing temporal anchor", str(result.skipped_temporal_anchor_missing))
    table.add_row("skipped micro-markets", str(result.skipped_micro_markets))
    table.add_row("skipped unclassified other", str(result.skipped_other_markets))
    for domain, count in result.domain_counts.items():
        table.add_row(f"  {domain.value}", str(count))
    if result.imbalance_warnings:
        table.add_row(
            "imbalance warnings",
            "\n".join(result.imbalance_warnings),
        )
    if result.skipped_for_domain:
        table.add_row("unfilled slots", str(result.skipped_for_domain))
    console.print(table)

    if dry_run:
        rprint("[yellow]--dry-run set, no parquet written.[/yellow]")
        return
    write_high_signal_markets(result.markets, target_path)
    rprint(f"[green]high-signal markets[/green] -> {target_path}")


@telegram_app.command("build-search-plan")
def social_telegram_build_search_plan(
    markets_path: Path = typer.Option(
        None,
        "--markets-path",
        help=(
            "Path to high_signal_markets.parquet. Defaults to "
            "data/analytical/dataset/high_signal_markets.parquet."
        ),
    ),
    channels_config: Path = typer.Option(
        Path("config/telegram_channels.yml"),
        "--channels-config",
        help="YAML file listing enabled public channels/groups.",
    ),
    queries_per_market: int = typer.Option(
        6, "--queries-per-market", min=1, help="Maximum queries per market."
    ),
    max_queries_per_channel: int = typer.Option(
        3,
        "--max-queries-per-channel",
        min=1,
        help="Cap on queries dispatched to a single channel per market.",
    ),
    lookback_days: str = typer.Option(
        "7,30,90",
        "--lookback-days",
        help="Comma-separated cumulative pre-close bands, for example 7,30,90.",
    ),
    require_channels: bool = typer.Option(
        False,
        "--require-channels/--allow-unassigned",
        help=(
            "Fail when no Telegram channels are configured. "
            "Default keeps planning available without channels so the "
            "operator can still audit the planned queries."
        ),
    ),
    output_parquet: Path = typer.Option(None, "--output-parquet"),
    output_json: Path = typer.Option(None, "--output-json"),
    output_dir: Path = typer.Option(None, "--output-dir", help="Override the data root."),
) -> None:
    """Generate the Telegram search plan from the selected market set."""
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    source = markets_path or paths.high_signal_markets_parquet
    try:
        markets = _load_high_signal_markets(source)
    except FileNotFoundError as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc

    channels: list = []
    channels_loaded_from = None
    if channels_config and channels_config.exists():
        try:
            channels = list(load_telegram_channels(channels_config))
            channels_loaded_from = channels_config
        except (FileNotFoundError, ValueError) as exc:
            rprint(f"[red]ERROR[/red] {exc}")
            raise typer.Exit(code=1) from exc
    elif require_channels:
        rprint(
            f"[red]ERROR[/red] --require-channels set but {channels_config} is missing."
        )
        raise typer.Exit(code=1)
    else:
        rprint(
            f"[yellow]WARNING[/yellow] no channels config at {channels_config}. "
            "Plan rows will have channel_id=None — they document the queries "
            "but cannot be ingested until a channels YAML is provided."
        )

    try:
        parsed_lookbacks = _parse_lookback_days(lookback_days)
    except ValueError as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc
    entries = build_telegram_search_plan(
        markets,
        channels=channels,
        queries_per_market=queries_per_market,
        max_queries_per_channel=max_queries_per_channel,
        lookback_days=parsed_lookbacks,
    )

    parquet_target = output_parquet or paths.telegram_search_plan_parquet
    json_target = output_json or paths.telegram_search_plan_json
    write_telegram_search_plan(entries, parquet_target, json_target)

    table = Table(title="Telegram search plan")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("selected markets", str(len(markets)))
    table.add_row("channels configured", str(len(channels)))
    if channels_loaded_from is not None:
        table.add_row("channels source", str(channels_loaded_from))
    table.add_row("plan entries", str(len(entries)))
    table.add_row(
        "entries without channel",
        str(sum(1 for e in entries if e.channel_id is None)),
    )
    console.print(table)
    rprint(f"[green]plan parquet[/green] -> {parquet_target}")
    rprint(f"[green]plan json[/green]    -> {json_target}")


def _parse_lookback_days(value: str) -> tuple[int, ...]:
    try:
        days = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    except ValueError as exc:
        raise ValueError("--lookback-days must contain comma-separated positive integers") from exc
    if not days or any(day <= 0 for day in days):
        raise ValueError("--lookback-days must contain comma-separated positive integers")
    return days


@telegram_app.command("search-markets")
def social_telegram_search_markets(
    plan_path: Path = typer.Option(None, "--plan-path", help="Persisted Telegram search plan."),
    config: Path = typer.Option(
        Path("config/telegram_channels.yml"),
        "--config",
        help="YAML file listing enabled public channels/groups.",
    ),
    limit_per_query: int = typer.Option(100, "--limit-per-query", min=1, max=1500),
    max_markets: int = typer.Option(50, "--max-markets", min=1),
    append: bool = typer.Option(True, "--append/--replace"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run"),
    output_dir: Path = typer.Option(None, "--output-dir", help="Override the data root."),
) -> None:
    """Execute the persisted pre-close market-window Telegram plan."""
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    try:
        entries = _load_telegram_search_plan(plan_path or paths.telegram_search_plan_parquet)
        channels = load_telegram_channels(config)
        result = search_telegram_market_plan(
            entries,
            channels,
            limit_per_query=limit_per_query,
            max_markets=max_markets,
            resume=resume,
            dry_run=dry_run,
            paths=paths,
        )
        if not dry_run and result.raw_dir is not None:
            records = normalize_telegram_raw_directory(
                result.raw_dir,
                paths=paths,
                channels=channels,
                require_text=True,
                append=append,
            )
            rprint(f"[green]social posts[/green] -> {paths.social_posts_parquet} ({len(records)} rows)")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc
    mode = "dry-run" if dry_run else "live"
    rprint(
        f"[green]Telegram market search {mode}[/green]: "
        f"attempted={result.work_items_attempted}, completed={result.work_items_completed}, "
        f"empty={result.work_items_empty}, skipped={result.work_items_skipped}, "
        f"failed={result.work_items_failed}, messages={result.posts_collected}"
    )
    if result.raw_dir is not None:
        rprint(f"[green]raw snapshots[/green] -> {result.raw_dir}")
    if result.manifest_path is not None:
        rprint(f"[green]search manifest[/green] -> {result.manifest_path}")


@telegram_app.command("coverage-report")
def social_telegram_coverage_report(
    markets_path: Path = typer.Option(
        None,
        "--markets-path",
        help="Path to high_signal_markets.parquet.",
    ),
    plan_path: Path = typer.Option(
        None,
        "--plan-path",
        help="Path to telegram_search_plan.parquet.",
    ),
    messages_path: Path = typer.Option(
        None,
        "--messages-path",
        help="Path to social_posts.parquet (Telegram rows are filtered automatically).",
    ),
    segments_path: Path = typer.Option(
        None,
        "--segments-path",
        help="Path to social_post_segments.parquet.",
    ),
    channels_config: Path = typer.Option(
        Path("config/telegram_channels.yml"),
        "--channels-config",
    ),
    output_path: Path = typer.Option(None, "--output-path"),
    output_dir: Path = typer.Option(None, "--output-dir"),
) -> None:
    """Compute and persist the Telegram coverage report."""
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    market_source = markets_path or paths.high_signal_markets_parquet
    plan_source = plan_path or paths.telegram_search_plan_parquet
    msg_source = messages_path or paths.social_posts_parquet
    seg_source = segments_path or paths.social_post_segments_parquet
    try:
        markets = _load_high_signal_markets(market_source)
        plan = _load_telegram_search_plan(plan_source)
    except FileNotFoundError as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc

    messages = read_parquet(msg_source, SocialPostRecord) if msg_source.exists() else []
    segments = read_parquet(seg_source, SocialPostSegment) if seg_source.exists() else []
    channels: list = []
    if channels_config and channels_config.exists():
        try:
            channels = list(load_telegram_channels(channels_config))
        except (FileNotFoundError, ValueError) as exc:
            rprint(f"[red]ERROR[/red] {exc}")
            raise typer.Exit(code=1) from exc

    limitations: list[str] = []
    if not messages:
        limitations.append("No Telegram messages collected yet.")
    if not segments:
        limitations.append("No Telegram segments computed yet.")
    if not channels:
        limitations.append("No Telegram channels configured.")

    report = build_coverage_report(
        selected=markets,
        plan=plan,
        messages=messages,
        segments=segments,
        channels_configured=len(channels),
        enabled_channels=[c.username for c in channels],
        limitations=limitations,
    )
    target = output_path or paths.telegram_coverage_report_json
    write_coverage_report(report, target)

    table = Table(title="Telegram coverage report")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("selected markets", str(report.selected_markets_total))
    for domain, count in sorted(report.selected_markets_by_domain.items()):
        table.add_row(f"  {domain}", str(count))
    table.add_row("plan entries", str(report.plan_entries_total))
    table.add_row("raw telegram messages", str(report.raw_messages_collected))
    table.add_row("normalized telegram", str(report.normalized_messages))
    table.add_row("segments", str(report.segments_generated))
    table.add_row("channels configured", str(report.telegram_channels_configured))
    table.add_row(
        "markets w/ insufficient coverage",
        str(len(report.markets_with_insufficient_coverage)),
    )
    if report.limitations:
        table.add_row("limitations", "\n".join(report.limitations))
    console.print(table)
    rprint(f"[green]coverage report[/green] -> {target}")


@dataset_app.command("build-manifest")
def dataset_build_manifest(
    raw_run_dirs: str = typer.Option(
        "",
        "--raw-run-dirs",
        help=(
            "Comma-separated raw Telegram run directories produced by "
            "`telegram fetch-*` commands. Empty by default."
        ),
    ),
    output_path: Path = typer.Option(None, "--output-path"),
    output_dir: Path = typer.Option(None, "--output-dir"),
    notes: str = typer.Option(
        "",
        "--notes",
        help="Comma-separated free-text note(s) to embed in the manifest.",
    ),
) -> None:
    """Write the high-signal Telegram manifest."""
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    targets = DATASET_TARGETS
    if not paths.high_signal_markets_parquet.exists():
        rprint(
            "[red]ERROR[/red] selected markets parquet missing. "
            "Run `polysignal dataset select-high-signal-markets` first."
        )
        raise typer.Exit(code=1)
    if not paths.telegram_search_plan_parquet.exists():
        rprint(
            "[red]ERROR[/red] telegram search plan parquet missing. "
            "Run `polysignal social telegram build-search-plan` first."
        )
        raise typer.Exit(code=1)
    coverage_path = (
        paths.telegram_coverage_report_json
        if paths.telegram_coverage_report_json.exists()
        else None
    )
    normalized_path = (
        paths.social_posts_parquet if paths.social_posts_parquet.exists() else None
    )
    segments_path = (
        paths.social_post_segments_parquet
        if paths.social_post_segments_parquet.exists()
        else None
    )
    channels_path = (
        Path("config/telegram_channels.yml")
        if Path("config/telegram_channels.yml").exists()
        else None
    )
    raw_dirs = tuple(Path(p.strip()) for p in raw_run_dirs.split(",") if p.strip())
    note_list = tuple(n.strip() for n in notes.split(",") if n.strip())
    payload = build_manifest(
        selected_markets_path=paths.high_signal_markets_parquet,
        telegram_plan_parquet_path=paths.telegram_search_plan_parquet,
        telegram_plan_json_path=paths.telegram_search_plan_json,
        coverage_report_path=coverage_path,
        raw_run_dirs=raw_dirs,
        normalized_messages_path=normalized_path,
        segments_path=segments_path,
        channels_config_path=channels_path,
        targets=targets,
        notes=note_list,
    )
    target = output_path or paths.high_signal_telegram_manifest_json
    write_manifest(payload, target)
    rprint(f"[green]manifest[/green] -> {target}")


@dataset_app.command("export")
def dataset_export(
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Dataset root to read normalized parquets from."
    ),
    export_dir: Path | None = typer.Option(
        None, "--export-dir", help="Where to write the canonical bundle (default <root>/dataset_export)."
    ),
    sample_rows: int = typer.Option(100, "--sample-rows", min=1, help="Rows per CSV sample."),
    partition: bool = typer.Option(
        False, "--partition/--no-partition", help="Write Hive-partitioned datasets (by platform/category)."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan only; write nothing."),
) -> None:
    """Export the canonical Traccia_2 §2 dataset bundle + DATA_DICTIONARY.md.

    Joins the normalized tables (contracts, prices, social posts + enrichment +
    comments, links) into ``data/dataset_export/`` as Parquet + CSV samples.
    """
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    markets_path = paths.polymarket_markets_parquet
    if not markets_path.exists():
        rprint(
            "[red]ERROR[/red] normalized markets parquet missing. "
            "Run the Polymarket ingestion first."
        )
        raise typer.Exit(code=1)

    def _read(path: Path) -> pd.DataFrame:
        return read_dataframe(path) if path.exists() else pd.DataFrame()

    markets = _read(paths.polymarket_markets_parquet)
    outcomes = _read(paths.polymarket_outcomes_parquet)
    prices = _read(paths.polymarket_price_points_parquet)
    posts = _read(paths.social_posts_parquet)
    enrichment = _read(paths.normalized_social / "post_enrichment.parquet")
    linked = _read(paths.linked_posts_parquet)
    comments = _read(paths.post_comments_parquet)

    target_dir = export_dir or (paths.root / "dataset_export")
    if dry_run:
        rprint(
            f"[yellow]--dry-run[/yellow]: would export to {target_dir} — "
            f"contracts={len(markets)} prices={len(prices)} posts={len(posts)} "
            f"links={len(linked)} comments={len(comments)}"
        )
        return

    result = export_dataset(
        markets=markets,
        outcomes=outcomes,
        prices=prices,
        posts=posts,
        enrichment=enrichment,
        linked=linked,
        comments=comments,
        output_dir=target_dir,
        sample_rows=sample_rows,
        partition=partition,
    )
    table = Table(title=f"Canonical dataset export -> {target_dir}")
    table.add_column("Entity", style="bold")
    table.add_column("Rows", justify="right")
    for name in ("contracts", "price_points", "social_posts", "links", "comments"):
        table.add_row(name, str(result.row_counts.get(name, 0)))
    console.print(table)
    if result.partitioned:
        for name, cols in result.partitioned.items():
            rprint(f"[cyan]{name}[/cyan] partitioned by {'/'.join(cols)}")
    rprint(f"[green]data dictionary[/green] -> {result.written['data_dictionary']}")


@processing_app.command("social-aggregate")
def process_social_aggregate(
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Dataset root to read normalized parquets from."
    ),
    out_path: Path | None = typer.Option(
        None, "--out", help="Output parquet (default <root>/analytical/processing/...)."
    ),
    use_spark: bool = typer.Option(
        True, "--spark/--no-spark", help="Use PySpark when a JVM is available (else pandas)."
    ),
) -> None:
    """Dedup + per-market social aggregation over the full corpus (PySpark/pandas).

    Big-data 'summarize + join' job: deduplicates posts by ``(platform,
    post_id)``, resolves each post's market via ``linked_posts``, and emits
    per-market post counts, unique authors, platform breadth, and engagement
    summaries. Runs on Spark (``local[*]``) when ``JAVA_HOME`` + PySpark are
    present, otherwise the identical pandas reference path.
    """
    import os

    from polysignal.processing import is_spark_available, run_dedup_aggregate_job

    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    posts_path = paths.social_posts_parquet
    if not posts_path.exists():
        rprint("[red]ERROR[/red] normalized social_posts parquet missing.")
        raise typer.Exit(code=1)
    links_path = paths.linked_posts_parquet if paths.linked_posts_parquet.exists() else None
    target = out_path or (paths.analytical / "processing" / "social_aggregate.parquet")

    engine = "spark" if (use_spark and is_spark_available()) else "pandas"
    rprint(f"[cyan]engine[/cyan]={engine} (JAVA_HOME set: {bool(os.environ.get('JAVA_HOME'))})")
    info = run_dedup_aggregate_job(
        posts_path=posts_path,
        output_path=target,
        links_path=links_path,
        use_spark=use_spark,
    )
    table = Table(title="Per-market social aggregate")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("engine", str(info["engine"]))
    table.add_row("input posts", f"{info['input_posts']:,}")
    table.add_row("markets", f"{info['markets']:,}")
    console.print(table)
    rprint(f"[green]OK[/green] wrote {info['output_path']}")


@analytics_app.command("run")
def analytics_run(
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Dataset root to read normalized parquets from."
    ),
    analytics_dir: Path | None = typer.Option(
        None, "--analytics-dir", help="Where to write artifacts (default <root>/analytical/analytics)."
    ),
) -> None:
    """Run all four Task-2 analyses and persist artifacts to data/analytical/analytics/.

    Computes the descriptive contract stats + price-informativeness curve, the
    per-platform/domain discourse (volume/sentiment/topics/NER), the lead/lag +
    spike-event-study correlation, and the cross-platform comparison.
    """
    from polysignal.analytics.runner import run_all

    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    if not paths.polymarket_markets_parquet.exists():
        rprint("[red]ERROR[/red] normalized markets parquet missing. Run ingestion first.")
        raise typer.Exit(code=1)
    result = run_all(paths=paths, output_dir=analytics_dir)

    table = Table(title=f"Task-2 analytics -> {result.output_dir}")
    table.add_column("artifact", style="bold")
    table.add_column("rows", justify="right")
    for name, count in sorted(result.summary.get("row_counts", {}).items()):
        table.add_row(name, f"{count:,}")
    console.print(table)

    info = result.summary.get("price_informativeness")
    if info:
        curve = Table(title="Price informativeness vs horizon")
        for col in ("horizon_days", "n_contracts", "brier", "accuracy", "auc"):
            curve.add_column(col, justify="right")
        for row in info:
            curve.add_row(*[str(row.get(c)) for c in ("horizon_days", "n_contracts", "brier", "accuracy", "auc")])
        console.print(curve)
    corr = result.summary.get("correlation", {})
    if corr:
        rprint(f"[cyan]correlation[/cyan] markets={corr.get('n_markets')} "
               f"relations={corr.get('relation_counts')} alignment_rate={corr.get('alignment_rate')}")
    xp = result.summary.get("cross_platform", {})
    if xp:
        rprint(f"[cyan]cross-platform[/cyan] state={xp.get('state')} "
               f"multi_platform_contracts={xp.get('multi_platform_contracts')} "
               f"platforms={xp.get('platforms_present')}")
    rprint(f"[green]OK[/green] summary -> {result.written['analytics_summary']}")


# ---------------------------------------------------------------------------
# Branch 06A — outcome-aware deterministic stance baseline
# ---------------------------------------------------------------------------


def _load_human_stance_labels(path: Path) -> dict[str, HumanStanceLabel]:
    """Parse a 2-column ``stance_id,human_stance_label`` CSV.

    Rows whose label is empty or whose ``review_status`` column is set
    to ``pending`` are skipped — they are unreviewed.
    """
    if not path.exists():
        return {}
    labels: dict[str, HumanStanceLabel] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return labels
        for row in reader:
            stance_id = (row.get("stance_id") or "").strip()
            raw_label = (row.get("human_stance_label") or "").strip().lower()
            status = (row.get("review_status") or "").strip().lower()
            if not stance_id or not raw_label or status == "pending":
                continue
            try:
                labels[stance_id] = HumanStanceLabel(raw_label)
            except ValueError:
                continue
    return labels


def _build_cli_stance_judge(
    *,
    paths,
    provider: str | None,
    model: str | None,
    confidence_floor: float,
    cache_path: Path | None,
    dry_run: bool,
) -> LLMStanceJudge:
    """Build the Branch 06B LLM stance judge from CLI flags.

    The CLI never instantiates the fake client itself — tests inject a
    fake judge at this builder boundary or call
    :func:`run_stance_pipeline` directly. Production callers must have
    a configured provider; otherwise we exit early with a clear
    message rather than silently returning the deterministic verdict.
    """
    settings = get_settings()
    resolved_provider = provider or settings.llm.provider
    resolved_model = model or settings.llm.model or "unconfigured"
    client = NullLLMClient()
    if not dry_run:
        if resolved_provider == "openai":
            api_key = settings.llm.openai_api_key or settings.llm.api_key
        elif resolved_provider == "anthropic":
            api_key = settings.llm.anthropic_api_key or settings.llm.api_key
        else:
            api_key = ""
        if not api_key:
            rprint(
                "[red]ERROR[/red] --use-llm requires a configured provider "
                "API key. Set POLYSIGNAL_LLM_PROVIDER and the matching "
                "POLYSIGNAL_*_API_KEY, or pass --llm-dry-run to verify "
                "selection logic without spending tokens."
            )
            raise typer.Exit(code=2)
        try:
            client = build_llm_client(
                provider=resolved_provider,
                api_key=api_key,
                model_id=resolved_model,
                temperature=settings.llm.temperature,
                timeout_seconds=settings.llm.timeout_seconds,
                max_tokens=settings.llm.max_tokens,
            )
        except RuntimeError as exc:
            rprint(f"[red]ERROR[/red] {exc}")
            raise typer.Exit(code=2) from exc
    cache_dir = cache_path or paths.stance_llm_cache_dir
    cache = JSONCache(cache_dir)
    return LLMStanceJudge(
        client,
        cache,
        provider=resolved_provider,
        model_id=resolved_model,
        policy=LLMStanceJudgePolicy(confidence_floor=confidence_floor),
        dry_run=dry_run,
    )


@stance_app.command("score")
def stance_score(
    input_path: Path = typer.Option(
        None,
        "--input",
        "--linked-posts-path",
        help=(
            "Path to linked_posts.parquet. "
            "Defaults to data/normalized/social/linked_posts.parquet."
        ),
    ),
    segments_path: Path = typer.Option(
        None,
        "--segments-path",
        help=(
            "Path to social_post_segments.parquet. "
            "When present, segment_text is used instead of the "
            "240-char preview stored on linked_posts.parquet."
        ),
    ),
    outcomes_path: Path = typer.Option(
        None,
        "--outcomes-path",
        help=(
            "Path to outcomes.parquet for the Branch 06C market metadata join. "
            "Defaults to data/normalized/polymarket/outcomes.parquet."
        ),
    ),
    markets_path: Path = typer.Option(
        None,
        "--markets-path",
        help="Path to markets.parquet for historical temporal filtering.",
    ),
    output_path: Path = typer.Option(
        None,
        "--output",
        "--output-path",
        help=(
            "Where to write stance_labels.parquet. Defaults to "
            "data/normalized/social/stance_labels.parquet."
        ),
    ),
    include_ambiguous: bool = typer.Option(
        False,
        "--include-ambiguous/--no-ambiguous",
        help=(
            "Also score rows whose effective relevance label is "
            "'ambiguous'. Off by default — ambiguous rows usually "
            "belong in the review CSV instead."
        ),
    ),
    min_link_confidence: float = typer.Option(
        ACCEPTED_WEAK_LINK_MIN_CONFIDENCE,
        "--min-link-confidence",
        min=0.0,
        max=1.0,
        help="Only score linked rows whose effective confidence is >= this floor.",
    ),
    limit: int = typer.Option(
        0,
        "--limit",
        min=0,
        help="Maximum eligible rows to score (0 = no cap).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--no-dry-run",
        help="Score and print the summary, but do not write the parquet.",
    ),
    force: bool = typer.Option(
        False,
        "--force/--no-force",
        help="Write an empty parquet when no rows are scored (default: skip the write).",
    ),
    output_dir: Path = typer.Option(None, "--output-dir", help="Override the data root."),
    use_llm: bool = typer.Option(
        False,
        "--use-llm/--no-llm",
        help=(
            "Branch 06B: run the cached LLM stance judge on rows whose "
            "deterministic verdict is ambiguous, insufficient_context, "
            "or below the confidence floor."
        ),
    ),
    llm_confidence_floor: float = typer.Option(
        0.5,
        "--llm-confidence-floor",
        min=0.0,
        max=1.0,
        help="Confidence below which directional baseline rows are sent to the LLM.",
    ),
    llm_cache_path: Path = typer.Option(
        None,
        "--llm-cache-path",
        help=(
            "Directory for the LLM stance cache (defaults to "
            "data/cache/llm/stance/)."
        ),
    ),
    llm_provider: str = typer.Option(
        None,
        "--llm-provider",
        help="LLM provider id (audit only). Defaults to the configured provider.",
    ),
    llm_model: str = typer.Option(
        None,
        "--llm-model",
        help="LLM model id (audit only). Defaults to the configured model.",
    ),
    llm_dry_run: bool = typer.Option(
        False,
        "--llm-dry-run/--no-llm-dry-run",
        help=(
            "Select eligible rows for LLM judging but do not call any "
            "provider — useful for verifying selection logic without "
            "spending tokens."
        ),
    ),
) -> None:
    """Score eligible linked rows and persist ``stance_labels.parquet``."""
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    judge: LLMStanceJudge | None = None
    if use_llm:
        try:
            judge = _build_cli_stance_judge(
                paths=paths,
                provider=llm_provider,
                model=llm_model,
                confidence_floor=llm_confidence_floor,
                cache_path=llm_cache_path,
                dry_run=llm_dry_run,
            )
        except typer.Exit:
            raise
    try:
        result = run_stance_pipeline(
            linked_posts_path=input_path,
            segments_path=segments_path,
            outcomes_path=outcomes_path,
            markets_path=markets_path,
            include_ambiguous=include_ambiguous,
            min_link_confidence=min_link_confidence,
            limit=limit if limit > 0 else None,
            paths=paths,
            output_path=output_path,
            force=force,
            dry_run=dry_run,
            llm_judge=judge,
        )
    except FileNotFoundError as exc:
        rprint(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Stance scoring summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("linked rows read", str(result.rows_read))
    table.add_row("eligible rows", str(result.rows_eligible))
    table.add_row("rows scored", str(result.rows_scored))
    table.add_row("skipped (label)", str(result.rows_skipped_label))
    table.add_row("skipped (confidence)", str(result.rows_skipped_confidence))
    table.add_row("skipped (empty text)", str(result.rows_skipped_text))
    table.add_row("skipped (post-close)", str(result.rows_skipped_post_close))
    table.add_row(
        "skipped (temporal anchor missing)",
        str(result.rows_skipped_temporal_anchor_missing),
    )
    table.add_row(
        "skipped (post timestamp missing)",
        str(result.rows_skipped_post_timestamp_missing),
    )
    for label in sorted(result.by_label):
        table.add_row(f"  {label}", str(result.by_label[label]))
    if judge is not None:
        table.add_row("llm considered", str(result.llm_considered))
        table.add_row("llm selected", str(result.llm_selected))
        table.add_row("llm calls", str(result.llm_calls))
        table.add_row("llm cache hits", str(result.llm_cache_hits))
        table.add_row("llm applied", str(result.llm_applied))
        table.add_row("llm invalid outputs", str(result.llm_invalid_outputs))
        table.add_row("llm label changes", str(result.llm_label_changes))
    console.print(table)

    if dry_run:
        rprint("[yellow]--dry-run set; no parquet written.[/yellow]")
        return
    if result.output_path is None:
        rprint(
            "[yellow]No stance rows produced; parquet was not written.[/yellow] "
            "Re-run with --force to materialise an empty file."
        )
        return
    rprint(f"[green]stance labels[/green] -> {result.output_path}")


@stance_app.command("inspect")
def stance_inspect(
    input_path: Path = typer.Option(
        None, "--input", "--stance-path", help="Path to stance_labels.parquet."
    ),
    limit: int = typer.Option(20, "--limit", min=1, help="Maximum rows to display."),
    label: str = typer.Option(
        None,
        "--label",
        help="Filter rows by stance_label (e.g. supports_yes, ambiguous).",
    ),
    min_confidence: float = typer.Option(
        0.0, "--min-confidence", min=0.0, max=1.0, help="Filter by confidence floor."
    ),
) -> None:
    """Inspect the contents of ``stance_labels.parquet``."""
    paths = get_data_paths()
    target = input_path or paths.stance_labels_parquet
    if not target.exists():
        rprint(
            f"[yellow]No stance file at {target}.[/yellow] "
            "Run `polysignal stance score` first."
        )
        raise typer.Exit(code=2)
    records = read_stance_records(target)
    if not records:
        rprint(f"[yellow]File at {target} is empty.[/yellow]")
        raise typer.Exit(code=2)
    if label:
        try:
            wanted = StanceLabel(label.strip().lower())
        except ValueError as exc:
            rprint(f"[red]Unknown stance label:[/red] {label!r}")
            raise typer.Exit(code=2) from exc
        records = [r for r in records if r.stance_label is wanted]
    if min_confidence > 0:
        records = [r for r in records if r.confidence >= min_confidence]
    records = sorted(records, key=lambda r: r.confidence, reverse=True)

    summary = Table(title=f"stance_labels.parquet — {target}")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value")
    summary.add_row("rows after filter", str(len(records)))
    counts: dict[str, int] = {}
    for record in records:
        counts[record.stance_label.value] = counts.get(record.stance_label.value, 0) + 1
    summary.add_row("by label", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    console.print(summary)

    shown = records[:limit]
    table = Table(title=f"Top {len(shown)} by confidence")
    table.add_column("#")
    table.add_column("market")
    table.add_column("post / segment")
    table.add_column("label")
    table.add_column("score")
    table.add_column("conf.")
    for index, record in enumerate(shown, start=1):
        token = record.segment_id or record.post_id
        table.add_row(
            str(index),
            _compact_id(record.market_id),
            _compact_id(token),
            record.stance_label.value,
            f"{record.stance_score:+.2f}",
            f"{record.confidence:.2f}",
        )
    console.print(table)
    for index, record in enumerate(shown, start=1):
        console.print(
            f"{index}. market: {_compact_id(record.market_id)} | "
            f"{_terminal_text(record.market_question, limit=100)}",
            markup=False,
        )
        console.print(
            f"   text:   {_terminal_text(record.segment_text_preview, limit=160)}",
            markup=False,
        )
        console.print(
            f"   why:    {_terminal_text(record.explanation, limit=140)}",
            markup=False,
        )


@stance_app.command("export-review")
def stance_export_review(
    input_path: Path = typer.Option(None, "--input", "--stance-path"),
    output_path: Path = typer.Option(None, "--output", "--output-path"),
    top_per_label: int = typer.Option(50, "--top-per-label", min=0),
    top_neutral: int = typer.Option(20, "--top-neutral", min=0),
) -> None:
    """Export a curated stance review CSV for human spot-checking."""
    paths = get_data_paths()
    source = input_path or paths.stance_labels_parquet
    if not source.exists():
        rprint(
            f"[yellow]No stance file at {source}.[/yellow] "
            "Run `polysignal stance score` first."
        )
        raise typer.Exit(code=2)
    records = read_stance_records(source)
    if not records:
        rprint(f"[yellow]File at {source} is empty.[/yellow]")
        raise typer.Exit(code=2)
    target = output_path or paths.manual_stance_review_csv
    _, written = write_manual_stance_review(
        records,
        target,
        top_per_label=top_per_label,
        top_neutral=top_neutral,
    )
    rprint(f"[green]manual stance review[/green] -> {target} ({written} rows)")


@stance_app.command("evaluate")
def stance_evaluate(
    input_path: Path = typer.Option(
        None, "--input", "--stance-path", help="Path to stance_labels.parquet."
    ),
    human_labels_path: Path = typer.Option(
        None,
        "--human-labels",
        help=(
            "CSV with columns stance_id,human_stance_label[,review_status]. "
            "Defaults to data/analytical/stance/manual_stance_review.csv "
            "when present."
        ),
    ),
    output_path: Path = typer.Option(None, "--output", "--output-path"),
    no_report: bool = typer.Option(
        False,
        "--no-report/--write-report",
        help="Skip writing the JSON report.",
    ),
) -> None:
    """Score the stance dataset against optional human labels."""
    paths = get_data_paths()
    source = input_path or paths.stance_labels_parquet
    if not source.exists():
        rprint(
            f"[yellow]No stance file at {source}.[/yellow] "
            "Run `polysignal stance score` first."
        )
        raise typer.Exit(code=2)
    records = read_stance_records(source)
    label_source = human_labels_path or paths.manual_stance_review_csv
    human_labels = _load_human_stance_labels(label_source)
    report = evaluate_stance(records, human_labels)

    summary = Table(title="Stance evaluation")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value")
    summary.add_row("predictions", str(report.total_predictions))
    summary.add_row("with human labels", str(report.total_with_labels))
    summary.add_row(
        "accuracy",
        f"{report.accuracy:.3f}" if report.accuracy is not None else "n/a",
    )
    summary.add_row(
        "macro F1",
        f"{report.macro_f1:.3f}" if report.macro_f1 is not None else "n/a",
    )
    summary.add_row("coverage (directional)", f"{report.coverage:.3f}")
    summary.add_row("abstention rate", f"{report.abstention_rate:.3f}")
    if report.per_label_counts:
        summary.add_row(
            "label counts",
            ", ".join(f"{k}={v}" for k, v in sorted(report.per_label_counts.items())),
        )
    if report.warnings:
        summary.add_row("warnings", "\n".join(report.warnings))
    console.print(summary)

    if not no_report:
        target = output_path or paths.stance_evaluation_report_json
        write_stance_evaluation_report(report, target)
        rprint(f"[green]stance evaluation report[/green] -> {target}")


# ---------------------------------------------------------------------------
# Features (Branch 08) — Social Pressure Index
# ---------------------------------------------------------------------------


def _validate_dataset_mode(value: str) -> str:
    mode = (value or "").strip().lower()
    if mode not in {"real", "auto"}:
        rprint(
            f"[red]Invalid --dataset-mode:[/red] {value!r} "
            "(choose from: real, auto)"
        )
        raise typer.Exit(code=2)
    return mode


def _parse_window_days(value: str) -> tuple[int, ...]:
    try:
        days = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        rprint(f"[red]Invalid --window-days:[/red] {value!r} (use e.g. 7,30,90)")
        raise typer.Exit(code=2) from exc
    if not days or any(d <= 0 for d in days) or list(days) != sorted(days):
        rprint(
            f"[red]Invalid --window-days:[/red] {value!r} "
            "(must be positive, ascending, e.g. 7,30,90)"
        )
        raise typer.Exit(code=2)
    return days


def _render_spi_summary(result: SPIResult) -> None:
    summary = result.summary
    table = Table(title=f"Social Pressure Index — {result.dataset_mode} dataset")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("markets evaluated", str(summary["markets_evaluated"]))
    direction = summary["pressure_direction_counts"]
    table.add_row(
        "pressure direction",
        ", ".join(f"{k}={v}" for k, v in direction.items()),
    )
    totals = summary["totals"]
    table.add_row("accepted links", str(totals["accepted_links"]))
    table.add_row("linked posts", str(totals["linked_posts"]))
    table.add_row("supports_yes / supports_no", f"{totals['supports_yes']} / {totals['supports_no']}")
    table.add_row("neutral / ambiguous", f"{totals['neutral']} / {totals['ambiguous']}")
    table.add_row("directional stances", str(totals["directional"]))
    console.print(table)


def _render_spi_comparison(comparison: dict) -> None:
    table = Table(title="SPI vs synthetic expected-results")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("markets evaluated", str(comparison["markets_evaluated"]))
    table.add_row(
        "expected direction",
        ", ".join(f"{k}={v}" for k, v in comparison["expected_direction_counts"].items()),
    )
    table.add_row(
        "direction agreement (exact)",
        f"{comparison['agreement_exact']}/{comparison['markets_evaluated']}",
    )
    table.add_row(
        "direction agreement (lenient MIXED)",
        f"{comparison['agreement_lenient']}/{comparison['markets_evaluated']} "
        f"({comparison['agreement_rate']:.1%})",
    )
    table.add_row("supports_yes count match", str(comparison["supports_yes_count_match"]))
    table.add_row("supports_no count match", str(comparison["supports_no_count_match"]))
    console.print(table)
    if comparison["incorrect_examples"]:
        rprint("\n[bold]Disagreements:[/bold]")
        for ex in comparison["incorrect_examples"]:
            console.print(
                f"  - {_compact_id(ex['market_id'])}: expected {ex['expected']}, "
                f"got {ex['predicted']} (yes={ex['supports_yes']}, no={ex['supports_no']}) "
                f"| {ex['question']}",
                markup=False,
            )


@features_app.command("compute-social-pressure")
def features_compute_social_pressure(
    output_dir: Path = typer.Option(
        None, "--output-dir", help="Dataset root to read from and write SPI features under."
    ),
    dataset_mode: str = typer.Option(
        "auto", "--dataset-mode", help="synthetic | real | auto (auto detects synthetic markers)."
    ),
    window_days: str = typer.Option(
        "7,30,90", "--window-days", help="Ascending day cut points for temporal windows."
    ),
    decay_half_life_days: float = typer.Option(
        30.0, "--decay-half-life-days", min=0.0, help="Half-life for temporal decay pressure."
    ),
    min_evidence: int = typer.Option(
        3, "--min-evidence", min=1, help="Minimum directional stances before calling a direction."
    ),
    compare_expected: bool = typer.Option(
        False, "--compare-expected", help="Compare against synthetic expected-results (synthetic only)."
    ),
    export_csv: bool = typer.Option(
        False, "--export-csv", help="Also write a CSV mirror of the feature table."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Compute and print summary without writing any files."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Replace an existing social_pressure.parquet."
    ),
) -> None:
    """Compute the deterministic per-market Social Pressure Index feature table.

    Reads the canonical linked-evidence, stance, post and market artefacts
    under --output-dir and writes
    ``analytical/features/social_pressure.parquet``. Accepted evidence and
    directional stances reuse the Branch 07 graph-load policy verbatim; the
    resolved YES/NO label is never consumed (only the close timestamp).
    """
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    mode = _validate_dataset_mode(dataset_mode)
    cfg = SPIConfig(
        window_days=_parse_window_days(window_days),
        decay_half_life_days=decay_half_life_days,
        min_evidence=min_evidence,
    )
    try:
        result = compute_social_pressure(paths=paths, dataset_mode=mode, config=cfg)
    except FileNotFoundError as exc:
        rprint(f"[red]ERROR[/red] missing required dataset: {exc}")
        raise typer.Exit(code=1) from exc
    _render_spi_summary(result)

    comparison = None
    if compare_expected:
        expected_path = paths.root / "metadata" / "synthetic_expected_results.json"
        if not expected_path.exists():
            rprint(
                f"[yellow]No expected-results at {expected_path}.[/yellow] "
                "Comparison is only available for the synthetic benchmark."
            )
        else:
            comparison = compare_with_expected(result.rows, expected_path)
            _render_spi_comparison(comparison)

    if result.dataset_mode == "synthetic":
        rprint(
            "[yellow]NOTE[/yellow] synthetic dataset — engineering validation only, "
            "not a real predictive-performance claim."
        )

    if dry_run:
        rprint("[yellow]--dry-run set; no files written.[/yellow]")
        return

    target = paths.social_pressure_features_parquet
    if target.exists() and not overwrite:
        rprint(
            f"[red]Refusing to overwrite[/red] {target}. "
            "Re-run with --overwrite to replace it."
        )
        raise typer.Exit(code=2)
    write_social_pressure_features(result.rows, target)
    rprint(f"[green]SPI features[/green] -> {target} ({len(result.rows)} rows)")

    summary_payload = dict(result.summary)
    if comparison is not None:
        summary_payload["comparison"] = comparison
    summary_path = paths.social_pressure_summary_json
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    rprint(f"[green]SPI summary[/green]  -> {summary_path}")

    if export_csv:
        csv_path = paths.social_pressure_features_csv
        frame = pd.DataFrame([r.model_dump(mode="json") for r in result.rows])
        frame.to_csv(csv_path, index=False)
        rprint(f"[green]SPI csv[/green]      -> {csv_path}")


@features_app.command("inspect-social-pressure")
def features_inspect_social_pressure(
    output_dir: Path = typer.Option(
        None, "--output-dir", help="Dataset root whose SPI feature table to inspect."
    ),
    input_path: Path = typer.Option(
        None, "--input", help="Explicit path to social_pressure.parquet (overrides --output-dir)."
    ),
    limit: int = typer.Option(20, "--limit", min=1, help="Maximum rows to display."),
    direction: str = typer.Option(
        None, "--direction", help="Filter rows by pressure_direction (e.g. YES, NO, MIXED)."
    ),
) -> None:
    """Inspect a computed Social Pressure Index feature table."""
    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    target = input_path or paths.social_pressure_features_parquet
    if not target.exists():
        rprint(
            f"[yellow]No SPI feature table at {target}.[/yellow] "
            "Run `polysignal features compute-social-pressure` first."
        )
        raise typer.Exit(code=2)
    rows = read_social_pressure_features(target)
    if not rows:
        rprint(f"[yellow]File at {target} is empty.[/yellow]")
        raise typer.Exit(code=2)

    if direction:
        wanted = direction.strip().upper()
        rows = [r for r in rows if r.pressure_direction == wanted]
        if not rows:
            rprint(f"[yellow]No rows with pressure_direction={wanted!r}.[/yellow]")
            raise typer.Exit(code=2)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row.pressure_direction] = counts.get(row.pressure_direction, 0) + 1
    modes = {row.dataset_mode for row in rows}

    summary = Table(title=f"social_pressure.parquet — {target}")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value")
    summary.add_row("rows", str(len(rows)))
    summary.add_row("dataset_mode", ", ".join(sorted(modes)))
    summary.add_row("by direction", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    summary.add_row("supports_yes total", str(sum(r.supports_yes_count for r in rows)))
    summary.add_row("supports_no total", str(sum(r.supports_no_count for r in rows)))
    summary.add_row("accepted links total", str(sum(r.accepted_link_count for r in rows)))
    console.print(summary)

    shown = sorted(rows, key=lambda r: abs(r.stance_balance), reverse=True)[:limit]
    table = Table(title=f"Top {len(shown)} by pressure strength")
    table.add_column("market")
    table.add_column("dir")
    table.add_column("balance")
    table.add_column("yes/no")
    table.add_column("acc.links")
    table.add_column("decay")
    table.add_column("quality")
    table.add_column("question")
    for row in shown:
        table.add_row(
            _compact_id(row.market_id),
            row.pressure_direction,
            f"{row.stance_balance:+.2f}",
            f"{row.supports_yes_count}/{row.supports_no_count}",
            str(row.accepted_link_count),
            f"{row.temporal_decay_pressure:+.2f}",
            f"{row.evidence_quality_score:.2f}",
            _terminal_text(row.question, limit=44),
        )
    console.print(table)


@features_app.command("build-timelines")
def features_build_timelines(
    output_dir: Path = typer.Option(
        None, "--output-dir", help="Dataset root to read from and write timelines under."
    ),
    half_life_days: float = typer.Option(
        3.0, "--half-life-days", help="EWMA half-life for the social signal."
    ),
    laplace_alpha: float = typer.Option(
        0.75, "--laplace-alpha", help="Pseudo-counts anchoring the signal at 0.5."
    ),
    source_daily_cap: float = typer.Option(
        3.0,
        "--source-daily-cap",
        help="Max stance weight one author/channel contributes per market-day.",
    ),
) -> None:
    """Build per-market daily timelines: Polymarket probability vs social signal.

    Reads the curated price points, accepted evidence, and stance labels,
    then writes ``analytical/timelines/market_timelines.parquet`` — the
    artifact behind the dashboard's Market Timeline graph. The social
    signal is smoothed and accepted-evidence-weighted; it never fabricates
    direction on days without evidence (it decays toward 0.5 with
    ``signal_strength`` near 0).
    """
    from polysignal.features.timeline import TimelineConfig, build_market_timelines

    paths = get_data_paths(root=output_dir) if output_dir else get_data_paths()
    if not paths.polymarket_price_points_parquet.exists():
        rprint(
            "[red]ERROR[/red] price_points.parquet not found — run "
            "`polysignal polymarket ingest-curated` first."
        )
        raise typer.Exit(code=1)
    result = build_market_timelines(
        paths=paths,
        config=TimelineConfig(
            half_life_days=half_life_days,
            laplace_alpha=laplace_alpha,
            source_daily_cap=source_daily_cap,
        ),
    )
    table = Table(title="Market timelines")
    table.add_column("Market", style="bold")
    table.add_column("Days")
    table.add_column("Price days")
    table.add_column("Evidence")
    table.add_column("Max strength")
    for row in result.summary["markets"]:
        table.add_row(
            _terminal_text(row["question"], limit=52),
            str(row["days"]),
            str(row["price_days_observed"]),
            str(row["evidence_total"]),
            f"{row['max_signal_strength']:.2f}",
        )
    console.print(table)
    rprint(f"[green]timelines[/green] -> {result.parquet_path} ({len(result.frame)} rows)")
    rprint(f"[green]summary[/green]   -> {result.summary_path}")


if __name__ == "__main__":  # pragma: no cover - convenience
    app()

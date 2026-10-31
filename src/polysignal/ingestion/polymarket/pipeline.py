"""End-to-end Polymarket ingestion orchestration.

Brings together the HTTP client, the binary YES/NO filter, the
normalizer, the raw store, and the Parquet writer. Designed to be
called either from the CLI or programmatically.

The pipeline records *why* every skipped market was dropped, so the
caller can render a faithful summary instead of a "kept N" headline
that hides the filter behaviour.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from polysignal.config import Settings, get_settings
from polysignal.domain import Market, Outcome, PricePoint
from polysignal.ingestion.polymarket.binary_filter import BinaryMarket, split_binary
from polysignal.ingestion.polymarket.client import PolymarketAPIError, PolymarketClient
from polysignal.ingestion.polymarket.normalizer import (
    normalize_market,
    normalize_price_history,
)
from polysignal.ingestion.polymarket.raw_store import RawRunWriter, generate_run_id
from polysignal.storage.parquet_store import upsert_parquet, write_parquet
from polysignal.storage.paths import DataPaths, get_data_paths

logger = logging.getLogger(__name__)


@dataclass
class SkippedMarket:
    """Audit record for a market that the binary filter rejected."""

    market_ref: str
    reason: str
    outcomes: list[str] = field(default_factory=list)


@dataclass
class PriceFetchOutcome:
    """Audit record for one (market, outcome) price-history attempt."""

    market_id: str
    outcome_id: str
    label: str
    fetched: bool
    point_count: int = 0
    reason: str | None = None  # populated on failure


@dataclass
class IngestionResult:
    """Summary returned by :func:`run_market_ingestion`."""

    fetched: int = 0
    events_fetched: int = 0
    markets_discovered: int = 0
    kept: int = 0
    skipped: list[SkippedMarket] = field(default_factory=list)
    markets: list[Market] = field(default_factory=list)
    outcomes: list[Outcome] = field(default_factory=list)
    price_points: list[PricePoint] = field(default_factory=list)
    price_attempts: list[PriceFetchOutcome] = field(default_factory=list)
    raw_markets_dir: Path | None = None
    raw_events_dir: Path | None = None
    raw_prices_dir: Path | None = None
    markets_parquet: Path | None = None
    outcomes_parquet: Path | None = None
    price_points_parquet: Path | None = None
    run_id: str = ""
    discovery_method: str = "events"
    order: str | None = None
    ascending: bool = False
    max_markets_per_event: int | None = None


def run_market_ingestion(
    *,
    limit: int = 50,
    binary_only: bool = True,
    save_raw: bool = True,
    fetch_prices: bool = False,
    dry_run: bool = False,
    page_size: int = 100,
    discovery: Literal["events", "markets"] = "events",
    order: str | None = "volume_24hr",
    ascending: bool = False,
    market_status: Literal["active", "closed", "all"] = "active",
    max_markets_per_event: int | None = 2,
    extra_params: dict[str, Any] | None = None,
    client: PolymarketClient | None = None,
    settings: Settings | None = None,
    paths: DataPaths | None = None,
    run_id: str | None = None,
) -> IngestionResult:
    """Fetch, filter, normalize, and persist Polymarket markets.

    Parameters
    ----------
    limit:
        Maximum number of markets to fetch from the upstream API.
    binary_only:
        When True (default), markets that are not binary YES/NO are
        dropped with a recorded reason. When False, the filter still
        runs (so non-binary markets can be reported) but they are NOT
        normalized — this milestone explicitly does not support them.
    save_raw:
        Persist raw API pages under ``data/raw/polymarket/markets/...``.
    fetch_prices:
        Also fetch the price history for each kept binary market and
        write ``data/normalized/polymarket/price_points.parquet``.
    dry_run:
        Run the full pipeline up to normalization but do not write any
        Parquet files. Raw payloads are still saved (so the run is
        auditable) unless ``save_raw`` is False.
    """
    settings = settings or get_settings()
    paths = paths or get_data_paths()
    paths.ensure()
    rid = run_id or generate_run_id()
    if max_markets_per_event is not None and max_markets_per_event <= 0:
        max_markets_per_event = None
    result = IngestionResult(
        run_id=rid,
        discovery_method=discovery,
        order=order,
        ascending=ascending,
        max_markets_per_event=max_markets_per_event,
    )

    owned_client = client is None
    polymarket_client = client or PolymarketClient(settings=settings)

    try:
        markets_payloads = _fetch_market_payloads(
            polymarket_client,
            paths=paths,
            run_id=rid,
            limit=limit,
            page_size=page_size,
            discovery=discovery,
            order=order,
            ascending=ascending,
            market_status=market_status,
            max_markets_per_event=max_markets_per_event,
            extra_params=extra_params,
            save_raw=save_raw,
            user_agent=settings.polymarket.user_agent,
            result=result,
        )

        kept_pairs, skipped_pairs = split_binary(markets_payloads)
        if len(kept_pairs) > limit:
            kept_pairs = kept_pairs[:limit]

        result.fetched = len(markets_payloads)
        for payload, reason in skipped_pairs:
            result.skipped.append(
                SkippedMarket(
                    market_ref=str(
                        payload.get("conditionId")
                        or payload.get("id")
                        or payload.get("slug")
                        or "unknown"
                    ),
                    reason=reason,
                    outcomes=_safe_outcomes(payload),
                )
            )

        normalize_iter: Iterable[tuple[dict[str, Any], BinaryMarket]] = kept_pairs
        if not binary_only:
            # The flag is exposed for future-compatibility: this milestone
            # cannot normalize non-binary markets (the downstream stance /
            # social-pressure code is YES/NO-only), so they still do not
            # reach Parquet. We just log that the operator opted in.
            for payload, reason in skipped_pairs:
                logger.info(
                    "polymarket: --include-non-binary set but %s still cannot "
                    "be normalized (%s); excluded from parquet outputs",
                    payload.get("conditionId") or payload.get("id"),
                    reason,
                )

        for payload, binary in normalize_iter:
            try:
                market = normalize_market(payload, binary=binary)
            except Exception as exc:
                logger.warning(
                    "polymarket: normalization failed for %s: %s",
                    payload.get("conditionId") or payload.get("id"),
                    exc,
                )
                result.skipped.append(
                    SkippedMarket(
                        market_ref=str(
                            payload.get("conditionId") or payload.get("id") or "unknown"
                        ),
                        reason=f"normalization error: {exc}",
                        outcomes=_safe_outcomes(payload),
                    )
                )
                continue
            result.markets.append(market)
            result.outcomes.extend(market.outcomes)

        result.kept = len(result.markets)

        if fetch_prices and result.markets:
            _fetch_prices_for_markets(
                polymarket_client,
                paths=paths,
                run_id=rid,
                markets=result.markets,
                save_raw=save_raw,
                user_agent=settings.polymarket.user_agent,
                result=result,
            )

        if not dry_run:
            _persist_normalized(result, paths, price_points_mode="overwrite")

    finally:
        if owned_client:
            polymarket_client.close()

    return result


def _fetch_market_payloads(
    client: PolymarketClient,
    *,
    paths: DataPaths,
    run_id: str,
    limit: int,
    page_size: int,
    discovery: Literal["events", "markets"],
    order: str | None,
    ascending: bool,
    market_status: Literal["active", "closed", "all"],
    max_markets_per_event: int | None,
    extra_params: dict[str, Any] | None,
    save_raw: bool,
    user_agent: str,
    result: IngestionResult,
) -> list[dict[str, Any]]:
    """Discover raw Gamma market payloads, optionally persisting raw pages."""
    if discovery == "events":
        return _fetch_market_payloads_from_events(
            client,
            paths=paths,
            run_id=run_id,
            limit=limit,
            page_size=page_size,
            order=order,
            ascending=ascending,
            market_status=market_status,
            max_markets_per_event=max_markets_per_event,
            extra_params=extra_params,
            save_raw=save_raw,
            user_agent=user_agent,
            result=result,
        )

    return _fetch_market_payloads_from_markets(
        client,
        paths=paths,
        run_id=run_id,
        limit=limit,
        page_size=page_size,
        order=order,
        ascending=ascending,
        market_status=market_status,
        extra_params=extra_params,
        save_raw=save_raw,
        user_agent=user_agent,
        result=result,
    )


def _fetch_market_payloads_from_markets(
    client: PolymarketClient,
    *,
    paths: DataPaths,
    run_id: str,
    limit: int,
    page_size: int,
    order: str | None,
    ascending: bool,
    market_status: Literal["active", "closed", "all"],
    extra_params: dict[str, Any] | None,
    save_raw: bool,
    user_agent: str,
    result: IngestionResult,
) -> list[dict[str, Any]]:
    """Walk the Gamma ``/markets`` pagination, optionally persisting pages."""
    writer: RawRunWriter | None = None
    request_params = _market_listing_params(
        order=order,
        ascending=ascending,
        market_status=market_status,
        extra_params=extra_params,
    )
    if save_raw:
        directory = paths.raw_run_dir("polymarket/markets", run_id)
        writer = RawRunWriter(
            directory=directory,
            endpoint="GET /markets",
            params={
                "limit": limit,
                "page_size": page_size,
                "extra_params": extra_params or {},
            },
            user_agent=user_agent,
        )

    payloads: list[dict[str, Any]] = []
    try:
        for page_index, page in client.iter_markets(
            limit=limit,
            page_size=page_size,
            extra_params=request_params,
        ):
            payloads.extend(page)
            if writer is not None:
                writer.write_page(page_index, page)
    except PolymarketAPIError as exc:
        logger.error("polymarket: aborting market fetch — %s", exc)
        if writer is not None:
            writer.extra["error"] = str(exc)
            writer.write_manifest()
            result.raw_markets_dir = writer.directory
        raise

    if writer is not None:
        writer.extra["limit"] = limit
        writer.write_manifest()
        result.raw_markets_dir = writer.directory

    return payloads


def _fetch_market_payloads_from_events(
    client: PolymarketClient,
    *,
    paths: DataPaths,
    run_id: str,
    limit: int,
    page_size: int,
    order: str | None,
    ascending: bool,
    market_status: Literal["active", "closed", "all"],
    max_markets_per_event: int | None,
    extra_params: dict[str, Any] | None,
    save_raw: bool,
    user_agent: str,
    result: IngestionResult,
) -> list[dict[str, Any]]:
    """Walk Gamma ``/events`` and extract markets across event families."""
    writer: RawRunWriter | None = None
    request_params = _event_listing_params(
        order=order,
        ascending=ascending,
        market_status=market_status,
        extra_params=extra_params,
    )
    if save_raw:
        directory = paths.raw_run_dir("polymarket/events", run_id)
        writer = RawRunWriter(
            directory=directory,
            endpoint="GET /events",
            params={
                "limit": limit,
                "page_size": page_size,
                "max_markets_per_event": max_markets_per_event,
                "request_params": request_params,
            },
            user_agent=user_agent,
        )

    payloads: list[dict[str, Any]] = []
    try:
        for page_index, page in client.iter_events(
            limit=None,
            page_size=page_size,
            extra_params=request_params,
        ):
            result.events_fetched += len(page)
            if writer is not None:
                writer.write_page(page_index, page)

            for event in page:
                event_payloads = _extract_event_markets(
                    event,
                    market_status=market_status,
                    max_markets_per_event=max_markets_per_event,
                    result=result,
                )
                payloads.extend(event_payloads)
                if len(payloads) >= limit:
                    payloads = payloads[:limit]
                    break
            if len(payloads) >= limit:
                break
    except PolymarketAPIError as exc:
        logger.error("polymarket: aborting event fetch — %s", exc)
        if writer is not None:
            writer.extra["error"] = str(exc)
            writer.write_manifest()
            result.raw_events_dir = writer.directory
        raise

    if writer is not None:
        writer.extra["limit"] = limit
        writer.extra["events_fetched"] = result.events_fetched
        writer.extra["markets_discovered"] = result.markets_discovered
        writer.write_manifest()
        result.raw_events_dir = writer.directory

    return payloads


def _fetch_prices_for_markets(
    client: PolymarketClient,
    *,
    paths: DataPaths,
    run_id: str,
    markets: list[Market],
    save_raw: bool,
    user_agent: str,
    result: IngestionResult,
) -> None:
    writer: RawRunWriter | None = None
    if save_raw:
        directory = paths.raw_run_dir("polymarket/prices", run_id)
        writer = RawRunWriter(
            directory=directory,
            endpoint="GET /prices-history",
            params={"interval": "max", "fidelity": 60},
            user_agent=user_agent,
        )

    for market in markets:
        for outcome in market.outcomes:
            audit = PriceFetchOutcome(
                market_id=market.market_id,
                outcome_id=outcome.outcome_id,
                label=outcome.label,
                fetched=False,
            )
            if _looks_synthesized(outcome.outcome_id, market.market_id):
                audit.reason = "no CLOB token id exposed by Gamma"
                result.price_attempts.append(audit)
                continue
            try:
                envelope = client.fetch_price_history(outcome.outcome_id)
            except PolymarketAPIError as exc:
                audit.reason = f"clob error: {exc}"
                result.price_attempts.append(audit)
                logger.warning(
                    "polymarket: skipping prices for %s/%s — %s",
                    market.market_id,
                    outcome.outcome_id,
                    exc,
                )
                continue

            if writer is not None:
                writer.write_named(
                    f"{_safe_filename(market.market_id)}__{_safe_filename(outcome.outcome_id)}",
                    envelope,
                )

            points = normalize_price_history(
                market_id=market.market_id,
                outcome_id=outcome.outcome_id,
                history=envelope,
            )
            audit.fetched = True
            audit.point_count = len(points)
            if not points:
                audit.reason = "empty history"
            result.price_attempts.append(audit)
            result.price_points.extend(points)

    if writer is not None:
        writer.write_manifest()
        result.raw_prices_dir = writer.directory


def _persist_normalized(
    result: IngestionResult,
    paths: DataPaths,
    *,
    price_points_mode: Literal["overwrite", "upsert"] = "overwrite",
    markets_mode: Literal["overwrite", "upsert"] = "overwrite",
) -> None:
    if result.markets:
        if markets_mode == "upsert":
            # Incremental curation must not wipe previously ingested
            # markets — fetch_prices_for_market used to overwrite the
            # whole table with the single refreshed market.
            result.markets_parquet = upsert_parquet(
                result.markets,
                paths.polymarket_markets_parquet,
                key_columns=("market_id",),
            )
        else:
            result.markets_parquet = write_parquet(
                result.markets,
                paths.polymarket_markets_parquet,
            )
    if result.outcomes:
        if markets_mode == "upsert":
            result.outcomes_parquet = upsert_parquet(
                result.outcomes,
                paths.polymarket_outcomes_parquet,
                key_columns=("market_id", "outcome_id"),
            )
        else:
            result.outcomes_parquet = write_parquet(
                result.outcomes,
                paths.polymarket_outcomes_parquet,
            )
    if result.price_points:
        if price_points_mode == "upsert":
            # Single-market refreshes append without duplicating samples
            # from repeated calls over the same CLOB history interval.
            result.price_points_parquet = upsert_parquet(
                result.price_points,
                paths.polymarket_price_points_parquet,
                key_columns=("market_id", "outcome_id", "timestamp"),
            )
        else:
            result.price_points_parquet = write_parquet(
                result.price_points,
                paths.polymarket_price_points_parquet,
            )


def _event_listing_params(
    *,
    order: str | None,
    ascending: bool,
    market_status: Literal["active", "closed", "all"],
    extra_params: dict[str, Any] | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if order:
        params["order"] = order
        params["ascending"] = str(ascending).lower()
    if market_status == "active":
        params.update({"active": "true", "closed": "false"})
    elif market_status == "closed":
        params.update({"closed": "true"})
    if extra_params:
        params.update(extra_params)
    return params


def _market_listing_params(
    *,
    order: str | None,
    ascending: bool,
    market_status: Literal["active", "closed", "all"],
    extra_params: dict[str, Any] | None,
) -> dict[str, Any]:
    params = _event_listing_params(
        order=order,
        ascending=ascending,
        market_status=market_status,
        extra_params=extra_params,
    )
    return params


def _extract_event_markets(
    event: dict[str, Any],
    *,
    market_status: Literal["active", "closed", "all"],
    max_markets_per_event: int | None,
    result: IngestionResult,
) -> list[dict[str, Any]]:
    raw_markets = event.get("markets")
    if not isinstance(raw_markets, list):
        return []

    event_markets: list[dict[str, Any]] = []
    context = _event_context(event)
    for raw_market in raw_markets:
        if not isinstance(raw_market, dict):
            continue
        result.markets_discovered += 1
        if not _market_status_allowed(raw_market, market_status):
            continue
        if max_markets_per_event is not None and len(event_markets) >= max_markets_per_event:
            break

        market = dict(raw_market)
        if not market.get("events"):
            market["events"] = [context]
        event_markets.append(market)
    return event_markets


def _event_context(event: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for key in (
        "id",
        "slug",
        "title",
        "category",
        "subcategory",
        "tags",
        "active",
        "closed",
        "volume",
        "volume24hr",
        "liquidity",
        "startDate",
        "endDate",
    ):
        if key in event:
            context[key] = event[key]
    return context


def _market_status_allowed(
    market: dict[str, Any],
    market_status: Literal["active", "closed", "all"],
) -> bool:
    if market_status == "all":
        return True
    is_closed = bool(market.get("closed"))
    if market_status == "closed":
        return is_closed
    # Active dataset discovery should exclude settled/closed markets and
    # markets explicitly marked inactive. Missing ``active`` is treated as
    # usable because older Gamma payloads do not always include it.
    return not is_closed and market.get("active") is not False


def _looks_synthesized(outcome_id: str, market_id: str) -> bool:
    return outcome_id == f"{market_id}-yes" or outcome_id == f"{market_id}-no"


def _safe_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in value)[:64]


def _safe_outcomes(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("outcomes")
    if isinstance(raw, list):
        return [str(o) for o in raw]
    if isinstance(raw, str):
        return [raw]
    return []


def fetch_prices_for_market(
    market_id: str,
    *,
    client: PolymarketClient | None = None,
    settings: Settings | None = None,
    paths: DataPaths | None = None,
    save_raw: bool = True,
    run_id: str | None = None,
) -> IngestionResult:
    """Fetch price history for a single binary market by id / slug.

    Convenience entry point for the ``polysignal polymarket fetch-prices
    --market-id <id>`` command. Reuses the same audit and storage
    machinery as the bulk ingestion.
    """
    settings = settings or get_settings()
    paths = paths or get_data_paths()
    paths.ensure()
    rid = run_id or generate_run_id()

    owned = client is None
    polymarket_client = client or PolymarketClient(settings=settings)
    result = IngestionResult(run_id=rid, discovery_method="market_lookup")
    try:
        payload = polymarket_client.fetch_market(market_id)
        if payload is None:
            raise PolymarketAPIError(f"market not found: {market_id}")

        kept, skipped = split_binary([payload])
        if not kept:
            reason = skipped[0][1] if skipped else "unknown"
            result.skipped.append(
                SkippedMarket(
                    market_ref=market_id,
                    reason=reason,
                    outcomes=_safe_outcomes(payload),
                )
            )
            return result

        _, binary = kept[0]
        market = normalize_market(payload, binary=binary)
        result.fetched = 1
        result.kept = 1
        result.markets.append(market)
        result.outcomes.extend(market.outcomes)

        _fetch_prices_for_markets(
            polymarket_client,
            paths=paths,
            run_id=rid,
            markets=[market],
            save_raw=save_raw,
            user_agent=settings.polymarket.user_agent,
            result=result,
        )
        _persist_normalized(result, paths, price_points_mode="upsert", markets_mode="upsert")
    finally:
        if owned:
            polymarket_client.close()
    return result


# ---------------------------------------------------------------------------
# Curated ingestion — explicit market list with full-lifetime price history
# ---------------------------------------------------------------------------


@dataclass
class CuratedMarketReport:
    """Per-market audit for one curated ingestion attempt."""

    market_ref: str
    market_id: str | None = None
    question: str = ""
    ingested: bool = False
    resolution: str | None = None
    volume: float | None = None
    close_time: str | None = None
    price_points: int = 0
    price_span_days: float = 0.0
    timeline_ok: bool = False
    reason: str | None = None


def ingest_curated_markets(
    market_refs: list[str],
    *,
    min_price_points: int = 100,
    min_price_span_days: float = 7.0,
    fidelity: int = 60,
    save_raw: bool = True,
    client: PolymarketClient | None = None,
    settings: Settings | None = None,
    paths: DataPaths | None = None,
    run_id: str | None = None,
) -> tuple[IngestionResult, list[CuratedMarketReport]]:
    """Ingest an explicit list of markets with **full-lifetime** price history.

    For every reference (condition id, numeric id, or slug) this:

    1. fetches and normalizes the market (binary YES/NO only);
    2. walks the CLOB ``/prices-history`` endpoint in bounded windows from
       the market's creation to its close (resolved markets expose at most
       ~3 weeks via ``interval=max``, so the chunked range fetch is the
       only way to recover the real timeline);
    3. validates timeline quality against ``min_price_points`` /
       ``min_price_span_days`` — markets without a usable timeline are
       reported (``timeline_ok=False``) and **not** persisted, so the
       frontend can rely on every curated market having a real timeline;
    4. upserts ``markets`` / ``outcomes`` / ``price_points`` parquet
       tables (existing curated markets are preserved).
    """
    settings = settings or get_settings()
    paths = paths or get_data_paths()
    paths.ensure()
    rid = run_id or generate_run_id()

    owned = client is None
    polymarket_client = client or PolymarketClient(settings=settings)
    result = IngestionResult(run_id=rid, discovery_method="curated_lookup")
    reports: list[CuratedMarketReport] = []

    writer: RawRunWriter | None = None
    prices_writer: RawRunWriter | None = None
    if save_raw:
        writer = RawRunWriter(
            directory=paths.raw_run_dir("polymarket/markets", rid),
            endpoint="GET /markets (curated lookup)",
            params={"market_refs": list(market_refs)},
            user_agent=settings.polymarket.user_agent,
        )
        prices_writer = RawRunWriter(
            directory=paths.raw_run_dir("polymarket/prices", rid),
            endpoint="GET /prices-history (chunked range)",
            params={"fidelity": fidelity},
            user_agent=settings.polymarket.user_agent,
        )

    try:
        for index, ref in enumerate(market_refs):
            report = CuratedMarketReport(market_ref=ref)
            reports.append(report)
            try:
                payload = polymarket_client.fetch_market(ref)
            except PolymarketAPIError as exc:
                report.reason = f"lookup failed: {exc}"
                continue
            if payload is None:
                report.reason = "market not found"
                continue
            if writer is not None:
                writer.write_page(index, [payload])

            kept, skipped = split_binary([payload])
            if not kept:
                report.reason = skipped[0][1] if skipped else "not a binary YES/NO market"
                continue
            _, binary = kept[0]
            try:
                market = normalize_market(payload, binary=binary)
            except Exception as exc:  # same guard as the bulk pipeline
                report.reason = f"normalization error: {exc}"
                continue

            report.market_id = market.market_id
            report.question = market.question
            report.resolution = market.resolution.value if market.resolution else None
            report.volume = market.volume
            report.close_time = market.close_time.isoformat() if market.close_time else None

            market_points = _fetch_curated_prices(
                polymarket_client,
                market=market,
                fidelity=fidelity,
                prices_writer=prices_writer,
                result=result,
            )
            yes_points = [
                p for p in market_points if p.outcome_id == market.outcomes[0].outcome_id
            ]
            report.price_points = len(yes_points)
            report.price_span_days = _span_days(yes_points)
            report.timeline_ok = (
                report.price_points >= min_price_points
                and report.price_span_days >= min_price_span_days
            )
            if not report.timeline_ok:
                report.reason = (
                    f"unusable price timeline ({report.price_points} points, "
                    f"{report.price_span_days:.1f}d span) — replace this market"
                )
                continue

            report.ingested = True
            result.fetched += 1
            result.kept += 1
            result.markets.append(market)
            result.outcomes.extend(market.outcomes)
            result.price_points.extend(market_points)

        if writer is not None:
            writer.extra["reports"] = [report.__dict__ for report in reports]
            writer.write_manifest()
            result.raw_markets_dir = writer.directory
        if prices_writer is not None:
            prices_writer.write_manifest()
            result.raw_prices_dir = prices_writer.directory

        if result.markets:
            _persist_normalized(
                result,
                paths,
                price_points_mode="upsert",
                markets_mode="upsert",
            )
    finally:
        if owned:
            polymarket_client.close()
    return result, reports


def _fetch_curated_prices(
    client: PolymarketClient,
    *,
    market: Market,
    fidelity: int,
    prices_writer: RawRunWriter | None,
    result: IngestionResult,
) -> list[PricePoint]:
    """Chunked full-lifetime price fetch for every real CLOB outcome token."""
    from datetime import datetime, timedelta, timezone

    start_anchor = market.created_at
    end_anchor = market.close_time or market.resolved_at or datetime.now(tz=timezone.utc)
    # One day of margin on each side keeps boundary samples.
    start_ts = int((start_anchor - timedelta(days=1)).timestamp())
    end_ts = int((end_anchor + timedelta(days=1)).timestamp())

    points: list[PricePoint] = []
    for outcome in market.outcomes:
        audit = PriceFetchOutcome(
            market_id=market.market_id,
            outcome_id=outcome.outcome_id,
            label=outcome.label,
            fetched=False,
        )
        if _looks_synthesized(outcome.outcome_id, market.market_id):
            audit.reason = "no CLOB token id exposed by Gamma"
            result.price_attempts.append(audit)
            continue
        try:
            envelope = client.fetch_price_history_range(
                outcome.outcome_id,
                start_ts=start_ts,
                end_ts=end_ts,
                fidelity=fidelity,
            )
        except PolymarketAPIError as exc:
            audit.reason = f"clob error: {exc}"
            result.price_attempts.append(audit)
            logger.warning(
                "polymarket: curated price fetch failed for %s/%s — %s",
                market.market_id,
                outcome.outcome_id,
                exc,
            )
            continue
        if prices_writer is not None:
            prices_writer.write_named(
                f"{_safe_filename(market.market_id)}__{_safe_filename(outcome.outcome_id)}",
                envelope,
            )
        normalized = normalize_price_history(
            market_id=market.market_id,
            outcome_id=outcome.outcome_id,
            history=envelope,
        )
        audit.fetched = True
        audit.point_count = len(normalized)
        if not normalized:
            audit.reason = "empty history"
        result.price_attempts.append(audit)
        points.extend(normalized)
    return points


def _span_days(points: list[PricePoint]) -> float:
    if len(points) < 2:
        return 0.0
    stamps = sorted(p.timestamp for p in points)
    return round((stamps[-1] - stamps[0]).total_seconds() / 86400, 1)


__all__ = [
    "CuratedMarketReport",
    "IngestionResult",
    "PriceFetchOutcome",
    "SkippedMarket",
    "fetch_prices_for_market",
    "ingest_curated_markets",
    "run_market_ingestion",
]

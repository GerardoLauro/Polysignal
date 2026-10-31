"""Collect a domain-balanced set of resolved binary Polymarket markets.

Targets ~100 resolved binary YES/NO markets per Traccia_2 thematic bucket
(politics / finance / sports) with verified CLOB price history, and writes
them to the canonical normalized Polymarket parquet tree. Reuses the
project's real client, binary filter, domain classifier, normalizer, and
storage helpers so the output is schema-identical to ``polysignal polymarket
ingest``.

Existing market/outcome/price parquets are backed up under ``_backups/``
before being overwritten (the project's standard data-safety convention).

Usage:
    python scripts/collect_balanced_markets.py --per-bucket 100 \
        --candidate-multiplier 1.6 --min-points 120 --min-span-days 7
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datetime import datetime, timezone  # noqa: E402

from polysignal.domain import PricePoint  # noqa: E402
from polysignal.ingestion.polymarket.binary_filter import classify_market  # noqa: E402
from polysignal.ingestion.polymarket.client import PolymarketAPIError, PolymarketClient  # noqa: E402
from polysignal.ingestion.polymarket.normalizer import (  # noqa: E402
    _classify_domain,
    normalize_market,
    normalize_price_history,
)
from polysignal.storage.parquet_store import backup_parquet, write_parquet  # noqa: E402
from polysignal.storage.paths import get_data_paths  # noqa: E402

BUCKET = {
    "politics": "politics",
    "crypto": "finance",
    "economy": "finance",
    "sports": "sports",
    "culture": "sports",
}


def _settled(binary) -> bool:
    yp, np_ = binary.yes_price, binary.no_price
    if yp is None or np_ is None:
        return False
    return max(yp, np_) >= 0.99 and min(yp, np_) <= 0.01


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bucket", type=int, default=100)
    ap.add_argument("--candidate-multiplier", type=float, default=2.4)
    ap.add_argument("--scan-pages", type=int, default=200)
    ap.add_argument("--scan-end-date-min", default="2025-01-01",
                    help="only scan markets resolving on/after this date (price-history availability)")
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("--min-points", type=int, default=50)
    ap.add_argument("--min-span-days", type=float, default=3.0)
    ap.add_argument("--max-window-days", type=int, default=180,
                    help="cap the price-history lookback before close to bound API load")
    ap.add_argument("--pause", type=float, default=0.25)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    paths = get_data_paths(Path(args.output_dir)) if args.output_dir else get_data_paths()
    paths.ensure()
    per_bucket = args.per_bucket
    want_candidates = int(per_bucket * args.candidate_multiplier)

    client = PolymarketClient()
    print("Scanning Gamma /markets (closed, by volume) for binary resolved candidates...")
    candidates: dict[str, list[dict]] = {"politics": [], "finance": [], "sports": []}
    scanned = 0
    seen: set[str] = set()
    try:
        for _page_index, page in client.iter_markets(
            limit=args.scan_pages * args.page_size,
            page_size=args.page_size,
            extra_params={
                "closed": "true",
                "order": "volumeNum",
                "ascending": "false",
                "end_date_min": args.scan_end_date_min,
            },
        ):
            for m in page:
                scanned += 1
                cid = m.get("conditionId") or ""
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                res = classify_market(m)
                if not res.is_binary or res.market is None or not _settled(res.market):
                    continue
                if not (res.market.yes_token_id and res.market.no_token_id):
                    continue
                bucket = BUCKET.get(_classify_domain(m).value)
                if bucket is None:
                    continue
                if len(candidates[bucket]) < want_candidates:
                    candidates[bucket].append(m)
            if all(len(v) >= want_candidates for v in candidates.values()):
                break
    except PolymarketAPIError as exc:
        print(f"discovery error: {exc}")

    for b, items in candidates.items():
        items.sort(key=lambda m: -float(m.get("volumeNum") or m.get("volume") or 0))
        print(f"  candidate pool {b}: {len(items)}")

    # Fetch price history and accept up to per_bucket markets per bucket.
    kept_markets = []
    kept_outcomes = []
    kept_prices = []
    accepted_by_bucket = {"politics": 0, "finance": 0, "sports": 0}
    max_window = args.max_window_days * 86400
    for bucket, items in candidates.items():
        for m in items:
            if accepted_by_bucket[bucket] >= per_bucket:
                break
            try:
                market = normalize_market(m)
            except ValueError:
                continue
            yes_outcome = market.outcomes[0]
            no_outcome = market.outcomes[1] if len(market.outcomes) > 1 else None
            if yes_outcome.outcome_id.endswith(("-yes", "-no")):
                continue
            # Full-lifetime window: created_at -> close_time, capped to bound load.
            end_dt = market.close_time or market.resolved_at or datetime.now(tz=timezone.utc)
            start_dt = market.created_at or end_dt
            end_ts = int(end_dt.timestamp())
            start_ts = int(start_dt.timestamp())
            if end_ts - start_ts > max_window:
                start_ts = end_ts - max_window
            if end_ts <= start_ts:
                continue
            try:
                env = client.fetch_price_history_range(
                    yes_outcome.outcome_id,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    pause_seconds=args.pause,
                )
            except PolymarketAPIError:
                continue
            yes_points = normalize_price_history(
                market_id=market.market_id,
                outcome_id=yes_outcome.outcome_id,
                history=env,
            )
            if len(yes_points) < args.min_points:
                continue
            ts = sorted(p.timestamp for p in yes_points)
            span_days = (ts[-1] - ts[0]).total_seconds() / 86400 if len(ts) >= 2 else 0.0
            if span_days < args.min_span_days:
                continue
            # Derive the complementary NO series (binary: NO = 1 - YES).
            no_points = []
            if no_outcome is not None and not no_outcome.outcome_id.endswith(("-yes", "-no")):
                no_points = [
                    PricePoint(
                        market_id=p.market_id,
                        outcome_id=no_outcome.outcome_id,
                        timestamp=p.timestamp,
                        price=round(max(0.0, min(1.0, 1.0 - p.price)), 6),
                        volume=p.volume,
                    )
                    for p in yes_points
                ]
            kept_markets.append(market)
            kept_outcomes.extend(market.outcomes)
            kept_prices.extend(yes_points)
            kept_prices.extend(no_points)
            accepted_by_bucket[bucket] += 1
            print(
                f"  [{bucket}] {accepted_by_bucket[bucket]:3d}/{per_bucket} "
                f"{market.question[:60]!r} pts={len(yes_points)} span={span_days:.0f}d",
                flush=True,
            )
            time.sleep(args.pause)

    print("\n===== COLLECTED =====")
    for b, n in accepted_by_bucket.items():
        print(f"  {b:10s} {n}")
    print(f"  total markets : {len(kept_markets)}")
    print(f"  total prices  : {len(kept_prices)}")

    if not kept_markets:
        print("No markets collected; leaving existing parquets untouched.")
        return

    # Back up existing parquets, then overwrite with the balanced set.
    for p in (
        paths.polymarket_markets_parquet,
        paths.polymarket_outcomes_parquet,
        paths.polymarket_price_points_parquet,
    ):
        b = backup_parquet(p)
        if b:
            print(f"  backed up {p.name} -> {b}")

    write_parquet(kept_markets, paths.polymarket_markets_parquet)
    write_parquet(kept_outcomes, paths.polymarket_outcomes_parquet)
    write_parquet(kept_prices, paths.polymarket_price_points_parquet)
    print(f"\nwrote markets   -> {paths.polymarket_markets_parquet}")
    print(f"wrote outcomes  -> {paths.polymarket_outcomes_parquet}")
    print(f"wrote prices    -> {paths.polymarket_price_points_parquet}")


if __name__ == "__main__":
    main()

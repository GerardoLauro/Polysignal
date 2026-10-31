# PolySignal — canonical pipeline & data layout

This document is the single source of truth for the **product** pipeline:
the curated resolved-market dataset, the staged social collection, and the
artifacts that feed the dashboard's Market Timeline graph.

## Canonical data layout (`data/`)

Everything the product depends on lives under one root: `data/`.

```
data/
├── raw/                       # API payloads + raw social snapshots (run-scoped, lineage)
│   ├── polymarket/{markets,events,prices}/<date>/<run>/
│   └── social/telegram/<date>/<run>/
├── normalized/
│   ├── polymarket/
│   │   ├── markets.parquet          # 6 curated resolved markets
│   │   ├── outcomes.parquet         # YES/NO outcomes + CLOB token ids
│   │   └── price_points.parquet     # FULL-LIFETIME price/probability timelines
│   └── social/
│       ├── social_posts.parquet     # real Reddit + Telegram posts (NO demo rows)
│       ├── social_post_segments.parquet
│       ├── linked_posts.parquet     # Branch 05 relevance verdicts
│       └── stance_labels.parquet    # Branch 06 stance
├── analytical/
│   ├── evidence/                    # accepted_evidence.parquet, rejected_evidence.parquet, evidence_summary.json
│   ├── features/                    # social_pressure.parquet (per-market SPI)
│   ├── timelines/                   # market_timelines.parquet  <-- the PolySignal graph
│   └── backtests/<profile>/         # Branch 09 backtest + dashboard_prediction_view
├── staging/                         # large-scale collection scratch (reddit/, telegram_web/, enriched/)
├── state/                           # resumable collection checkpoints
├── reports/                         # collection summaries + run logs
└── cache/                           # LLM cache, embeddings, Telegram Telethon session
```

Old / demo / synthetic datasets were moved to `archive/` (see
`archive/README.md` and `archive/cleanup_manifest_2026-06-13.json`).
Nothing under `archive/` feeds the live pipeline. In particular the 900
`ui_stress_test_12E` synthetic rows were **excluded** when the real social
posts were imported into `data/`.

## The 6 curated markets

Selected for: resolved, binary YES/NO, high volume / hype, strong
Reddit + Telegram discussion, and a **verified full-lifetime CLOB price
timeline**. Defined in `config/curated_markets.txt`.

| Market | Resolution | Volume | Price points | Span |
| --- | --- | --- | --- | --- |
| US forces enter Iran by April 30? | YES | ~$269M | 318 | 21d |
| Fed decreases interest rates by 50+ bps after January 2026 meeting? | NO | ~$235M | 3,187 | 133d |
| Russia x Ukraine ceasefire by May 31, 2026? | YES | ~$141M | 732 | 39d |
| Will Arsenal win the 2025-26 English Premier League? | YES | ~$12M | 6,631 | 300d |
| Will Bitcoin reach $150,000 in February? | NO | ~$29M | 669 | 29d |
| Will the next Prime Minister of Hungary be Péter Magyar? | YES | ~$25M | 6,112 | 262d |

The resolved markets span geopolitics, macro, sports, crypto, and
elections — all topics covered by the configured Reddit/Telegram sources.

### Why chunked price history matters

Polymarket's CLOB `/prices-history` endpoint only returns ~3 weeks of data
for a settled market via `interval=max`, and rejects `startTs/endTs`
windows longer than ~2 weeks. The curated ingester
(`polysignal polymarket ingest-curated`) walks the market's full lifetime
in 14-day windows and merges them, so the frontend always has a complete
probability timeline. **Markets without a usable timeline are reported and
not persisted** — replace them with a better market.

## Rebuilding the canonical artifacts

Run in order from the repo root (all default to the `data/` root):

```bash
# 1. Markets + full-lifetime price timelines (network)
polysignal polymarket ingest-curated --ids-file config/curated_markets.txt

# 2. Relevance linking (which posts are about which market)
polysignal linking score-candidates --candidate-mode full-scan

# 3. Outcome-aware stance (does a post support YES or NO)
polysignal stance score

# 4. Accepted / rejected evidence with reasons + provenance
polysignal linking export-evidence

# 5. Per-market Social Pressure Index features
polysignal features compute-social-pressure --overwrite

# 6. Market timelines (the PolySignal graph payload)
polysignal features build-timelines

# 7. Frontend backtest + dashboard prediction view
polysignal evaluate build-dataset
polysignal evaluate backtest --profile real_curated_main
polysignal evaluate build-dashboard-view --profile real_curated_main
```

Then launch the dashboard:

```bash
streamlit run src/polysignal/dashboard/app.py
```

## Evidence logic (accepted vs rejected)

`polysignal linking export-evidence` applies **exactly** the graph-loader
acceptance policy, row by row, and writes auditable tables to
`data/analytical/evidence/`:

* `accepted_evidence.parquet` — the rows that count as real evidence;
* `rejected_evidence.parquet` — every dropped row with a `rejection_stage`
  and `rejection_reason`;
* `evidence_summary.json` — counts by reason / market / platform.

Acceptance requires: an effective relevance label of `relevant` or
`weakly_relevant` (weak links need confidence ≥ 0.6), a resolvable parent
post + market, temporal admissibility (pre-close for resolved markets,
timestamp present), and survival of deduplication. Duplicate handling:
posts flagged `is_duplicate` are rejected; per `(market, post)` only the
highest-confidence row is accepted, the rest are rejected as
`duplicate_evidence`. Every row carries provenance: retrieval query +
type, the post's `query_provenance` / `retrieval_run_id`, channel/author,
raw snapshot path, and the scorer explanation.

## The PolySignal graph (Market Timeline)

`polysignal features build-timelines` writes
`data/analytical/timelines/market_timelines.parquet`: one row per market
per day with

* `market_prob` — the real Polymarket YES price (implied probability),
  forward-filled on no-trade days;
* `social_signal` — a smoothed, accepted-evidence-weighted estimate of
  P(YES) in `[0, 1]`;
* `signal_strength` — how much directional evidence backs the signal that
  day, in `[0, 1)`.

The social signal is deliberately **not** a raw daily post ratio (which
jumps 0↔100 on quiet days). Instead:

1. only **accepted** evidence joined with directional stance feeds it;
2. each author/channel contributes at most `source_daily_cap` of weight
   per market-day (source diversity, anti-spam);
3. each stance is weighted `confidence × (1 + log1p(engagement))`;
4. daily YES/NO weight is **EWMA-smoothed** (half-life 3 days);
5. the signal is **Laplace-anchored** at 0.5, so thin-evidence days stay
   near 0.5 with `signal_strength ≈ 0` instead of spiking.

The dashboard's "Market Timeline" section plots `market_prob` vs
`social_signal` with the per-day accepted-evidence count as bars, and only
treats the social line as meaningful where `signal_strength` is well above
zero.

## Large-scale social collection (up to ~1M posts)

Both collectors are staged, resumable, rate-limited, and logged. They
write to `data/staging/` and `data/state/`; promote vetted output into
`data/normalized/social/` only after review.

### 1. Scrapling public-web (Reddit + Telegram preview pages)

```bash
polysignal social collect-large-enriched \
  --config configs/social_enrichment_sources.example.yaml \
  --markets-path data/normalized/polymarket/markets.parquet \
  --reddit-root data/staging/reddit \
  --telegram-root data/staging/telegram_web \
  --output-root data/staging/enriched \
  --target-total-records 1000000 \
  --target-reddit-records 700000 \
  --target-telegram-records 300000 \
  --max-pages-per-source 200 --max-pages-per-query 8 \
  --rate-limit-seconds 4 --resume --stop-when-target-reached
```

Resumable via page-level checkpoints in `data/staging/enriched/state/`;
per-run logs in `data/staging/enriched/reports/`. Re-run the same command
to continue after a Reddit cooldown / crash.

### 2. Telegram API (Telethon, full channel history)

One-time auth (interactive, no secrets in code — set
`POLYSIGNAL_TELEGRAM_API_ID` / `POLYSIGNAL_TELEGRAM_API_HASH` in `.env`):

```bash
polysignal social telegram auth
```

Then the staged overnight collector:

```bash
polysignal social telegram collect-large \
  --config config/telegram_channels.yml \
  --target-total 1000000 \
  --batch-per-channel 1000 \
  --max-rounds 1000 \
  --pause-seconds 1 \
  --resume
```

Pages backwards through each channel's history by message id, checkpointing
`data/state/telegram_collection_state.json` after every round, normalizing
incrementally into `data/normalized/social/social_posts.parquet`, and
logging to `data/reports/telegram_large_<run>.log`. A killed run resumes
exactly where it stopped; saved messages are never re-collected. Telethon
honours Telegram's `FloodWait` automatically.

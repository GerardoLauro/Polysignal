# Branch 12 — Social dataset enrichment (Telegram + Reddit public web)

Branch 12 widens PolySignal's **real** social evidence base with two responsible
public-web datasets, collected with [Scrapling](https://github.com/D4Vinci/Scrapling):

- a **Telegram** public-channel dataset from `https://t.me/s/<channel>` preview pages;
- a **Reddit** public-web dataset from `https://old.reddit.com/r/<sub>/new/` listing pages.

Both are normalized into the existing `SocialPostRecord` schema, deduplicated
deterministically, and stored under their own data roots, then merged into the
combined enriched dataset used by the `real_social_enriched_main` Branch 09 profile.

## Why these sources, why Scrapling

- Reddit API / PRAW approval was **not available before the deadline**, so Reddit
  is collected **only** from public `old.reddit.com` pages. This is **not** official
  Reddit API access and **not** a complete crawl.
- Telegram public channels expose recent messages without login and are a dense,
  timestamped source of crypto/finance/breaking-news commentary.
- Scrapling is used purely as a responsible public-web extraction framework: only
  the plain `Fetcher` (no stealth/bypass fetchers), explicit rate limits, local raw
  snapshots, deterministic parsing, and deterministic deduplication.

## Data ethics & policy constraints

Allowed and used: public pages without login, conservative low-volume crawling,
explicit rate limits, raw snapshot storage, deterministic parsing/dedup, normalized
export with transparent `collection_method` provenance.

Not used (forbidden): login automation, CAPTCHA/Cloudflare/Turnstile bypass, proxy
rotation to evade limits, aggressive infinite scroll, scraping private/deleted/
protected content. Raw Reddit/Telegram content is used **only** to build a normalized
research dataset for PolySignal — never to train LLMs or general AI models.

If a source blocks, fails, or returns unusable content, the run fails that source
gracefully, records it in the coverage report, and continues with the others.

## Install

```cmd
pip install -e ".[scraping]"
```

This pulls `scrapling`, `curl-cffi`, `feedparser`, and `pyyaml`. The package still
imports without the extra; the adapters raise an actionable error only when used.

## Source configuration

Sources live in `configs/social_enrichment_sources.example.yaml` under two sections,
`reddit_scrapling` and `telegram_public_web`. Each source carries a `name`, target
`url`, `domains`, and `priority`. Sources may be marked `status: needs_validation`;
they are attempted and, if unavailable, recorded as failed without aborting the run.
The loader validates sections and sources at runtime and skips invalid ones.

## How to (re)run collection

```cmd
:: Reddit public web -> data_reddit_scrapling_eval
python -m polysignal social collect-reddit-scrapling ^
  --config configs/social_enrichment_sources.example.yaml ^
  --output-root data_reddit_scrapling_eval --max-pages 3 --limit 200 --rate-limit 3

:: Telegram public web -> data_telegram_enriched_eval
python -m polysignal social collect-telegram-public ^
  --config configs/social_enrichment_sources.example.yaml ^
  --output-root data_telegram_enriched_eval --max-pages 4 --limit 250 --rate-limit 2

:: Combine -> data_social_enriched_eval
python -m polysignal social merge-enriched ^
  --reddit-root data_reddit_scrapling_eval ^
  --telegram-root data_telegram_enriched_eval ^
  --output-root data_social_enriched_eval

:: Coverage report + artifact index + verdict
python -m polysignal social build-enriched-profile ^
  --input-root data_social_enriched_eval --profile real_social_enriched_main
```

Useful flags: `--only name1,name2` (subset of sources), `--skip-needs-validation`,
`--dry-run` (list planned sources), `--max-pages`, `--limit`, `--rate-limit`.

## Branch 12B large market-aware collection

Branch 12B adds a resumable collector without removing the smaller Branch 12A
commands. It uses the saved Polymarket timelines to build deterministic Reddit
search work, combines that work with source `/new/` pages and Telegram public
preview backfill, and counts only unique normalized post identities toward the
target. Repeated retrieval rows remain stored for provenance.

Start with a filesystem-free dry run:

```cmd
python -m polysignal social collect-large-enriched ^
  --markets-path data_social_enriched_eval/normalized/polymarket/markets.parquet ^
  --dry-run
```

Then run resumable stages:

```cmd
python -m polysignal social collect-large-enriched ^
  --target-total-records 10000 --target-reddit-records 7000 --target-telegram-records 3000
python -m polysignal social collect-large-enriched ^
  --target-total-records 25000 --target-reddit-records 17500 --target-telegram-records 7500
python -m polysignal social collect-large-enriched ^
  --target-total-records 50000 --target-reddit-records 35000 --target-telegram-records 15000
```

The default split is Reddit=35,000 and Telegram=15,000. If public Telegram
preview pages are exhausted below target, the missing volume transfers to
Reddit. Each successful page flushes raw snapshots and normalized Parquet/CSV
before atomically advancing checkpoints:

- `data_social_enriched_eval/state/collection_state.json`
- `data_social_enriched_eval/state/source_progress.json`
- `data_social_enriched_eval/state/query_progress.json`
- `data_social_enriched_eval/queries/market_queries.{parquet,csv}`
- `data_social_enriched_eval/reports/collection_summary.json`

Platform-filtered mirrors of the three checkpoint files are also written under
`data_reddit_scrapling_eval/state/` and `data_telegram_enriched_eval/state/`.
Resume state includes a fingerprint of the resolved source config: target and
page-cap increases are accepted, while source-config or root changes are
rejected.

Rows with missing, invalid, outside-window, or post-close timestamps are
retained for audit but excluded before segmentation, linking, stance, graph
projection, and SPI evidence use.

If Reddit returns `403` or `429`, Branch 12B opens a persisted six-hour Reddit
circuit breaker and stops issuing Reddit requests. A later resumable invocation
continues only after that cooldown expires.

## Outputs

| Dataset | Raw snapshots | Normalized |
| --- | --- | --- |
| Reddit | `data_reddit_scrapling_eval/raw/social/reddit_scrapling/` | `…/normalized/social/social_posts.{parquet,csv}` |
| Telegram | `data_telegram_enriched_eval/raw/social/telegram_public_web/` | `…/normalized/social/social_posts.{parquet,csv}` |
| Combined | — | `data_social_enriched_eval/normalized/social/social_posts.{parquet,csv}` |

Each run also writes a JSON collection summary under
`<root>/analytical/enrichment/`. Raw HTML page snapshots and per-post JSON snapshots
(with `raw_snapshot_path`, `source_name`, `source_url`, `collection_method`,
`extraction_status`) are stored under the raw zone for full provenance.

## How to inspect a dataset

```cmd
python -m polysignal social inspect-posts ^
  --posts-path data_social_enriched_eval/normalized/social/social_posts.parquet
```

## Downstream compatibility

The normalized rows are ordinary `SocialPostRecord` rows, so the existing pipeline
runs against the enriched root with `--output-dir data_social_enriched_eval`:
contract-post linking, stance classification, the Social Pressure Index, the Branch
09 backtest (`--benchmark-profile real_social_enriched_main`), and the Branch 10
dashboard view. Public-web posts are not market-targeted, so contract-linking
coverage is bounded by topical overlap with resolved markets; when coverage is too
sparse for a meaningful Branch 09 evaluation, no results are fabricated — the
datasets and the honest coverage report (`BRANCH_12_DATASET_ENRICHMENT.md`) are still
produced.

# High-signal Telegram dataset

This workflow focuses Telegram retrieval on resolved Polymarket markets with
auditable pre-close windows. The target is useful linked evidence, not raw
channel volume.

## Selection

`dataset select-high-signal-markets`:

- deduplicates by `market_id`;
- selects resolved markets only by default;
- requires `close_time`, then `resolved_at`;
- applies conservative question-led domain corrections when upstream metadata
  is stale;
- excludes obvious micro-markets and unclassified `other` markets by default;
- prefers searchability, liquidity, volume, price-history availability, and
  recent searchable windows without forcing priced markets into the subset.

Winning outcomes and resolved-label values are never retrieval inputs.
Meaningful outcome aliases may contribute queries; generic `Yes` and `No` do
not.

## Search plan

`social telegram build-search-plan` emits one stable work item per market,
channel, query, and non-overlapping pre-close band:

- `0-7d`
- `8-30d`
- `31-90d`

Bands are clamped to market creation time. Routing uses every YAML
`domain_tags` value. When no domain-specific channel exists, only configured
general-news channels are used as best effort; the planner never fans out to
all channels.

## Live search

```powershell
python -m polysignal dataset select-high-signal-markets --output-dir data_resolved_eval --target-markets 50 --resolved-only
python -m polysignal social telegram build-search-plan --output-dir data_resolved_eval --channels-config config/telegram_channels.yml --lookback-days 7,30,90 --queries-per-market 6
python -m polysignal social telegram search-markets --output-dir data_resolved_eval --config config/telegram_channels.yml --limit-per-query 100 --max-markets 50 --append --resume
python -m polysignal social segment-posts --posts-path data_resolved_eval/normalized/social/social_posts.parquet --output-path data_resolved_eval/normalized/social/social_post_segments.parquet
python -m polysignal social telegram coverage-report --output-dir data_resolved_eval --channels-config config/telegram_channels.yml
```

Telegram search uses server-side `search=` and `offset_date=` for the upper
bound and retains client-side lower/upper timestamp checks as the final
leakage guard.

## Outputs

| path | description |
| --- | --- |
| `data/analytical/dataset/high_signal_markets.parquet` | Curated resolved markets with scores and temporal anchors. |
| `data/analytical/dataset/telegram_search_plan.parquet` | Stable market-window work items. |
| `data/analytical/dataset/telegram_search_plan.json` | Human-readable plan dump. |
| `data/analytical/dataset/telegram_search_manifest.json` | Resumable per-work-item outcomes and errors. |
| `data/raw/social/telegram/<date>/<run-id>/` | Run-scoped raw targeted snapshots. |
| `data/normalized/social/social_posts.parquet` | Unified normalized social retrieval rows. |
| `data/normalized/social/social_post_segments.parquet` | Derived digest segments. |
| `data/analytical/dataset/telegram_coverage_report.json` | Targeted coverage report. |
| `data/analytical/dataset/high_signal_telegram_manifest.json` | Bundle manifest. |

## Provenance schema

Targeted `SocialPostRecord` rows preserve:

- `market_id_candidate`
- `retrieval_run_id`
- `retrieval_work_id`
- `retrieval_window_start`
- `retrieval_window_end`
- `retrieval_lookback_days`
- `retrieval_window_label`
- `retrieval_mode`
- `temporal_valid_candidate`

`market_id_candidate` means only that a query generated from this market
retrieved the post. Semantic relevance is decided later by Branch 05.

Exact repeated retrieval tuples are idempotent. Distinct query, market, or
window provenance remains auditable even when the platform post ID repeats.

## Measured recovery

The 2026-06-01 recovery produced:

| metric | value |
| --- | ---: |
| selected resolved markets | 49 |
| plan work items | 2,769 |
| targeted retrieval rows | 2,786 |
| unique targeted messages | 2,679 |
| targeted segments | 4,775 |
| markets with targeted rows | 9 |
| provenance-only links scored | 4,395 |
| temporally valid non-unrelated links | 1,197 across 7 markets |
| graph-accepted links | 13 across 5 markets |

The result is **CONDITIONALLY READY**: strict semantics work and accepted
evidence exists, but retrieval breadth remains below target.

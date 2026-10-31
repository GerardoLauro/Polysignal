# Telegram sources (Branch 07 social-evidence run)

Telegram is the primary empirical social source. Reddit and X/Twitter remain
deferred because usable credentials/API access are unavailable. The pipeline
stays platform-agnostic.

## Configuration

Sources live in [`config/telegram_channels.yml`](../config/telegram_channels.example.yml).
Supported keys are `handle`, `source_name`, `domain_tags`, `language`,
`enabled`, `limit`, `access_status`, and `notes`. Legacy `username`, `name`,
and `domain` aliases remain accepted. Python does not hardcode channels.

Every enabled handle is a candidate until Telethon validates and fetches it at
runtime. Private, inaccessible, fake, rate-limited, or empty targets are
recorded with a persisted reason.

## Legacy channel-wide crawl

The original Branch 07 crawl remains available as raw evidence:

| metric | value |
| --- | ---: |
| channels attempted / succeeded / skipped | 22 / 19 / 3 |
| raw messages fetched | 22,880 |
| deduplicated usable-text normalized rows | 20,534 |

Those rows are useful channel history, but they have no market-window
provenance and are not sufficient by themselves for resolved-market graph
evidence.

## Targeted pre-close recovery

The recovery uses resolved-market search-plan rows rather than channel-wide
history:

```powershell
python -m polysignal dataset select-high-signal-markets --output-dir data_resolved_eval --target-markets 50 --resolved-only
python -m polysignal social telegram build-search-plan --output-dir data_resolved_eval --channels-config config/telegram_channels.yml --lookback-days 7,30,90 --queries-per-market 6
python -m polysignal social telegram search-markets --output-dir data_resolved_eval --config config/telegram_channels.yml --limit-per-query 100 --max-markets 50 --append --resume
```

Measured result:

| metric | value |
| --- | ---: |
| selected resolved markets | 49 |
| persisted search-plan work items | 2,769 |
| targeted retrieval rows | 2,786 |
| unique targeted messages | 2,679 |
| markets with targeted rows | 9 |
| channels with targeted rows | 11 |
| completed / empty / error work items | 110 / 2,581 / 78 |

The `78` errors are persisted failures for `@Polymarket`: Telegram reported
that it is private or inaccessible to the authenticated account. No local
history fallback was reported as server-side search.

Raw snapshots:

- `data_resolved_eval/raw/social/telegram/2026-06-01/20260601T204537Z-204537Z-98a9a2/`
- `data_resolved_eval/raw/social/telegram/2026-06-01/20260601T204555Z-204555Z-3c21a0/`

Resume state:

- `data_resolved_eval/analytical/dataset/telegram_search_manifest.json`

## Provenance

Targeted wrappers and normalized rows preserve:

- retrieval run and stable work-item IDs;
- market candidate and query type;
- search-window start and end;
- lookback band and window label;
- `retrieval_mode=server_side_search`;
- candidate temporal-validity flag.

Exact retrieval tuples are idempotent. The same post remains auditable when
surfaced by a different query, market, or time window.

## Engagement and replies

Normalized rows preserve channel/source, author/sender, timestamp, views,
forwards, reactions, and reply counts when Telegram exposes them. The adapter
does not fetch discussion-group reply text and does not fabricate comments.

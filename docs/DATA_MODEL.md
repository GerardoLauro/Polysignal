# PolySignal — data model

The canonical domain model lives in
[`src/polysignal/domain/models.py`](../src/polysignal/domain/models.py)
as Pydantic v2 classes. This document is a stable, human-readable
description of the same model. When the code changes, this document
must be updated.

All timestamps are **timezone-aware UTC**. All identifiers are strings.

## Polymarket identifier mapping (Branch 03)

The Polymarket-side identifiers are now wired up and verified against
the live Gamma + CLOB APIs:

| Generic field             | Polymarket source                                                                                          |
| ------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `Market.market_id`        | `conditionId` (falls back to the numeric `id` only when `conditionId` is absent).                          |
| `Market.condition_id`     | named alias of the above; always set when known.                                                           |
| `Market.event_id`         | `events[0].id` (or `eventId` if exposed flat). May be `None`.                                              |
| `Market.slug`             | URL slug from Gamma.                                                                                       |
| `Outcome.market_id`       | parent `Market.market_id`, repeated on flat outcome rows for joins.                                        |
| `Outcome.outcome_id`      | per-outcome `clobTokenIds[i]` — also the join key for the CLOB `/prices-history` endpoint.                 |
| `Outcome.current_price`   | per-outcome `outcomePrices[i]`, clamped to `[0, 1]`.                                                       |
| `PricePoint.market_id` / `outcome_id` | pair of the above. Timestamps are unix seconds → timezone-aware UTC.                           |

Rules followed:

* **Binary scope.** Branch 03 only normalizes binary YES/NO markets.
  See [`polysignal.ingestion.polymarket.binary_filter`](../src/polysignal/ingestion/polymarket/binary_filter.py)
  and the "Polymarket ingestion" section of the README for the
  rationale and the skip reasons.
* **Named optional fields.** The richer Polymarket-specific data
  (`event_id`, `is_closed`, `volume`, `liquidity`,
  `Outcome.current_price`) lives in dedicated optional fields on the
  domain models — none of the existing identifier fields were
  overloaded.
* **Resolution mapping.** Conservative: `closed=true` plus
  `outcomePrices[YES] >= 0.99` ⇒ `ResolutionResult.YES`; the symmetric
  rule applies for NO. Explicit UMA dispute / invalid statuses map to
  `INVALID`. Anything ambiguous is left as `None` (with `is_closed`
  still set correctly).

## Core entities

### `Market`

A Polymarket contract.

| Field           | Type                  | Notes                                                            |
| --------------- | --------------------- | ---------------------------------------------------------------- |
| `market_id`     | `str`                 | Polymarket `conditionId` (see [identifier mapping](#polymarket-identifier-mapping-branch-03)). |
| `slug`          | `str`                 | URL-safe slug.                                                   |
| `question`      | `str`                 | Human-readable question.                                         |
| `description`   | `str \| None`         | Long description, if any.                                        |
| `domain`        | `MarketDomain`        | Coarse topical category (politics, sports, crypto, ...).         |
| `created_at`    | `datetime`            | Market creation time (UTC).                                      |
| `close_time`    | `datetime \| None`    | Time at which trading closes.                                    |
| `resolved_at`   | `datetime \| None`    | Resolution time, if resolved.                                    |
| `is_resolved`   | `bool`                | True once Polymarket has settled the outcome.                    |
| `is_closed`     | `bool`                | True once trading has closed (independent of resolution).        |
| `resolution`    | `ResolutionResult \| None` | `YES`, `NO`, `INVALID`, or `MULTI_OUTCOME`.                |
| `outcomes`      | `list[Outcome]`       | At least 2 entries (for binary markets: exactly 2).              |
| `tags`          | `list[str]`           | Free-form tags from Polymarket.                                  |
| `source_url`    | `str \| None`         | Canonical Polymarket URL.                                        |
| `event_id`      | `str \| None`         | First Polymarket event the market belongs to, if any.            |
| `condition_id`  | `str \| None`         | Mirrors `market_id` when sourced from Polymarket.                |
| `volume`        | `float \| None`       | Cumulative trading volume (USDC).                                |
| `liquidity`     | `float \| None`       | Order-book liquidity (USDC).                                     |

### `Outcome`

One possible outcome of a market.

| Field           | Type            | Notes                                                          |
| --------------- | --------------- | -------------------------------------------------------------- |
| `market_id`     | `str \| None`   | Parent market id. Populated for Polymarket-normalized rows.    |
| `outcome_id`    | `str`           | Per-outcome Polymarket CLOB token id (the join key for the CLOB `prices-history` endpoint). |
| `label`         | `str`           | Human-readable label (e.g. "Yes", "No"). Branch 03 only emits YES/NO. |
| `index`         | `int`           | Index inside `Market.outcomes` (YES = 0, NO = 1).              |
| `current_price` | `float \| None` | Last Gamma `outcomePrices[i]`, in `[0, 1]`.                    |

### `PricePoint`

A single price observation for a (market, outcome) pair.

| Field         | Type        | Notes                                            |
| ------------- | ----------- | ------------------------------------------------ |
| `market_id`   | `str`       | Same identifier as `Market.market_id` (see above).|
| `outcome_id`  | `str`       | Same identifier as `Outcome.outcome_id` (see above).|
| `timestamp`   | `datetime`  | UTC.                                             |
| `price`       | `float`     | In `[0, 1]`, interpretable as implied probability. |
| `volume`      | `float \| None` | Cumulative or incremental volume if available. |

### `MarketQuery` (Branch 04)

A search query generated from a market for social ingestion. One
`Market` typically expands into 3–6 queries.

| Field            | Type           | Notes                                                              |
| ---------------- | -------------- | ------------------------------------------------------------------ |
| `market_id`      | `str`          | Parent `Market.market_id`.                                         |
| `query`          | `str`          | The literal search string sent to the social platform.             |
| `query_type`     | `str`          | One of `core`, `entity`, `constraint`, `constraint_entity`, `anchor`, `slug`, `domain`, `tag`, `polymarket`, `description_anchor`. |
| `rank`           | `int`          | Stable 0-based rank in the per-market family.                      |
| `source_fields`  | `list[str]`    | Which market fields contributed (e.g. `question`, `tags`, `domain`). |
| `generated_at`   | `datetime`     | UTC.                                                               |

### `SocialPostRecord` (Branch 04)

The denormalized, parquet-friendly view of a collected social post.
Query-driven rows represent one `(post, query, market_candidate)` tuple.
Direct Telegram channel rows may omit query and market provenance.

| Field                  | Type           | Notes                                                              |
| ---------------------- | -------------- | ------------------------------------------------------------------ |
| `post_id`              | `str`          | Unique per `platform`.                                             |
| `platform`             | `Platform`     | `REDDIT`, `TWITTER`, `TELEGRAM`.                                   |
| `source_id`            | `str`          | Platform-native id (currently equals `post_id`).                   |
| `url`                  | `str \| None`  |                                                                    |
| `author_id`            | `str \| None`  |                                                                    |
| `author_name`          | `str \| None`  | Display handle.                                                    |
| `created_at`           | `datetime`     | UTC.                                                               |
| `text`                 | `str`          | Title + body (Reddit) or the message text.                         |
| `title`                | `str \| None`  | Reddit submission title; absent for platforms without titles.      |
| `subreddit_or_channel` | `str \| None`  | Subreddit (Reddit) or channel name (Telegram).                     |
| `query`                | `str \| None`  | Search query, when one surfaced this post.                         |
| `query_type`           | `str \| None`  | Retrieval provenance category, when available.                    |
| `market_id_candidate`  | `str \| None`  | Market that generated `query`. **Not** a relevance claim — see below. |
| `engagement_score`     | `float`        | Combined likes/upvotes/replies/views — single number for plots.    |
| `upvotes`              | `int \| None`  | Platform upvotes / score where applicable.                         |
| `comments_count`       | `int \| None`  |                                                                    |
| `shares_count`         | `int \| None`  |                                                                    |
| `views_count`          | `int \| None`  | Telegram views, when available.                                    |
| `reactions_count`      | `int \| None`  | Telegram reaction total, when available.                           |
| `language`             | `str \| None`  | ISO 639-1 if known.                                                |
| `raw_path`             | `str \| None`  | Absolute path to the per-post raw JSON payload.                    |
| `collected_at`         | `datetime`     | UTC. When this post was retrieved from the platform.               |
| `is_duplicate`         | `bool`         | `True` on rows after the first occurrence of `(platform, post_id)`. |
| `text_length`          | `int`          | `len(text)` precomputed for cheap filtering.                       |

**Semantic rule for `market_id_candidate`:** this field says only that
the post was *retrieved using a query generated from this market*. It
is **not** an assertion that the post is semantically relevant to the
market. True relevance is computed by Branch 05 and stored in
`data/normalized/social/linked_posts.parquet` or an equivalent
documented normalized output.

Targeted Telegram recovery rows also preserve `retrieval_run_id`,
`retrieval_work_id`, `retrieval_window_start`, `retrieval_window_end`,
`retrieval_lookback_days`, `retrieval_window_label`, `retrieval_mode`, and
`temporal_valid_candidate`. Exact retrieval tuples are idempotent, while the
same platform post remains auditable when surfaced by a different market,
query, or time window.

Direct Telegram channel rows may leave `market_id_candidate=None`. Branch 05
then uses bounded deterministic prefiltering by default. Full cross-scoring is
available only through explicit `--candidate-mode full-scan` debugging.

**Platform rule:** `social_posts.parquet` is platform-agnostic. Reddit,
Telegram, and X/Twitter rows use the same schema, and Branch 05 must not
depend on source-specific logic. The current enum values are `reddit`,
`telegram`, and `twitter` (`twitter` is the serialized value for
X/Twitter). Telegram public-channel ingestion is implemented through the
optional Telethon adapter. It fetches explicit configured targets only and
does not collect private chats.

`social_posts.parquet` is written only after live ingestion produces
real posts. Dry-run writes `social_queries.parquet` but never creates
fake social-post rows.

### `SocialPost`

A piece of content collected from a social platform.

| Field            | Type             | Notes                                                          |
| ---------------- | ---------------- | -------------------------------------------------------------- |
| `post_id`        | `str`            | Unique per `platform`.                                         |
| `platform`       | `Platform`       | `REDDIT`, `TWITTER`, `TELEGRAM`.                               |
| `author_id`      | `str \| None`    |                                                                |
| `author_handle`  | `str \| None`    |                                                                |
| `created_at`     | `datetime`       | UTC.                                                           |
| `text`           | `str`            | Raw text.                                                      |
| `text_normalized`| `str \| None`    | Output of `polysignal.nlp.normalize`.                          |
| `language`       | `str \| None`    | ISO 639-1.                                                     |
| `url`            | `str \| None`    |                                                                |
| `parent_post_id` | `str \| None`    | For threaded replies / comments.                               |
| `engagement`     | `Engagement`     | Likes/upvotes/replies/views.                                   |
| `entities`       | `list[Entity]`   | Output of NER (may be empty).                                  |
| `extra`          | `dict[str, Any]` | Platform-specific raw fields.                                  |

### `Engagement`

| Field      | Type        |
| ---------- | ----------- |
| `likes`    | `int \| None` |
| `reposts`  | `int \| None` |
| `replies`  | `int \| None` |
| `views`    | `int \| None` |
| `score`    | `int \| None` |

### `Entity`

A named entity mentioned in a post.

| Field    | Type         | Notes                                  |
| -------- | ------------ | -------------------------------------- |
| `text`   | `str`        | Surface form in the post.              |
| `label`  | `EntityType` | `PERSON`, `ORG`, `LOC`, `EVENT`, ...   |
| `start`  | `int`        | Character offset in the normalized text.|
| `end`    | `int`        |                                        |

### `Link`

A lightweight legacy link between a `SocialPost` and a `Market`.
Branch 05 semantic-relevance output uses `LinkedPost` below. Final
outcome-aware stance remains separate.

| Field            | Type               | Notes                                                                   |
| ---------------- | ------------------ | ----------------------------------------------------------------------- |
| `post_id`        | `str`              |                                                                         |
| `platform`       | `Platform`         |                                                                         |
| `market_id`      | `str`              |                                                                         |
| `method`         | `LinkMethod`       | `HEURISTIC`, `LLM`, `MANUAL`, `HYBRID`, `DETERMINISTIC_BASELINE`.       |
| `confidence`     | `float`            | In `[0, 1]`.                                                            |
| `created_at`     | `datetime`         | When the link was inferred.                                             |
| `evidence`       | `list[str]`        | Tokens / phrases / keywords that justified the link.                    |

### `LinkedPost` (Branch 05)

One scored `(market, post)` pair written to
`data/normalized/social/linked_posts.parquet`. Branch 05A populates the
deterministic relevance fields. Optional Branch 05B embeddings populate
the additive embedding diagnostics without changing offline behavior.
Optional Branch 05C LLM verification adds audit and final-decision fields;
the original 05A/05B fields are never overwritten.

| Field                   | Type                 | Notes                                                        |
| ----------------------- | -------------------- | ------------------------------------------------------------ |
| `link_id`               | `str`                | Stable hash of market, post, and retrieval query.             |
| `market_id`             | `str`                | Candidate market being scored.                               |
| `post_id`               | `str`                | Social post id.                                              |
| `platform`              | `Platform`           | Source platform.                                             |
| `query`                 | `str \| None`        | Branch 04 retrieval query, if known.                          |
| `query_type`            | `str \| None`        | Branch 04 query family, if known.                             |
| `relevance_score`       | `float`              | Deterministic 05A score in `[0, 1]`.                          |
| `relevance_label`       | `RelevanceLabel`     | `relevant`, `weakly_relevant`, `ambiguous`, or `unrelated`.   |
| `confidence`            | `float`              | Confidence in `[0, 1]`, separate from relevance score.        |
| `method`                | `LinkMethod`         | `DETERMINISTIC_BASELINE` for 05A/05B output.                  |
| `explanation`           | `str`                | Short auditable rationale.                                   |
| `matched_entities`      | `list[str]`          | Matched entity terms.                                        |
| `matched_terms`         | `list[str]`          | Matched contract-specific terms.                             |
| `constraint_match`      | `ConstraintMatch`    | `full`, `partial`, `missing`, `mismatch`, or `unknown`.       |
| `created_at`            | `datetime`           | UTC scoring timestamp.                                       |
| `post_created_at`       | `datetime \| None`   | Original post timestamp.                                    |
| `market_question`       | `str`                | Denormalized market question.                                |
| `post_text_preview`     | `str`                | Short denormalized preview.                                  |
| `raw_post_path`         | `str \| None`        | Branch 04 raw snapshot path, if available.                   |
| `signal_count`          | `int`                | Number of matched contract-specific categories.              |
| `anchor_score`          | `float`              | Additive 05E meaningful-anchor diagnostic.                    |
| `constraint_score`      | `float`              | Additive 05E gated-constraint diagnostic.                     |
| `domain_mismatch`       | `bool`               | Explicit post-domain evidence conflicts with the market.      |
| `rejection_reason`      | `str \| None`        | Auditable hard-rejection reason, when applicable.             |
| `semantic_similarity`  | `float \| None`      | Optional 05B cosine similarity.                              |
| `embedding_model`       | `str \| None`        | Optional 05B model id.                                       |
| `embedding_enabled`     | `bool`               | Whether the optional 05B overlay ran.                        |
| `llm_selected`          | `bool`               | Whether 05C selected the row for verification.               |
| `llm_judged`            | `bool`               | Whether a valid cached or fresh 05C judgment exists.         |
| `llm_cache_hit`         | `bool`               | Whether 05C reused a cached judgment.                        |
| `llm_invalid_output`    | `bool`               | Whether invalid LLM output triggered safe fallback.          |
| `llm_applied`           | `bool`               | Whether the conservative 05C merge affected final fields.    |
| `llm_provider`          | `str \| None`        | Optional provider audit id.                                  |
| `llm_model`             | `str \| None`        | Optional model audit id.                                     |
| `llm_relevance_label`   | `RelevanceLabel \| None` | Validated LLM relevance-only label.                       |
| `llm_confidence`        | `float \| None`      | Validated and clamped LLM confidence.                        |
| `llm_constraint_match`  | `ConstraintMatch \| None` | Validated LLM constraint match.                          |
| `llm_reason`            | `str \| None`        | Short validated LLM rationale.                               |
| `final_relevance_label` | `RelevanceLabel \| None` | Conservative merged relevance label.                     |
| `final_relevance_score` | `float \| None`      | Final score; preserves deterministic score today.            |
| `final_confidence`      | `float \| None`      | Conservative merged confidence.                              |
| `final_constraint_match`| `ConstraintMatch \| None` | Conservative merged constraint match.                    |
| `final_method`          | `LinkMethod \| None` | `DETERMINISTIC_BASELINE` or `HYBRID`.                        |

### `Stance`

Outcome-aware stance for one post on one market.

| Field            | Type               | Notes                                                                   |
| ---------------- | ------------------ | ----------------------------------------------------------------------- |
| `post_id`        | `str`              |                                                                         |
| `platform`       | `Platform`         |                                                                         |
| `market_id`      | `str`              |                                                                         |
| `outcome_id`     | `str \| None`      | If the stance points at a specific outcome.                             |
| `label`          | `StanceLabel`      | `SUPPORTS_YES`, `SUPPORTS_NO`, `NEUTRAL`, `AMBIGUOUS`, `UNRELATED`, `INSUFFICIENT_CONTEXT`. |
| `confidence`     | `float`            | In `[0, 1]`.                                                            |
| `method`         | `StanceMethod`     | `HEURISTIC`, `LLM`, `HYBRID`.                                           |
| `rationale`      | `str \| None`      | Short text explanation (mostly populated by LLM-assisted method).       |

### `SocialPressureRecord`

The Social Pressure Index computed for a (market, outcome, time window).

| Field            | Type             | Notes                                                          |
| ---------------- | ---------------- | -------------------------------------------------------------- |
| `market_id`      | `str`            |                                                                |
| `outcome_id`     | `str`            |                                                                |
| `window_start`   | `datetime`       |                                                                |
| `window_end`     | `datetime`       |                                                                |
| `index_value`    | `float`          | In `[0, 1]`.                                                   |
| `n_posts`        | `int`            | Number of posts contributing.                                  |
| `drivers`        | `list[Driver]`   | Decomposition of the index into named drivers.                 |

### `Driver`

| Field         | Type    | Notes                                            |
| ------------- | ------- | ------------------------------------------------ |
| `name`        | `str`   | E.g. `"stance"`, `"engagement"`, `"recency"`.    |
| `contribution`| `float` | Signed contribution to the index.                |
| `weight`      | `float` | Weight applied to the underlying signal.         |

### `ProbabilityEstimate`

The output of the forecasting module for a (market, outcome, timestamp).

| Field                  | Type                | Notes                                                     |
| ---------------------- | ------------------- | --------------------------------------------------------- |
| `market_id`            | `str`               |                                                           |
| `outcome_id`           | `str`               |                                                           |
| `timestamp`            | `datetime`          |                                                           |
| `market_probability`   | `float`             | From `PricePoint.price` at the corresponding time.        |
| `social_probability`   | `float \| None`     | From the Social Pressure Index.                           |
| `adjusted_probability` | `float`             | Final social-adjusted probability.                        |
| `confidence`           | `ConfidenceBand`    | `LOW`, `MEDIUM`, `HIGH`.                                  |
| `delta`                | `float`             | `adjusted - market`.                                      |
| `explanation`          | `list[str]`         | Human-readable drivers / reasons.                         |
| `method`               | `ForecastMethod`    | `BASELINE_BLEND`, `MODEL_X`, ...                          |

## Enumerations

These live in `polysignal.domain.enums`:

- `Platform`: `REDDIT`, `TWITTER`, `TELEGRAM`, `OTHER`.
- `MarketDomain`: `POLITICS`, `SPORTS`, `CRYPTO`, `ECONOMY`, `CULTURE`,
  `SCIENCE`, `OTHER`.
- `ResolutionResult`: `YES`, `NO`, `INVALID`, `MULTI_OUTCOME`.
- `EntityType`: `PERSON`, `ORG`, `LOC`, `EVENT`, `WORK_OF_ART`,
  `PRODUCT`, `MISC`.
- `LinkMethod`: `HEURISTIC`, `LLM`, `MANUAL`, `HYBRID`,
  `DETERMINISTIC_BASELINE`.
- `RelevanceLabel`: `RELEVANT`, `WEAKLY_RELEVANT`, `AMBIGUOUS`,
  `UNRELATED`.
- `ConstraintMatch`: `FULL`, `PARTIAL`, `MISSING`, `MISMATCH`, `UNKNOWN`.
- `StanceLabel`: `SUPPORTS_YES`, `SUPPORTS_NO`, `NEUTRAL`, `AMBIGUOUS`,
  `UNRELATED`, `INSUFFICIENT_CONTEXT`.
- `StanceMethod`: `HEURISTIC`, `LLM`, `HYBRID`.
- `ConfidenceBand`: `LOW`, `MEDIUM`, `HIGH`.
- `ForecastMethod`: `BASELINE_BLEND`, `MARKET_ONLY`, `SOCIAL_ONLY`,
  `MODEL_LEARNED`.

## Storage layout

```
data/
  raw/
    polymarket/events/YYYY-MM-DD/...
    polymarket/markets/YYYY-MM-DD/...
    polymarket/prices/YYYY-MM-DD/...
    social/reddit/YYYY-MM-DD/<run-dir>/<market>__<post>__<query_hash>.json
    social/twitter/...   # honored layout; future X/Twitter adapter stub
    social/telegram/...  # honored layout; first-class source deferred to 04B
  normalized/
    polymarket/
      markets.parquet         # one row per binary YES/NO Market
      outcomes.parquet        # one row per Outcome, including parent market_id
      price_points.parquet    # one row per (market, outcome, timestamp)
    social/
      social_posts.parquet    # unified social rows; provenance optional
      social_queries.parquet  # one row per MarketQuery
      social_post_segments.parquet # branch 05G segment-level evidence
      linked_posts.parquet    # branch 05 semantic relevance output
      stance_labels.parquet   # branch 06 outcome-aware stance output
    posts.parquet             # legacy path (reserved for branch 05+ outputs)
    links.parquet             # legacy/scaffold path
    stances.parquet
    entities.parquet
  analytical/
    linking/evaluation_report.json
    evaluation/resolved_market_labels.parquet  # branch 07 resolved-only labels (eval targets ONLY)
    review/top_linked_social_evidence.{csv,parquet} # branch 07 human review pack
    social_pressure.parquet
    probability_estimates.parquet
  cache/
    llm/<sha256>.json
    embeddings/<model>.json
    feature/<name>.parquet
    social/<future>           # reserved
  graph_exports/
    nodes/*.csv
    rels/*.csv
```

**Current implementation:** each entity is persisted as a single flat
Parquet file (`markets.parquet`, `prices.parquet`, ...) via
[`polysignal.storage.parquet_store`](../src/polysignal/storage/parquet_store.py).
There is no partitioned writer yet.

**Timestamp encoding.** All timezone-aware UTC `datetime` fields
(`Market.created_at`, `Market.close_time`, `Market.resolved_at`,
`PricePoint.timestamp`, ...) are serialized to Parquet as **ISO 8601
UTC strings** via Pydantic's `model_dump(mode="json")`. They round-trip
back to timezone-aware `datetime` objects through `dataframe_to_models`,
but downstream analytics that want a native pandas `datetime64[ns, UTC]`
column must parse the string explicitly:

```python
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
```

## Graph Loader v1 projection

The graph loader consumes canonical Parquet outputs without changing their
source-of-truth role. Required inputs are `markets.parquet`,
`outcomes.parquet`, `social_posts.parquet`, and `linked_posts.parquet`.
`social_post_segments.parquet` and `stance_labels.parquet` are optional.

Linked rows may identify segment-level evidence. When
`linked_posts.post_id` matches `social_post_segments.segment_id`, the loader
attaches `:LINKED_TO` to the segment's parent `:Post` and keeps segment
provenance on the relationship. No `:Segment` node is created.

Loader v1 keeps `price_points.parquet` analytical. It also skips entities,
topics, narratives, and Social Pressure Index data until real upstream
contracts exist.

A future change can switch to a native Arrow `timestamp[us, tz=UTC]`
column without changing the public domain model — see the docstring of
[`polysignal.storage.parquet_store`](../src/polysignal/storage/parquet_store.py).

**Planned (next milestones):** a Hive-style partitioned layout
(`platform=...`, `domain=...`, `date=YYYY-MM-DD`) for the normalized
zone, written via `pyarrow.dataset.write_dataset`. The exact partition
keys will live in `polysignal.storage.paths` so they can be referenced
everywhere. Until that change lands, do not assume partition pruning
in queries.

## Resolved-market evaluation labels (Branch 07)

`data/analytical/evaluation/resolved_market_labels.parquet` is the
canonical **resolved-only** view, built by
[`polysignal.evaluation.resolved_labels`](../src/polysignal/evaluation/resolved_labels.py)
from the normalized markets/outcomes via
`polysignal polymarket export-resolved-labels`. One row per resolved
market: `market_id`, `condition_id`, `question`, `domain`, `created_at`,
`close_time`, `resolved_at`, `is_resolved`, `is_closed`, `resolution`,
`winning_outcome_label/id/index`, `binary_yes_no`, `n_outcomes`, and
`source_fields` (which upstream fields determined the result).

> **Evaluation targets only.** These labels must never enter social
> retrieval, query generation, linking, stance classification, the Social
> Pressure Index, feature engineering, or any prediction input. They exist
> so a later strictly-temporal evaluation can score forecasts against
> ground truth (see [EVALUATION.md](EVALUATION.md)). A market is only
> emitted when the upstream normalizer already marked it resolved — the
> builder never fabricates a winner.

The project currently focuses on already-resolved markets for evaluation;
active/open markets are deferred until after the full pipeline is complete.
The human review pack
`data/analytical/review/top_linked_social_evidence.{csv,parquet}`
(`polysignal linking export-review`) is for manual QC only and is never a
training or label-leakage path.
It includes source message ID, market cutoff, retrieval run/work IDs, exact
search window, lookback band, retrieval mode, candidate temporal-validity
flag, and `accepted_by_graph_policy`.

**Telegram-primary social evidence.** The current empirical run uses
Telegram as the primary social source (22 candidates attempted, 19
succeeded, 3 skipped with persisted reasons; 22,880 raw messages fetched and
20,534 deduplicated usable-text rows normalized; Reddit/X deferred). The
targeted Branch 07 recovery adds 2,786 pre-close retrieval rows across 9
markets with server-side Telegram search provenance. Sources are configured in
[`config/telegram_channels.yml`](../config/telegram_channels.example.yml) and ingested
with `polysignal social telegram ingest` for channel history and `polysignal
social telegram search-markets` for resolved-market windows. Both paths record
persisted provenance. See
[TELEGRAM_SOURCES.md](TELEGRAM_SOURCES.md). The graph read layer over the
resulting evidence is documented in [GRAPH_OPERATIONS.md](GRAPH_OPERATIONS.md).

**Resolved-market dataset root.** Already-resolved markets are collected
non-destructively into a separate data root (`data_resolved_eval/`) so the
active Branch 03–06 demo under `data/normalized/` is preserved. Collect
them with `polysignal polymarket ingest --discovery markets --market-status
closed --order liquidityNum --end-date-min <ISO> --output-dir
data_resolved_eval`: `--discovery markets` walks `/markets` directly (far
more binary YES/NO than `events` discovery), and `--end-date-min` favours
recent resolved markets that still expose CLOB price history. The social
evidence is then linked to that resolved root via `linking
score-candidates --markets-path data_resolved_eval/...`; resolution labels
never participate in linking or stance.

For historical graph evidence, `LINKED_TO` is accepted-only: `relevant`, or
`weakly_relevant` with confidence `>= 0.60`, and source-post timestamp at or
before the market `close_time` (falling back to `resolved_at`). Candidate rows
remain in Parquet for audit but are not automatically graph evidence.

## Raw source persistence

Every ingestion run that calls an external API **must persist a replayable
source artifact before normalization**. This rule applies to Polymarket,
Reddit, Telegram, and any future provider.

Design rules:

1. **Raw before normalized.** Persist the untouched API response when it
   is JSON-serializable. Polymarket stores exact paginated API chunks.
   Reddit stores a documented JSON snapshot because PRAW `Submission`
   objects contain live-client references and cannot be dumped verbatim.
   If normalization fails, the source artifact remains available for
   debugging and re-processing.
2. **Run-scoped directories.** Each ingestion run gets its own
   subdirectory under the appropriate raw zone, named after the run
   start time in UTC and the source:

   ```
   data/raw/polymarket/markets/<YYYY-MM-DD>/<YYYYMMDDTHHMMSSZ>-<run_id>/
       page-0001.json
       page-0002.json
       ...
       _manifest.json   # endpoint, params, page count, started_at, finished_at
   data/raw/polymarket/events/<YYYY-MM-DD>/<YYYYMMDDTHHMMSSZ>-<run_id>/
       page-0001.json
       page-0002.json
       ...
       _manifest.json
   data/raw/polymarket/prices/<YYYY-MM-DD>/<YYYYMMDDTHHMMSSZ>-<run_id>/
       <market_id>__<token_id>.json
   data/raw/social/reddit/<YYYY-MM-DD>/<YYYYMMDDTHHMMSSZ>-<run_id>/
       <market_id>__<post_id>__<query_hash>.json
       _manifest.json
   ```

3. **Append-only run directories.** Re-running the same fetch produces
   a new run directory. Within one social run, an exact repeated
   `(market_id, post_id, query_hash)` retrieval is an idempotent rewrite
   of the same snapshot path. This keeps ingestion runs reproducible
   while avoiding duplicate files for an identical retrieval event.
4. **Self-describing.** Each run directory contains a `_manifest.json`.
   Polymarket manifests record endpoint, request parameters, page/object
   counts, start/finish timestamps (UTC), and the upstream `User-Agent`.
   Social manifests record platform, issued queries, object count,
   start/finish timestamps, and run metadata.
5. **Normalized Parquet is derived.** When social raw snapshots are enabled
   (the default), each collected social-post record traces back to a raw
   file through its `raw_path`. Operators may explicitly disable snapshot
   writes with `--no-save-raw`, in which case `raw_path` is `None`.
6. **Format.** Raw files are stored as **JSON** (one file per page or
   per object) for human inspection, not JSONL — pagination is
   already encoded by file naming.

Helpers that already exist:

- [`polysignal.storage.paths.DataPaths.raw_polymarket_markets`](../src/polysignal/storage/paths.py)
- [`polysignal.storage.paths.DataPaths.raw_polymarket_events`](../src/polysignal/storage/paths.py)
- [`polysignal.storage.paths.DataPaths.raw_polymarket_prices`](../src/polysignal/storage/paths.py)
- [`polysignal.storage.paths.DataPaths.raw_social`](../src/polysignal/storage/paths.py)
- [`polysignal.storage.paths.DataPaths.raw_run_dir(source, run_id, when=None)`](../src/polysignal/storage/paths.py)
  — returns a run-scoped subdirectory under `data/raw/<source>/<YYYY-MM-DD>/<YYYYMMDDTHHMMSSZ>-<run_id>/`.

`raw_run_dir` is shared by the Polymarket and social pipelines. The
materialised raw files plus `_manifest.json` are written via
[`polysignal.ingestion.polymarket.raw_store.RawRunWriter`](../src/polysignal/ingestion/polymarket/raw_store.py)
or
[`polysignal.ingestion.social.raw_store.SocialRawWriter`](../src/polysignal/ingestion/social/raw_store.py).

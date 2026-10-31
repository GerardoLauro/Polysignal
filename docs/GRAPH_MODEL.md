# PolySignal - knowledge graph model

PolySignal uses Neo4j as a queryable projection of the canonical Parquet
datasets. Parquet remains the source of truth. The graph is intended for
relationship analysis, not as a replacement for analytical time-series
storage.

The authoritative schema statements live in
[`src/polysignal/graph/schema.py`](../src/polysignal/graph/schema.py). Batched
upsert queries live in
[`src/polysignal/graph/queries.py`](../src/polysignal/graph/queries.py).

## Loader v1 scope

Graph Loader v1 loads these nodes:

| Label | Key |
| --- | --- |
| `:Contract` | `market_id` |
| `:Domain` | `name` |
| `:Outcome` | `(market_id, outcome_id)` |
| `:Post` | `(platform, post_id)` |
| `:Author` | `(platform, author_id)` |
| `:Platform` | `name` |

It loads these relationships:

| Type | Direction | Evidence properties |
| --- | --- | --- |
| `:HAS_OUTCOME` | `(:Contract)->(:Outcome)` | - |
| `:BELONGS_TO` | `(:Contract)->(:Domain)` | - |
| `:POSTED_ON` | `(:Post)->(:Platform)` | - |
| `:WROTE` | `(:Author)->(:Post)` | - |
| `:LINKED_TO` | `(:Post)->(:Contract)` | `link_id`, provenance, relevance fields, `accepted`, `temporal_valid`, source-post timestamp, temporal anchor |
| `:SUPPORTS` | `(:Post)->(:Outcome)` | `stance_id`, confidence, method, rationale, source provenance, `created_at` |
| `:OPPOSES` | `(:Post)->(:Outcome)` | same as `:SUPPORTS` |

`:Entity`, `:Topic`, and future `:Narrative` nodes are deliberately deferred
until real extraction datasets exist. `PricePoint` rows remain in Parquet
because time-series analysis does not need tens of thousands of graph nodes.

## Inputs

Required Parquet datasets:

- `data/normalized/polymarket/markets.parquet`
- `data/normalized/polymarket/outcomes.parquet`
- `data/normalized/social/social_posts.parquet`
- `data/normalized/social/linked_posts.parquet`

Optional Parquet datasets:

- `data/normalized/social/social_post_segments.parquet`
- `data/normalized/social/stance_labels.parquet`

The planner reports missing optional datasets and skipped graph components.
An alternate stance file can be passed explicitly for smoke planning only.

## Accepted LINKED_TO policy

`LINKED_TO` means accepted historical evidence, not every scored candidate.
The loader includes `relevant` rows and includes `weakly_relevant` rows only
when confidence is at least `0.60`. It excludes `unrelated`, `ambiguous`,
low-confidence weak rows, missing endpoints, post-close evidence, and
historical rows without a close/resolution anchor. Each exclusion is counted
in the dry-run summary. Scored candidates remain available in
`linked_posts.parquet`; they are not projected as graph evidence.

Measured 2026-06-01 recovery dry-run: `4,395` provenance-only candidates
produce `13 LINKED_TO`, `7 SUPPORTS`, and `0 OPPOSES`. Skips are
`3,048 unrelated`, `150 ambiguous`, and `1,184 low-confidence weak` rows.
There are zero accepted post-close or missing-anchor links.

## Segment-aware links

Branch 05 linked rows may use a segment ID in `linked_posts.post_id`. Loader
v1 resolves each link as follows:

1. If the linked post ID matches `social_post_segments.segment_id`, use that
   segment's `parent_post_id`.
2. Otherwise use the linked post ID directly.
3. Attach `:LINKED_TO` to the parent `:Post` and preserve segment provenance
   on the relationship.

Loader v1 does not create `:Segment` nodes.

## Directional stance projection

Only `SUPPORTS_YES` and `SUPPORTS_NO` stance labels produce graph
relationships. `NEUTRAL`, `AMBIGUOUS`, `INSUFFICIENT_CONTEXT`, and `UNRELATED`
are reported as skipped rather than projected as directional edges.
Directional rows must also map to an accepted link and pass the same
source-post temporal-admissibility rule.

For current binary YES/NO contracts, directional rows target the normalized
Yes or No outcome. For future multi-outcome rows, the loader requires an
explicit outcome target; ambiguous mappings are skipped and reported.

## Constraints and indexes

The local infrastructure uses Neo4j Community. Schema initialization therefore
uses Community-compatible uniqueness constraints rather than Enterprise-only
node key constraints. Initialization is idempotent through `IF NOT EXISTS`.

```cypher
CREATE CONSTRAINT contract_key IF NOT EXISTS
FOR (c:Contract) REQUIRE c.market_id IS UNIQUE;

CREATE CONSTRAINT outcome_key IF NOT EXISTS
FOR (o:Outcome) REQUIRE (o.market_id, o.outcome_id) IS UNIQUE;
```

The planner validates required identifiers before live loading. Write queries
use ordered batched `UNWIND` payloads and `MERGE`, making repeated loads
idempotent.

## Commands

Preview schema initialization and graph loading without contacting Neo4j:

```bash
python -m polysignal graph init --dry-run
python -m polysignal graph load --dry-run
python -m polysignal graph inspect
```

Apply the schema, load data, and inspect live counts:

```bash
docker compose up -d neo4j
python -m polysignal graph init --no-dry-run
python -m polysignal graph load --no-dry-run
python -m polysignal graph inspect --live
```

## Future extensions

Entity mentions, topics, narratives, graph-backed dashboard exploration, and
bulk `neo4j-admin import` remain future work. Social Pressure Index,
forecasting, backtesting, and price-point graph nodes are outside Loader v1.

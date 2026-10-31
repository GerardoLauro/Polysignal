# Graph operations

How to plan, inspect, and query the PolySignal knowledge graph. Everything
here runs **offline** (no Neo4j, no Docker) in `--dry-run` mode; live
execution is optional and explicitly opt-in.

All commands target the resolved-market dataset with `--output-dir
data_resolved_eval` (see [DATA_MODEL.md](DATA_MODEL.md)).

## Schema (Branch 07, Neo4j Community-compatible)

`polysignal graph init --dry-run` plans 11 statements: composite `IS UNIQUE`
constraints + indexes for `Contract`, `Outcome`, `Post`, `Author`,
`Platform`, `Domain` (no `IS NODE KEY` — that is Enterprise-only). Nodes:
`Contract`, `Domain`, `Outcome`, `Post`, `Author`, `Platform`. Relationships:
`BELONGS_TO`, `HAS_OUTCOME`, `POSTED_ON`, `WROTE`, `LINKED_TO` (segment-aware),
`SUPPORTS`/`OPPOSES` (directional stance → `Outcome`). `Entity`, `Topic`,
`Narrative`, `Segment`, and `PricePoint` nodes are documented but deferred.

## Plan / inspect the load (no Neo4j)

```powershell
python -m polysignal graph init --dry-run
python -m polysignal graph load --dry-run --output-dir data_resolved_eval
python -m polysignal graph inspect --output-dir data_resolved_eval
```

The loader reads canonical Parquet, validates required columns, resolves
segment→parent posts for `LINKED_TO`, maps binary `supports_yes`/`supports_no`
to the YES/NO `Outcome`, and **skips** non-directional stance rows. `LINKED_TO`
is accepted evidence only: `relevant`, or `weakly_relevant` with confidence
`>= 0.60`, with a valid pre-close source timestamp for historical markets.
Rejected categories are reported explicitly. It never fabricates nodes/edges.

Current plan on `data_resolved_eval` (2026-06-01):

| node | count | | relationship | count |
| --- | ---: | --- | --- | ---: |
| Contract | 122 | | HAS_OUTCOME | 244 |
| Outcome | 244 | | WROTE | 20,534 |
| Post | 20,534 | | LINKED_TO | 0 |
| Author | 19 | | SUPPORTS | 0 |
| Platform | 1 | | OPPOSES | 0 |

Skipped candidate rows: `unrelated=38,535`, `ambiguous=1,042`,
`low_confidence=1,407`, `post_close=46`. The current strict stance parquet is
empty, so `SUPPORTS=0` and `OPPOSES=0`.

## Read queries (parameterized, dry-run prints Cypher)

```powershell
python -m polysignal graph queries list
python -m polysignal graph queries contract-coverage --dry-run --limit 50
python -m polysignal graph queries top-contracts --dry-run --limit 20
python -m polysignal graph queries author-activity --dry-run --limit 50
python -m polysignal graph queries platform-coverage --dry-run --limit 100
python -m polysignal graph queries contract-neighborhood --market-id <id> --dry-run
python -m polysignal graph queries stance-summary --dry-run
python -m polysignal graph queries health --dry-run
```

| query | answers |
| --- | --- |
| `contract-coverage` | per-contract linked posts, avg link confidence, author/channel breadth, stance counts |
| `top-contracts` | contracts ranked by linked-posts × avg confidence |
| `author-activity` | per author/source: posts, distinct linked contracts, avg confidence |
| `platform-coverage` | per platform/channel: messages, linked messages, covered contracts |
| `contract-neighborhood` | one contract's outcomes, linked posts, authors, channels, stance edges |
| `stance-summary` | SUPPORTS/OPPOSES counts per contract outcome (empty-safe) |
| `health` | orphan posts, contracts without outcomes/links, orphan outcomes, … |

Every builder lives in
[`polysignal.graph.queries`](../src/polysignal/graph/queries.py), returns a
`GraphQuery(name, description, cypher, params)`, is **read-only** (no
`MERGE`/`CREATE`/`DELETE`), and routes all user input through `$params`
(`$market_id`, `$limit`) — never string interpolation. They are unit-tested
in `tests/test_graph_queries.py` without a live database.

## Optional live execution (not required for offline verification)

```powershell
docker compose up -d neo4j
python -m polysignal graph init --no-dry-run
python -m polysignal graph load --no-dry-run --output-dir data_resolved_eval
python -m polysignal graph inspect --output-dir data_resolved_eval --live
python -m polysignal graph queries stance-summary --no-dry-run
```

Live Neo4j and Docker are **not** required for offline graph verification.
Branch 07 still cannot close until valid accepted historical evidence exists.

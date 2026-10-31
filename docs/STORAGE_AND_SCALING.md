# Storage & Scaling — design choices (Traccia_2 Task 1 / Big-Data axis)

This document records *why* PolySignal stores data the way it does, how the
layout maps to the analytics query patterns, and how the heavy work scales out.
It backs the "design choices" + "creativity" criteria of the assignment.

The corpus this is reasoned against (current run):

| Entity | Rows | On disk |
| --- | --- | --- |
| Polymarket contracts | 300 | `data/normalized/polymarket/markets.parquet` |
| Outcomes | 600 | `outcomes.parquet` |
| Price points | ~1.35M | `price_points.parquet` |
| Social posts | ~1.357M (Telegram ~99%, Reddit 8.3K, X 0) | `data/normalized/social/social_posts.parquet` |
| Post enrichment (NER/sentiment/lang/hashtags) | ~1.357M | `post_enrichment.parquet` |
| Links (post ↔ contract) | ~357K | `linked_posts.parquet` |
| Stance labels | ~30.7K | `stance_labels.parquet` |

## 1. Primary store: columnar Parquet

The canonical normalized tables and the §2 export bundle are **Apache Parquet**.

**Why columnar fits the workload.** Every analytics query in the project is a
*scan-and-aggregate over a few columns of many rows*: price/social time-series
overlays, per-domain / per-platform distributions, linking-coverage fractions,
SPI feature windows, backtest label joins. These touch 2–6 columns out of
30–40. A columnar format reads only the projected columns and skips the rest,
and its per-column encoding (dictionary + RLE for the low-cardinality
`platform`/`category`/`language` columns, delta for timestamps) shrinks the
1.357M-post table dramatically versus row-oriented CSV/JSON. Predicate
push-down + min/max page statistics let a reader skip whole row-groups that
fall outside a filter (e.g. one domain, one time window) without decoding them.

**Immutability + safety.** Derived tables (`post_enrichment`, `linked_posts`,
`stance_labels`) are written *beside* the raw corpus and joined by key, so the
multi-hundred-MB raw post table is never rewritten. This is the project's
data-safety convention (see the Telegram 0-byte-loss incident and the
strict-read / quarantine / backup guards) and it also keeps each pipeline stage
independently reproducible.

## 2. Partitioning (`polysignal dataset export --partition`)

The export writes **Hive-style partitioned** datasets via
`pyarrow.parquet.write_to_dataset`:

| Entity | Partition keys | Rationale |
| --- | --- | --- |
| `social_posts` | `platform` | Cross-platform comparison reads one platform at a time; X/Reddit slices stay tiny next to Telegram. |
| `links` | `platform` | Coverage + join queries are platform-scoped. |
| `contracts` | `category` | Per-domain (politics/finance/sports) analytics prune to one folder. |

Layout produced (one folder per partition value):

```
data/dataset_export_partitioned/
  social_posts/platform=telegram/*.parquet
  social_posts/platform=reddit/*.parquet
  links/platform=telegram/*.parquet
  contracts/category=politics/*.parquet
  contracts/category=finance/*.parquet
```

**Partition pruning** means a query like "all Reddit posts" or "finance
contracts" opens only the matching folder — the reader never touches the other
partitions. A natural third axis is **year-month of `published_at`**
(`platform=…/ym=2025-11/…`) for the time-windowed SPI/backtest scans; the export
helper accepts additional partition columns the same way, and the doc records it
as the scale-out layout once the corpus grows beyond single-node convenience.

## 3. Why not (only) X — the 5 V's mapped to candidate stores

| Store | Model | Best for | 5-V fit here | Verdict |
| --- | --- | --- | --- | --- |
| **Parquet on disk** (chosen) | Columnar files | Scan/aggregate analytics, ML feature builds | **Volume** (compresses 1.35M×40 cols), **Variety** (nested lists: hashtags, entities), **Value** (cheap column scans) | **Primary** analytical store |
| Relational (Postgres) | Row tables + SQL | Transactional point reads/writes, strict joins | Strong **Veracity** (constraints), weak on wide-column scans + nested arrays | Good for a serving layer; overkill offline |
| MongoDB | Document | Heterogeneous semi-structured docs | **Variety** (raw API JSON), flexible schema | Used conceptually for the *raw* snapshot tier; analytics still export to Parquet |
| Cassandra | Wide-column | High-**Velocity** append, time-series at scale | **Velocity**/**Volume** for a always-on firehose | Not needed: ingestion is batch, not a 24/7 stream |
| **Neo4j** (chosen, complementary) | Property graph | Multi-hop relationship queries | **Value** of *connections*: post→author→contract→outcome, stance edges | **Secondary** graph store for relationship analytics |

The deliberate choice is **polyglot persistence**: columnar Parquet as the
analytical backbone, a Neo4j property graph for the relationship questions that
are awkward in tabular form (author/contract co-activity, stance propagation,
coverage neighbourhoods), and a raw document tier for provenance snapshots.

## 4. Big-data processing: Spark "summarize + join", with a pandas fallback

The heavy job is per-market social aggregation over the full ~1.357M-post
corpus: deduplicate by `(platform, post_id)`, resolve each post's market via the
`linked_posts` join (the real corpus carries no inline `market_id`), then
summarize post counts, unique authors, platform breadth, and engagement.

Implemented in `src/polysignal/processing/social_aggregate.py` and wired to
**`polysignal process social-aggregate`**:

- **Spark path** (`local[*]`): activates when `JAVA_HOME` + PySpark are present.
  The dedup/market-resolution is a distributed `groupBy(market_id).agg(...)`
  after a `post_id → market_id` join — the canonical MapReduce **shuffle +
  reduce** pattern, partitioned across cores (`spark.sql.shuffle.partitions`).
- **pandas path**: the identical aggregate spec on a single node, always
  available so the package runs without a JVM. Both paths are asserted
  equivalent in `tests/test_processing_spark.py`.

**Scale-out plan.** The same job generalises from `local[*]` to a cluster
unchanged: the input Parquet is already partitioned (cheap parallel reads), the
aggregation is a partition-local map (count/sum/distinct per market) followed by
a reduce-side merge keyed on `market_id`, and the post↔contract resolution is a
broadcast-or-shuffle join (the contracts side is tiny: 300 rows, broadcastable;
the links side, ~357K, shuffles). MapReduce framing: **map** emits
`(market_id, {post, author, platform, engagement})`, **combine** pre-aggregates
per partition, **reduce** merges per market. Nothing in the job assumes a single
machine.

## 5. Neo4j graph — runbook (Task E2, reproducible)

The property graph is loaded end-to-end on a live instance (no Docker required
on this machine — a native Neo4j Community install under `.tools/` + Temurin
JDK 17 is used; an AuraDB Bolt URL works identically). Credentials live in
`.env` (`NEO4J_*`), never versioned.

```bash
# 0. One-time: set the DB password to match Neo4jSettings (default polysignal-dev)
JAVA_HOME=.tools/jdk-17.0.13+11 \
  .tools/neo4j-community-5.20.0/bin/neo4j-admin dbms set-initial-password polysignal-dev

# 1. Start Neo4j (console or service); Bolt on localhost:7687
JAVA_HOME=.tools/jdk-17.0.13+11 .tools/neo4j-community-5.20.0/bin/neo4j console

# 2. Apply schema (constraints + indexes)
polysignal graph init --no-dry-run

# 3. Load Parquet → graph (idempotent UNWIND/MERGE batches)
polysignal graph load --no-dry-run --batch-size 2000

# 4. Run the read queries against the live DB
polysignal graph queries contract-coverage --no-dry-run
polysignal graph queries author-activity   --no-dry-run
polysignal graph queries platform-coverage --no-dry-run
polysignal graph queries stance-summary    --no-dry-run

# 5. Capture the query results to data/graph_exports/ (reproducibility artifact)
python scripts/capture_graph_queries.py
```

**Loaded graph (current run):** 300 Contract · 600 Outcome · 5 Domain ·
2 Platform · 1,357,296 Post · 4,663 Author nodes; relationships
BELONGS_TO 300 · HAS_OUTCOME 600 · POSTED_ON 1,357,296 · WROTE 1,357,217 ·
LINKED_TO 32,149 · SUPPORTS 5,797 (4,116,225 rows in 2,066 batches). Query
captures (real rows) live in `data/graph_exports/query_*.json`; a tiny-fixture
load is asserted in `tests/test_smoke_graph.py`.

## 6. Summary of design choices

1. **Columnar Parquet** primary store — matches scan/aggregate analytics; nested
   list columns hold hashtags/entities without flattening.
2. **Hive partitioning** by `platform` / `category` (year-month ready) for
   partition pruning on the exact axes the dashboard/analytics filter on.
3. **Derived-table-beside-raw** immutability — the big corpus is never rewritten.
4. **PySpark summarize+join** for the 1.35M-post aggregation, with a verified
   pandas fallback — distributed MapReduce shape, single-node convenience.
5. **Neo4j property graph** for relationship analytics — polyglot persistence,
   loaded and queried live, in the runbook above.

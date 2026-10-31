# PolySignal — architecture

This document describes the **conceptual modules** of PolySignal, the
data flow between them, and the design principles that keep the system
clean, modular, and extensible.

## Design principles

1. **Separation of concerns.** Each module has a single, well-defined
   responsibility, a typed input contract, a typed output contract, and
   no knowledge of how upstream/downstream modules are implemented.
2. **Deterministic core, AI-assisted enhancements.** The pipeline must
   run end-to-end without any LLM call. Where LLMs add value they are
   wrapped behind a stable interface, their outputs are structured and
   cached, and the deterministic baseline always remains available.
3. **No invented data.** The system never fabricates social posts,
   markets, or stance labels to pad a demo. Empty datasets are visible.
4. **No invented credentials.** Real ingestion code is added only when
   the corresponding service is properly configured.
5. **Time discipline.** Every record carries an explicit, timezone-aware
   timestamp. Forecasting and backtesting code enforces a strict
   "no information from the future" rule.
6. **Reproducibility.** Configuration is centralized. Random seeds are
   exposed. LLM calls are cached by a deterministic key (prompt + model
   + version) so that the same input always yields the same output.

## Modules

```
config       → environment, paths, seeds, feature flags
domain       → Pydantic data classes shared by every module
storage      → Parquet IO, raw JSON IO, LLM/feature caches
ingestion    → Polymarket + social platforms (Reddit/X/Telegram)
linking      → contract ↔ post matching (heuristic + LLM-assisted)
nlp          → normalization, NER, outcome-aware stance detection
graph        → Neo4j schema, loaders, graph queries
features     → Social Pressure Index, graph features, time features
forecasting  → social-adjusted probability, explanation, baselines
analytics    → descriptive, temporal, correlation, cross-platform
evaluation   → metrics, calibration, leak-free backtests
scheduling   → fast/heavy/manual refresh policy
llm          → provider-agnostic client + structured-output helper
dashboard    → Streamlit application
cli          → Typer entry point that orchestrates the modules
```

### Boundaries

The arrow `A → B` means "A produces typed objects consumed by B".

```
ingestion.polymarket  → storage  → linking
ingestion.social      → storage  → linking
linking               → nlp      → features
nlp                   → graph    → features
features              → forecasting → evaluation
forecasting           → dashboard
evaluation            → dashboard
scheduling            ↔ dashboard, CLI, ingestion
llm                   ← linking, nlp        (optional, opt-in)
```

The dashboard depends only on the **outputs** of upstream modules; it
should never reach into raw API responses.

## Branch 04 social-ingestion boundary

Branch 04 is complete for the current milestone. It produces candidate
social retrieval evidence, not semantic links:

- Reddit is the primary live-ready source. It is credential-gated,
  supports an offline dry-run, never fabricates posts, and preserves
  query lineage in raw snapshots.
- Telegram public-channel ingestion is implemented with optional Telethon
  user sessions, explicit YAML-configured targets, raw snapshots, stable ids,
  and unified parquet normalization. It is the immediate social source while
  Reddit credentials are pending. Private chats are not collected.
- X/Twitter remains a future adapter stub and does not block Branch 05.

`data/normalized/social/social_queries.parquet` is a Branch 04 output.
`data/normalized/social/social_posts.parquet` is created only when live
ingestion returns real posts. Every social row is platform-agnostic.
`market_id_candidate` means only "retrieved using a query generated from
this market"; it is not a semantic-relevance decision. Direct Telegram
channel rows may omit it; Branch 05 cross-scores those rows against markets.

Branch 05A consumes those candidate rows and writes deterministic
semantic-relevance verdicts to
`data/normalized/social/linked_posts.parquet`. Entity overlap alone is
insufficient: the hardened scorer tracks contract-specific categories
and temporal/contextual constraint matches. Branch 05B adds an optional
cached embedding overlay that is disabled by default and never breaks
offline mode. Branch 05C adds an opt-in cached structured LLM verifier
for borderline relevance rows. It validates JSON, merges conservatively,
and falls back to the 05A/05B result on malformed output. Optional OpenAI
and Anthropic live clients are available only when explicitly configured.
Branch 06 adds outcome-aware deterministic stance plus an optional cached
structured LLM judge. The default path remains offline and writes
`data/normalized/social/stance_labels.parquet`.

## Data zones

The data directory is partitioned by maturity:

| Zone                  | Purpose                                                | Format          |
| --------------------- | ------------------------------------------------------ | --------------- |
| `data/raw/`           | API payloads; Reddit and Telegram snapshots with lineage | JSON / JSONL    |
| `data/normalized/`    | Validated, typed, deduplicated records                 | Parquet         |
| `data/analytical/`    | Feature tables ready for analytics and modeling        | Parquet         |
| `data/cache/llm/`     | LLM responses keyed by prompt+model+version            | JSON            |
| `data/cache/feature/` | Expensive intermediate features                        | Parquet / JSON  |
| `data/graph_exports/` | Offline graph dumps (CSV / JSONL) for backup           | CSV / JSONL     |

**Current implementation (v0.1.x):** the Parquet store
([`polysignal.storage.parquet_store`](../src/polysignal/storage/parquet_store.py))
persists each normalized entity to a **single flat Parquet file**
(`markets.parquet`, `social_queries.parquet`, `social_posts.parquet`,
...). This is fine at the
dataset sizes targeted by the first ingestion milestone and lets
analytics read the file directly with pandas.

**Planned (real ingestion milestone and beyond):** as the corpus
grows, the normalized zone will move to a **partitioned Parquet
layout** with directory keys such as `platform=...`, `domain=...`,
and `date=YYYY-MM-DD` (Hive-style), readable both by pandas/pyarrow
and by Spark. The write path will switch to
`pyarrow.dataset.write_dataset` behind the same `parquet_store`
interface so call sites do not change.

Until then, do not assume partitioned writes in code or downstream
queries.

## Refresh policy

See [`REFRESH_POLICY.md`](REFRESH_POLICY.md). In short, three modes:

- **fast** — cheap operations only (recompute Social Pressure Index from
  cached features, update UI views, refresh latest market prices);
- **heavy** — expensive operations (new social ingestion, LLM relevance
  checks, stance recomputation, NER, graph updates);
- **manual** — user-triggered, runs whatever the user picks.

The policy is a small deterministic state machine living in
`polysignal.scheduling.refresh`.

## Forecasting baseline

The first iteration of "social-adjusted probability" is a
**deterministic blend** between the market-implied probability `p_m`
and a social signal `p_s` derived from the Social Pressure Index:

```
p_final = (1 - w) * p_m + w * p_s
```

where `w ∈ [0, 1]` is a confidence-weighted blending factor. The blend
is intentionally simple and explainable — every input is logged so the
dashboard can show the contribution of each driver. The module exposes
the same signature that a future learned model can implement, so it can
be swapped in without changing callers.

See [`src/polysignal/forecasting/baseline.py`](../src/polysignal/forecasting/baseline.py)
and the accompanying `explain.py`.

## Evaluation

See [`EVALUATION.md`](EVALUATION.md). Branch 09 compares four baselines:
constant/base-rate, market-only, social-only, and market+social. The current
policy evaluates one final pre-close prediction per resolved binary market.
Learned predictions use deterministic out-of-fold stratified splits and never
include information available only after the prediction cutoff. Metrics are
Brier score, log loss, accuracy, AUC when both classes are present, Expected
Calibration Error, calibration bins, and the confusion matrix at threshold
`0.5`. Time-sliced 7d / 3d / 1d evaluation remains deferred until price and
social-feature history are broad enough to support it cleanly.

## Design guardrails

The decisions below are load-bearing — every contributor should keep
them in mind before adding new modules or dependencies. If a change
would violate one of them, that is a *design discussion*, not a
silent code change.

1. **Deterministic core first.** The end-to-end pipeline must run
   without any LLM. AI-assisted steps are *enhancements*, never hard
   dependencies.
2. **Optional AI-assisted modules.** Every LLM-touching component
   lives behind an interface defined in `polysignal.llm` and is
   guarded by the `POLYSIGNAL_LLM_PROVIDER` setting. `disabled` is the default.
3. **Structured & cached LLM output.** LLM calls *always* return a
   Pydantic-validated object and are cached by a deterministic key
   (`sha256(prompt + model + prompt_version)`). Free-text outputs are
   not allowed in the pipeline.
4. **Parquet for analytical data.** Normalized datasets live as
   Parquet files under `data/normalized/`. The current implementation
   writes one **flat Parquet file** per entity; a **partitioned
   Hive-style layout** (`platform=...`, `domain=...`, `date=...`) is
   the planned next step for real ingestion and analytical scaling,
   designed to remain pandas/pyarrow and **Spark-compatible** without
   changing the `parquet_store` interface. See "Data zones" above.
5. **Neo4j for graph representation.** The graph layer is canonical
   for relational analysis (centrality, narrative communities,
   author influence). All edges carry typed properties.
6. **JSON cache today, Redis tomorrow.** The current cache
   (`polysignal.storage.cache.JSONCache`) is a simple file-backed
   store. When the dashboard needs sub-100ms response times for
   shared sessions, the same interface will be implemented on top of
   **Redis** without changing call sites.
7. **Kafka is future work, not a current dependency.** Streaming
   ingestion (Reddit firehose, Telegram channels) will eventually
   move from periodic batches to an Apache Kafka / Redpanda topic
   feeding the existing ingestors. The interface in
   `polysignal.ingestion.social.base.SocialIngestor` does not
   preclude this evolution.
8. **Lambda-inspired fast / heavy / manual refresh.** The
   `RefreshPolicy` state machine separates cheap recomputations
   (Lambda's "speed layer" idea) from heavy ingestion + enrichment
   (the "batch layer"). The dashboard reads from the cheapest
   available layer at any moment.
9. **Single-machine first, scalable later.** The reference
   implementation runs on a developer laptop. Every storage and
   ingestion decision is chosen so that scaling out (Spark over the
   Parquet zone, Kafka in front of ingestion, Redis as hot cache,
   Neo4j cluster) is an *additive* change, not a rewrite.

## Extensibility

Every module exposes its work through a small abstract class or
Protocol. To replace a baseline with an advanced version:

1. Create a new class implementing the same interface.
2. Wire it in via configuration (`config.settings`), not by editing
   downstream callers.
3. Run the test suite. Tests pin the interface, not the implementation.

The same principle applies to ingestion backends, LLM providers, and
graph stores.

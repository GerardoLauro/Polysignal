# Polysignal

Social intelligence layer for Polymarket prediction markets. Progetto Big Data Engineering di Francesco Simone Pirozzi e Gerardo Lauro.

Questo repository contiene codice, test, dashboard, documentazione, report PDF/LaTeX e piccoli campioni CSV. **Il dataset completo e i risultati necessari alla dashboard restano fuori da GitHub.** Le statistiche nelle sezioni storiche descrivono il lavoro originale; non sono dati inclusi né risultati ricalcolati durante la preparazione del repository.

## Avvio rapido

Da questa cartella, con Python 3.12 (ambiente verificato):

```text
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.lock
.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
run_app.cmd
```

Su Linux/macOS usare `.venv/bin/python` e `bash run_app.sh`. L'installazione richiede Internet; le librerie JavaScript della dashboard sono incluse. Senza dati privati l'interfaccia si apre e indica come configurarli, senza presentare risultati inventati. I campioni CSV non sostituiscono gli artefatti completi della dashboard.

- [Guida italiana e materiali](LEGGIMI.md)
- [Configurazione della dashboard e dei dati esterni](docs/RUN_APP.md)
- [Ambiente riproducibile e controlli automatici](docs/REPRODUCIBILITY.md)
- [Report PDF](pdf_latex/main.pdf), [sorgenti LaTeX](pdf_latex/main.tex), [campioni](data/examples/DATA_DICTIONARY.md)
- [Licenza MIT del progetto](LICENSE), [librerie di terzi](THIRD_PARTY_NOTICES.md)

`python scripts/make_release.py` esporta per impostazione predefinita soltanto sorgenti, documenti e campioni. `.env`, sessioni, dataset ripristinati e output generati sono ignorati. Gli strumenti di raccolta sono opzionali e non vengono avviati durante installazione o test.

---

## Documentazione tecnica originale e storico

Le sezioni seguenti conservano la descrizione del progetto e delle sue fasi precedenti. I riferimenti a dataset e conteggi non implicano che tali dati siano contenuti nel clone. Per l'avvio corrente valgono le istruzioni sopra e le guide collegate.

# PolySignal

**A social intelligence layer for Polymarket prediction markets.**

PolySignal is a research-oriented system that studies the relationship between
**social media signals** (Reddit, X/Twitter, Telegram) and **Polymarket
prediction market outcomes**. It collects resolved contracts and their price
histories, links them to social discussion, enriches the textual data, builds
a knowledge graph of the domain, and exposes the result through an interactive
dashboard.

The scientific question is deliberately conservative:

> **Do social media signals provide incremental predictive value over
> market-implied probabilities in specific domains, platforms, and time
> windows?**

PolySignal is *not* an attempt to "beat Polymarket". It is an analytical layer
that quantifies social context and exposes it for inspection, comparison, and
controlled evaluation.

---

## Table of contents

1. [Status](#status)
2. [What this repository contains today](#what-this-repository-contains-today)
3. [High-level architecture](#high-level-architecture)
4. [Repository layout](#repository-layout)
5. [Installation](#installation)
6. [Local infrastructure](#local-infrastructure)
7. [Running tests](#running-tests)
8. [Launching the dashboard](#launching-the-dashboard)
9. [CLI](#cli)
10. [Configuration and secrets](#configuration-and-secrets)
11. [Next implementation step](#next-implementation-step)
12. [Academic requirements coverage](#academic-requirements-coverage)
13. [License](#license)

---

## Status

### Current dataset & pipeline (300-market build)

The canonical `data/` set now contains a **domain-balanced 300 resolved
contracts — 100 politics, 100 finance, 100 sports** — each with its
**full-lifetime** Polymarket price series (**1,353,114** price points,
median ~4,900 points/market). The full social→signal pipeline has been run
end-to-end on this set:

| Stage | Result |
| --- | --- |
| Markets (resolved, balanced) | 300 (politics/finance/sports = 100/100/100) |
| Price time series | 1,353,114 points, full lifetime, 300/300 markets |
| Social corpus | 1,368,041 posts (Telegram 1,348,965 + Reddit 16,254 + X/Twitter 2,822) |
| NLP enrichment | sentiment + NER + hashtags + language on **all 1.37M** posts (`post_enrichment.parquet`) |
| Contract↔post linking | 26,337 relevant + 412,875 weakly-relevant across the 300 markets (37,562 accepted as evidence over 147 markets) |
| Outcome-aware stance | 35,679 scored (4,605 YES / 1,964 NO / 26,231 neutral) |
| Topic modeling | LDA over linked posts (`topics.parquet`) |
| Backtest (real_optional, n=300) | market_only AUC 0.989; market+social AUC 0.975; social_only AUC 0.404 |
| Task-3 prediction (temporal CV, 4 model families) | market macro-F1 0.90–0.93; social-only ≈ chance (macro-F1 0.47–0.51); no lift from social |

**Honest coverage caveat:** social evidence is dense for politics/crypto
markets (the Telegram channels' topics) and sparse for sports; 104/300
markets have directional social evidence (147 carry at least one linked
post), but its direction does not track outcomes, so the social signal does
**not** add incremental value over the market price. The corpus is
Telegram-dominated (98.6%); Reddit (1.2%) and X/Twitter (0.2%) are small
secondary platforms whose scale is capped by their anti-automation defences.
See [`docs/NLP_ENRICHMENT.md`](docs/NLP_ENRICHMENT.md) and
`scripts/verify_dataset.py`.

---

The sections below describe the original branch-by-branch baseline
(**implemented through the Branch 09 quality gate**). The repository
defines:

- the project structure;
- the domain model (Pydantic);
- the module interfaces for every conceptual component;
- deterministic, testable utilities (text normalization, evaluation metrics,
  Social Pressure Index baseline, social-adjusted probability baseline,
  heuristic linker, refresh policy);
- the Neo4j graph schema;
- a React dashboard (`webapp/`, served by `run_app.cmd`);
- a Typer CLI skeleton;
- a docker-compose for local Neo4j;
- a tests folder with unit tests for the deterministic components.

**Polymarket** ingestion is real for binary YES/NO markets (Branch 03 —
see ["Polymarket ingestion"](#polymarket-ingestion) below).
**Branch 04 - Social Ingestion: Reddit + Telegram** supports real
collection. Telegram uses Telethon; **Reddit is collected via Scrapling
public-web only (old.reddit.com) — the Reddit OAuth API is not used.**
**X/Twitter** is collected as a small third platform via authenticated
browser automation (Playwright); its scale is capped by X's anti-automation defences.
**Branch 05 - Contract-Post Linking & Semantic Relevance** is complete through
05C: the hardened deterministic 05A baseline writes `linked_posts.parquet`,
the 05B embedding overlay is optional, and the 05C cached structured LLM
relevance judge is an opt-in verifier for borderline rows. Default behavior
remains offline. **Branch 06 - Outcome-Aware Stance** is complete through its
quality gate: the deterministic pipeline writes `stance_labels.parquet`, and
the optional structured LLM stance judge is cached and opt-in.
**Branch 07 - Knowledge Graph & Neo4j** is **CONDITIONALLY READY** after the
targeted Telegram recovery. The resolved evaluation set contains **110 real
resolved markets** with winners (`data/analytical/evaluation/resolved_market_labels.parquet`,
**evaluation targets only**) plus a thin price-history sample. The curated
resolved-only selector produced **49** high-signal markets and a persisted
**2,769**-row Telegram market-window plan. Server-side pre-close searches
added **2,786** auditable retrieval rows across **9** markets. Provenance-only
re-linking scored **4,395** candidates; the unchanged graph policy accepts
**13 `LINKED_TO`** edges across **5** markets and projects **7 `SUPPORTS`**
edges. Coverage remains below the requested breadth target, so real-data
Branch 08 and Branch 09 results remain conditional. Reddit/X stay deferred (access/API limits); the
architecture is platform-agnostic. See
[`docs/TELEGRAM_SOURCES.md`](docs/TELEGRAM_SOURCES.md),
and [`docs/GRAPH_OPERATIONS.md`](docs/GRAPH_OPERATIONS.md)
for the available graph operations documentation. The original branch-specific
quality-gate documents were not included in the supplied archive.

**Branch 09 - Forecasting, Backtesting & Beating Polymarket** runs on
**real data only**. The final-pre-close evaluator uses explicit benchmark
profiles. Its no-flag default, `real_optional`, joins real resolved contracts
to the canonical Branch 08 social-pressure features and the latest valid
pre-close YES price. `real_social_enriched_main` runs the identical evaluation
against the Branch 12 public-web enriched social root. The earlier synthetic
benchmark pathway has been removed entirely. Current real-data coverage is
still thin: only **9/110** resolved rows have a valid pre-close price and only
**1/110** has sufficient accepted social evidence. See
[`docs/EVALUATION.md`](docs/EVALUATION.md).

## What this repository contains today

> The table below reflects the **current 300-market build** (the figures in
> *Current dataset & pipeline* above are authoritative).

| Data concern                         | Status                          |
| ------------------------------------ | ------------------------------- |
| Resolved-only evaluation labels      | ✅ 300 resolved markets (eval targets only) |
| Resolved-market price history        | ✅ 1,353,114 points, full lifetime, 300/300 |
| Telegram multi-channel ingestion     | ✅ ~1.35M posts across 68 channels |
| Social linked to resolved markets    | ✅ 357,691 candidate links; 32,149 accepted evidence rows |
| Outcome-aware stance                 | ✅ 30,680 stance rows (134 markets with accepted evidence) |
| Top linked-evidence review pack      | ✅ accepted evidence across 134 markets |
| Live Neo4j graph                     | ✅ loaded end-to-end (4.1M rows); query captures in `data/graph_exports/` |
| Graph read-query helpers + CLI       | ✅ 7 parameterized queries, run live or `--dry-run` |
| Social evidence *breadth*            | ⚠️ single platform (Telegram); Reddit/X deferred to the scrape |



| Concern                              | Status                          |
| ------------------------------------ | ------------------------------- |
| Repository structure and tooling     | ✅ implemented                   |
| Domain models (Pydantic)             | ✅ implemented                   |
| Configuration (pydantic-settings)    | ✅ implemented                   |
| Deterministic NLP normalization      | ✅ implemented + tested          |
| Legacy heuristic contract↔post linker| ✅ scaffold baseline + tests     |
| Branch 05A semantic relevance linker | ✅ hardened deterministic baseline |
| Branch 05B embedding overlay         | ✅ optional, offline-safe        |
| Branch 05C LLM relevance judge       | ✅ optional, cached, relevance-only |
| Outcome-aware stance                 | ✅ Branch 06 quality gate passed |
| Social Pressure Index (baseline)     | ✅ implemented + tested          |
| Social-adjusted probability baseline | ✅ implemented + tested          |
| Evaluation metrics                   | ✅ implemented + tested          |
| Refresh policy                       | ✅ implemented + tested          |
| Neo4j schema definition              | ✅ documented + cypher constants |
| React dashboard (`webapp/`)          | ✅ served by `run_app.cmd`       |
| Task-2 analytics layer               | ✅ `polysignal analytics run` (14 artifacts) |
| Task-3 prediction (temporal CV)      | ✅ `polysignal evaluate task3` (4 model families, ablation) |
| Typer CLI skeleton                   | ✅ implemented                   |
| Polymarket ingestion (real)          | ✅ binary YES/NO markets         |
| Reddit ingestion (real)              | ✅ Scrapling public-web (old.reddit.com); API not used |
| Social query generation              | ✅ constraint-preserving         |
| Social normalization / dedup         | ✅ SocialPostRecord + parquet    |
| X/Twitter ingestion (live)           | 🚧 future adapter stub (no scraping) |
| Telegram ingestion (live)            | ✅ Telethon public-channel adapter |
| LLM provider adapters                | ✅ optional OpenAI / Anthropic    |
| Graph construction (loaders)         | ✅ Loader v1 planner + batched upserts |
| Backtesting orchestration            | ✅ final-pre-close Branch 09 evaluator |

## High-level architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          External data sources                           │
│   Polymarket Gamma/CLOB   Reddit   X/Twitter   Telegram   LLM provider   │
└──────────────────────────────────────────────────────────────────────────┘
            │              │           │            │             │
            ▼              ▼           ▼            ▼             ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                       Ingestion layer                           │
   │  polysignal.ingestion.polymarket  /  polysignal.ingestion.social │
   └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                     Normalization & storage                     │
   │   data/raw  →  data/normalized  (Parquet)  +  data/cache (LLM)  │
   └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  Linking  →  NLP enrichment  →  Stance  →  Feature engineering  │
   │  (heuristic + LLM-assisted, controlled, cached)                 │
   └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │              Knowledge graph (Neo4j) construction                │
   │  Contracts • Outcomes • Posts • Authors • Entities • Topics     │
   └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │     Analytics  +  Social Pressure Index  +  Forecasting         │
   │  Social-adjusted probability vs. Polymarket-implied probability │
   └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │   Evaluation & backtesting  (Brier, log loss, AUC, calibration) │
   └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │              Interactive dashboard (React app, run_app.cmd)     │
   │   browse 300 contracts • price+social overlay • per-post inspect│
   └─────────────────────────────────────────────────────────────────┘
```

For deeper detail see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
[`docs/DATA_MODEL.md`](docs/DATA_MODEL.md),
[`docs/GRAPH_MODEL.md`](docs/GRAPH_MODEL.md),
[`docs/REFRESH_POLICY.md`](docs/REFRESH_POLICY.md), and
[`docs/EVALUATION.md`](docs/EVALUATION.md).

## Repository layout

```
.
├── pyproject.toml            # package metadata + dependencies
├── docker-compose.yml        # local Neo4j
├── .env.example              # configuration template
├── Makefile                  # convenience commands
├── data/
│   ├── raw/                  # API payloads / social snapshots (gitignored)
│   ├── normalized/           # Parquet datasets (gitignored)
│   ├── analytical/           # derived feature tables (gitignored)
│   ├── cache/                # LLM / feature caches (gitignored)
│   └── graph_exports/        # offline graph dumps (gitignored)
├── docs/                     # technical documentation
├── src/polysignal/
│   ├── config/               # pydantic-settings configuration
│   ├── domain/               # Pydantic schemas + enums
│   ├── storage/              # paths, parquet IO, caches
│   ├── ingestion/
│   │   ├── polymarket/       # Gamma + CLOB client, binary YES/NO filter,
│   │   │                     #   normalizer, raw store, pipeline (real)
│   │   └── social/           # Reddit + Telegram real; X/Twitter stub
│   ├── linking/              # contract ↔ post linking (heuristic + LLM)
│   ├── nlp/                  # normalization, stance, NER
│   ├── graph/                # Neo4j schema, planner, client, upsert queries
│   ├── features/             # Social Pressure Index, graph features
│   ├── forecasting/          # social-adjusted probability baseline
│   ├── analytics/            # descriptive / temporal / correlation
│   ├── evaluation/           # metrics + backtesting
│   ├── scheduling/           # refresh policy
│   ├── llm/                  # provider-agnostic client interface
│   └── cli/                  # Typer entry point
├── webapp/                   # React dashboard (served by run_app.cmd)
└── tests/                    # pytest unit tests
```

## Installation

PolySignal targets **Python 3.11+**. We strongly recommend a virtual
environment. Pick the snippet that matches your shell — the commands
inside each block are safe to copy-paste verbatim.

**Linux / macOS (bash, zsh):**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

**Windows CMD:**

```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Optional extras

`scraping` adds the Scrapling stack used for Reddit + Telegram public-web
collection, `telegram` adds Telethon, `nlp` adds VADER (sentiment), and
`llm` adds the OpenAI and Anthropic SDKs. All are optional. **Reddit is
collected via Scrapling public-web only — the Reddit OAuth API is not used.**

```bash
python -m pip install -e ".[scraping]"
python -m pip install -e ".[telegram]"
python -m pip install -e ".[nlp]"
python -m pip install -e ".[llm]"
```

## Local infrastructure

Neo4j is the knowledge-graph store. A local instance is provided via
[`docker-compose.yml`](docker-compose.yml). The container exposes the
HTTP browser on port `7474`, Bolt on port `7687`, and uses the
development credentials `neo4j / polysignal-dev` (defined in the
compose file).

Start it:

```bash
docker compose up -d neo4j
```

Stop it:

```bash
docker compose down
```

If Docker is not available on your machine you can still develop the
deterministic core of the project. Graph schema previews, load planning, and
offline inspection do not contact Neo4j.

### Applying the graph schema

The schema (labels, relationship types, constraints, indexes) is defined
in [`src/polysignal/graph/schema.py`](src/polysignal/graph/schema.py)
and described in [`docs/GRAPH_MODEL.md`](docs/GRAPH_MODEL.md). The CLI
can either print the Cypher statements without contacting Neo4j
(`--dry-run`) or execute them against the running database.

Preview the statements:

```bash
polysignal graph init --dry-run
```

Apply them (requires Neo4j running):

```bash
polysignal graph init --no-dry-run
```

Preview the idempotent Parquet-to-Neo4j load plan and inspect the available
datasets:

```bash
polysignal graph load --dry-run
polysignal graph inspect
```

Load the planned nodes and relationships after Neo4j is running:

```bash
polysignal graph load --no-dry-run
polysignal graph inspect --live
```

Graph Loader v1 resolves segment-level linked evidence back to parent posts.
It loads contracts, domains, outcomes, posts, authors, platforms, core
relationships, and optional directional stance edges. Price points remain in
Parquet; entity, topic, narrative, and Social Pressure Index loading are
deferred.

## Running tests

```bash
python -m pytest
```

With coverage:

```bash
python -m pytest --cov=polysignal --cov-report=term-missing
```

The suite covers the deterministic components (text normalization,
evaluation metrics, heuristic linker, heuristic stance, Social Pressure
Index, social-adjusted probability, refresh policy) plus a set of
lightweight smoke tests (package import, CLI commands via Typer's
`CliRunner`, configuration loading, storage paths, graph dry-run,
analytics, and the Task-3 temporal-CV runner). Smoke tests do not require network
access, Docker, Neo4j, or an LLM provider.

## Running lint

```bash
python -m ruff check src tests
```

Apply auto-fixes:

```bash
python -m ruff check --fix src tests
python -m ruff format src tests
```

## Inspecting the configuration

The resolved configuration (with secrets implicitly masked — only
booleans like `reddit.configured` are exposed) can be displayed with:

```bash
polysignal config-show
```

To create the `data/` layout under the configured root:

```bash
polysignal init-data-dirs
```

## Canonical data layout & pipeline

The product pipeline now lives under one canonical root, **`data/`**, and is
documented end to end in [`docs/PIPELINE.md`](docs/PIPELINE.md): the 6 curated
resolved markets, their full-lifetime Polymarket price timelines, the social
evidence policy, the Market-Timeline graph, and the two large-scale collection
commands. Legacy / demo dataset roots were moved to `archive/`
during the 2026-06-13 cleanup (see `archive/README.md`); nothing under
`archive/` feeds the live pipeline.

## Launching the dashboard

The dashboard is the **React app** in `webapp/` (the earlier Streamlit skeleton was
removed). One command, from the project root:

```bash
# Cross-platform (macOS / Linux / Windows):
python scripts/serve_app.py
# Windows one-click equivalent:
run_app.cmd
# macOS / Linux convenience wrapper:
./run_app.sh
```

It starts a single local server (`scripts/serve_app.py`) and opens
<http://127.0.0.1:8000/>, serving both the UI and the live data endpoint
(`/api/social-enriched/ps-data`, built by `scripts/build_ps_data.py`) from the real
pipeline artifacts on one origin — no internet, no CORS (React/Babel are vendored).
See [`docs/RUN_APP.md`](docs/RUN_APP.md) for the full walkthrough.

The app lets you browse all 300 contracts by domain / resolution date / outcome, view
each contract's price line with the social-activity overlay, explore the per-platform /
per-domain sentiment & engagement distributions (the Task-2 `analytics` artifacts), and
open a **per-post inspector** (NER entities, hashtags, language, sentiment, stance,
engagement, and a source link). It reads `data/reports/ps_data.json` plus the persisted
artifacts under `data/analytical/` (`analytics/`, `backtests/<profile>/`, `timelines/`,
`features/`); missing optional files degrade to honest "no coverage" panels instead of
crashing.

The dashboard never recomputes forecasting/SPI, retrains models, or calls any
API/LLM. The `real_optional` profile carries explicit sparse-coverage
warnings — see [dashboard setup and boundaries](docs/RUN_APP.md) for the
full claim boundaries. To point the loader elsewhere, set
`POLYSIGNAL_DATA_DIR=/path/to/data`.

## CLI

The project ships three equivalent ways to invoke the CLI:

| Invocation                                 | When to use it                                  |
| ------------------------------------------ | ----------------------------------------------- |
| `polysignal ...`                           | Installed console script — the preferred form. |
| `python -m polysignal ...`                 | Works without relying on the script being on `PATH`. |
| `from polysignal.cli.main import app`      | Programmatic access from another Python program. |

> Do **not** use `python -m polysignal.cli.main`. It still works but
> triggers a Python `RuntimeWarning` from runpy because the parent
> package would have already imported the module. The supported
> entry points above avoid that path.

Useful commands at this stage:

```bash
polysignal --help
polysignal version
polysignal config-show
polysignal init-data-dirs
polysignal graph init --dry-run
polysignal graph load --dry-run
polysignal graph inspect
polysignal refresh status --mode fast
polysignal refresh status --mode heavy
```

Branch 07 upstream-data commands:

```bash
# Offline dataset summaries
polysignal polymarket inspect                      # all markets
polysignal polymarket inspect --resolved-only      # resolved-only (evaluation focus)

# Resolved-only evaluation labels (evaluation targets ONLY — never features)
polysignal polymarket export-resolved-labels --source-dir data_resolved_eval

# Collect recent already-resolved markets WITH price history (needs outbound network).
# 'markets' discovery + liquidity order + a recent end-date floor maximise the
# yield of binary YES/NO resolved markets that still expose CLOB price history.
polysignal polymarket ingest --discovery markets --market-status closed \
    --order liquidityNum --end-date-min 2024-06-01 --limit 360 --output-dir data_resolved_eval

# Canonical deterministic stance output (no LLM by default)
polysignal stance score

# Human review pack: top linked social evidence per market (+ stance join)
polysignal linking export-review

# Multi-channel Telegram ingestion (primary social source; needs a Telethon session)
polysignal social telegram ingest --config config/telegram_channels.yml \
    --output-dir data_resolved_eval --limit-per-channel 400 --max-total 6000

# Targeted resolved-market recovery (selection and plan are offline; search needs Telethon)
polysignal dataset select-high-signal-markets --output-dir data_resolved_eval --target-markets 50 --resolved-only
polysignal social telegram build-search-plan --output-dir data_resolved_eval \
    --channels-config config/telegram_channels.yml --lookback-days 7,30,90 --queries-per-market 6
polysignal social telegram search-markets --output-dir data_resolved_eval \
    --config config/telegram_channels.yml --limit-per-query 100 --max-markets 50 --append --resume

# Graph plan + parameterized read queries (all offline / dry-run)
polysignal graph load --dry-run --output-dir data_resolved_eval
polysignal graph queries list
polysignal graph queries top-contracts --dry-run
polysignal graph queries stance-summary --dry-run
polysignal graph queries contract-neighborhood --market-id <id> --dry-run
polysignal graph queries health --dry-run

# Task-2 analytics (descriptive, discourse, correlation, cross-platform)
polysignal analytics run

# Canonical Section-2 dataset export + data dictionary
polysignal dataset export --overwrite

# Branch 09 forecasting/backtesting (offline)
polysignal evaluate list-profiles
polysignal evaluate inspect-profile --benchmark-profile real_optional
polysignal evaluate build-dataset --overwrite
polysignal evaluate backtest --overwrite --export-csv
polysignal evaluate inspect-backtest --benchmark-profile real_optional

# Task-3 outcome prediction: feature matrix + temporal-CV model ablation
polysignal evaluate task3 --overwrite
```

### End-to-end reproducibility runbook

From a populated `data/` (the canonical 300-market build), the full study reproduces with:

```bash
# 1. Enrichment + linking + stance (LLM optional; cached on disk under data/cache/llm)
polysignal nlp enrich
polysignal linking score-candidates --overwrite
polysignal stance score --overwrite

# 2. Features → analytics → prediction
polysignal features compute-social-pressure --overwrite   # Social Pressure Index
polysignal features build-timelines --overwrite           # per-market timelines
polysignal analytics run                                  # Task-2 (14 artifacts)
polysignal evaluate backtest --overwrite --export-csv     # final pre-close backtest
polysignal evaluate task3 --overwrite                     # Task-3 temporal-CV ablation

# 3. Canonical dataset export + dashboard
polysignal dataset export --overwrite
run_app.cmd                                               # React dashboard
```

Each step is offline and re-runnable; LLM-assisted linking/stance reuse the on-disk cache,
so a re-run over the current corpus incurs no new API cost. See
[`docs/SUBMISSION_CHECKLIST.md`](docs/SUBMISSION_CHECKLIST.md) for the deliverable map.

`polysignal social ingest` is real — see
["Social ingestion (Branch 04)"](#social-ingestion-branch-04) below.

The **Polymarket** commands are real (Branch 03):

```
polysignal polymarket fetch-markets --limit 50 --binary-only
polysignal polymarket inspect-market --market-id <slug-or-condition-id>
polysignal polymarket fetch-prices  --market-id <slug-or-condition-id>
polysignal polymarket ingest        --limit 50 --binary-only
```

See ["Polymarket ingestion"](#polymarket-ingestion) below.

The current implementation status of each module is tracked in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Configuration and secrets

All runtime configuration is loaded from environment variables or an `.env`
file via [`pydantic-settings`](https://docs.pydantic.dev/latest/usage/pydantic_settings/).
Copy the template and edit it:

```bash
cp .env.example .env
```

**Never commit `.env`.** The example file contains only placeholders and
documents which credentials are required for each ingestion source.

LLM usage is **disabled by default** (`POLYSIGNAL_LLM_PROVIDER=disabled`). The pipeline
must function end-to-end without an LLM — LLM-assisted steps are optional
enhancements, not hard dependencies.

## Polymarket ingestion

Branch 03 implements real Polymarket ingestion against the public
[Gamma](https://gamma-api.polymarket.com) (market metadata) and
[CLOB](https://clob.polymarket.com) (price history) APIs.

### Scope: binary YES/NO only

This milestone supports **only binary YES/NO markets**. Multi-outcome
markets (e.g. "Who wins the 2028 election?" with four candidates) and
non-YES/NO binary markets (e.g. "Team A vs Team B") are detected,
**skipped with a recorded reason**, and surfaced in the CLI summary.

The restriction is deliberate. Every downstream PolySignal component
is YES/NO-aware:

* **stance detection** (`SUPPORTS_YES` / `SUPPORTS_NO`);
* **social pressure** (`SocialPressureRecord` per outcome);
* **social-adjusted probability** (`market` vs. `social` for YES);
* **backtesting** (market-only vs. market + social on YES).

Generalising to multi-outcome markets is captured as future work
(see ["Known limitations"](#known-limitations) below) — it requires
broadening stance labels, the pressure index, and the forecast blender
together, which is out of scope for this milestone.

### Pipeline

```
Gamma /events   ──►  binary YES/NO filter  ──►  normalizer  ──►  Parquet
       │                       │                                       │
       ▼                       ▼                                       ▼
  data/raw/polymarket/    skip-reason log                  data/normalized/polymarket/
    events/<run-dir>/                                        markets.parquet
   prices/<run-dir>/                                        outcomes.parquet
                                                            price_points.parquet
```

The HTTP client lives in
[`polysignal.ingestion.polymarket.client`](src/polysignal/ingestion/polymarket/client.py),
the filter in
[`polysignal.ingestion.polymarket.binary_filter`](src/polysignal/ingestion/polymarket/binary_filter.py),
the pure-function normalizer in
[`polysignal.ingestion.polymarket.normalizer`](src/polysignal/ingestion/polymarket/normalizer.py),
the raw store (run-scoped directories + `_manifest.json`) in
[`polysignal.ingestion.polymarket.raw_store`](src/polysignal/ingestion/polymarket/raw_store.py),
and the orchestration in
[`polysignal.ingestion.polymarket.pipeline`](src/polysignal/ingestion/polymarket/pipeline.py).

### Running it

```powershell
# Inspect one market — no parquet writes, just classification + normalization preview.
polysignal polymarket inspect-market --market-id "will-candidate-x-win-2025"

# Fetch up to 50 markets (metadata only).
polysignal polymarket fetch-markets --limit 50 --binary-only

# Fetch price history for one binary market.
polysignal polymarket fetch-prices --market-id "will-candidate-x-win-2025"

# End-to-end: markets + prices for a diverse active/open event sample.
polysignal polymarket ingest --limit 20 --binary-only

# Larger dataset: cap each event family to avoid one event dominating the sample.
polysignal polymarket ingest --limit 50 --binary-only --order volume_24hr --descending --max-markets-per-event 2
```

Useful flags for `fetch-markets` and `ingest`:

| Flag                                    | Meaning                                                         |
| --------------------------------------- | --------------------------------------------------------------- |
| `--limit N`                             | Max markets to fetch from Gamma (default: 50).                  |
| `--binary-only` / `--include-non-binary`| Keep only YES/NO markets (default). The opt-in flag still cannot |
|                                         | normalize non-binary markets — it just logs them.               |
| `--save-raw` / `--no-save-raw`          | Persist raw API pages (default: on).                            |
| `--dry-run`                             | Skip Parquet writes (raw still saved unless `--no-save-raw`).   |
| `--output-dir`                          | Override the data root (defaults to `POLYSIGNAL_DATA_DIR`).     |
| `--page-size N`                         | Upstream Gamma event page size (default: 100).                  |
| `--order FIELD`                         | Gamma event ordering field (default: `volume_24hr`).            |
| `--ascending` / `--descending`          | Sort direction for Gamma event discovery (default: descending). |
| `--max-markets-per-event N`             | Cap markets selected from one event family; use `0` for no cap. |

### Where the output goes

```
data/
├── raw/polymarket/
│   ├── events/<YYYY-MM-DD>/<YYYYMMDDTHHMMSSZ>-<run_id>/
│   │     page-0001.json
│   │     page-0002.json
│   │     _manifest.json            # endpoint, params, page/object counts, UA
│   └── prices/<YYYY-MM-DD>/<YYYYMMDDTHHMMSSZ>-<run_id>/
│         <market_id>__<token_id>.json
│         _manifest.json
└── normalized/polymarket/
    ├── markets.parquet             # one row per kept Market
    ├── outcomes.parquet            # one row per Outcome (market_id + YES/NO)
    └── price_points.parquet        # one row per (market, outcome, timestamp)
```

`price_points.parquet` is written with an upsert keyed by
`(market_id, outcome_id, timestamp)`, so repeated fetches of the
same market do not duplicate samples.

### Identifier mapping (verified for this milestone)

| Generic field          | Polymarket source                                                                          |
| ---------------------- | ------------------------------------------------------------------------------------------ |
| `Market.market_id`     | `conditionId` (falls back to `id` only when `conditionId` is absent).                      |
| `Market.condition_id`  | `conditionId` (named alias of the above).                                                  |
| `Market.event_id`      | `events[0].id` or `eventId` if present.                                                    |
| `Outcome.outcome_id`   | per-outcome `clobTokenIds[i]` (CLOB token id) — also the join key for `prices-history`.    |
| `Outcome.market_id`    | parent `Market.market_id`, so `outcomes.parquet` can be joined without reading nested data.|
| `Outcome.current_price`| `outcomePrices[i]` from the Gamma payload, clamped to `[0, 1]`.                            |
| `PricePoint.*`         | the corresponding `(market_id, outcome_id)` pair; timestamps are unix seconds → UTC.       |

### Normalized schemas

`markets.parquet`:
`market_id`, `slug`, `question`, `description`, `domain`,
`created_at`, `close_time`, `resolved_at`, `is_resolved`,
`is_closed`, `resolution`, `outcomes`, `tags`, `source_url`,
`event_id`, `condition_id`, `volume`, `liquidity`.

`outcomes.parquet`:
`market_id`, `outcome_id`, `label`, `index`, `current_price`.

`price_points.parquet`:
`market_id`, `outcome_id`, `timestamp`, `price`, `volume`.

If `clobTokenIds` is absent for a market (rare), the outcome falls
back to a synthesized id (`<market_id>-yes` / `-no`). Markets with
synthesized ids cannot be joined to the CLOB price-history endpoint
and are reported in the per-outcome audit (`price_attempts`) with the
reason "no CLOB token id exposed by Gamma".

### Known limitations

* **Multi-outcome markets are not normalized.** They are detected,
  logged, and counted in the "skipped" bucket of every CLI summary.
  Lifting this requires generalising stance labels, the Social
  Pressure Index, and the forecast blender together — a separate
  branch.
* **Non-YES/NO binary markets** (e.g. "Team A vs Team B") are also
  skipped — same rationale.
* **Resolution mapping is conservative.** A market is marked
  `RESOLVED YES` / `RESOLVED NO` when `closed=true` and the
  outcome prices show one side at ≥ 0.99. More nuanced UMA dispute
  states map to `INVALID` only when explicit.
* **Live integration tests are off by default.** The Polymarket
  catalog changes constantly, so CI runs only the fixture-based
  suite. Set `POLYSIGNAL_RUN_NETWORK_TESTS=1` to opt into the live
  Gamma + CLOB smoke tests.

### Future work

* **Multi-outcome support.** Extend `BinaryMarket` → `CategoricalMarket`
  and rework outcome-aware stance / pressure / forecasting in lockstep.
* **Partitioned Parquet layout** under `data/normalized/polymarket/`
  by `domain=` / `date=` once dataset size justifies it
  (see ["Data zones"](docs/ARCHITECTURE.md#data-zones)).
* **Incremental fetch by `updatedAt`** rather than full re-pulls, to
  reduce upstream pressure on the Gamma API.

## Social ingestion (Branch 04)

Branch 04 implements real **candidate social-post collection** keyed
off the Polymarket markets produced by Branch 03. It is complete for
the current milestone. Reddit and Telegram are live-ready backends.
Telegram public-channel ingestion is the immediate path while Reddit
credentials are pending. X/Twitter remains a future adapter stub.

### Scope and explicit non-goals

Branch 04 is responsible for *collecting* candidate social posts and
preparing them for downstream analysis. It explicitly does **not** do:

- LLM-assisted semantic relevance (branch 05);
- outcome-aware stance, sentiment, NER (later branches);
- Neo4j graph loading, Social Pressure Index, forecasting, dashboards.

The most important semantic detail: query-driven rows may carry a
``market_id_candidate`` field. This means *"the post was retrieved
using a query generated from this market"*, **not** *"the post is
truly relevant to that market"*. Direct Telegram channel rows may omit
it; Branch 05 then cross-scores those posts against available markets.

### Reddit via Scrapling (public web only)

Reddit is text-rich and produces useful posts for prediction-market
discussion. It is collected via **Scrapling public-web** (old.reddit.com),
not the Reddit OAuth API — the API is intentionally not used by this
project. Public-web Reddit can return HTTP 403/429 to automated fetches;
the collector reports and respects the block (it does not evade it), so
Reddit coverage is built up incrementally. X/Twitter is out of scope.

### Telegram public-channel ingestion

Telegram uses an optional Telethon user session under
`data/cache/telegram/`. Only explicit configured public channel/group
usernames are fetched. See
[`docs/TELEGRAM_INGESTION.md`](docs/TELEGRAM_INGESTION.md).

### High-signal Telegram dataset

The `polysignal dataset` and `polysignal social telegram` commands
prepare a focused dataset of ~50 Polymarket markets and their Telegram
chatter — the design and CLI workflow are in
[`docs/HIGH_SIGNAL_TELEGRAM_DATASET.md`](docs/HIGH_SIGNAL_TELEGRAM_DATASET.md).
Outputs land under `data/analytical/dataset/`:
`high_signal_markets.parquet`, `telegram_search_plan.{parquet,json}`,
`telegram_coverage_report.json`, and
`high_signal_telegram_manifest.json`.

### Outcome-aware stance (Branch 06A)

`polysignal stance {score, inspect, export-review, evaluate}` runs
the deterministic outcome-aware stance baseline on top of Branch 05's
`linked_posts.parquet`. Stance is **not** sentiment: every cue is
interpreted relative to the market question and outcomes. The full
design and workflow live in
[`docs/NLP_ENRICHMENT.md`](docs/NLP_ENRICHMENT.md). Outputs land at
`data/normalized/social/stance_labels.parquet`,
`data/analytical/stance/manual_stance_review.csv`, and
`data/analytical/stance/stance_evaluation_report.json`.

### How query generation works

Polymarket questions are deliberately precise — and that precision
often carries a load-bearing constraint that a naive token-stripper
would discard. For example, "Will China invade Taiwan before GTA VI?"
has both an *event core* ("China invade Taiwan") and a *contextual
anchor* ("GTA VI" — which doubles as a temporal proxy). The generator
in [`polysignal.ingestion.social.query_gen`](src/polysignal/ingestion/social/query_gen.py)
decomposes each question into:

| Component        | Example                       |
| ---------------- | ----------------------------- |
| event core       | `China invade Taiwan`         |
| constraints      | `before GTA VI`               |
| context anchors  | `GTA VI`, `GTA 6` (variant)   |

…and then emits a small family of queries (deterministic, deduplicated):

- **core** — the event core verbatim;
- **entity** — the salient nouns / action verbs from the core;
- **constraint** — the event core paired with the *full* constraint
  phrase, keeping the relation word (`"China invade Taiwan before
  GTA VI"`);
- **constraint_entity** — same as `constraint` but anchored on the
  entity tokens (`"Trump out President before GTA VI"`);
- **anchor** — entity tokens plus the bare anchor (no relation word),
  including Roman ↔ arabic numeral variants (`"China invade Taiwan
  GTA 6"`);
- **slug** — content-bearing tokens recovered from the market slug
  (when the question is terse but the slug carries the date / build);
- **domain** — entities plus a domain keyword (`politics`, `crypto`, …);
- **tag** — entities plus the first two tags from the Gamma payload;
- **polymarket** — entities plus `"prediction market"` for capturing
  meta-discussion of the contract itself;
- **description_anchor** — entities plus a single capitalized phrase
  pulled from the description (capped at three, so we never inflate a
  query with resolution boilerplate).

Each emitted query is short (≤ 96 chars) so Reddit accepts it and
carries trace metadata: `market_id`, `query_type`, `rank`,
`source_fields`, `generated_at`.

#### Worked examples

| Market question                                       | Sample emitted queries                                                                                                                                                                            |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Will China invade Taiwan before GTA VI?               | `China invade Taiwan` · `China invade Taiwan before GTA VI` · `China invade Taiwan GTA VI` · `China invade Taiwan GTA 6` · `China invade Taiwan politics`                                         |
| GTA VI released before June 2026?                     | `GTA VI released` · `GTA VI` · `GTA VI released before June 2026` · `GTA VI before June 2026` · `GTA VI June 2026` · `GTA VI culture`                                                             |
| Will Trump be out as President before GTA VI?         | `Trump be out as President` · `Trump out President` · `Trump be out as President before GTA VI` · `Trump out President before GTA VI` · `Trump out President GTA VI` · `Trump out President GTA 6` |
| Will MicroStrategy sell any Bitcoin by December 31, 2026? | `MicroStrategy sell any Bitcoin` · `MicroStrategy sell Bitcoin` · `MicroStrategy sell any Bitcoin by December 31, 2026` · `MicroStrategy sell Bitcoin by December 31, 2026` · `MicroStrategy sell Bitcoin crypto` |

Each table row shows the first ~5 queries; the actual cap is
`--queries-per-market` (default `6`).

### Collecting Reddit via Scrapling

Reddit needs no credentials here — it is collected via the Scrapling
public-web stack:

```powershell
python -m pip install -e ".[scraping]"
scripts\collect_scrapling_web.cmd            # market-aware Reddit + Telegram public web
```

The collector generates market-aware queries from
`data/normalized/polymarket/markets.parquet`, fetches old.reddit.com, and
writes to `data/staging/reddit`. If Reddit returns 403/429 it records the
block and respects a cooldown (no evasion); Telegram public-web continues
regardless.

### Running it

```powershell
# Generate queries from data/normalized/polymarket/markets.parquet.
python -m polysignal social generate-queries --limit-markets 20 --queries-per-market 6

# Same, but only for one market (and just preview — no parquet write).
python -m polysignal social generate-queries --market-id "kraken-ipo-by-december-31-2026-513" --dry-run

# Market-aware Reddit + Telegram public-web collection (Scrapling, resumable).
python -m polysignal social collect-large-enriched --resume

# Inspect what was collected.
python -m polysignal social inspect-posts --limit 20
python -m polysignal social inspect-posts --include-duplicates
```

Useful flags for `social ingest`:

| Flag                          | Meaning                                                  |
| ----------------------------- | -------------------------------------------------------- |
| `--platform`                  | `reddit`, `twitter`, `telegram`, or `all` (default: reddit). |
| `--markets-path PATH`         | Override the markets parquet (default: branch-03 output). |
| `--limit-markets N`           | Maximum markets to expand into queries (default: 10).    |
| `--queries-per-market N`      | Max queries per market (default: 6).                     |
| `--posts-per-query N`         | Max posts requested per query (default: 20).             |
| `--save-raw` / `--no-save-raw`| Persist raw per-post snapshots (default: on).            |
| `--dry-run`                   | Offline plan: persist generated queries without credentials, API calls, raw snapshots, or post writes. |
| `--output-dir PATH`           | Override the data root.                                  |

### Where the output goes

```
data/
├── raw/social/reddit/<YYYY-MM-DD>/<YYYYMMDDTHHMMSSZ>-<run_id>/
│     <market_id>__<post_id>__<query_hash>.json   # one file per retrieval event
│     _manifest.json                              # platform, queries, file count, run start/finish
├── normalized/social/
│     social_posts.parquet                        # unified social rows; provenance optional
│     social_queries.parquet                      # one row per generated MarketQuery
└── ...
```

**About "raw" payloads.** Each per-post JSON is *not* a verbatim dump of
the PRAW `Submission` object (those carry unpickleable references to
the live PRAW client). It is a **snapshot of the fields we keep**:
`id`, `title`, `selftext`, `subreddit`, `author`, `created_utc`,
`score`, `ups`, `upvote_ratio`, `num_comments`, `permalink`, `url`,
`over_18` — wrapped with the retrieval context (`query`, `query_hash`,
`market_id`, `post_id`, `collected_at`) so the file is self-describing.
A future milestone can switch to a fuller snapshot without changing
the directory layout.

**Filename embeds query lineage.** Including the SHA-256 prefix of the
query in the filename means the same post retrieved by two different
queries lands in two separate files — query lineage is preserved on
disk, not only inside the JSON.

`social_posts.parquet` schema (see
[`polysignal.domain.SocialPostRecord`](src/polysignal/domain/models.py)):

`post_id`, `platform`, `source_id`, `url`, `author_id`, `author_name`,
`created_at`, `text`, `title`, `subreddit_or_channel`, `query`,
`market_id_candidate`, `engagement_score`, `upvotes`, `comments_count`,
`shares_count`, `language`, `raw_path`, `collected_at`,
`is_duplicate`, `text_length`.

`social_queries.parquet` schema:
`market_id`, `query`, `query_type`, `rank`, `source_fields`,
`generated_at`.

`social_posts.parquet` is intentionally created only after a live
ingestion run produces real posts. It may not exist in an offline-only
workspace. The schema is platform-agnostic: Reddit, Telegram, and
X/Twitter rows share the same columns, so Branch 05 must not depend on
platform-specific linking logic. The current `Platform` enum serializes
X/Twitter as `twitter`.

### Deduplication design

A given Reddit post can be surfaced by several queries from several
markets. The pipeline preserves **one row per (post, query,
market_candidate, retrieval_window) tuple** and sets `is_duplicate = True` on rows
*after* the first occurrence of `(platform, post_id)`. Branch 05 can:

- filter to `is_duplicate == False` for a unique-posts view;
- or treat duplicate rows as *additional retrieval evidence* when
  scoring relevance against multiple candidate markets.

The fallback key — used only when `post_id` is missing — is
`sha256(platform + url + normalize(text) + created_at)`.

### What X/Twitter and Telegram do today

The X/Twitter class is a future adapter stub. It deliberately does not
scrape and does not block the milestone.

Telegram is implemented with optional Telethon user-session support for
explicit public channels/groups. It writes raw message snapshots under
`data/raw/social/telegram/`, appends stable `telegram:<channel>:<message>`
rows to `social_posts.parquet`, supports resumable pre-close market-window
search, and can feed Branch 05 without Reddit.
See [`docs/TELEGRAM_INGESTION.md`](docs/TELEGRAM_INGESTION.md).

```powershell
python -m polysignal social telegram ingest --config config\telegram_channels.yml --limit-per-channel 200
python -m polysignal linking score-candidates
```

Branch 05 defaults to bounded `auto` candidate generation: retrieval
provenance when present, otherwise deterministic anchor prefiltering. Use
`--candidate-mode full-scan` only for explicit debugging comparisons.

### Running tests offline

```powershell
python -m pytest tests/test_social_query_gen.py tests/test_social_reddit_normalize.py tests/test_social_raw_store.py tests/test_social_normalizer.py tests/test_social_pipeline.py
```

The full suite (`python -m pytest`) also runs them. **No network is
required** — the Reddit ingestor accepts an injected PRAW-like client
and tests use a fake one. Live Reddit calls can be enabled with:

```powershell
$env:POLYSIGNAL_RUN_LIVE_REDDIT_TESTS = "1"
python -m pytest tests/test_social_reddit_live.py
```

The live tests are skipped by default and require Reddit credentials in
your `.env`.

### Known limitations

* **Comments are not fetched.** Branch 04 covers Reddit *posts* only;
  expanding to top comments is a clean follow-up at
  `RedditIngestor._iter_submissions`.
* **No language detection.** `language` is left `None`; a heuristic /
  fastText pass can land in a later branch without schema changes.
* **No incremental fetch.** Each run re-issues every query; dedup
  handles the practical overlap but does not skip the network calls.
* **`market_id_candidate` is retrieval-only.** Real semantic relevance
  scoring is branch 05.
* **Telegram public channels/groups only.** Private chats are not
  collected. Keep fetch limits modest and respect Telegram rate limits.

## Next implementation step

Branch 09 forecasting and backtesting runs on real data (`real_optional` and
`real_social_enriched_main` profiles), dashboard work, and demos. Branch 10 is the next implementation step: wire the
read-only dashboard views to the existing Branch 08 and Branch 09 artifacts,
without adding new analytics. In parallel, the next data priority is broader
accepted pre-close Telegram evidence and broader historical YES-price coverage
so `real_optional` comparisons contain both resolved classes. Time-sliced 7d /
3d / 1d evaluation remains deferred until that history exists.

## Academic requirements coverage

| Requirement                                | Where it lives                                                          |
| ------------------------------------------ | ----------------------------------------------------------------------- |
| Collect resolved Polymarket contracts      | `polysignal.ingestion.polymarket.client`; domain-balanced 100/bucket via `scripts/collect_balanced_markets.py` |
| Collect Polymarket price time series       | `client.fetch_price_history_range` (full lifetime, 14-day windows)      |
| Three thematic domains (politics/finance/sports) | `polysignal.domain.enums.thematic_category`; `Market.thematic_category` |
| Collect candidate social posts             | Telegram (Telethon) + Reddit (Scrapling) + X/Twitter (Playwright, small third platform) |
| Link contracts ↔ social posts              | `polysignal.linking.relevance` + `polysignal.linking.linking_pipeline`  |
| Preprocess and normalize textual data      | `polysignal.nlp.normalize`                                              |
| Sentiment analysis (Task 2/3)              | `polysignal.nlp.sentiment` (VADER default) → `nlp enrich`              |
| NER + hashtags + language (§2 post fields) | `polysignal.nlp.extract` → `post_enrichment.parquet`                   |
| Topic modeling (Task 2.2)                  | `polysignal.nlp.topics` (sklearn LDA) → `nlp topics`                   |
| Store dataset in queryable format          | `polysignal.storage.parquet_store` + `polysignal.graph.*`               |
| Descriptive / temporal / correlation analysis | `polysignal.analytics.*`                                             |
| Cross-platform comparison                  | `polysignal.analytics.correlation` + dashboard cross-platform view      |
| Interactive dashboard / app                | `webapp/` (served by `run_app.cmd` / `python scripts/serve_app.py`)     |
| Predictive models (optional)               | `polysignal.forecasting.*` + `polysignal.evaluation.*`                  |

### Scope decision: all three platforms, Telegram-dominated

This project collects **all three** social sources named by the assignment —
**Telegram** (Telethon/MTProto), **Reddit** (Scrapling public-web), and
**X/Twitter** (authenticated Playwright automation). The corpus is heavily
**Telegram-dominated** (98.6%), with Reddit (1.2%) and X/Twitter (0.2%) as
small secondary platforms whose scale is capped by their anti-automation
defences. All three flow through one platform-agnostic `SocialPostRecord`
schema, so the cross-platform analysis is a genuine three-platform
comparison (114 markets carry posts from all three). Broadening the Reddit
and X passes would scale their share without any code change.

## License

MIT — see [`LICENSE`](LICENSE).

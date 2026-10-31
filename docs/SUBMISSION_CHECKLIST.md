# Submission checklist — `Traccia_2 (1).pdf`

Maps every Traccia_2 deliverable (§4), task requirement (§3) and evaluation axis (§6) to
where it is satisfied in this repository. Status legend: ✅ done · 🟡 done but
single-platform/Telegram-heavy (auto-completes after the multi-platform scrape) · ⏳ manual
step deferred to the end.

---

## §4 Deliverables

| Deliverable | Where | Status |
| --- | --- | --- |
| **1. Technical report** | `pdf_latex/` (`main.tex` + `part1..part4`, `references.bib`); pre-compiled `main.pdf`. Task-3 chapter in `part4_prediction.tex`. Rebuild with `latexmk -pdf main.tex` (needs a TeX toolchain). | ✅ |
| **2. Dataset** (Section-2 schema) | `polysignal dataset export` → `data/dataset_export/` (`contracts`, `price_points`, `social_posts`, `links`, `comments`) + `DATA_DICTIONARY.md`. Hive-partitioned mirror in `data/dataset_export_partitioned/`. | ✅ |
| **3. Source code** | This repo: `src/polysignal/` (pipeline), `scripts/`, `webapp/` (dashboard), `tests/` (725 passing). | ✅ |
| **4. Dashboard (running)** | React app: `run_app.cmd` → <http://127.0.0.1:8000/>. See `docs/RUN_APP.md`. | ✅ |
| **5. Slides (optional)** | `docs/PRESENTATION.md` (outline). | 🟡 optional |

---

## §3 Task 1 — Data ingestion pipeline

| Requirement | Where | Status |
| --- | --- | --- |
| Polymarket resolved contracts + full price series | `data/normalized/polymarket/` (300 markets, 100/100/100; 1.35M price points). `polysignal polymarket …`. | ✅ |
| Social collection from 3 platforms, longitudinal | Telegram (~1.35M) + Reddit; X adapter auth-gated (`scripts/collect_twitter_playwright.py`). | 🟡 Telegram-heavy |
| Contract→post linking (keyword/entity/LLM) + coverage stats | `polysignal linking score-candidates`; `data/analytical/linking/coverage_report.json`. | ✅ |
| Preprocessing: clean, dedup, language, timestamp norm, NER | `polysignal nlp enrich` → `post_enrichment.parquet` (VADER sentiment, spaCy/heuristic NER, language). | ✅ |
| Storage solution + motivation | Parquet (columnar + partitioning) + **live Neo4j** graph; `docs/STORAGE_AND_SCALING.md` (5 V's, Spark scale-out). | ✅ |

## §3 Task 2 — Analytical dashboard

All four analyses: `polysignal analytics run` → `data/analytical/analytics/` (14 artifacts + `analytics_summary.json`).

| Requirement | Where | Status |
| --- | --- | --- |
| Descriptive: domain, duration, volume, volatility, price evolution, informativeness-vs-horizon | `analytics/descriptive.py` → `domain_distribution`, `contract_summary`, `price_volatility`, `price_informativeness` (Brier 0.056→0.031 T-30→T-1). | ✅ |
| Discourse: volume, timing, sentiment, topics, NER, per platform/domain | `analytics/discourse.py` → `discourse_volume`, `activity_timeline`, `sentiment_distribution`, `topic_summary`, `entity_frequency`. | 🟡 |
| Correlation: lead/lag, sentiment vs price, spike event study | `analytics/correlation.py` → `correlation_per_market` (159), `spike_event_study` (900). | ✅ |
| Cross-platform comparison | `analytics/cross_platform.py` → `cross_platform_comparison` (228). | 🟡 auto-completes post-scrape |
| Dashboard: browse by domain/date/outcome; price+social overlay; distributions; per-post inspector; customization | `webapp/` + `scripts/build_ps_data.py` (`run_app.cmd`). | ✅ |

## §3 Task 3 — Outcome prediction (optional)

`polysignal evaluate task3` → feature matrix + temporal-CV ablation.

| Requirement | Where | Status |
| --- | --- | --- |
| Features: volume, engagement, sentiment, **linguistic (TF-IDF)** | `evaluation/task3_features.py` → `data/analytical/features/task3_feature_matrix.parquet` (+ `_summary.json`). | ✅ |
| Models (binary), ≥3 families | `evaluation/task3.py` → logistic regression, linear SVM, naive Bayes, gradient boosting. | ✅ |
| Temporal cross-validation (no leakage) | `evaluation/temporal_cv.py` (expanding window over close date); TF-IDF re-fit per fold. | ✅ |
| Metrics: accuracy, macro-F1, AUC-ROC | `evaluation/metrics.py` (`macro_f1`, `f1_per_class`) + Brier/ECE. | ✅ |
| Ablation (market / social / combined) + feature importance | `data/analytical/backtests/real_optional/task3_results.{json,csv}` + `task3_feature_importance.csv`. | ✅ |

**Headline result (honest):** the market price is the strong predictor (macro-F1 0.90–0.93,
AUC up to 0.95 across all four families); social-only is near chance (macro-F1 0.47–0.53,
AUC ≈ 0.5); combining adds no consistent lift.

---

## §6 Evaluation axes

| Axis | Evidence |
| --- | --- |
| **Design choices** | Storage (Parquet + Neo4j, `docs/STORAGE_AND_SCALING.md`); linking strategy + coverage report; Task-3 feature engineering (four categories + leakage-safe per-fold TF-IDF). |
| **Experiments** | Correlation rigor (lead/lag + spike event study); predictive soundness (temporal CV, 4 families, calibration); ablation (market/social/combined). |
| **Creativity** | Neo4j knowledge graph; optional cached LLM linking/stance; PySpark aggregation job; per-fold linguistic SVD. |
| **Dashboard** (priority/accuracy/conciseness/usability/customization) | `webapp/` — tiered browse-all-300, price+social overlay, AnalyticsPanel distributions, per-post inspector, tweaks panel. |

---

## ⏳ Deferred manual step (after this)

The pipeline and every view are cross-platform-ready. The only remaining work is the
**large multi-platform scrape** (X/Twitter via `X_AUTH_TOKEN`/`X_CT0`, more Reddit with
comments, more Telegram). Re-run the runbook in `README.md`; the per-platform discourse
comparison and cross-platform analysis then populate automatically, completing Task 2.4,
and Task 3 can be re-evaluated on the richer corpus.

## Pre-submission hygiene

- [ ] Blank the token in `.env.example`; keep `.env` out of the submitted bundle (the repo
      is not git-tracked, so secrets are not auto-ignored). Rotate any exposed token.
- [ ] Rebuild `pdf_latex/main.pdf` with a TeX toolchain (`latexmk -pdf main.tex`) so the
      Task-3 chapter and `fig_task3_models.pdf` are included.
- [ ] `python -m pytest` green; `run_app.cmd` opens the dashboard.

# PolySignal: A Social Intelligence Layer for Prediction Markets

**Progetto Big Data Engineering**

---

## 1. Title Slide

**PolySignal**  
*A Social Intelligence Layer for Polymarket Prediction Markets*

**Research Question:**  
Do social media signals provide incremental predictive value over market-implied probabilities?

**Team:** [Your Names]  
**Date:** June 2026  
**Course:** Big Data Engineering

---

## 2. Project Overview

**What is PolySignal?**

A research-oriented system that studies the relationship between:
- **Social media signals** (Telegram, Reddit)
- **Polymarket prediction market outcomes**

**Key Philosophy:**
- Not an attempt to "beat Polymarket"
- An analytical layer that quantifies social context
- Controlled evaluation with explicit claim boundaries

**Architecture:** End-to-end pipeline from data collection → enrichment → linking → knowledge graph → backtesting → interactive dashboard

---

## 3. Research Objectives

**Primary Goal:**  
Quantify the relationship between social discourse and prediction market outcomes

**Specific Objectives:**
1. Collect resolved Polymarket contracts with full price histories
2. Gather social media discussions from multiple platforms
3. Link social posts to relevant prediction markets
4. Extract sentiment, stance, and topics from social discourse
5. Build a queryable knowledge graph of the domain
6. Evaluate predictive value through rigorous backtesting
7. Expose findings through an interactive dashboard

**Scientific Approach:**  
Conservative evaluation with no-leakage controls and explicit synthetic/real separation

---

## 4. Dataset: 300 Balanced Markets

**Market Selection:**
- **300 resolved binary YES/NO contracts**
- Domain-balanced: 100 politics / 100 finance / 100 sports
- Median ~4,900 price points per market

**Social Corpus:**
- **1,357,296 total posts** (Telegram 1.35M + Reddit)
- Post-2024 focus for relevance
- Deduplicated and normalized

**Price Data:**
- **1,353,114 price points** across 300 markets
- Full-lifetime series from Polymarket CLOB API
- 300/300 markets with complete pricing

**Visual:** Bar chart showing corpus size by platform (Telegram: 1.35M, Reddit: ~7k)

---

## 5. Data Collection Pipeline

**Polymarket Ingestion:**
- Gamma API (market metadata) + CLOB API (price history)
- Binary YES/NO filter (multi-outcome markets excluded)
- Full-lifetime price series with 14-day rolling windows

**Social Collection:**
- **Telegram:** Telethon public-channel adapter (20 curated channels)
- **Reddit:** Scrapling public-web (old.reddit.com)
- **X/Twitter:** Intentionally out of scope (no scraping, paid API)

**Query Generation:**
- Market-aware queries from contract questions
- 10 query types: core, entity, constraint, anchor, domain, tag, etc.
- Constraint-preserving (e.g., "before GTA VI" retained)

**Visual:** Flow diagram showing Polymarket API → Binary Filter → Normalization → Parquet

---

## 6. NLP Enrichment Pipeline

**Four-Layer Enrichment:**

1. **Sentiment Analysis** (VADER)
   - Compound, positive, negative, neutral scores
   - Market-agnostic baseline

2. **Named Entity Recognition** (spaCy)
   - Persons, organizations, locations, events
   - Entity frequency tracking

3. **Metadata Extraction**
   - Hashtags, language detection
   - Text length, engagement metrics

4. **Topic Modeling** (LDA)
   - 20 topics over linked posts
   - Top terms per topic for interpretability

**Output:** `post_enrichment.parquet` with 1.35M enriched rows

**Visual:** Pipeline stages with input/output counts at each step

---

## 7. Linking Methodology

**Challenge:** Match 1.35M posts to 300 markets with precision

**Approach:**
- **Deterministic baseline** (Branch 05A): keyword matching + TF-IDF relevance
- **Optional LLM judge** (Branch 05C): structured relevance scoring (cached)
- **Provenance tracking:** Query lineage preserved

**Linking Results:**
- **357,691 candidate pairs** across 234 markets
- **21,989 relevant** / 64,913 ambiguous / 270,789 weakly relevant
- After evidence filters: **32,149 accepted** / 325,542 rejected
- **134 markets with accepted evidence**

**Rejection Reasons:**
- Weak confidence (<0.6 threshold): majority
- Temporal violations (post-close): 15,593 posts
- Ambiguous labels

**Visual:** Sankey diagram showing linking funnel from 1.35M posts → 357k candidates → 32k accepted

---

## 8. Stance Detection

**Definition:** Outcome-aware directional positioning

**NOT sentiment** — every cue interpreted relative to market question and outcomes

**Pipeline:**
- Input: `linked_posts.parquet` (357k pairs)
- Deterministic rule-based baseline (no LLM by default)
- Labels: `SUPPORTS_YES`, `SUPPORTS_NO`, `NEUTRAL`, `AMBIGUOUS`

**Stance Results:**
- **30,680 stance labels** across 131 markets
- **Directional:** 5,825 posts (4,082 YES / 1,743 NO)
- **Neutral:** 22,427 posts
- **Directional agreement with outcomes:** 0.19 (weak)

**Visual:** Stacked bar chart showing stance distribution (YES/NO/Neutral) and resolution split

---

## 9. Knowledge Graph

**Schema (Neo4j):**
- **Nodes:** Contract, Outcome, Post, Author, Platform, Domain, Topic, Entity
- **Relationships:** `LINKED_TO`, `SUPPORTS`, `OPPOSES`, `HAS_OUTCOME`, `POSTED_BY`, `MENTIONS`

**Graph Population:**
- Loader v1: contracts, outcomes, posts, authors, core relationships
- Directional stance edges: `SUPPORTS`/`OPPOSES` from stance labels
- Price points remain in Parquet (time-series optimized)

**Query Capabilities:**
- Contract neighborhood exploration
- Stance summary per market
- Top contracts by social volume
- Health checks and coverage reports

**Visual:** Graph diagram showing Contract → Post → Author connections with stance edges

---

## 10. Backtest Results

**Evaluation Setup:**
- **Benchmark profile:** `real_optional` (no leakage)
- **300 markets:** 57 resolved YES / 243 resolved NO
- **Models:** market_only (baseline) vs market_social (market + social features) vs social_only

**Performance Metrics:**

| Model | Brier Score ↓ | AUC ↑ | Accuracy |
|-------|--------------|-------|----------|
| **market_only** | **0.031** | **0.989** | **0.963** |
| market_social | 0.040 | 0.970 | 0.957 |
| social_only | 0.160 | 0.456 | 0.810 |

**Visual:** Side-by-side bar chart comparing Brier scores and AUC across three models

**Δ Brier:** market_social is **0.009 worse** than market_only  
**Interpretation:** Social signals do **not** add predictive lift over market prices

---

## 11. Dashboard Demo

**Interactive Streamlit Application**

**Sections:**
1. **Market Timeline Graph:** Real Polymarket probability vs smoothed social signal over time
2. **Benchmark Overview:** Dataset statistics and coverage metrics
3. **Model Comparison:** AUC, Brier, accuracy across baselines
4. **Prediction Explorer:** Filterable market-level drill-down
5. **Social Signal Inspector:** Post-level evidence with stance
6. **Calibration View:** Predicted vs observed outcome rates
7. **Feature Importance:** Top predictive features (spoiler: price dominates)
8. **Warnings/Claim Boundaries:** Explicit coverage caveats

**Demo Command:** `streamlit run src/polysignal/dashboard/app.py`

**Visual:** Screenshot mockup of Market Timeline graph showing price curve vs social signal

---

## 12. Key Findings (Honest Assessment)

**Primary Finding:**  
**Social media signals do NOT provide incremental predictive value over Polymarket prices**

**Evidence:**
1. **Market-only model dominates:** Brier 0.031, AUC 0.989
2. **Adding social features degrades performance:** Δ Brier +0.009
3. **Social-only model worse than random:** AUC 0.456
4. **Directional agreement weak:** 0.19 with actual outcomes
5. **Stance-target correlation:** ≈ 0 (no signal)

**Why Social Fails:**
- **Hype bias:** Social predicts YES when 82% of markets resolve NO
- **Coverage mismatch:** Dense for politics/crypto, sparse for sports
- **Market efficiency:** Polymarket prices already incorporate public signals

**This is not a failure** — it validates market efficiency and provides empirical evidence

---

## 13. Root Cause Analysis

**Why didn't social signals help?**

**1. Channel-Market Mismatch**
- Telegram channels: crypto/politics-focused
- Market sample: 1/3 sports (low social coverage)
- Only 54/300 markets have dense social evidence

**2. Hype vs Reality**
- Social discourse is aspirational/emotional
- YES-biased discussion (4,082 YES vs 1,743 NO stance)
- Markets resolve 81% NO in our sample

**3. Market Efficiency Hypothesis**
- Polymarket prices aggregate diverse information sources
- Public social signals already priced in
- Incremental edge would require private/early signals

**4. Temporal Lag**
- Social discourse often reacts to price movements (not predictive)
- Post-close contamination: 15,593 posts filtered

**Visual:** Venn diagram showing overlap between social-dense domains and market sample

---

## 14. Limitations

**Data Coverage:**
- X/Twitter excluded (no scraping, paid API)
- Reddit blocked by old.reddit.com (HTTP 403)
- Telegram: 20 public channels (selection bias)

**Temporal Constraints:**
- Post-2024 social corpus (historical gaps)
- Price history unavailable for older markets
- No real-time streaming (batch-only)

**Methodological:**
- Binary YES/NO markets only (no multi-outcome)
- Deterministic linking baseline (LLM judge optional)
- No comments collected (post-level only)

**Platform-Specific:**
- Telegram: public channels only (no private chats)
- Reddit: incremental collection due to rate limits
- No cross-platform author identity resolution

---

## 15. Future Work

**Short-Term (Technical):**
1. Expand Telegram channel set (50+ channels)
2. Implement Reddit OAuth API (if approved)
3. Add multi-outcome market support
4. Incorporate comment threads (not just posts)
5. Time-sliced evaluation (7d/3d/1d pre-close windows)

**Medium-Term (Research):**
1. Test on markets with genuine information asymmetry
2. Early-signal detection (before price movement)
3. Cross-platform author identity resolution
4. Narrative tracking over time
5. Misinformation/manipulation detection

**Long-Term (System):**
1. Real-time streaming pipeline
2. Hybrid ensemble models (market + alternative data)
3. Domain-specific fine-tuning (politics vs sports)
4. Causal inference framework (social → price vs price → social)

---

## 16. Technical Architecture

**Stack:**
- **Language:** Python 3.11+
- **Data:** Parquet (storage), Pandas (processing)
- **NLP:** spaCy (NER), VADER (sentiment), scikit-learn (LDA topics)
- **Graph:** Neo4j (knowledge graph)
- **Dashboard:** Streamlit
- **APIs:** Polymarket Gamma/CLOB, Telethon (Telegram)

**Repository Structure:**
```
polysignal/
├── ingestion/     # Polymarket + social collectors
├── linking/       # Contract↔post relevance
├── nlp/           # Sentiment, NER, topics, stance
├── graph/         # Neo4j schema + loaders
├── features/      # Social Pressure Index
├── forecasting/   # Social-adjusted probability
├── evaluation/    # Metrics + backtesting
└── dashboard/     # Streamlit app
```

**Testing:** 80%+ coverage on deterministic components

---

## 17. Conclusions

**What We Built:**
- End-to-end social intelligence pipeline for prediction markets
- 300-market dataset with 1.35M social posts
- Rigorous NLP enrichment + knowledge graph
- Interactive dashboard with controlled evaluation

**What We Learned:**
- **Polymarket prices are highly efficient** (hard to beat with public signals)
- Social media reflects **sentiment, not predictive edge**
- Hype bias dominates social discourse
- Coverage quality > coverage quantity

**Research Contribution:**
- **Empirical evidence** of market efficiency in crypto prediction markets
- Reproducible methodology for social-market analysis
- Honest null result (valuable for the field)

**Key Takeaway:**  
PolySignal doesn't beat Polymarket — **and that's exactly the finding**

---

## 18. Academic Requirements Coverage

**Traccia 2 Requirements:**

✅ **Data Collection:**
- Resolved Polymarket contracts with price histories
- Social media posts (Telegram, Reddit)
- Three thematic domains (politics, finance, sports)

✅ **Data Processing:**
- Text normalization and deduplication
- Sentiment analysis (VADER)
- NER + hashtags + language detection
- Topic modeling (LDA)

✅ **Linking:**
- Contract↔post relevance scoring
- Outcome-aware stance detection

✅ **Storage:**
- Parquet (analytical queries)
- Neo4j (graph traversal)

✅ **Analysis:**
- Descriptive, temporal, correlation analytics
- Predictive modeling + backtesting

✅ **Visualization:**
- Interactive dashboard with multiple views

---

## 19. Demonstration

**Live Demo Components:**

1. **Dashboard Launch:**
   ```bash
   streamlit run src/polysignal/dashboard/app.py
   ```

2. **Market Timeline View:**
   - Select a curated market (e.g., politics)
   - Show Polymarket price evolution
   - Overlay social signal (SPI)
   - Highlight divergences (if any)

3. **Social Inspector:**
   - Drill into linked posts
   - Show stance labels
   - Display NER entities and topics

4. **Backtest Results:**
   - Compare model performance
   - Show calibration curves
   - Feature importance (price >> social features)

**Fallback:** Screenshots + recorded walkthrough if live demo fails

---

## 20. Questions & Discussion

**Open Questions:**

1. **Does social discourse have value in different market types?**
   - Information-asymmetric events?
   - Niche communities with domain expertise?

2. **Is the timing wrong?**
   - Real-time streaming vs batch analysis
   - Early signals before market formation

3. **What about alternative data sources?**
   - News APIs, on-chain data, expert forums
   - Private signals vs public social media

**Contact:**
- Repository: [GitHub link]
- Documentation: `docs/` folder
- Dashboard: [Deployed URL if available]

**Thank you for your attention!**

---

## Appendix: Key Metrics Summary

**Dataset Scale:**
- Markets: 300 (100 politics / 100 finance / 100 sports)
- Price points: 1,353,114
- Social posts: 1,357,296
- Linked pairs: 357,691
- Accepted evidence: 32,149
- Stance labels: 30,680

**Pipeline Performance:**
- Linking chunked over 721k corpus (8.5GB RAM-bounded)
- End-to-end runtime: ~8 minutes (link → stance → SPI → backtest)
- Markets with evidence: 134 / 300

**Backtest (real_optional, 300 markets):**
- market_only: Brier 0.031, AUC 0.989, Acc 0.963
- market_social: Brier 0.040, AUC 0.970, Acc 0.957
- social_only: Brier 0.160, AUC 0.456, Acc 0.810

**Coverage:**
- Directional stance: 5,825 / 30,680 (19%)
- Markets with sufficient evidence: 54 / 300 (18%)
- YES stance bias: 4,082 YES / 1,743 NO (2.3:1 ratio)
- Actual resolution: 57 YES / 243 NO (0.23:1 ratio) ← **mismatch**

---

## Appendix: Technical Decisions

**Why Telegram > Twitter?**
- X/Twitter API requires paid tier
- No scraping (legal/ethical reasons)
- Telegram: public channels, Telethon-friendly

**Why Parquet > SQL?**
- Columnar storage (analytics-optimized)
- Schema evolution support
- Portable (no DB instance required)
- Neo4j used for graph-specific queries only

**Why deterministic baseline > LLM-first?**
- Reproducibility (no API key dependence)
- Cost control (1.35M posts × LLM = expensive)
- LLM judge as optional verifier (cached)

**Why no live trading?**
- Academic project (research, not production)
- Polymarket API limits
- Backtesting sufficient for research question

---

## Appendix: Relevant Files

**Key Documentation:**
- `README.md` — Project overview
- `docs/PIPELINE.md` — End-to-end pipeline
- `docs/NLP_ENRICHMENT.md` — Sentiment, NER, topics
- `docs/EVALUATION.md` — Evaluation methodology
- `docs/BRANCH_10_DASHBOARD.md` — Dashboard architecture

**Key Artifacts:**
- `data/normalized/polymarket/markets.parquet`
- `data/normalized/social/social_posts.parquet`
- `data/analytical/linking/linked_posts.parquet`
- `data/analytical/stance/stance_labels.parquet`
- `data/analytical/backtests/real_optional/metrics_summary.json`

**Scripts:**
- `scripts/collect_balanced_markets.py` — 300-market dataset builder
- `scripts/relink_chunked.py` — RAM-bounded linking
- `run_app.cmd` — Dashboard launcher (Windows)

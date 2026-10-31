# NLP Enrichment — Sentiment, NER, Hashtags, Topics

This document covers the text-enrichment layer added to satisfy the
Traccia_2 requirements for **per-post fields** (§2: `hashtags`,
`mentioned_entities`, `language`), the **sentiment-analysis** technique
(Task 2.2), the **sentiment features** for prediction (Task 3), and
**topic modeling** (Task 2.2).

It is deliberately separate from the outcome-aware *stance* layer
(`docs/NLP_ENRICHMENT.md`): **stance** is interpreted relative to a
market's YES/NO outcomes; **sentiment** is the polarity of the language
itself. The assignment asks for both.

## Design: derived tables, not a corpus rewrite

The raw `social_posts.parquet` (~1.3M rows, ~300 MB) is **never
rewritten** — that is the project's data-safety convention (see the 0-byte
parquet incident guards in `polysignal.storage.parquet_store`). Instead,
enrichment is a derived table keyed by `post_id`, exactly like
`linked_posts.parquet` and `stance_labels.parquet`:

| File | Produced by | Grain |
| --- | --- | --- |
| `data/normalized/social/post_enrichment.parquet` | `polysignal nlp enrich` | one row per post |
| `data/normalized/social/topics.parquet` | `polysignal nlp topics` | one row per LDA topic |

`post_enrichment.parquet` columns: `post_id`, `platform`, `language`,
`hashtags` (list), `mentions` (list), `mentioned_entities`
(list of `{text,label}`), `sentiment_label`, `sentiment_compound`,
`sentiment_pos/neu/neg`, `sentiment_method`, `entity_method`,
`enriched_at`.

## Backends (tiered, offline-safe)

Every component degrades gracefully so the pipeline always runs without
network or heavy models. The selected backend is recorded per row.

### Sentiment — `polysignal.nlp.sentiment`
1. `transformer` — HuggingFace `cardiffnlp/twitter-roberta-base-sentiment-latest` (opt-in, `--sentiment-backend transformer`).
2. `vader` — VADER (`vaderSentiment`), tuned for social text (emoji, caps, slang). **Default.**
3. `lexicon` — a small built-in opinion lexicon with negation handling; zero third-party deps, always available.

Decision band on the compound score: `>= 0.05` positive, `<= -0.05`
negative, else neutral (standard VADER thresholds).

### Entities / hashtags / language — `polysignal.nlp.extract`
* **entities** — spaCy `en_core_web_sm` when installed (`--entity-backend spacy`); otherwise a heuristic backend that combines a domain gazetteer (politicians, central banks, crypto assets, leagues, …, compiled into a single alternation regex for speed) with capitalized n-gram detection. Labels map to `EntityType` (person / org / loc / product / …).
* **hashtags / mentions** — deterministic regex.
* **language** — `langdetect` when installed; otherwise a Cyrillic + stop-word heuristic that separates the corpus's dominant languages (en / ru / it / es / fr).

Throughput on the reference machine: ~3,500 posts/s (VADER + heuristic
NER), so the full ~1.3M-row corpus enriches in ~6–7 minutes single-core.

### Topic modeling — `polysignal.nlp.topics`
Latent Dirichlet Allocation over `CountVectorizer` features
(scikit-learn, already a core dependency). Runs over the *linked /
relevant* posts by default (the discourse that actually concerns the
contracts). Output: top terms + mass per topic, plus an in-memory
per-document dominant-topic assignment.

## CLI

```bash
# Enrich every post (sentiment + NER + hashtags + language).
polysignal nlp enrich

# Quick run / pick backends.
polysignal nlp enrich --limit 20000 --sentiment-backend vader --entity-backend heuristic

# Topic model over the relevant linked posts.
polysignal nlp topics --n-topics 10 --source linked
```

## Dependencies

`vaderSentiment` is added under the `nlp` extra (lightweight, pure
Python). spaCy / langdetect / transformers are optional upgrades under
`nlp-full`; the backends auto-detect them at runtime.

```bash
pip install -e ".[nlp]"        # VADER default
pip install -e ".[nlp-full]"   # + spaCy + langdetect
```

## Thematic domain mapping

`MarketDomain` (6 values) is mapped to the three Traccia_2 thematic
buckets via `polysignal.domain.enums.thematic_category`:

| Thematic category | MarketDomain values |
| --- | --- |
| `politics` | POLITICS (incl. geopolitics) |
| `finance` | CRYPTO + ECONOMY/macro |
| `sports` | SPORTS + CULTURE/entertainment |

`Market.thematic_category` is materialized at ingestion so the dashboard
and the "≥100 contracts per domain" coverage check can group by the
assignment's three categories directly. The domain classifier uses
word-boundary matching (so "russia" no longer matches "Bo**russia**"
Dortmund, etc.) plus a game-pattern heuristic for league-less sports
markets ("Will <Team> win on <date>?").

## Collection coverage notes (honest status)

* **Reddit = Scrapling public-web only.** The Reddit OAuth API is not used.
  `old.reddit.com` currently returns **HTTP 403** to automated fetches; the
  collector opens a circuit breaker and respects a cooldown (it does not
  evade the block). Reddit coverage is therefore built up opportunistically;
  Telegram public-web (`t.me/s/<channel>`) collection is unaffected and
  continues to broaden coverage of the 300 markets.
* **Comments / replies (Traccia_2 §2).** The per-post `comments_count`
  engagement metric is captured (144,975 posts carry it). Per-comment *text*
  is not collected from the current sources: Telegram channels are broadcast
  (their comments live in a separate linked discussion group that is not
  fetched), and Reddit comment threads are behind the same 403 block. The
  schema is ready for it — a comment is representable as a `SocialPost` with
  `parent_post_id` set — so this is a collection extension (Telegram
  discussion groups / Reddit comments when reachable), not a schema change.
* **Social signal is topic-matched, not market-matched.** The Telegram
  channels are crypto / politics / geopolitics / macro, so politics and
  finance markets link densely while sports markets (no sports channels) are
  sparse. This is why only 18/300 markets currently reach sufficient
  directional evidence, and why broadening Reddit/sports coverage is the
  active data priority.

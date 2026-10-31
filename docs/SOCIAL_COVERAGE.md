# Social Coverage: the breadth↔depth problem and how it's handled

## The problem (diagnosed 2026-06-16)

When the dataset moved from a hand-curated handful of markets to the
**balanced 300** (100 politics / 100 finance / 100 sports), the social
evidence looked like it had collapsed. It had not — it had been **exposed**.

Directional stances on the 300, by thematic domain:

| domain | directional stances | markets with ≥3 |
| --- | --- | --- |
| politics | 1,793 | 13 |
| finance | 102 | 5 |
| **sports** | **0** | **0** |

The evidence is concentrated on ~6 politics/crypto markets (Iran, Bitcoin,
Fed) that still carry 191–344 directional stances each — they did *not* lose
coverage. The real issues were two:

1. **Topic mismatch (fundamental).** The social corpus was collected from
   crypto + politics/geopolitics Telegram channels (chosen to match the old
   curated markets). The new 300 are volume-balanced, so most of them — and
   **all 100 sports markets** — are simply not discussed by those channels.
   No amount of re-linking conjures sports posts from crypto channels.
2. **A sampling shortcut (mine).** The first 300-market re-link ran on a
   200k recency sample of the 1.35M corpus. The correct scope is the
   **post-2024 window** (a post created before a market existed cannot be
   relevant evidence for it) — ~617k posts — which the proper re-link uses.

## The fix (two parts)

### A. Collect domain-matched social data
`config`-level: 11 validated public Telegram channels were added to
[`configs/social_enrichment_sources.example.yaml`](../configs/social_enrichment_sources.example.yaml)
so the corpus covers all three domains, not just crypto/politics:

* **sports** — FabrizioRomanoNews, espn, TheAthletic, bleacherreport,
  skysports, OneFootball (football: Champions League / Premier League /
  World Cup; US sports via ESPN / Bleacher Report);
* **entertainment** — Variety, netflix, Culture_Crave (film / awards / GTA);
* **finance / macro** — Bloomberg, Investing.com (Fed / rate / economy).

Each was validated live via `t.me/s/<handle>`. Collection is via Scrapling
public-web (`polysignal social collect-large-enriched`); Telethon is not used
here (its session needs interactive re-auth) and Reddit stays 403-blocked.
`t.me/s/` paginates back through years of history, so older markets benefit
too. This is an **ongoing** collection — coverage grows with each run.

### B. Two-tier analysis (collection breadth, analysis depth)
All **300** markets satisfy the "≥100 per domain" *collection* requirement and
carry the **descriptive + price** analyses (which need no social data). The
**social** analyses (discourse, sentiment, signal–market correlation, and the
Task-3 predictor) are reported on the **curated high-coverage subset** — the
markets that actually have sufficient social evidence — produced by
[`scripts/curated_analysis_subset.py`](../scripts/curated_analysis_subset.py)
into `data/analytical/curated_analysis_markets.parquet`. This keeps every
social claim honest (computed where evidence exists) without dropping the
balanced 300.

## Reproduce

```bash
# 1) collect domain-matched social data (ongoing; resumable)
scripts\collect_scrapling_web.cmd

# 2) build the correct linking corpus (post-2024 window + new staging) and re-link
python scripts/build_linking_corpus.py
polysignal linking score-candidates --posts-path data/normalized/social/_linking_corpus.parquet

# 3) re-run downstream + curate
polysignal stance score
polysignal features compute-social-pressure --overwrite
polysignal features build-timelines
polysignal evaluate build-dataset  --output-dir data --benchmark-profile real_optional --overwrite
polysignal evaluate backtest       --output-dir data --benchmark-profile real_optional --overwrite --export-csv
python scripts/curated_analysis_subset.py
```

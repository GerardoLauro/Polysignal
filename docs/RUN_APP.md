# Polysignal — local dashboard

Install the environment from the [README](../README.md), then run `run_app.cmd` on Windows or `bash run_app.sh` on Linux/macOS. The server binds only to `127.0.0.1` and opens a browser. Stop with Ctrl+C.

```text
python scripts/serve_app.py --port 8129 --no-browser
python scripts/serve_app.py --check
```

Use your virtual environment's Python. `--check` validates available data and exits: 0 when valid, 2 when the private dataset is missing. It does not start a server. If the preferred port is occupied, the normal launcher tries nearby ports and prints the actual URL.

## Source-only checkout

The full dataset is intentionally not on GitHub. Only small original CSV examples are included. Without the private pipeline artifacts the UI still loads and shows a setup message; the data endpoint returns HTTP 503 with an explanation. No market results are fabricated. This is expected setup behavior, not a demonstration dataset.

If you already have a private data directory, point `POLYSIGNAL_DATA_DIR` at it **before starting the server**. Its root must contain `analytical/`, `normalized/`, and any required `reports/`, rather than another nested `data/` directory. For example in PowerShell:

```powershell
$env:POLYSIGNAL_DATA_DIR = 'D:\Polysignal-data'
.venv\Scripts\python.exe scripts/serve_app.py
```

On Linux/macOS use `POLYSIGNAL_DATA_DIR=/path/to/data .venv/bin/python scripts/serve_app.py`.

Alternatively, if you own the original companion archives listed in [BUNDLE_DATA.json](BUNDLE_DATA.json):

```text
python scripts/restore_bundle_data.py --archives "path/to/Dati-e-materiali" --dashboard-only --verify-only
python scripts/restore_bundle_data.py --archives "path/to/Dati-e-materiali" --dashboard-only
```

The script checks archive and member hashes and all existing-file conflicts before writing. It preserves differing local files. `--destination` may select another project root (restored files are under `data/` there); default is this repository. The full original dataset is not needed for a source audit. Keep private archives outside GitHub. Neither `.env` nor a Neo4j server nor API credentials are needed just to display already-generated dashboard artifacts.

The server rebuilds its data in memory at startup; restart after changing files. The UI and API share one origin and the React/Babel libraries are vendored locally, so no font/CDN requests are needed.

## Data sources mapped into the UI

| UI element | Source artifact |
|---|---|
| Probability timeline (poly / signal / pressure) | `data/analytical/timelines/market_timelines.parquet` |
| Pressure, stance %, source diversity, evidence quality, net pressure | `data/analytical/features/social_pressure.parquet` |
| Linked-evidence cards (channel, text, engagement, stance, why) | `data/analytical/evidence/accepted_evidence.parquet` + `data/normalized/social/stance_labels.parquet` |
| Status / close / resolution | `data/normalized/polymarket/markets.parquet` |
| System panel (corpus size, artifacts, coverage) | `data/reports/usefulness_report.json`, `timeline_summary.json` |

`drivers` and `divergenceExplain` are templated **from the real numbers** (no fabricated
quotes or accounts). All probabilities/percentages are the pipeline's own outputs.

## Which markets are shown — and why

Markets are shown in three honest **coverage tiers** (constants at the top of
`scripts/build_ps_data.py`); each market carries `domain` (fine chip: POLITICS / MACRO /
CRYPTO / SPORTS) and `category` (traccia domain: politics / finance / sports):

- **demo** — strong genuine evidence: ≥ `MIN_ON_SUBJECT` (5) accepted, in-window posts that
  strictly name the market's subject, plus a directional stance.
- **sparse** — genuine but thin on-subject evidence (≥ `MIN_ON_SUBJECT_SPARSE`, capped per
  domain). Shown with a **Sparse** badge and **only on-subject cards** — never off-subject
  noise. The on-subject test excludes generic political *and* sports words, so e.g. a cricket
  post cannot count as evidence for an NBA market.
- **none** — a resolved contract with price history but no on-subject social discourse. Shown
  for completeness across all three domains (this is how the 100 sports markets stay
  browsable, since the collected channels barely discuss sports). The social line is drawn
  **equal to the market price** and the evidence panel says "no social coverage". No
  fabrication.

Markets with **0** genuinely on-subject posts (e.g. Portugal–Ventura: 219 "accepted" links,
0 about Ventura) are still excluded from the evidence tiers — they appear, if at all, only in
the no-signal layer. `system.coverage` reports the per-domain totals and how many markets in
each domain actually carry evidence.

**The social-signal line is market-aware:** on days with little directional evidence it falls
back to the live Polymarket probability (the best prior at that moment in time); where social
evidence is strong it pulls toward the social stance and can diverge. Set
`INCLUDE_DEGENERATE = True` to also surface 0-on-subject markets.

Evidence cards are also filtered to the market's active window (close − 180 days … close) so
stale reposts don't leak in.

## Regenerate the data snapshot (optional)

```
.venv\Scripts\python.exe scripts\build_ps_data.py
```

Writes `data/reports/ps_data.json` and prints a per-market validation table. The server
rebuilds this in-memory on every start, so you only need this for inspection.

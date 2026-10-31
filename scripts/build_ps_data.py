"""Build the PolySignal frontend data contract from the real pipeline artifacts.

Produces the exact JSON shape the frontend reads as ``window.PS_DATA = { markets, system }``
(see ``webapp/data.js``). Everything here is derived from
on-disk analytical artifacts — no fabricated quotes, accounts, or numbers. Prose fields
(`drivers`, `divergenceExplain`) are templated from the real metrics with hedged,
decision-support language.

Run standalone to print a validation/diagnostic table:

    .venv/Scripts/python.exe scripts/build_ps_data.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:  # keep the diagnostic print safe on Windows cp1252 consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("POLYSIGNAL_DATA_DIR", str(ROOT / "data"))).expanduser().resolve()

TIMELINES = DATA / "analytical" / "timelines" / "market_timelines.parquet"
PRESSURE = DATA / "analytical" / "features" / "social_pressure.parquet"
STANCE = DATA / "normalized" / "social" / "stance_labels.parquet"
ACCEPTED = DATA / "analytical" / "evidence" / "accepted_evidence.parquet"
MARKETS = DATA / "normalized" / "polymarket" / "markets.parquet"
USEFULNESS = DATA / "reports" / "usefulness_report.json"
TIMELINE_SUMMARY = DATA / "analytical" / "timelines" / "timeline_summary.json"
# Task-2 analytics artifacts (produced by `polysignal analytics run`).
ANALYTICS_DIR = DATA / "analytical" / "analytics"
# Out-of-fold predictive backtest (Task-3, expanding-window temporal CV). This — NOT the
# in-sample dashboard scorecard — is the leakage-free headline predictive result.
BACKTESTS_DIR = DATA / "analytical" / "backtests"
ENRICHMENT = DATA / "normalized" / "social" / "post_enrichment.parquet"
POST_COMMENTS = DATA / "normalized" / "social" / "post_comments.parquet"

# A market must have at least this many directional stances to be a full "demo"
# market; below it the directional view is too thin to headline a detail page.
MIN_DIRECTIONAL_STANCE = 1
# A market must have at least this many accepted, in-window posts that actually name
# its subject (strict entity match). Below it, the "evidence" is generic false-positive
# noise (e.g. Portugal's market matched only UK/US/Venezuela election posts, 0 about
# Ventura/Portugal) and the detail page would show misleading cards.
MIN_ON_SUBJECT = 5
# Sparse coverage tier: markets with genuine-but-thin on-subject evidence (>= this many
# strictly on-subject posts, but below MIN_ON_SUBJECT). They are shown honestly with a
# "Sparse" evidence badge and ONLY on-subject cards — never off-subject noise — so every
# thematic domain (politics / finance / sports) is represented in the explorer without
# fabricating signal. Markets with 0 on-subject posts are still excluded.
MIN_ON_SUBJECT_SPARSE = 2
# Cap on sparse-tier markets per display domain, so the clean demo stays readable and no
# single weakly-covered domain floods the table. Richest-evidence markets are kept first.
SPARSE_PER_DOMAIN = 12
# Broad "no social coverage" layer: per thematic domain (politics/finance/sports), include
# up to this many resolved markets that have a price timeline but no genuine social
# evidence. They make every domain — including sports, which the collected channels barely
# discuss — fully browsable with price + resolution, honestly flagged as no-signal. No
# evidence is fabricated; the social line is exactly the market price for these markets.
# Each thematic domain holds 100 markets, so 120 exposes **all 300** contracts (Traccia_2
# Task 2 requires browsing every contract by domain/date/outcome); tiers stay honest.
NONE_PER_DOMAIN = 120
# Flip to True to also include thin/off-subject markets (Strait of Hormuz, Portugal).
INCLUDE_DEGENERATE = False

# Generic political words that must NOT count as on-subject evidence on their own.
GENERIC_POLITICAL = {
    "election", "elections", "win", "wins", "won", "presidential", "president",
    "vote", "votes", "poll", "polls", "primary", "parliament", "parliamentary",
    "next", "minister", "candidate", "race", "ballot", "may", "february",
    "january", "april", "march",
}
# Generic sports / competition words. A sports market is on-subject only when a post
# names the distinctive team/competitor token — not a shared league word. Without this,
# an IPL (cricket) post would wrongly count as evidence for an NBA Finals market.
GENERIC_SPORTS = {
    "win", "wins", "won", "final", "finals", "nba", "nfl", "mlb", "nhl", "mls",
    "league", "premier", "champions", "champion", "championship", "uefa", "cup",
    "title", "titles", "season", "super", "bowl", "playoff", "playoffs", "game",
    "match", "series", "team", "club", "win?", "fc",
}
# Combined stoplist for strict on-subject entity matching across all domains.
GENERIC_ON_SUBJECT = GENERIC_POLITICAL | GENERIC_SPORTS

POSTS_PER_MARKET = 6

# Fine display domain (drives the coloured chip + explorer filter). Keyed on the raw
# market `domain` so sports/economy are no longer silently collapsed into POLITICS.
DOMAIN_MAP = {
    "politics": "POLITICS",
    "sports": "SPORTS",
    "crypto": "CRYPTO",
    "economy": "MACRO",
    "macro": "MACRO",
    "finance": "MACRO",
    "culture": "CULTURE",
}
# Traccia-2 canonical thematic domain (politics / finance / sports), used for honest
# coverage accounting; carried alongside the fine chip on every market. Authoritative
# source is markets.thematic_category; this raw-domain fallback covers any gaps.
THEMATIC_FROM_RAW = {
    "politics": "politics",
    "sports": "sports",
    "crypto": "finance",
    "economy": "finance",
    "finance": "finance",
    "macro": "finance",
    "culture": "politics",
}
PLATFORM_LABEL = {"telegram": "Telegram", "reddit": "Reddit", "news": "News", "x": "X"}
STANCE_MAP = {
    "supports_yes": "YES",
    "supports_no": "NO",
    "neutral": "NEUTRAL",
    "ambiguous": "AMBIGUOUS",
    "insufficient_context": "NEUTRAL",
}

_STOPWORDS = {
    "will", "the", "by", "in", "of", "a", "an", "to", "and", "for", "be", "is",
    "after", "before", "next", "returns", "normal", "end", "reach", "win", "out",
    "as", "deal", "permanent", "decreases", "interest", "rates", "bps", "meeting",
    "traffic", "fall", "enter", "forces",
}


# --------------------------------------------------------------------------- helpers
def _f(x, default=0.0):
    try:
        v = float(x)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return default


def _i(x, default=0):
    try:
        return int(round(float(x)))
    except (TypeError, ValueError):
        return default


def _s(x):
    """Return a clean string, or ``None`` for missing/NaN values.

    Parquet reads surface missing text as a float ``NaN``; passing that straight
    into the payload yields a bare ``NaN`` token that the browser cannot parse.
    """
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    text = str(x).strip()
    return text or None


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _fmt_day(ts) -> str:
    dt = pd.Timestamp(ts)
    return f"{dt:%b} {dt.day}"


def _fmt_date_full(ts) -> str:
    dt = pd.Timestamp(ts)
    return f"{dt:%b} {dt.day}, {dt.year}"


def _iso_date(ts) -> str | None:
    """ISO date (YYYY-MM-DD) for client-side resolution-date sorting; None if absent."""
    dt = pd.Timestamp(ts)
    return None if pd.isna(dt) else f"{dt:%Y-%m-%d}"


def _humanize(n: float) -> str:
    n = _f(n)
    if n <= 0:
        return "—"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(int(round(n)))


def _ago(ts, now: datetime) -> str:
    try:
        dt = pd.Timestamp(ts).to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except Exception:
        return "—"
    days = (now - dt).days
    if days < 0:
        return "—"
    if days == 0:
        return "today"
    if days < 21:
        return f"{days}d"
    if days < 90:
        return f"{days // 7}w"
    return f"{days // 30}mo"


def _entities_from_question(question: str) -> list[str]:
    """Topical node labels for the narrative graph, derived from the question text."""
    out, seen = [], set()
    # multi-word proper-noun-ish runs and salient nouns
    for tok in re.findall(r"[A-Za-zÀ-ÿ$][\w$,]*", question):
        clean = tok.strip(",").strip()
        low = clean.lower()
        if len(clean) < 3 or low in _STOPWORDS:
            continue
        is_proper = clean[0].isupper() or clean[0] == "$"
        keyword = low in {"ceasefire", "peace", "regime", "election", "bitcoin", "rate"}
        if is_proper or keyword:
            key = low
            if key not in seen:
                seen.add(key)
                out.append(clean if clean.isupper() or clean[0] == "$" else clean.capitalize())
        if len(out) >= 4:
            break
    while len(out) < 2:
        out.append("Market")
    return out[:4]


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def _slugify(question: str) -> str:
    s = _fold(question)
    s = s.replace("$", " dollar ").replace("150,000", "150k").replace("50+", "50")
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:52].strip("-")


def _market_entities(question: str) -> set[str]:
    """Folded proper-noun tokens that pin a post to *this* market (not generic words)."""
    ents = set()
    for tok in re.findall(r"[A-Za-zÀ-ÿ$][\wÀ-ÿ$]*", question):
        clean = tok.strip()
        low = _fold(clean)
        if len(clean) < 3 or low in _STOPWORDS:
            continue
        if clean[0].isupper() or clean[0] == "$" or low in {"bitcoin", "ceasefire"}:
            ents.add(low)
    return ents


def _to_naive(ts):
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def _clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+", "", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _evidence_bucket(accepted: int) -> str:
    if accepted >= 300:
        return "High"
    if accepted >= 50:
        return "Moderate"
    return "Sparse"


# Decision-time direction band. A market is called Toward YES/NO only when the smoothed
# social estimate clears 0.5 by this margin AND there is enough evidence strength that
# day; otherwise it is Balanced. Mirrors the SPI ``direction_band`` (0.12) so the two
# stay conceptually consistent, but is evaluated at the snap instant, not whole-window.
DECISION_DIR_BAND = 0.12
DECISION_STRENGTH_FLOOR = 0.05

# Cap on how much the social estimate may pull the displayed probability away from the
# live market price. ``signalProb`` is a *market-anchored* estimate: the order book is the
# prior and social is a bounded nudge, never an override. This matters because the
# out-of-fold backtest shows social adds NO predictive lift — so letting a thin social
# stream swing the line 40-58pts (the raw ``signal_strength`` reaches ~0.96) would imply a
# predictive power it does not have. Capping keeps signalProb ≈ market with a small social
# adjustment. NOTE: this clamps only the blend weight in the probability line; the raw
# social pressure/direction fields (pressureNow/pressureDir/netPressure) keep their true,
# uncapped strength so the raw signal stays a distinct, honest quantity. Dashboard-only —
# the pipeline's ``signal_strength`` artifact is untouched.
SOCIAL_BLEND_CAP = 0.35


def _direction_from_signal(social_signal: float, strength: float) -> str:
    """Decision-time pressure direction from the snap-day social estimate + strength.

    ``social_signal`` is P(YES) in [0,1] (Laplace-anchored at 0.5 when evidence is thin);
    ``strength`` is the [0,1] evidence weight that day. When evidence is too thin the
    signal sits at the 0.5 anchor and we report Balanced rather than inventing a lean.
    """
    if strength < DECISION_STRENGTH_FLOOR:
        return "Balanced"
    if social_signal >= 0.5 + DECISION_DIR_BAND:
        return "Toward YES"
    if social_signal <= 0.5 - DECISION_DIR_BAND:
        return "Toward NO"
    return "Balanced"


def _trim_settlement(poly: list[float]) -> int:
    """Number of trailing settlement-spike days to drop (order book snapped to 0/1)."""
    lo, hi, n0, run = 0.03, 0.97, len(poly), 0
    for v in reversed(poly):
        if v < lo or v > hi:
            run += 1
        else:
            break
    return run if 0 < run <= max(2, round(0.15 * n0)) else 0


def _no_signal_market(mid, prow, tl, meta, thematic: str, now: datetime) -> dict:
    """A browsable resolved market with a price timeline but no genuine social evidence.

    Used for the broad coverage / no-signal layer: the social line is exactly the market
    price (no fabricated divergence), evidence is empty, and the card is clearly flagged.
    """
    question = str(prow["question"])
    domain = DOMAIN_MAP.get(str(prow["domain"]).lower(), "POLITICS")
    raw_poly = [_clip(_f(v, 0.5), 0.0, 1.0) for v in tl["market_prob"]]
    cut = _trim_settlement(raw_poly)
    if cut:
        tl = tl.iloc[: len(raw_poly) - cut]
    n = len(tl)
    poly = [round(_clip(_f(v, 0.5), 0.0, 1.0), 4) for v in tl["market_prob"]]
    signal = list(poly)  # no social evidence → signal line *is* the market price
    date_labels = [_fmt_day(d) for d in tl["date"]]
    snap = next((i for i in range(n - 1, -1, -1) if 0.03 <= poly[i] <= 0.97), n - 1)
    poly_prob = poly[snap]

    close_ts = pd.Timestamp(prow.get("close_time"))
    resolved = str(meta["resolution"]).upper() if meta is not None and pd.notna(meta.get("resolution")) else None
    is_resolved = (close_ts.tz_localize(None) if close_ts.tzinfo is None else close_ts) < (
        pd.Timestamp(now).tz_localize(None) if close_ts.tzinfo is None else pd.Timestamp(now)
    )
    if resolved in ("YES", "NO") and is_resolved:
        status, resolution = "RESOLVED", f"Resolved {resolved} · {_fmt_day(close_ts)}"
    elif is_resolved:
        status, resolution = "RESOLVED", f"Closed · {_fmt_day(close_ts)}"
    else:
        status, resolution = "ACTIVE", _fmt_date_full(close_ts)

    scorecard = None
    if resolved in ("YES", "NO"):
        y = 1.0 if resolved == "YES" else 0.0
        b = round((poly_prob - y) ** 2, 3)
        correct = (poly_prob >= 0.5) == (y == 1.0)
        scorecard = {"outcome": resolved, "kind": "descriptive_in_sample",
                     "marketBrier": b, "signalBrier": b,
                     "marketCorrect": bool(correct), "signalCorrect": bool(correct), "edge": "tie"}

    return {
        "id": _slugify(question),
        "scorecard": scorecard,
        "question": question,
        "short": question if len(question) <= 40 else question[:38] + "…",
        "domain": domain,
        "category": thematic,
        "coverageTier": "none",
        "status": status,
        "close": _fmt_date_full(close_ts),
        "closeTs": _iso_date(close_ts),
        "outcome": resolved if resolved in ("YES", "NO") else "UNRESOLVED",
        "resolution": resolution,
        "conf": 0,
        "drivers": [
            "No on-subject social evidence in the collected corpus",
            "Social line tracks the Polymarket price (no independent signal)",
            "Browsable for price, resolution and domain coverage only",
        ],
        "polyProb": poly_prob,
        "signalProb": poly_prob,
        "delta": 0.0,
        "pressureNow": 0.0,
        "pressureDir": "Balanced",
        "series": {"poly": poly, "signal": signal, "pressure": [0.0] * n},
        "dateLabels": date_labels,
        "eventMarks": [{"i": n - 1, "label": f"Resolved {resolved}" if resolved in ("YES", "NO") else "Market close", "kind": "resolve"}],
        "stance": {"yes": 0, "no": 0, "neutral": 0, "ambiguous": 0},
        "engagement": [[0, 0.0], [n - 1, 0.0]],
        "acceptedPosts": 0,
        "totalSeen": _i(prow.get("linked_post_count")),
        "sourceDiversity": 0.0,
        "evidenceBucket": "None",
        "evidenceQuality": 0,
        "netPressure": 0.0,
        "netStanceWindow": 0.0,
        "divergenceExplain": (
            "No social discourse in the collected corpus names this market's subject, so "
            "there is no independent social signal: the social line is drawn equal to the "
            "Polymarket price. This is a no-signal / coverage case — the contract, its "
            "price history and its resolution are shown for completeness."
        ),
        "posts": [],
        "entities": _entities_from_question(question),
        "platforms": [],
    }


# --------------------------------------------------------------------------- builder
def _load_analytics() -> dict:
    """Embed the Task-2 analytics artifacts into the payload (empty when absent).

    Reads the parquet tables produced by ``polysignal analytics run`` plus the
    JSON run summary, and projects them onto the small JSON shape the dashboard's
    Analytics/Distributions panel reads. All numbers are real; nothing fabricated.
    """
    if not ANALYTICS_DIR.exists():
        return {"available": False}

    def _tbl(name: str) -> list[dict]:
        path = ANALYTICS_DIR / f"{name}.parquet"
        if not path.exists():
            return []
        df = pd.read_parquet(path)
        return json.loads(df.to_json(orient="records"))

    summary_path = ANALYTICS_DIR / "analytics_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return {
        "available": True,
        "generatedAt": summary.get("generated_at"),
        "priceInformativeness": _tbl("price_informativeness"),
        "domainDistribution": _tbl("domain_distribution"),
        "contractSummary": _tbl("contract_summary"),
        "volatilityByDomain": _tbl("volatility_by_domain"),
        "discourseVolume": _tbl("discourse_volume"),
        "sentimentDistribution": _tbl("sentiment_distribution"),
        "activityTimeline": _tbl("activity_timeline"),
        "entityFrequency": [r for r in _tbl("entity_frequency")
                            if not _is_noise(r.get("entity"))],
        "topicSummary": _tbl("topic_summary"),
        "crossPlatformSummary": _tbl("cross_platform_summary"),
        "correlation": summary.get("correlation", {}),
        "crossPlatformState": summary.get("cross_platform", {}),
    }


def _load_predictive_benchmark() -> dict:
    """Load the leakage-free, out-of-fold predictive backtest (Task-3 temporal CV).

    This is the dashboard's **headline predictive claim** — distinct from the in-sample
    descriptive scorecard. It reports out-of-fold metrics for the three ablations
    (market_only / social_only / market_social) so the UI can state, honestly, what
    predicts the resolved outcome and that social adds no out-of-fold lift. Returns
    ``{"available": False}`` when the artifact is absent (fresh checkout).
    """
    if not BACKTESTS_DIR.exists():
        return {"available": False}
    # Prefer the real-corpus profile; fall back to whatever single profile exists.
    candidates = [BACKTESTS_DIR / "real_optional" / "task3_results.json"]
    candidates += sorted(BACKTESTS_DIR.glob("*/task3_results.json"))
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return {"available": False}
    try:
        res = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"available": False}

    cv = res.get("cross_validation", {})
    best = res.get("best_by_macro_f1", {})

    def _ablation_row(ablation: str) -> dict | None:
        # Pick the best model family for this ablation by Brier (lower = better).
        rows = []
        for family, abls in (res.get("models") or {}).items():
            m = (abls or {}).get(ablation)
            if isinstance(m, dict) and m.get("brier") is not None:
                rows.append((family, m))
        if not rows:
            return None
        family, m = min(rows, key=lambda fm: fm[1]["brier"])
        return {
            "model": family,
            "brier": round(_f(m.get("brier")), 3),
            "auc": round(_f(m.get("auc")), 3),
            "accuracy": round(_f(m.get("accuracy")), 3),
            "macroF1": round(_f(m.get("macro_f1")), 3),
        }

    ablations = {
        key: _ablation_row(key)
        for key in ("market_only", "social_only", "market_social")
    }
    ablations = {k: v for k, v in ablations.items() if v is not None}
    return {
        "available": bool(ablations),
        "scheme": cv.get("scheme"),
        "orderKey": cv.get("order_key"),
        "nFolds": cv.get("n_folds"),
        "nContracts": res.get("n_contracts"),
        "nWithSocialEvidence": res.get("n_with_social_evidence"),
        "ablations": ablations,
        "best": {
            "model": best.get("model"),
            "ablation": best.get("ablation"),
            "brier": round(_f(best.get("brier")), 3) if best.get("brier") is not None else None,
        },
        "conclusion": res.get("conclusion"),
    }


# Display-only stop-list: channel handles and broadcast boilerplate that the heuristic
# NER backend surfaces as pseudo-entities. Filtered out of the dashboard's entity views
# only (the underlying enrichment tables are untouched). Conservative on purpose.
NOISE_ENTITIES = {
    "firstsquaw", "walterbloomberg", "watcherguru", "osintdefender chat",
    "first class ucl", "league royale campaign", "j9lg", "crypto.com",
    "telegram", "whatsapp", "viber", "youtube", "subscribe",
    "just in", "breaking", "new", "news", "markets", "buy", "please",
    "details", "not", "round", "sell", "win", "high",
}


def _is_noise(text: object) -> bool:
    """True if ``text`` is a known non-entity boilerplate/channel token (display only)."""
    return text is not None and str(text).strip().lower() in NOISE_ENTITIES


def _enrichment_lookup() -> dict[str, dict]:
    """Map ``post_id -> extracted features`` for the per-post inspector."""
    if not ENRICHMENT.exists():
        return {}
    cols = ["post_id", "language", "hashtags", "mentioned_entities",
            "sentiment_label", "sentiment_compound"]
    df = pd.read_parquet(ENRICHMENT, columns=[c for c in cols if c])
    out: dict[str, dict] = {}
    for row in df.itertuples(index=False):
        ents = getattr(row, "mentioned_entities", None)
        ent_list = ents.tolist() if hasattr(ents, "tolist") else (ents or [])
        tags = getattr(row, "hashtags", None)
        tag_list = tags.tolist() if hasattr(tags, "tolist") else (tags or [])
        out[str(row.post_id)] = {
            "language": _s(getattr(row, "language", None)),
            "hashtags": [str(t) for t in tag_list][:8],
            "entities": [t for t in (str(e.get("text")) for e in ent_list
                                      if isinstance(e, dict)) if not _is_noise(t)][:8],
            "sentimentLabel": _s(getattr(row, "sentiment_label", None)),
            "sentimentScore": _f(getattr(row, "sentiment_compound", 0.0)),
        }
    return out


def _comment_counts() -> dict[str, int]:
    """Map ``post_id -> sampled-comment count`` (0 until the scrape populates it)."""
    if not POST_COMMENTS.exists():
        return {}
    df = pd.read_parquet(POST_COMMENTS, columns=["post_id"])
    if df.empty:
        return {}
    return df.groupby("post_id").size().to_dict()


def build_ps_data(verbose: bool = False) -> dict:
    timelines = pd.read_parquet(TIMELINES)
    pressure = pd.read_parquet(PRESSURE).set_index("market_id")
    stance = pd.read_parquet(STANCE)
    markets_meta = pd.read_parquet(MARKETS).set_index("condition_id")
    accepted = pd.read_parquet(ACCEPTED)
    enrich_lu = _enrichment_lookup()
    comment_lu = _comment_counts()

    # per-post stance: stance_labels.post_id == accepted_evidence.parent_post_id
    stance_lu: dict[tuple, tuple] = {}
    for r in stance.itertuples(index=False):
        key = (r.market_id, str(r.post_id))
        if key not in stance_lu:
            stance_lu[key] = (STANCE_MAP.get(str(r.stance_label), "NEUTRAL"), _f(r.confidence))

    now = datetime.now(timezone.utc)
    markets_out: list[dict] = []

    # ``id`` is the frontend's React key and de-dupe key. ``_slugify`` truncates to 52
    # chars, so near-identical questions whose only difference is past that cutoff (e.g.
    # several "Fed … after January 2025/2026 meeting?" contracts) collapse to the same
    # slug. Without a uniqueness guard the colliding contracts were silently dropped
    # (295 instead of 300). ``_unique_id`` appends a numeric suffix on collision; markets
    # are de-duped by ``market_id`` (``seen_mids``) so distinct contracts always survive.
    used_ids: set[str] = set()

    def _unique_id(base: str) -> str:
        uid, i = base, 2
        while uid in used_ids:
            uid, i = f"{base}-{i}", i + 1
        used_ids.add(uid)
        return uid

    seen_mids: set = set()

    # markets with accepted evidence, ordered by evidence volume
    candidates = (
        pressure[pressure["accepted_link_count"] > 0]
        .sort_values("accepted_link_count", ascending=False)
        .index.tolist()
    )

    diagnostics = []
    sparse_by_domain: dict[str, int] = {}
    for mid in candidates:
        prow = pressure.loc[mid]
        directional = _i(prow.get("directional_stance_count", 0))

        tl = timelines[timelines["market_id"] == mid].sort_values("date")
        if len(tl) < 2:
            diagnostics.append((str(prow["question"])[:42], "SKIP (<2 days)", len(tl)))
            continue

        # Trim only the trailing *settlement spike* — the 1-2 final days where a
        # resolved market's order book snaps to 0/1. A long confident tail (a market
        # that simply sat near 0/1 for weeks) is real divergence and is kept.
        LO, HI = 0.03, 0.97
        raw_poly = [_clip(_f(v, 0.5), 0.0, 1.0) for v in tl["market_prob"]]
        n0 = len(raw_poly)
        run = 0
        for v in reversed(raw_poly):
            if v < LO or v > HI:
                run += 1
            else:
                break
        if 0 < run <= max(2, round(0.15 * n0)):
            tl = tl.iloc[: n0 - run]

        question = str(prow["question"])
        raw_domain = str(prow["domain"]).lower()
        domain = DOMAIN_MAP.get(raw_domain, "POLITICS")
        n = len(tl)

        poly = [round(_clip(_f(v, 0.5), 0.0, 1.0), 4) for v in tl["market_prob"]]
        # Pure social estimate of P(YES) (Laplace-anchored at 0.5 when evidence is thin).
        signal_social = [round(_clip(_f(v, 0.5), 0.0, 1.0), 4) for v in tl["social_signal"]]
        # Daily directional-evidence weight in [0,1] (EWMA-smoothed, carries memory).
        sig_w = [_clip(_f(v), 0.0, 1.0) for v in tl["signal_strength"]]
        # Market-aware signal line: where social evidence is thin the line falls back to
        # the *live market-implied probability* (the best prior at that moment in time);
        # where social evidence is strong it pulls toward the social stance. The pull is
        # BOUNDED by ``SOCIAL_BLEND_CAP`` — social is a nudge on the market anchor, never an
        # override — because social has no out-of-fold predictive lift (see the constant).
        # This is an explicit shrink-to-market-prior so the graph tracks the real market
        # state over time instead of sitting on a flat 0.5 — decision-support only.
        blend_w = [min(w, SOCIAL_BLEND_CAP) for w in sig_w]
        signal = [
            round(_clip(blend_w[i] * signal_social[i] + (1.0 - blend_w[i]) * poly[i], 0.0, 1.0), 4)
            for i in range(n)
        ]
        # ``strength`` is the *true* (uncapped) social evidence weight — the raw pressure
        # signal — shown as-is; only the probability blend above is capped.
        strength = [_clip(w * 100.0, 0.0, 100.0) for w in sig_w]
        series_pressure = [round(v, 1) for v in strength]
        ev_count = [_f(v) for v in tl["evidence_count"]]
        date_labels = [_fmt_day(d) for d in tl["date"]]

        # Decision-time snapshot: last day the order book was still live (in [LO,HI]);
        # falls back to the final point for markets that were confidently pinned.
        snap = n - 1
        for i in range(n - 1, -1, -1):
            if LO <= poly[i] <= HI:
                snap = i
                break
        poly_prob = poly[snap]
        signal_prob = signal[snap]
        delta = round(signal_prob - poly_prob, 4)

        # Decision-time social pressure. The whole-window SPI counts (supports_yes_count
        # etc.) describe the *entire* lifetime of the market and can disagree in time with
        # the snap-aligned ``signal_prob`` — that mismatch is exactly what produced the
        # confusing "Toward YES but PolySignal < 50%" cards. So the direction/magnitude
        # shown next to the probability are taken from the SAME instant as the probability:
        # the smoothed social estimate and evidence strength on the decision-time day.
        #   social_signal[snap] in [0,1]  -> P(YES) the social stream implied that day
        #   net = 2*(social_signal-0.5)   -> signed lean in [-1,1], consistent with delta
        #   strength[snap] in [0,100]      -> how much directional evidence backed it
        social_at_snap = signal_social[snap]
        pressure_now = round(strength[snap], 1)
        net_pressure = round(_clip(2.0 * (social_at_snap - 0.5), -1.0, 1.0), 2)
        pressure_dir = _direction_from_signal(social_at_snap, sig_w[snap])

        # stance distribution (percent, sums ~100)
        y = _i(prow.get("supports_yes_count"))
        no = _i(prow.get("supports_no_count"))
        neu = _i(prow.get("neutral_stance_count"))
        amb = _i(prow.get("ambiguous_stance_count"))
        tot = y + no + neu + amb
        if tot > 0:
            sy, sn, sne = round(y / tot * 100), round(no / tot * 100), round(neu / tot * 100)
            sa = 100 - sy - sn - sne
            sa = max(0, sa)
        else:
            sy = sn = sne = sa = 0
        stance_dist = {"yes": sy, "no": sn, "neutral": sne, "ambiguous": sa}

        # Whole-window net stance (signed YES−NO share over the market's full lifetime).
        # Kept as a descriptive census of the corpus; distinct from the decision-time
        # ``net_pressure`` above. Surfaced so the dashboard can show both honestly.
        net_stance_window = (
            round(_clip((y - no) / (y + no), -1.0, 1.0), 2) if (y + no) > 0 else 0.0
        )

        # source diversity: Pielou evenness over channels, with channel-count fallback
        ch = _i(prow.get("unique_channel_count"))
        entropy = _f(prow.get("channel_entropy"))
        if ch > 1 and entropy > 0:
            evenness = _clip(entropy / math.log(ch), 0.0, 1.0)
            source_div = round(0.5 * evenness + 0.5 * _clip(ch / 20.0, 0.0, 1.0), 3)
        else:
            source_div = round(_clip(ch / 20.0, 0.0, 1.0), 3)

        conf = _clip(_i(_f(prow.get("confidence_mean")) * 100), 0, 100)
        evidence_quality = _clip(_i(_f(prow.get("evidence_quality_score")) * 100), 0, 100)
        accepted_posts = _i(prow.get("accepted_link_count"))
        bucket = _evidence_bucket(accepted_posts)

        # market metadata: status / close / resolution
        meta = markets_meta.loc[mid] if mid in markets_meta.index else None
        # Traccia-2 canonical domain (politics/finance/sports) — authoritative from the
        # market metadata, with a raw-domain fallback for any unmapped rows.
        thematic = (
            str(meta["thematic_category"]).lower()
            if meta is not None and pd.notna(meta.get("thematic_category"))
            else THEMATIC_FROM_RAW.get(raw_domain, "politics")
        )
        close_ts = pd.Timestamp(prow.get("close_time"))
        resolved = str(meta["resolution"]).upper() if meta is not None and pd.notna(meta.get("resolution")) else None
        is_resolved = close_ts.tz_localize(None) < pd.Timestamp(now).tz_localize(None) if close_ts.tzinfo is None else close_ts < pd.Timestamp(now)
        if resolved in ("YES", "NO") and is_resolved:
            status = "RESOLVED"
            resolution = f"Resolved {resolved} · {_fmt_day(close_ts)}"
        elif is_resolved:
            status = "RESOLVED"
            resolution = f"Closed · {_fmt_day(close_ts)}"
        else:
            status = "ACTIVE"
            resolution = _fmt_date_full(close_ts)
        close_display = _fmt_date_full(close_ts)

        # ---- event marks (resolution + largest market move + peak evidence day)
        marks, used_idx = [], set()
        moves = [abs(poly[i] - poly[i - 1]) for i in range(1, n)]
        if moves and max(moves) > 0.02:
            mi = int(np.argmax(moves)) + 1
            marks.append({"i": mi, "label": "Largest market move", "kind": "data"})
            used_idx.add(mi)
        if ev_count:
            ei = int(np.argmax(ev_count))
            if ei not in used_idx and ev_count[ei] > 0:
                marks.append({"i": ei, "label": "Peak social volume", "kind": "signal"})
                used_idx.add(ei)
        res_label = f"Resolved {resolved}" if resolved in ("YES", "NO") else "Market close"
        if (n - 1) not in used_idx:
            marks.append({"i": n - 1, "label": res_label, "kind": "resolve"})
        marks.sort(key=lambda m: m["i"])

        # ---- engagement keyframes from daily evidence volume (sparse, 0..100)
        ev_max = max(ev_count) or 1.0
        k = min(16, n)
        idxs = sorted(set(int(round(j * (n - 1) / (k - 1))) for j in range(k))) if n > 1 else [0]
        engagement = [[j, round(_clip(ev_count[j] / ev_max * 100, 0, 100), 1)] for j in idxs]
        if len(engagement) < 2:
            engagement = [[0, 0.0], [n - 1, 0.0]]

        # ---- evidence cards from the *accepted* set (clean channels + engagement),
        # ranked so strict-relevant, high-confidence, on-topic posts float to the top.
        all_ents = _market_entities(question)
        strict_ents = {e for e in all_ents if _fold(e) not in GENERIC_ON_SUBJECT}
        ents_match = {e[:4] for e in all_ents}
        close_naive = _to_naive(close_ts)
        window_lo = close_naive - pd.Timedelta(days=180)
        am = accepted[accepted["market_id"] == mid].copy()
        cards = []
        strict_hits = 0
        for r in am.itertuples(index=False):
            text = _clean_text(getattr(r, "post_text_preview", ""))
            if len(text) < 12:
                continue
            # Evidence must come from the market's active window (the accepted set
            # never enforced a lower time bound, so stale reposts leak in).
            pdate = getattr(r, "post_created_at", None)
            if pd.notna(pdate):
                pdn = _to_naive(pdate)
                if pdn < window_lo or pdn > close_naive + pd.Timedelta(days=1):
                    continue
            link = _clip(_f(getattr(r, "final_confidence", 0.0)), 0.0, 1.0)
            rel = str(getattr(r, "final_relevance_label", "") or "")
            folded = _fold(text)
            on_subject = any(_fold(e) in folded for e in strict_ents)
            if on_subject:
                strict_hits += 1
            score = (2.0 if rel == "relevant" else 0.0) + (3.0 if on_subject else 0.0) + link
            cards.append((score, text, link, rel, on_subject, r))
        cards.sort(key=lambda c: c[0], reverse=True)

        # ---- coverage tier (honest, never fabricated):
        #  * 0 on-subject posts            -> off-subject noise (e.g. Portugal) -> exclude
        #  * >= MIN_ON_SUBJECT + directional view -> full "demo" market (unchanged)
        #  * >= MIN_ON_SUBJECT_SPARSE      -> "sparse" tier: Sparse badge, only on-subject
        #                                     cards, capped per domain so every thematic
        #                                     domain (politics/finance/sports) is present.
        if strict_hits < MIN_ON_SUBJECT_SPARSE and not INCLUDE_DEGENERATE:
            diagnostics.append((question[:42], "SKIP (off-subject)", f"on-subject={strict_hits}"))
            continue
        if strict_hits >= MIN_ON_SUBJECT and directional >= MIN_DIRECTIONAL_STANCE:
            coverage_tier = "demo"
        else:
            coverage_tier = "sparse"
            if not INCLUDE_DEGENERATE and sparse_by_domain.get(domain, 0) >= SPARSE_PER_DOMAIN:
                diagnostics.append((question[:42], "SKIP (sparse cap)", f"{domain} full"))
                continue
            sparse_by_domain[domain] = sparse_by_domain.get(domain, 0) + 1
            cards = [c for c in cards if c[4]]  # only genuinely on-subject evidence
            bucket = "Sparse"
            # Report the genuine on-subject count, not the inflated raw accepted_link_count
            # (the same guard that keeps Portugal's 219 off-subject "links" out of the demo).
            accepted_posts = strict_hits

        posts, seen_text = [], set()
        platforms = []
        for _score, text, link, rel, _on_subject, r in cards:
            if text[:70] in seen_text:
                continue
            seen_text.add(text[:70])
            platform = PLATFORM_LABEL.get(str(getattr(r, "platform", "")).lower(), "News")
            if platform not in platforms:
                platforms.append(platform)
            channel = str(getattr(r, "post_channel", "") or "").strip()
            author = str(getattr(r, "post_author", "") or "").strip() or channel or "source"
            st, st_conf = stance_lu.get((mid, str(getattr(r, "parent_post_id", ""))), ("NEUTRAL", link))
            eng = _f(getattr(r, "post_engagement_score", 0.0))
            terms = re.search(r"terms=([^|]+)", str(getattr(r, "explanation", "") or ""))
            rel_disp = rel.replace("_", " ") or "relevant"
            if terms and terms.group(1).strip():
                why = f"Shares market terms ({terms.group(1).strip()}); {rel_disp} link at {link:.2f} confidence."
            else:
                why = f"{rel_disp.capitalize()} link to the market question at {link:.2f} confidence."
            # Per-post inspector: full metadata + extracted features (Traccia_2 Task 2).
            pid = str(getattr(r, "parent_post_id", "") or "")
            feats = enrich_lu.get(pid, {})
            posts.append({
                "platform": platform,
                "author": author[:40],
                "handle": channel[:36] if channel else platform.lower(),
                "text": text[:240],
                "stance": st,
                "conf": round(_clip(st_conf, 0.0, 1.0), 2),
                "eng": _humanize(eng),
                "ago": _ago(getattr(r, "post_created_at", None), now),
                "link": round(link, 2),
                "why": why[:180],
                "url": str(getattr(r, "post_url", "") or "") or None,
                "language": feats.get("language"),
                "hashtags": feats.get("hashtags", []),
                "entities": feats.get("entities", []),
                "sentimentLabel": feats.get("sentimentLabel"),
                "sentimentScore": round(_f(feats.get("sentimentScore")), 3),
                "commentSample": _i(comment_lu.get(pid, 0)),
            })
            if len(posts) >= POSTS_PER_MARKET:
                break

        if not platforms:
            platforms = ["Telegram"]
        entities = _entities_from_question(question)

        # ---- prose (templated from real metrics, hedged)
        # ``decision_lean`` is the snap-aligned direction (matches the displayed
        # probability); ``window_lean`` is the full-lifetime stance census. They can
        # differ, and the prose says so rather than implying they must agree.
        decision_lean = "YES" if net_pressure > 0.05 else "NO" if net_pressure < -0.05 else "balanced"
        window_lean = (
            "YES" if net_stance_window > 0.05
            else "NO" if net_stance_window < -0.05 else "balanced"
        )
        delta_pts = round(delta * 100)
        above_below = "above" if delta_pts >= 0 else "below"
        drivers = [
            f"{accepted_posts:,} accepted posts across {ch} channels",
            f"Full-window stance census: {sy}% YES / {sn}% NO (leans {window_lean})",
            f"At decision time the social stream leaned {decision_lean}; "
            f"market-anchored estimate {delta_pts:+d} pts {above_below} the order-book mid",
        ]
        divergence_explain = (
            f"Over {n} days, PolySignal linked {accepted_posts:,} accepted posts from {ch} "
            f"channels for this market (full-window stance census {sy}% YES vs {sn}% NO, "
            f"leaning {window_lean}). The displayed probability is a market-anchored social "
            f"estimate: where directional evidence is thin it sits on the live Polymarket "
            f"price, and where evidence is strong it pulls toward the social stance. At the "
            f"decision-time snapshot the recent social stream leaned {decision_lean}, placing "
            f"the estimate {delta_pts:+d} pts {above_below} the order-book mid. This is an "
            f"in-sample, descriptive divergence view — not the out-of-fold predictive "
            f"benchmark and not a trading recommendation."
        )

        slug = _unique_id(_slugify(question))
        seen_mids.add(mid)

        # ---- forecast scorecard: score the decision-time snapshot vs the real outcome
        scorecard = None
        if resolved in ("YES", "NO"):
            y = 1.0 if resolved == "YES" else 0.0
            m_brier = round((poly_prob - y) ** 2, 3)
            s_brier = round((signal_prob - y) ** 2, 3)
            m_correct = (poly_prob >= 0.5) == (y == 1.0)
            s_correct = (signal_prob >= 0.5) == (y == 1.0)
            if abs(m_brier - s_brier) < 0.02:
                edge = "tie"
            else:
                edge = "signal" if s_brier < m_brier else "market"
            scorecard = {
                "outcome": resolved,
                "kind": "descriptive_in_sample",
                "marketBrier": m_brier,
                "signalBrier": s_brier,
                "marketCorrect": bool(m_correct),
                "signalCorrect": bool(s_correct),
                "edge": edge,
            }

        markets_out.append({
            "id": slug,
            "scorecard": scorecard,
            "question": question,
            "short": question if len(question) <= 40 else question[:38] + "…",
            "domain": domain,
            "category": thematic,
            "coverageTier": coverage_tier,
            "status": status,
            "close": close_display,
            "closeTs": _iso_date(close_ts),
            "outcome": resolved if resolved in ("YES", "NO") else "UNRESOLVED",
            "resolution": resolution,
            "conf": conf,
            "drivers": drivers,
            "polyProb": poly_prob,
            "signalProb": signal_prob,
            "delta": delta,
            "pressureNow": pressure_now,
            "pressureDir": pressure_dir,
            "series": {"poly": poly, "signal": signal, "pressure": series_pressure},
            "dateLabels": date_labels,
            "eventMarks": marks,
            "stance": stance_dist,
            "engagement": engagement,
            "acceptedPosts": accepted_posts,
            "totalSeen": _i(prow.get("linked_post_count")) or accepted_posts,
            "sourceDiversity": source_div,
            "evidenceBucket": bucket,
            "evidenceQuality": evidence_quality,
            "netPressure": net_pressure,
            "netStanceWindow": net_stance_window,
            "divergenceExplain": divergence_explain,
            "posts": posts,
            "entities": entities,
            "platforms": platforms,
        })
        diagnostics.append((question[:42], status, f"poly={poly_prob:.2f} sig={signal_prob:.2f} d={delta_pts:+d} | on-subject={strict_hits} shown={len(posts)}"))

    # ---- Broad coverage / no-signal layer: every other resolved market with a price
    # timeline, capped per thematic domain so politics/finance/sports are all browsable.
    # De-dupe by market_id (not slug) so a contract whose slug collides with one already
    # shown is still added with a unique id, rather than silently dropped.
    none_by_cat: dict[str, int] = {}
    for mid in pressure.index:
        prow = pressure.loc[mid]
        question = str(prow["question"])
        if mid in seen_mids:
            continue
        tl = timelines[timelines["market_id"] == mid].sort_values("date")
        if len(tl) < 2:
            continue
        meta = markets_meta.loc[mid] if mid in markets_meta.index else None
        thematic = (
            str(meta["thematic_category"]).lower()
            if meta is not None and pd.notna(meta.get("thematic_category"))
            else THEMATIC_FROM_RAW.get(str(prow["domain"]).lower(), "politics")
        )
        if none_by_cat.get(thematic, 0) >= NONE_PER_DOMAIN:
            continue
        none_by_cat[thematic] = none_by_cat.get(thematic, 0) + 1
        entry = _no_signal_market(mid, prow, tl, meta, thematic, now)
        entry["id"] = _unique_id(entry["id"])
        seen_mids.add(mid)
        markets_out.append(entry)
        diagnostics.append((question[:42], "NO-SIGNAL", f"{thematic} price-only"))

    # Evidence-bearing markets first (by divergence), then the no-signal coverage layer.
    markets_out.sort(key=lambda m: (m["coverageTier"] == "none", -abs(m["delta"])))
    system = _build_system(markets_out)
    system["scorecard"] = _compute_scorecard(markets_out)
    system["analytics"] = _load_analytics()
    system["predictiveBenchmark"] = _load_predictive_benchmark()
    payload = {"markets": markets_out, "system": system}

    if verbose:
        print(f"\nBuilt {len(markets_out)} markets\n" + "-" * 96)
        for q, s, d in diagnostics:
            print(f"  {q:<44} {s:<18} {d}")
        print("-" * 96)
        _validate(payload)
        print("Validation: OK")
    return payload


def _build_system(markets_out: list[dict]) -> dict:
    use = {}
    if USEFULNESS.exists():
        use = json.loads(USEFULNESS.read_text(encoding="utf-8"))
    head = use.get("headline", {})
    posts_loaded = _i(head.get("collected_total"), sum(m["totalSeen"] for m in markets_out))
    # accepted links actually shown in the universe (consistent with the cards/metrics)
    accepted_links = sum(m["acceptedPosts"] for m in markets_out)

    generated = None
    if TIMELINE_SUMMARY.exists():
        try:
            generated = json.loads(TIMELINE_SUMMARY.read_text(encoding="utf-8")).get("generated_at")
        except Exception:
            generated = None
    gen_dt = pd.Timestamp(generated).to_pydatetime() if generated else datetime.now(timezone.utc)
    if gen_dt.tzinfo is None:
        gen_dt = gen_dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    delta_h = (now - gen_dt).total_seconds() / 3600.0
    ago = "just now" if delta_h < 1 else (f"{int(delta_h)}h ago" if delta_h < 48 else f"{int(delta_h // 24)}d ago")

    max_days = max((len(m["dateLabels"]) for m in markets_out), default=0)

    # Honest coverage breakdown by tier and thematic domain (traccia "coverage" view).
    tiers = {"demo": 0, "sparse": 0, "none": 0}
    by_cat: dict[str, dict[str, int]] = {}
    for m in markets_out:
        tier = m.get("coverageTier", "demo")
        tiers[tier] = tiers.get(tier, 0) + 1
        cat = m.get("category", "politics")
        slot = by_cat.setdefault(cat, {"total": 0, "withEvidence": 0})
        slot["total"] += 1
        if tier in ("demo", "sparse"):
            slot["withEvidence"] += 1

    artifacts = [
        ("market_timelines.parquet", TIMELINES),
        ("social_pressure.parquet", PRESSURE),
        ("stance_labels.parquet", STANCE),
        ("accepted_evidence.parquet", ACCEPTED),
        ("usefulness_report.json", USEFULNESS),
    ]
    return {
        "lastUpdate": f"{gen_dt:%Y-%m-%d %H:%M} UTC",
        "lastUpdateAgo": ago,
        "marketsLoaded": len(markets_out),
        "postsLoaded": posts_loaded,
        "acceptedLinks": accepted_links,
        "benchmark": "controlled-benchmark · real-corpus",
        "profile": "social-adjusted · balanced",
        "artifacts": [{"name": name, "ok": path.exists()} for name, path in artifacts],
        "mode": "READ-ONLY · BENCHMARK",
        "coverageWindow": f"{max_days} days",
        "coverage": {"tiers": tiers, "byCategory": by_cat},
    }


def _compute_scorecard(markets_out: list[dict]) -> dict:
    """**Descriptive** in-sample comparison: the market-anchored social estimate vs the
    order book on resolved markets.

    IMPORTANT — this is NOT the predictive headline. ``signalProb`` is an in-sample,
    market-anchored *descriptive* series (deterministic, never fitted on outcomes), so
    scoring it against the same outcomes it was drawn alongside is illustrative only and
    must not be read as evidence that PolySignal predicts outcomes. The leakage-free,
    out-of-fold predictive claim lives in ``system.predictiveBenchmark`` (the temporal
    backtest). Only evidence-bearing markets are included; the no-signal coverage layer
    (where the social line equals the price) would otherwise pad the table with ties.
    """
    scored = [m for m in markets_out if m.get("scorecard") and m.get("coverageTier") != "none"]
    n = len(scored)
    if n == 0:
        return {"n": 0, "kind": "descriptive_in_sample"}
    m_acc = sum(m["scorecard"]["marketCorrect"] for m in scored) / n
    s_acc = sum(m["scorecard"]["signalCorrect"] for m in scored) / n
    m_brier = sum(m["scorecard"]["marketBrier"] for m in scored) / n
    s_brier = sum(m["scorecard"]["signalBrier"] for m in scored) / n
    signal_wins = [m for m in scored if m["scorecard"]["edge"] == "signal"]
    market_wins = sum(1 for m in scored if m["scorecard"]["edge"] == "market")
    ties = sum(1 for m in scored if m["scorecard"]["edge"] == "tie")
    if signal_wins:
        names = ", ".join(m["short"].rstrip("…").strip().rstrip(",") for m in signal_wins)
        tail = f" The estimate was closer on {len(signal_wins)} ({names})."
    else:
        tail = " The estimate did not beat the order book on any market here."
    verdict = (
        f"Descriptive (in-sample) view on {n} resolved markets: the order book was more "
        f"accurate overall (Brier {m_brier:.2f} vs {s_brier:.2f}).{tail} This compares the "
        f"market-anchored social *estimate*, not a predictive model — see the out-of-fold "
        f"backtest for the headline predictive result."
    )
    return {
        "n": n,
        "kind": "descriptive_in_sample",
        "marketCorrect": sum(m["scorecard"]["marketCorrect"] for m in scored),
        "signalCorrect": sum(m["scorecard"]["signalCorrect"] for m in scored),
        "marketAcc": round(m_acc, 2),
        "signalAcc": round(s_acc, 2),
        "marketBrier": round(m_brier, 3),
        "signalBrier": round(s_brier, 3),
        "signalWins": len(signal_wins),
        "marketWins": market_wins,
        "ties": ties,
        "verdict": verdict,
    }


def _validate(payload: dict) -> None:
    assert isinstance(payload.get("markets"), list) and payload["markets"], "no markets"
    assert isinstance(payload.get("system"), dict), "no system"
    for m in payload["markets"]:
        s = m["series"]
        n = len(m["dateLabels"])
        assert n >= 2, f"{m['id']}: series too short"
        assert len(s["poly"]) == n == len(s["signal"]) == len(s["pressure"]), f"{m['id']}: ragged series"
        assert m["entities"], f"{m['id']}: empty entities"
        # No-signal coverage markets legitimately have no platforms (no linked posts).
        assert m["platforms"] or m.get("coverageTier") == "none", f"{m['id']}: empty platforms"
        assert len(m["engagement"]) >= 2, f"{m['id']}: engagement too short"
        for ev in m["eventMarks"]:
            assert 0 <= ev["i"] < n, f"{m['id']}: eventMark out of range"
        for v in s["poly"] + s["signal"]:
            assert 0.0 <= v <= 1.0, f"{m['id']}: prob out of range"
        for v in s["pressure"]:
            assert 0.0 <= v <= 100.0, f"{m['id']}: pressure out of range"
        st = m["stance"]
        assert abs(sum(st.values()) - 100) <= 2 or sum(st.values()) == 0, f"{m['id']}: stance sum {sum(st.values())}"


def main() -> None:
    payload = build_ps_data(verbose=True)
    out = ROOT / "data" / "reports" / "ps_data.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}  ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

"""Social media discourse analysis (Traccia_2 Task 2.2).

Analyzes the volume, timing, sentiment, topics, and named entities of the posts
*accepted-linked* to each contract, broken down per platform and per thematic
domain. Sentiment/entities come from ``post_enrichment``; topics from the LDA
``topics`` table.

Pure transforms over DataFrames; persistence lives in the runner.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from polysignal.analytics._common import accepted_links, domain_by_market, to_utc


def _with_domain(linked: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    dom = domain_by_market(markets)
    out = linked.copy()
    out["domain"] = out["market_id"].map(dom).astype("string")
    return out


def volume_by_platform_domain(linked: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """Accepted-linked post volume + contract coverage per (platform, domain)."""
    cols = ["platform", "domain", "n_posts", "n_contracts", "n_authors"]
    acc = accepted_links(linked)
    if acc.empty:
        return pd.DataFrame(columns=cols)
    acc = _with_domain(acc, markets)
    grouped = acc.groupby(["platform", "domain"], dropna=False)
    out = pd.DataFrame(
        {
            "n_posts": grouped.size(),
            "n_contracts": grouped["market_id"].nunique(),
            "n_authors": grouped["post_id"].nunique(),
        }
    ).reset_index()
    return out.sort_values(["platform", "domain"]).reset_index(drop=True)


def activity_timeline(
    linked: pd.DataFrame, markets: pd.DataFrame, *, bins: int = 10
) -> pd.DataFrame:
    """Distribution of post timing across each contract's normalized lifetime.

    Lifetime position 0.0 = contract creation, 1.0 = resolution. Returns, per
    domain, the share of accepted-linked posts falling in each of ``bins``
    equal-width phase buckets — i.e. *when* in a contract's life discourse peaks.
    """
    cols = ["domain", "phase_bucket", "n_posts", "share"]
    acc = accepted_links(linked)
    if acc.empty or markets.empty:
        return pd.DataFrame(columns=cols)
    created = dict(zip(markets["market_id"], to_utc(markets["created_at"]), strict=False))
    from polysignal.analytics._common import resolution_anchor

    anchor = dict(zip(markets["market_id"], resolution_anchor(markets), strict=False))
    acc = _with_domain(acc, markets)
    acc = acc.copy()
    acc["post_ts"] = to_utc(acc["post_created_at"])
    positions: list[dict[str, Any]] = []
    for _, row in acc.iterrows():
        mid = row["market_id"]
        c, a, p = created.get(mid), anchor.get(mid), row["post_ts"]
        if c is None or a is None or pd.isna(c) or pd.isna(a) or pd.isna(p) or a <= c:
            continue
        pos = (p - c).total_seconds() / (a - c).total_seconds()
        if 0.0 <= pos <= 1.0:
            bucket = min(int(pos * bins), bins - 1)
            positions.append({"domain": row["domain"], "phase_bucket": bucket})
    if not positions:
        return pd.DataFrame(columns=cols)
    frame = pd.DataFrame(positions)
    grouped = frame.groupby(["domain", "phase_bucket"]).size().rename("n_posts").reset_index()
    totals = grouped.groupby("domain")["n_posts"].transform("sum")
    grouped["share"] = (grouped["n_posts"] / totals).round(4)
    return grouped.sort_values(["domain", "phase_bucket"]).reset_index(drop=True)


def sentiment_distribution(
    linked: pd.DataFrame, enrichment: pd.DataFrame, markets: pd.DataFrame
) -> pd.DataFrame:
    """Sentiment label mix + mean compound per (platform, domain)."""
    cols = ["platform", "domain", "n_enriched", "pos_share", "neu_share",
            "neg_share", "mean_compound"]
    acc = accepted_links(linked)
    if acc.empty or enrichment.empty:
        return pd.DataFrame(columns=cols)
    acc = _with_domain(acc, markets)
    enr = enrichment[["post_id", "sentiment_label", "sentiment_compound"]].drop_duplicates("post_id")
    merged = acc.merge(enr, on="post_id", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=cols)
    merged["sentiment_label"] = merged["sentiment_label"].astype("string").str.lower()
    grouped = merged.groupby(["platform", "domain"], dropna=False)

    def _share(s: pd.Series, label: str) -> float:
        return round(float((s == label).mean()), 4)

    out = grouped.apply(
        lambda g: pd.Series(
            {
                "n_enriched": len(g),
                "pos_share": _share(g["sentiment_label"], "positive"),
                "neu_share": _share(g["sentiment_label"], "neutral"),
                "neg_share": _share(g["sentiment_label"], "negative"),
                "mean_compound": round(float(pd.to_numeric(g["sentiment_compound"], errors="coerce").mean()), 4),
            }
        ),
        include_groups=False,
    ).reset_index()
    return out.sort_values(["platform", "domain"]).reset_index(drop=True)


def _iter_entities(value: Any):
    """Yield ``(text, label)`` from a ``mentioned_entities`` cell (list of dicts)."""
    if value is None:
        return
    seq = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(seq, (list, tuple)):
        return
    for item in seq:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            label = str(item.get("label") or item.get("type") or "").strip()
        else:
            text, label = str(item).strip(), ""
        if text:
            yield text, label


def entity_frequency(
    linked: pd.DataFrame,
    enrichment: pd.DataFrame,
    markets: pd.DataFrame,
    *,
    top_n: int = 25,
) -> pd.DataFrame:
    """Top named entities (by mention count) per domain, from accepted posts."""
    cols = ["domain", "entity", "label", "mentions"]
    acc = accepted_links(linked)
    if acc.empty or enrichment.empty:
        return pd.DataFrame(columns=cols)
    acc = _with_domain(acc, markets)
    enr = enrichment[["post_id", "mentioned_entities"]].drop_duplicates("post_id")
    merged = acc.merge(enr, on="post_id", how="inner")
    counters: dict[str, Counter] = {}
    labels: dict[tuple[str, str], str] = {}
    for _, row in merged.iterrows():
        domain = row["domain"]
        for text, label in _iter_entities(row["mentioned_entities"]):
            counters.setdefault(domain, Counter())[text] += 1
            labels[(domain, text)] = label
    rows: list[dict[str, Any]] = []
    for domain, counter in counters.items():
        for entity, mentions in counter.most_common(top_n):
            rows.append({"domain": domain, "entity": entity,
                         "label": labels.get((domain, entity), ""), "mentions": int(mentions)})
    return pd.DataFrame(rows, columns=cols)


def topic_summary(topics: pd.DataFrame) -> pd.DataFrame:
    """Project the LDA ``topics`` table onto a compact summary."""
    cols = ["topic_id", "label", "top_terms", "weight"]
    if topics.empty:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame()
    out["topic_id"] = topics.get("topic_id")
    out["label"] = topics.get("label")
    out["top_terms"] = topics.get("top_terms")
    out["weight"] = pd.to_numeric(topics.get("weight"), errors="coerce").round(4)
    return out.sort_values("weight", ascending=False).reset_index(drop=True)

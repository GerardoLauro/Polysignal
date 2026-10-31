"""Canonical Traccia_2 §2 dataset export.

Produces a single documented bundle under ``data/dataset_export/`` that maps
1:1 to the assignment's Section-2 schema, joining the normalized parquet
tables (contracts, prices, social posts + enrichment + comments, links) into
inspectable Parquet + CSV-sample artifacts plus a ``DATA_DICTIONARY.md``.

The functions are pure transforms over DataFrames (no settings reads); the
:func:`export_dataset` orchestrator does the disk I/O. Reading is via the
project's ``read_dataframe`` so timestamps stay ISO-8601 UTC strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# --- Section-2 canonical column orders -------------------------------------

CONTRACT_COLUMNS = [
    "market_id",
    "question",
    "category",
    "outcomes",
    "resolution_outcome",
    "resolution_date",
    "creation_date",
    "close_date",
    "url",
]

PRICE_POINT_COLUMNS = ["market_id", "outcome", "timestamp", "price", "volume"]

# Section-2 social post fields (provenance + author + content + engagement).
SOCIAL_POST_COLUMNS = [
    "post_id",
    "platform",
    "market_id",
    "category",
    "search_query",
    "query_type",
    "collected_at",
    "url",
    "author_id",
    "author_name",
    "author_url",
    "author_followers",
    "published_at",
    "text",
    "language",
    "hashtags",
    "mentioned_entities",
    "sentiment_label",
    "sentiment_compound",
    "like_count",
    "reply_count",
    "repost_count",
    "view_count",
    "comment_sample_count",
]

LINK_COLUMNS = [
    "link_id",
    "market_id",
    "post_id",
    "platform",
    "method",
    "relevance_label",
    "relevance_score",
    "confidence",
    "constraint_match",
]

COMMENT_COLUMNS = [
    "post_id",
    "platform",
    "comment_id",
    "parent_comment_id",
    "author_id",
    "author_url",
    "text",
    "published_at",
    "like_count",
    "reply_count",
    "language",
]


@dataclass
class ExportResult:
    """Paths + row counts for one canonical export run."""

    output_dir: Path
    written: dict[str, Path] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    partitioned: dict[str, list[str]] = field(default_factory=dict)


# --- pure transforms --------------------------------------------------------


def build_contracts(markets: pd.DataFrame) -> pd.DataFrame:
    """Project the markets table onto the Section-2 contract schema."""
    out = pd.DataFrame()
    out["market_id"] = markets["market_id"]
    out["question"] = markets.get("question")
    out["category"] = markets.get("thematic_category")
    out["outcomes"] = markets.get("outcomes")
    out["resolution_outcome"] = markets.get("resolution")
    out["resolution_date"] = markets.get("resolved_at")
    out["creation_date"] = markets.get("created_at")
    out["close_date"] = markets.get("close_time")
    out["url"] = markets.get("source_url")
    return out[CONTRACT_COLUMNS]


def build_price_points(prices: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Map price rows to the §2 schema, resolving ``outcome`` to its label."""
    label_map = (
        outcomes[["market_id", "outcome_id", "label"]]
        .drop_duplicates(["market_id", "outcome_id"])
        if not outcomes.empty
        else pd.DataFrame(columns=["market_id", "outcome_id", "label"])
    )
    merged = prices.merge(label_map, on=["market_id", "outcome_id"], how="left")
    out = pd.DataFrame()
    out["market_id"] = merged["market_id"]
    out["outcome"] = merged["label"].fillna(merged["outcome_id"])
    out["timestamp"] = merged["timestamp"]
    out["price"] = merged["price"]
    out["volume"] = merged.get("volume")
    return out[PRICE_POINT_COLUMNS]


def _effective_label(row: pd.Series) -> Any:
    final = row.get("final_relevance_label")
    return final if final not in (None, "") and pd.notna(final) else row.get("relevance_label")


def _effective_score(row: pd.Series) -> Any:
    final = row.get("final_relevance_score")
    return final if final is not None and pd.notna(final) else row.get("relevance_score")


def resolve_best_links(linked: pd.DataFrame) -> pd.DataFrame:
    """One best (post -> market) link per post.

    "Best" = highest effective relevance score among RELEVANT/WEAKLY_RELEVANT
    rows; falls back to the highest-scoring row when none are accepted. Used to
    populate the resolved ``market_id``/``category`` on each social post.
    """
    if linked.empty:
        return pd.DataFrame(columns=["post_id", "market_id", "eff_label", "eff_score"])
    work = linked.copy()
    work["eff_label"] = work.apply(_effective_label, axis=1)
    work["eff_score"] = pd.to_numeric(
        work.apply(_effective_score, axis=1), errors="coerce"
    ).fillna(0.0)
    accepted_mask = work["eff_label"].astype(str).isin(["RELEVANT", "WEAKLY_RELEVANT"])
    work["_accepted"] = accepted_mask.astype(int)
    work = work.sort_values(
        ["post_id", "_accepted", "eff_score"], ascending=[True, False, False]
    )
    best = work.drop_duplicates(subset=["post_id"], keep="first")
    return best[["post_id", "market_id", "eff_label", "eff_score"]]


def build_links(linked: pd.DataFrame) -> pd.DataFrame:
    """Project linked_posts onto the §2 contract<->post mapping schema."""
    if linked.empty:
        return pd.DataFrame(columns=LINK_COLUMNS)
    out = pd.DataFrame()
    out["link_id"] = linked.get("link_id")
    out["market_id"] = linked["market_id"]
    out["post_id"] = linked["post_id"]
    out["platform"] = linked.get("platform")
    out["method"] = linked.get("final_method").fillna(linked.get("method")) if "final_method" in linked else linked.get("method")
    out["relevance_label"] = linked.apply(_effective_label, axis=1)
    out["relevance_score"] = linked.apply(_effective_score, axis=1)
    out["confidence"] = linked.get("final_confidence").fillna(linked.get("confidence")) if "final_confidence" in linked else linked.get("confidence")
    out["constraint_match"] = linked.get("constraint_match")
    return out[LINK_COLUMNS]


def build_social_posts(
    posts: pd.DataFrame,
    enrichment: pd.DataFrame,
    best_links: pd.DataFrame,
    *,
    category_by_market: dict[str, Any] | None = None,
    comment_counts: dict[str, int] | None = None,
    deduplicate: bool = True,
) -> pd.DataFrame:
    """Join posts + enrichment + resolved link into the §2 social schema."""
    work = posts.copy()
    if deduplicate and "is_duplicate" in work.columns:
        work = work[~work["is_duplicate"].fillna(False).astype(bool)]
    # Resolve linked market_id / category (best link per post).
    if not best_links.empty:
        work = work.merge(
            best_links[["post_id", "market_id"]].rename(columns={"market_id": "_linked_market"}),
            on="post_id",
            how="left",
        )
    else:
        work["_linked_market"] = None
    resolved_market = work.get("_linked_market")
    cat_map = category_by_market or {}
    # Enrichment join (hashtags / entities / language / sentiment).
    enrich_cols = [
        "post_id",
        "hashtags",
        "mentioned_entities",
        "language",
        "sentiment_label",
        "sentiment_compound",
    ]
    have = [c for c in enrich_cols if c in enrichment.columns]
    if not enrichment.empty and "post_id" in have:
        work = work.merge(
            enrichment[have].drop_duplicates("post_id"),
            on="post_id",
            how="left",
            suffixes=("", "_enr"),
        )
    counts = comment_counts or {}
    out = pd.DataFrame()
    out["post_id"] = work["post_id"]
    out["platform"] = work.get("platform")
    out["market_id"] = resolved_market
    out["category"] = [cat_map.get(m) for m in resolved_market] if resolved_market is not None else None
    out["search_query"] = work.get("query")
    out["query_type"] = work.get("query_type")
    out["collected_at"] = work.get("collected_at")
    out["url"] = work.get("url")
    out["author_id"] = work.get("author_id")
    out["author_name"] = work.get("author_name")
    out["author_url"] = work.get("author_url")
    out["author_followers"] = work.get("author_followers")
    out["published_at"] = work.get("created_at")
    out["text"] = work.get("text")
    out["language"] = work.get("language_enr") if "language_enr" in work else work.get("language")
    out["hashtags"] = work.get("hashtags")
    out["mentioned_entities"] = work.get("mentioned_entities")
    out["sentiment_label"] = work.get("sentiment_label")
    out["sentiment_compound"] = work.get("sentiment_compound")
    # Unified §2 engagement, falling back to platform-native columns.
    out["like_count"] = work.get("like_count")
    if "like_count" not in work or out["like_count"].isna().all():
        out["like_count"] = work.get("upvotes")
    out["reply_count"] = work.get("reply_count")
    if "reply_count" not in work or out["reply_count"].isna().all():
        out["reply_count"] = work.get("comments_count")
    out["repost_count"] = work.get("repost_count")
    if "repost_count" not in work or out["repost_count"].isna().all():
        out["repost_count"] = work.get("shares_count")
    out["view_count"] = work.get("view_count")
    if "view_count" not in work or out["view_count"].isna().all():
        out["view_count"] = work.get("views_count")
    out["comment_sample_count"] = [int(counts.get(p, 0)) for p in work["post_id"]]
    return out[SOCIAL_POST_COLUMNS]


# --- writers ----------------------------------------------------------------


def _write_table(
    df: pd.DataFrame,
    path: Path,
    *,
    partition_cols: list[str] | None = None,
) -> list[str]:
    """Write ``df`` to ``path``; if ``partition_cols`` given, write a Hive-style
    partitioned dataset directory instead of a single file. Returns the
    partition columns actually used (empty when monolithic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    usable = [c for c in (partition_cols or []) if c in df.columns and df[c].notna().any()]
    if usable:
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Partition values must be stringy and non-null for Hive layout.
        frame = df.copy()
        for col in usable:
            frame[col] = frame[col].fillna("__unknown__").astype(str)
        dataset_dir = path.with_suffix("")  # directory next to the .parquet name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        pq.write_to_dataset(
            pa.Table.from_pandas(frame, preserve_index=False),
            root_path=str(dataset_dir),
            partition_cols=usable,
        )
        return usable
    df.to_parquet(path, index=False)
    return []


def _write_csv_sample(df: pd.DataFrame, path: Path, *, max_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.head(max_rows).to_csv(path, index=False)


def export_dataset(
    *,
    markets: pd.DataFrame,
    outcomes: pd.DataFrame,
    prices: pd.DataFrame,
    posts: pd.DataFrame,
    enrichment: pd.DataFrame,
    linked: pd.DataFrame,
    comments: pd.DataFrame | None,
    output_dir: Path,
    sample_rows: int = 100,
    partition: bool = False,
) -> ExportResult:
    """Build and write the four §2 entities + comments + CSV samples + dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result = ExportResult(output_dir=output_dir)

    category_by_market = (
        dict(zip(markets["market_id"], markets.get("thematic_category"), strict=False))
        if not markets.empty
        else {}
    )
    comment_counts: dict[str, int] = {}
    if comments is not None and not comments.empty and "post_id" in comments.columns:
        comment_counts = comments.groupby("post_id").size().to_dict()

    best_links = resolve_best_links(linked)
    contracts = build_contracts(markets)
    price_points = build_price_points(prices, outcomes)
    social_posts = build_social_posts(
        posts,
        enrichment,
        best_links,
        category_by_market=category_by_market,
        comment_counts=comment_counts,
    )
    links = build_links(linked)
    comments_out = (
        comments[[c for c in COMMENT_COLUMNS if c in comments.columns]].copy()
        if comments is not None and not comments.empty
        else pd.DataFrame(columns=COMMENT_COLUMNS)
    )

    tables = {
        "contracts": (contracts, ["category"] if partition else None),
        "price_points": (price_points, None),
        "social_posts": (social_posts, ["platform"] if partition else None),
        "links": (links, ["platform"] if partition else None),
        "comments": (comments_out, None),
    }
    for name, (df, part_cols) in tables.items():
        target = output_dir / f"{name}.parquet"
        used = _write_table(df, target, partition_cols=part_cols)
        result.written[name] = target
        result.row_counts[name] = len(df)
        if used:
            result.partitioned[name] = used
        _write_csv_sample(df, output_dir / f"{name}_sample.csv", max_rows=sample_rows)

    dict_path = output_dir / "DATA_DICTIONARY.md"
    dict_path.write_text(_render_data_dictionary(result), encoding="utf-8")
    result.written["data_dictionary"] = dict_path
    return result


# --- data dictionary --------------------------------------------------------

_FIELD_DOC: dict[str, tuple[str, str, str]] = {
    # column -> (type, nullability, §2 mapping / source)
    "market_id": ("string", "required", "Polymarket condition_id; join key across all entities"),
    "question": ("string", "required", "§2 contract question text"),
    "category": ("string", "nullable", "thematic_category (politics/finance/sports)"),
    "outcomes": ("json-list", "required", "§2 outcome labels for the contract"),
    "resolution_outcome": ("string", "nullable", "§2 resolved outcome (YES/NO/...)"),
    "resolution_date": ("iso8601-utc", "nullable", "§2 resolution date"),
    "creation_date": ("iso8601-utc", "required", "§2 contract creation date"),
    "close_date": ("iso8601-utc", "nullable", "§2 contract close date"),
    "url": ("string", "nullable", "§2 contract/post source URL"),
    "outcome": ("string", "required", "§2 price-series outcome label"),
    "timestamp": ("iso8601-utc", "required", "§2 price-series timestamp"),
    "price": ("float[0,1]", "required", "§2 implied probability"),
    "volume": ("float", "nullable", "§2 traded volume"),
    "post_id": ("string", "required", "platform-native post id"),
    "platform": ("string", "required", "reddit | telegram | twitter"),
    "search_query": ("string", "nullable", "§2 provenance: query that surfaced the post"),
    "query_type": ("string", "nullable", "provenance: query category"),
    "collected_at": ("iso8601-utc", "required", "§2 provenance: collection timestamp"),
    "author_id": ("string", "nullable", "§2 author id/handle"),
    "author_name": ("string", "nullable", "§2 author display name"),
    "author_url": ("string", "nullable", "§2 author profile URL (synthesized per platform)"),
    "author_followers": ("int", "nullable", "§2 author follower count (platform-dependent)"),
    "published_at": ("iso8601-utc", "nullable", "§2 content published_at"),
    "text": ("string", "required", "§2 content text"),
    "language": ("string", "nullable", "§2 detected language (post_enrichment)"),
    "hashtags": ("json-list", "nullable", "§2 hashtags (post_enrichment)"),
    "mentioned_entities": ("json-list", "nullable", "§2 NER entities (post_enrichment)"),
    "sentiment_label": ("string", "nullable", "sentiment (post_enrichment)"),
    "sentiment_compound": ("float[-1,1]", "nullable", "VADER compound (post_enrichment)"),
    "like_count": ("int", "nullable", "§2 engagement (X likes / Reddit ups / TG reactions)"),
    "reply_count": ("int", "nullable", "§2 engagement (replies / num_comments)"),
    "repost_count": ("int", "nullable", "§2 engagement (retweets / shares)"),
    "view_count": ("int", "nullable", "§2 engagement (views)"),
    "comment_sample_count": ("int", "required", "count of sampled comments in comments.parquet"),
    "link_id": ("string", "required", "link record id"),
    "method": ("string", "nullable", "linking method (deterministic/LLM)"),
    "relevance_label": ("string", "nullable", "effective relevance verdict"),
    "relevance_score": ("float[0,1]", "nullable", "effective relevance score"),
    "confidence": ("float[0,1]", "nullable", "linker confidence"),
    "constraint_match": ("string", "nullable", "constraint match category"),
    "comment_id": ("string", "required", "platform-native comment id"),
    "parent_comment_id": ("string", "nullable", "parent comment id (thread structure)"),
}


def _entity_table(name: str, columns: list[str], rows: int) -> str:
    lines = [f"### `{name}.parquet` ({rows} rows)", "", "| Column | Type | Nullability | §2 mapping / source |", "| --- | --- | --- | --- |"]
    for col in columns:
        typ, nul, doc = _FIELD_DOC.get(col, ("string", "nullable", ""))
        lines.append(f"| `{col}` | {typ} | {nul} | {doc} |")
    lines.append("")
    return "\n".join(lines)


def _render_data_dictionary(result: ExportResult) -> str:
    rc = result.row_counts
    header = [
        "# PolySignal canonical dataset — data dictionary",
        "",
        "Generated by `polysignal dataset export`. Maps 1:1 to the Traccia_2 §2",
        "schema. Timestamps are ISO-8601 UTC strings; list columns are JSON arrays.",
        "",
        "| Entity | File | Rows |",
        "| --- | --- | --- |",
        f"| Contracts | `contracts.parquet` | {rc.get('contracts', 0)} |",
        f"| Price series | `price_points.parquet` | {rc.get('price_points', 0)} |",
        f"| Social posts | `social_posts.parquet` | {rc.get('social_posts', 0)} |",
        f"| Links | `links.parquet` | {rc.get('links', 0)} |",
        f"| Comments | `comments.parquet` | {rc.get('comments', 0)} |",
        "",
    ]
    if result.partitioned:
        header.append("**Partitioned outputs:** " + ", ".join(
            f"`{k}` by {'/'.join(v)}" for k, v in result.partitioned.items()
        ))
        header.append("")
    body = [
        _entity_table("contracts", CONTRACT_COLUMNS, rc.get("contracts", 0)),
        _entity_table("price_points", PRICE_POINT_COLUMNS, rc.get("price_points", 0)),
        _entity_table("social_posts", SOCIAL_POST_COLUMNS, rc.get("social_posts", 0)),
        _entity_table("links", LINK_COLUMNS, rc.get("links", 0)),
        _entity_table("comments", COMMENT_COLUMNS, rc.get("comments", 0)),
    ]
    return "\n".join(header) + "\n" + "\n".join(body)


__all__ = [
    "COMMENT_COLUMNS",
    "CONTRACT_COLUMNS",
    "LINK_COLUMNS",
    "PRICE_POINT_COLUMNS",
    "SOCIAL_POST_COLUMNS",
    "ExportResult",
    "build_contracts",
    "build_links",
    "build_price_points",
    "build_social_posts",
    "export_dataset",
    "resolve_best_links",
]

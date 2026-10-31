"""Run NLP enrichment (sentiment + NER + hashtags + language) over posts.

Produces the derived ``post_enrichment.parquet`` table (one row per post)
without rewriting the large raw ``social_posts.parquet``. Streams the
corpus in Parquet row-group batches so memory stays bounded even on the
~1.3M-row Telegram corpus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from polysignal.nlp.extract import (
    detect_language,
    extract_hashtags,
    extract_mentions,
    get_entity_backend,
)
from polysignal.nlp.sentiment import get_sentiment_analyzer
from polysignal.storage.parquet_store import write_dataframe

logger = logging.getLogger(__name__)

_READ_COLUMNS = ["post_id", "platform", "text", "title", "language"]


@dataclass
class EnrichmentSummary:
    posts_in: int = 0
    posts_enriched: int = 0
    sentiment_method: str = ""
    entity_method: str = ""
    label_counts: dict[str, int] = field(default_factory=dict)
    total_hashtags: int = 0
    total_entities: int = 0
    languages: dict[str, int] = field(default_factory=dict)
    out_path: Path | None = None


def _clean_str(value: object) -> str:
    """Coerce a parquet/pandas cell to a plain string (NaN/None → "")."""
    if value is None:
        return ""
    if isinstance(value, float):  # pandas NaN for missing text/title
        return ""
    return str(value)


def _combined_text(row: dict) -> str:
    title = _clean_str(row.get("title"))
    text = _clean_str(row.get("text"))
    if title and title not in text:
        return f"{title}. {text}".strip()
    return text


def run_enrichment(
    posts_path: Path,
    out_path: Path,
    *,
    sentiment_prefer: str = "auto",
    entity_prefer: str = "auto",
    limit: int | None = None,
    batch_size: int = 20_000,
    progress: bool = True,
) -> EnrichmentSummary:
    """Enrich posts at ``posts_path`` → ``out_path`` (post_enrichment parquet)."""
    if not posts_path.exists():
        raise FileNotFoundError(f"social posts parquet not found: {posts_path}")

    analyzer = get_sentiment_analyzer(sentiment_prefer)
    entities = get_entity_backend(entity_prefer)
    summary = EnrichmentSummary(
        sentiment_method=analyzer.method, entity_method=entities.name
    )
    now = datetime.now(tz=timezone.utc).isoformat()

    pf = pq.ParquetFile(posts_path)
    available = [c for c in _READ_COLUMNS if c in pf.schema_arrow.names]
    rows: list[dict] = []
    stop = False
    for batch in pf.iter_batches(batch_size=batch_size, columns=available):
        bdf = batch.to_pandas()
        for row in bdf.to_dict(orient="records"):
            text = _combined_text(row)
            sent = analyzer.score(text)
            ents = entities.entities(text)
            tags = extract_hashtags(text)
            lang = row.get("language") or detect_language(text)
            rows.append(
                {
                    "post_id": row.get("post_id"),
                    "platform": row.get("platform"),
                    "language": lang,
                    "hashtags": tags,
                    "mentions": extract_mentions(text),
                    "mentioned_entities": [e.as_dict() for e in ents],
                    **sent.as_row(),
                    "entity_method": entities.name,
                    "enriched_at": now,
                }
            )
            summary.posts_in += 1
            summary.label_counts[sent.label.value] = (
                summary.label_counts.get(sent.label.value, 0) + 1
            )
            summary.total_hashtags += len(tags)
            summary.total_entities += len(ents)
            if lang:
                summary.languages[lang] = summary.languages.get(lang, 0) + 1
            if limit is not None and summary.posts_in >= limit:
                stop = True
                break
        if progress and summary.posts_in % 100_000 < batch_size:
            logger.info("enrichment: processed %d posts", summary.posts_in)
        if stop:
            break

    out_df = pd.DataFrame(rows)
    summary.posts_enriched = len(out_df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_dataframe(out_df, out_path)
    summary.out_path = out_path
    return summary


__all__ = ["EnrichmentSummary", "run_enrichment"]

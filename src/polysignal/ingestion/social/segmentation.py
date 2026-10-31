"""Deterministic multi-topic digest segmentation for Telegram/social posts.

Branch 05G adds a lightweight, conservative segmenter that splits long
multi-topic digest posts (a single Telegram message that bundles many
independent news items) into smaller semantic segments before candidate
generation and scoring.

The original ``social_posts.parquet`` is **not** modified. Segments live in
a derived dataset (``social_post_segments.parquet``) keyed back to the
parent post via ``parent_post_id``. Each segment gets a stable
``segment_id`` of the form::

    <parent_post_id>#seg000

so auditability flows from a linked row back to the exact paragraph used
for scoring.

The segmenter is intentionally minimal — it does not attempt NLP topic
modeling. It splits on blank-line boundaries and on lines that look like
bullet markers (a leading emoji, ``*``/``-``/``•``, or a numbered list
item). Short posts pass through as one segment.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from polysignal.domain import Platform, SocialPostRecord, SocialPostSegment
from polysignal.storage.parquet_store import read_parquet, write_parquet

# A "long" post is one that is plausibly a multi-topic digest. Below this
# we always return one segment — the original post.
_MIN_LENGTH_FOR_SPLIT = 320
# Minimum char length of a segment after trimming. Sub-threshold fragments
# are dropped (they cannot carry meaningful anchor evidence).
_MIN_SEGMENT_LENGTH = 25
# Hard cap to avoid unbounded blow-up from pathological inputs.
_MAX_SEGMENTS = 40

_BLANK_LINE_RE = re.compile(r"\n\s*\n")
_BULLET_LINE_RE = re.compile(
    r"^(?:[\s -⁯]*"
    r"(?:[\*\-•‣◦⁃∙]"
    r"|\d{1,2}[\.\)]"
    r"|[\U0001F300-\U0001FAFF☀-➿←-⇿⬀-⯿])"
    r"\s+)",
    re.UNICODE,
)
# Digest markers that, when repeated, strongly suggest multi-topic content.
_DIGEST_MARKERS = (
    "latest",
    "update",
    "breaking",
    "new",
    "today",
    "ipo",
    "huge",
    "insight",
    "opinion",
    "adoption",
)
# Header lines we treat as a non-content cover and drop from the first
# segment if present (e.g. "Catch up on the news over the last 24 hours!").
_HEADER_PHRASES = (
    "catch up on the news",
    "news over the last",
    "daily digest",
    "top stories",
    "morning briefing",
    "evening briefing",
)


@dataclass(frozen=True)
class PostSegment:
    """One semantic segment derived from a parent social post.

    ``segment_id`` is stable: same parent + same index → same id.
    """

    segment_id: str
    parent_post_id: str
    platform: str
    source: str | None
    segment_index: int
    segment_text: str
    created_at: datetime
    url: str | None
    raw_path: str | None
    text_length: int
    segmentation_method: str

    def to_record(self, *, base: SocialPostRecord) -> SocialPostRecord:
        """Return a ``SocialPostRecord`` carrying this segment's text.

        ``post_id`` is replaced by ``segment_id`` so downstream pipelines
        treat each segment as an independent row, while ``raw_path`` and
        the original retrieval columns (``query``, ``query_type``,
        ``market_id_candidate``) are preserved verbatim from ``base``.
        """
        return base.model_copy(
            update={
                "post_id": self.segment_id,
                "text": self.segment_text,
                "text_length": self.text_length,
                # ``source_id`` keeps the parent's stable platform id so a
                # join back to social_posts.parquet still works.
                "source_id": base.source_id,
            }
        )


def _looks_like_digest(text: str) -> bool:
    """Heuristic: is this likely a multi-topic digest worth splitting?

    A post is treated as a digest when *either* the text is long enough to
    plausibly carry several stories *and* shows structural cues (blank
    lines, bullets, repeated digest markers), or when it has three or
    more independent bullet/emoji-prefixed lines regardless of length.
    """
    bullet_lines = sum(1 for ln in text.splitlines() if _BULLET_LINE_RE.match(ln))
    if bullet_lines >= 3:
        return True
    if len(text) < _MIN_LENGTH_FOR_SPLIT:
        return False
    blank_groups = len(_BLANK_LINE_RE.findall(text))
    if blank_groups >= 2:
        return True
    lowered = text.lower()
    repeated_markers = sum(lowered.count(marker) >= 2 for marker in _DIGEST_MARKERS)
    return repeated_markers >= 2


def _strip_header(first: str) -> str:
    """Drop a single leading "Catch up on the news..." cover line."""
    lowered = first.strip().lower()
    if not lowered:
        return first
    for phrase in _HEADER_PHRASES:
        if phrase in lowered and len(first.strip()) <= 120:
            return ""
    return first


def _split_blank_lines(text: str) -> list[str]:
    return [chunk.strip() for chunk in _BLANK_LINE_RE.split(text) if chunk.strip()]


def _split_bullets(text: str) -> list[str]:
    """Split on lines that start with a bullet/emoji marker.

    A new segment starts whenever a bullet line is encountered; preceding
    non-bullet lines are attached to the previous segment.
    """
    out: list[str] = []
    buffer: list[str] = []
    for line in text.splitlines():
        if _BULLET_LINE_RE.match(line) and buffer:
            joined = "\n".join(buffer).strip()
            if joined:
                out.append(joined)
            buffer = [line]
        else:
            buffer.append(line)
    if buffer:
        joined = "\n".join(buffer).strip()
        if joined:
            out.append(joined)
    return out


def _segment_text(text: str) -> tuple[list[str], str]:
    """Return ``(segments, method)`` for one input string."""
    raw = (text or "").strip()
    if not raw:
        return [], "empty"
    if not _looks_like_digest(raw):
        return [raw], "single"

    parts = _split_blank_lines(raw)
    method = "blank-line"
    if len(parts) < 2:
        parts = _split_bullets(raw)
        method = "bullet"
    if parts:
        head = _strip_header(parts[0])
        parts = ([head] if head else []) + parts[1:]

    out: list[str] = []
    for part in parts:
        for sub in _split_bullets(part):
            cleaned = sub.strip()
            if len(cleaned) >= _MIN_SEGMENT_LENGTH:
                out.append(cleaned)
            if len(out) >= _MAX_SEGMENTS:
                break
        if len(out) >= _MAX_SEGMENTS:
            break
    if not out:
        return [raw], "single"
    return out, method


def segment_post(record: SocialPostRecord) -> list[PostSegment]:
    """Split one post record into one or more :class:`PostSegment` rows.

    Short or non-digest posts come back as a single segment whose
    ``segment_text`` equals the original text. Empty/media-only rows
    return an empty list.
    """
    if record.created_at is None or record.temporal_valid_candidate is False:
        return []
    segments, method = _segment_text(record.text or "")
    if not segments:
        return []
    return [
        PostSegment(
            segment_id=f"{record.post_id}#seg{idx:03d}",
            parent_post_id=record.post_id,
            platform=record.platform.value,
            source=record.subreddit_or_channel,
            segment_index=idx,
            segment_text=text,
            created_at=record.created_at,
            url=record.url,
            raw_path=record.raw_path,
            text_length=len(text),
            segmentation_method=method if len(segments) > 1 else "single",
        )
        for idx, text in enumerate(segments)
    ]


def segment_records(records: Iterable[SocialPostRecord]) -> list[PostSegment]:
    """Apply :func:`segment_post` over an iterable of records."""
    out: list[PostSegment] = []
    for record in records:
        out.extend(segment_post(record))
    return out


def segments_to_records(
    segments: Sequence[PostSegment],
    base_lookup: dict[str, SocialPostRecord],
) -> list[SocialPostRecord]:
    """Materialize segments as ``SocialPostRecord`` rows for scoring."""
    out: list[SocialPostRecord] = []
    for segment in segments:
        base = base_lookup.get(segment.parent_post_id)
        if base is None:
            continue
        out.append(segment.to_record(base=base))
    return out


def to_segment_models(segments: Sequence[PostSegment]) -> list[SocialPostSegment]:
    """Convert :class:`PostSegment` dataclasses to parquet-friendly models."""
    out: list[SocialPostSegment] = []
    for segment in segments:
        try:
            platform = Platform(segment.platform)
        except ValueError:
            platform = Platform.TELEGRAM
        out.append(
            SocialPostSegment(
                segment_id=segment.segment_id,
                parent_post_id=segment.parent_post_id,
                platform=platform,
                source=segment.source,
                segment_index=segment.segment_index,
                segment_text=segment.segment_text,
                created_at=segment.created_at,
                url=segment.url,
                raw_path=segment.raw_path,
                text_length=segment.text_length,
                segmentation_method=segment.segmentation_method,
            )
        )
    return out


def write_segments(segments: Sequence[PostSegment], path: Path) -> Path:
    """Write segments to a parquet file at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(to_segment_models(segments), path)
    return path


def read_segments(path: Path) -> list[SocialPostSegment]:
    """Read a segments parquet file back into :class:`SocialPostSegment`."""
    if not path.exists():
        return []
    return read_parquet(path, SocialPostSegment)


def segment_models_to_records(
    segments: Sequence[SocialPostSegment],
    base_lookup: dict[str, SocialPostRecord],
) -> list[SocialPostRecord]:
    """Materialize parquet segment rows as scorer-facing :class:`SocialPostRecord`."""
    out: list[SocialPostRecord] = []
    for segment in segments:
        base = base_lookup.get(segment.parent_post_id)
        if base is None:
            continue
        out.append(
            base.model_copy(
                update={
                    "post_id": segment.segment_id,
                    "text": segment.segment_text,
                    "text_length": segment.text_length,
                }
            )
        )
    return out


__all__ = [
    "PostSegment",
    "read_segments",
    "segment_models_to_records",
    "segment_post",
    "segment_records",
    "segments_to_records",
    "to_segment_models",
    "write_segments",
]

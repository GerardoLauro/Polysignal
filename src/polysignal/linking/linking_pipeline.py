"""High-level Branch 05 pipeline: candidates → scorer → ``LinkedPost`` rows.

The pipeline is intentionally thin. It does three things:

1. Decide which (market, post) pairs to score (see
   :mod:`polysignal.linking.candidates`).
2. Run the deterministic
   :class:`polysignal.linking.relevance.RelevanceScorer` on each pair.
3. Optionally verify borderline rows with the cached Branch 05C LLM judge.
4. Wrap each verdict in a :class:`polysignal.domain.LinkedPost` row,
   ready for parquet export.

All policy that doesn't belong in the scorer (label thresholds,
constraint penalties, etc.) is centralized in
:class:`RelevanceConfig`. Production callers pass that config in;
tests and the fixture demo use the defaults.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from polysignal.domain import (
    LinkedPost,
    LinkMethod,
    Market,
    SocialPost,
    SocialPostRecord,
)
from polysignal.linking.candidates import (
    CandidateMode,
    CandidatePair,
    build_pairs_cross_product,
    build_pairs_from_records,
)
from polysignal.linking.embeddings import EmbeddingProvider
from polysignal.linking.llm_assisted import LLMJudgeResult, LLMRelevanceJudge
from polysignal.linking.relevance import (
    RelevanceConfig,
    RelevanceJudgement,
    RelevanceScorer,
)
from polysignal.storage.parquet_store import read_parquet, write_parquet


@dataclass(frozen=True)
class LinkingRunResult:
    """Lightweight summary of one linking pass."""

    pairs_scored: int
    linked_rows: int
    output_path: Path | None
    by_label: dict[str, int]
    used_fixtures: bool


def _link_id(market_id: str, post_id: str, query: str | None) -> str:
    """Stable, deterministic id derived from the triple."""
    seed = f"{market_id}|{post_id}|{query or ''}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:24]


def _judgement_to_linked_post(
    pair: CandidatePair,
    judgement: RelevanceJudgement,
    *,
    now: datetime,
    llm_result: LLMJudgeResult | None = None,
    llm_judge: LLMRelevanceJudge | None = None,
    segment_lookup: dict[str, tuple[str, int, str]] | None = None,
) -> LinkedPost:
    text = pair.post.text or ""
    preview = text.replace("\n", " ").strip()[:240]
    segment_info = (segment_lookup or {}).get(pair.post.post_id)
    final_label = (
        llm_result.final_relevance_label if llm_result is not None else judgement.relevance_label
    )
    final_score = (
        llm_result.final_relevance_score if llm_result is not None else judgement.relevance_score
    )
    final_confidence = (
        llm_result.final_confidence if llm_result is not None else judgement.confidence
    )
    final_constraint = (
        llm_result.final_constraint_match if llm_result is not None else judgement.constraint_match
    )
    final_method = (
        llm_result.final_method if llm_result is not None else LinkMethod.DETERMINISTIC_BASELINE
    )
    llm_verdict = llm_result.verdict if llm_result is not None else None
    return LinkedPost(
        link_id=_link_id(pair.market.market_id, pair.post.post_id, pair.query),
        market_id=pair.market.market_id,
        post_id=pair.post.post_id,
        platform=pair.post.platform,
        query=pair.query,
        query_type=pair.query_type,
        relevance_score=judgement.relevance_score,
        relevance_label=judgement.relevance_label,
        confidence=judgement.confidence,
        method=LinkMethod.DETERMINISTIC_BASELINE,
        explanation=judgement.explanation,
        matched_entities=list(judgement.matched_entities),
        matched_terms=list(judgement.matched_terms),
        constraint_match=judgement.constraint_match,
        created_at=now,
        post_created_at=pair.post.created_at,
        market_question=pair.market.question,
        post_text_preview=preview,
        raw_post_path=pair.raw_post_path,
        signal_count=judgement.signal_count,
        anchor_score=judgement.anchor_score,
        constraint_score=judgement.constraint_score,
        domain_mismatch=judgement.domain_mismatch,
        rejection_reason=judgement.rejection_reason,
        semantic_similarity=judgement.semantic_similarity,
        embedding_model=judgement.embedding_model,
        embedding_enabled=judgement.embedding_enabled,
        llm_selected=llm_result.selected if llm_result is not None else False,
        llm_judged=llm_result.judged if llm_result is not None else False,
        llm_cache_hit=llm_result.cache_hit if llm_result is not None else False,
        llm_invalid_output=llm_result.invalid_output if llm_result is not None else False,
        llm_applied=llm_result.applied if llm_result is not None else False,
        llm_provider=llm_judge.provider if llm_judge is not None else None,
        llm_model=llm_judge.model_id if llm_judge is not None else None,
        llm_relevance_label=llm_verdict.relevance_label if llm_verdict is not None else None,
        llm_confidence=llm_verdict.confidence if llm_verdict is not None else None,
        llm_constraint_match=llm_verdict.constraint_match if llm_verdict is not None else None,
        llm_reason=llm_verdict.reason if llm_verdict is not None else None,
        final_relevance_label=final_label,
        final_relevance_score=final_score,
        final_confidence=final_confidence,
        final_constraint_match=final_constraint,
        final_method=final_method,
        parent_post_id=segment_info[0] if segment_info is not None else None,
        segment_id=pair.post.post_id if segment_info is not None else None,
        segment_index=segment_info[1] if segment_info is not None else None,
        segment_text_preview=(segment_info[2] if segment_info is not None else None),
    )


def score_pairs(
    pairs: Sequence[CandidatePair],
    *,
    config: RelevanceConfig | None = None,
    now: datetime | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    llm_judge: LLMRelevanceJudge | None = None,
    segment_lookup: dict[str, tuple[str, int, str]] | None = None,
) -> list[LinkedPost]:
    """Score every candidate pair and return a list of ``LinkedPost`` rows.

    ``segment_lookup`` (segment_id → (parent_post_id, segment_index,
    preview)) is optional Branch 05G audit metadata. When supplied the
    scorer-facing ``post_id`` is taken as a segment id; the output
    ``LinkedPost`` then carries ``parent_post_id`` / ``segment_id`` /
    ``segment_index`` / ``segment_text_preview`` so a downstream reader
    can trace the verdict back to the original Telegram message.
    """
    scorer = RelevanceScorer(config=config)
    moment = (now or datetime.now(tz=timezone.utc)).astimezone(timezone.utc)
    out: list[LinkedPost] = []
    for pair in pairs:
        verdict = scorer.score(
            pair.post,
            pair.market,
            query=pair.query,
            embedding_provider=embedding_provider,
        )
        llm_result = (
            llm_judge.judge(
                pair.post,
                pair.market,
                verdict,
                query=pair.query,
                query_type=pair.query_type,
            )
            if llm_judge is not None
            else None
        )
        out.append(
            _judgement_to_linked_post(
                pair,
                verdict,
                now=moment,
                llm_result=llm_result,
                llm_judge=llm_judge,
                segment_lookup=segment_lookup,
            )
        )
    return out


def run_linking_from_records(
    records: Sequence[SocialPostRecord],
    markets: Sequence[Market],
    *,
    config: RelevanceConfig | None = None,
    include_duplicates: bool = False,
    candidate_mode: CandidateMode = CandidateMode.AUTO,
    max_candidate_markets_per_post: int = 10,
    now: datetime | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    llm_judge: LLMRelevanceJudge | None = None,
) -> list[LinkedPost]:
    """Score the (market_id_candidate, post) pairs produced by Branch 04."""
    pairs = build_pairs_from_records(
        records,
        markets,
        include_duplicates=include_duplicates,
        candidate_mode=candidate_mode,
        max_candidate_markets_per_post=max_candidate_markets_per_post,
    )
    return score_pairs(
        pairs,
        config=config,
        now=now,
        embedding_provider=embedding_provider,
        llm_judge=llm_judge,
    )


def run_linking_cross_product(
    posts: Sequence[SocialPost],
    markets: Sequence[Market],
    *,
    config: RelevanceConfig | None = None,
    now: datetime | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    llm_judge: LLMRelevanceJudge | None = None,
) -> list[LinkedPost]:
    """Score every (market, post) pair — used by tests and the fixture demo."""
    pairs = build_pairs_cross_product(posts, markets)
    return score_pairs(
        pairs,
        config=config,
        now=now,
        embedding_provider=embedding_provider,
        llm_judge=llm_judge,
    )


def write_linked_posts(rows: Sequence[LinkedPost], path: Path) -> Path:
    """Write the linked-posts dataset to ``path``.

    The schema matches :class:`LinkedPost`. Empty input produces an empty
    parquet file — callers should check the row count themselves.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(rows, path)
    return path


def read_linked_posts(path: Path) -> list[LinkedPost]:
    """Read the linked-posts dataset back into model objects."""
    return read_parquet(path, LinkedPost)


def summarize_by_label(rows: Sequence[LinkedPost]) -> dict[str, int]:
    """Return a count of effective labels, falling back to the 05A/05B label."""
    counts: dict[str, int] = {}
    for row in rows:
        key = (row.final_relevance_label or row.relevance_label).value
        counts[key] = counts.get(key, 0) + 1
    return counts


__all__ = [
    "LinkingRunResult",
    "read_linked_posts",
    "run_linking_cross_product",
    "run_linking_from_records",
    "score_pairs",
    "summarize_by_label",
    "write_linked_posts",
]

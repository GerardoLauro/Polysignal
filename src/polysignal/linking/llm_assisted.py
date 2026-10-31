"""Optional cached LLM verifier for Branch 05C contract-post relevance.

The judge is deliberately narrow: it may refine borderline semantic
relevance labels, but it never infers stance, sentiment, or outcomes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from polysignal.domain import ConstraintMatch, Link, LinkMethod, Market, RelevanceLabel, SocialPost
from polysignal.linking.constraints import extract_market_constraints
from polysignal.linking.relevance import RelevanceJudgement
from polysignal.llm.client import LLMClient
from polysignal.llm.structured import parse_structured_response
from polysignal.storage.cache import JSONCache, stable_key

PROMPT_VERSION = "branch05c-v1"
_MAX_REASON_LENGTH = 180
_MAX_LIST_ITEMS = 8
_BORDERLINE_LABELS = {RelevanceLabel.WEAKLY_RELEVANT, RelevanceLabel.AMBIGUOUS}
_LABEL_RANK = {
    RelevanceLabel.UNRELATED: 0,
    RelevanceLabel.AMBIGUOUS: 1,
    RelevanceLabel.WEAKLY_RELEVANT: 2,
    RelevanceLabel.RELEVANT: 3,
}


class LLMRelevanceVerdict(BaseModel):
    """Validated relevance-only response expected from the LLM."""

    model_config = ConfigDict(extra="forbid")

    relevance_label: RelevanceLabel
    confidence: float = Field(ge=0.0, le=1.0)
    constraint_match: ConstraintMatch = ConstraintMatch.UNKNOWN
    reason: str = ""
    matched_market_elements: list[str] = Field(default_factory=list)
    missing_market_elements: list[str] = Field(default_factory=list)
    is_stance: Literal[False] = False

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: Any) -> float:
        number = float(value)
        return min(1.0, max(0.0, number))

    @field_validator("reason")
    @classmethod
    def _short_reason(cls, value: str) -> str:
        return value.strip()[:_MAX_REASON_LENGTH]

    @field_validator("matched_market_elements", "missing_market_elements")
    @classmethod
    def _short_lists(cls, values: list[str]) -> list[str]:
        return [item.strip()[:80] for item in values[:_MAX_LIST_ITEMS] if item.strip()]


@dataclass(frozen=True)
class LLMJudgePolicy:
    """Cost-control and conservative-merge thresholds."""

    borderline_only: bool = True
    min_score: float = 0.35
    max_score: float = 0.75
    low_confidence: float = 0.60
    apply_confidence: float = 0.85
    force_confidence: float = 0.95

    def should_judge(self, baseline: RelevanceJudgement) -> bool:
        if not self.borderline_only:
            return True
        return (
            baseline.relevance_label in _BORDERLINE_LABELS
            or self.min_score <= baseline.relevance_score <= self.max_score
            or baseline.confidence < self.low_confidence
        )


@dataclass
class LLMJudgeStats:
    considered: int = 0
    selected: int = 0
    calls: int = 0
    cache_hits: int = 0
    invalid_outputs: int = 0
    applied: int = 0
    label_changes: int = 0


@dataclass(frozen=True)
class LLMJudgeResult:
    selected: bool
    judged: bool
    cache_hit: bool
    invalid_output: bool
    applied: bool
    verdict: LLMRelevanceVerdict | None
    final_relevance_label: RelevanceLabel
    final_relevance_score: float
    final_confidence: float
    final_constraint_match: ConstraintMatch
    final_method: LinkMethod


class LLMRelevanceJudge:
    """Optional relevance verifier with deterministic cache and safe fallback."""

    def __init__(
        self,
        llm: LLMClient,
        cache: JSONCache,
        *,
        provider: str | None = None,
        model_id: str | None = None,
        policy: LLMJudgePolicy | None = None,
        dry_run: bool = False,
    ) -> None:
        self.llm = llm
        self.cache = cache
        self.provider = provider or type(llm).__name__
        self.model_id = model_id or llm.model_id
        self.policy = policy or LLMJudgePolicy()
        self.dry_run = dry_run
        self.stats = LLMJudgeStats()

    def judge(
        self,
        post: SocialPost,
        market: Market,
        baseline: RelevanceJudgement,
        *,
        query: str | None = None,
        query_type: str | None = None,
    ) -> LLMJudgeResult:
        """Judge one pair if selected; otherwise return the unchanged baseline."""
        self.stats.considered += 1
        if not self.policy.should_judge(baseline):
            return self._baseline_result(baseline)

        self.stats.selected += 1
        if self.dry_run:
            return self._baseline_result(baseline, selected=True)

        key = self._cache_key(post, market)
        cached = self.cache.get(key)
        if cached is not None:
            try:
                verdict = LLMRelevanceVerdict.model_validate(cached)
            except Exception:
                self.cache.delete(key)
                self.stats.invalid_outputs += 1
                return self._baseline_result(baseline, selected=True, invalid_output=True)
            self.stats.cache_hits += 1
            return self._merge(baseline, verdict, cache_hit=True)

        prompt = self._prompt(post, market, baseline, query=query, query_type=query_type)
        self.stats.calls += 1
        try:
            raw = self.llm.complete_structured(prompt, schema=LLMRelevanceVerdict)
            verdict = self._validate_response(raw)
        except Exception:
            self.stats.invalid_outputs += 1
            return self._baseline_result(baseline, selected=True, invalid_output=True)

        self.cache.set(key, verdict.model_dump(mode="json"))
        return self._merge(baseline, verdict)

    def _cache_key(self, post: SocialPost, market: Market) -> str:
        return stable_key(
            {
                "prompt_version": PROMPT_VERSION,
                "provider": self.provider,
                "model": self.model_id,
                "market_id": market.market_id,
                "post_id": post.post_id,
                "market_text": {
                    "question": market.question,
                    "description": market.description,
                    "domain": market.domain.value,
                    "outcomes": [outcome.label for outcome in market.outcomes],
                },
                "post_text": post.text_normalized or post.text,
            }
        )

    def _prompt(
        self,
        post: SocialPost,
        market: Market,
        baseline: RelevanceJudgement,
        *,
        query: str | None,
        query_type: str | None,
    ) -> str:
        constraints = [
            {
                "relation": constraint.relation_phrase,
                "anchor": constraint.anchor,
                "raw": constraint.raw,
            }
            for constraint in extract_market_constraints(market.question)
        ]
        context = {
            "market_id": market.market_id,
            "market_question": market.question,
            "market_description": market.description,
            "outcomes": [outcome.label for outcome in market.outcomes],
            "domain": market.domain.value,
            "constraints": constraints,
            "candidate_post_text": (post.text_normalized or post.text)[:4000],
            "post_timestamp": post.created_at.isoformat() if post.created_at else None,
            "query_provenance": {"query": query, "query_type": query_type},
            "baseline": {
                "label": baseline.relevance_label.value,
                "score": baseline.relevance_score,
                "confidence": baseline.confidence,
                "constraint_match": baseline.constraint_match.value,
                "embedding_similarity": baseline.semantic_similarity,
            },
        }
        return (
            "Decide only whether this social post is semantically relevant to the "
            "Polymarket market. Do not infer stance, sentiment, or outcome probability. "
            "Return JSON only with relevance_label, confidence, constraint_match, "
            "reason, matched_market_elements, missing_market_elements, and "
            "is_stance=false.\nCONTEXT:\n" + json.dumps(context, ensure_ascii=True, sort_keys=True)
        )

    @staticmethod
    def _validate_response(raw: Any) -> LLMRelevanceVerdict:
        if isinstance(raw, LLMRelevanceVerdict):
            return raw
        if isinstance(raw, str):
            return parse_structured_response(raw, LLMRelevanceVerdict)
        if isinstance(raw, BaseModel):
            raw = raw.model_dump(mode="json")
        return LLMRelevanceVerdict.model_validate(raw)

    def _merge(
        self,
        baseline: RelevanceJudgement,
        verdict: LLMRelevanceVerdict,
        *,
        cache_hit: bool = False,
    ) -> LLMJudgeResult:
        label = baseline.relevance_label
        confidence = baseline.confidence
        constraint_match = baseline.constraint_match
        applied = False

        if verdict.confidence >= self.policy.apply_confidence:
            baseline_is_borderline = baseline.relevance_label in _BORDERLINE_LABELS
            very_high_confidence = verdict.confidence >= self.policy.force_confidence
            if baseline_is_borderline or very_high_confidence:
                label_gap = abs(
                    _LABEL_RANK[baseline.relevance_label] - _LABEL_RANK[verdict.relevance_label]
                )
                if label_gap > 1 and not very_high_confidence:
                    label = RelevanceLabel.AMBIGUOUS
                else:
                    label = verdict.relevance_label
                confidence = max(baseline.confidence, verdict.confidence)
                constraint_match = verdict.constraint_match
                applied = True

        if applied:
            self.stats.applied += 1
        if label != baseline.relevance_label:
            self.stats.label_changes += 1

        return LLMJudgeResult(
            selected=True,
            judged=True,
            cache_hit=cache_hit,
            invalid_output=False,
            applied=applied,
            verdict=verdict,
            final_relevance_label=label,
            final_relevance_score=baseline.relevance_score,
            final_confidence=confidence,
            final_constraint_match=constraint_match,
            final_method=LinkMethod.HYBRID if applied else LinkMethod.DETERMINISTIC_BASELINE,
        )

    @staticmethod
    def _baseline_result(
        baseline: RelevanceJudgement,
        *,
        selected: bool = False,
        invalid_output: bool = False,
    ) -> LLMJudgeResult:
        return LLMJudgeResult(
            selected=selected,
            judged=False,
            cache_hit=False,
            invalid_output=invalid_output,
            applied=False,
            verdict=None,
            final_relevance_label=baseline.relevance_label,
            final_relevance_score=baseline.relevance_score,
            final_confidence=baseline.confidence,
            final_constraint_match=baseline.constraint_match,
            final_method=LinkMethod.DETERMINISTIC_BASELINE,
        )


class LLMRelevanceLinker:
    """Legacy ``Link`` adapter kept for the pre-Branch-05 pipeline."""

    method = LinkMethod.LLM

    def __init__(self, llm: LLMClient, cache: JSONCache) -> None:
        self.llm = llm
        self.cache = cache

    def verdict(self, post: SocialPost, market: Market) -> LLMRelevanceVerdict:
        key = stable_key(
            {
                "prompt_version": PROMPT_VERSION,
                "legacy_linker": True,
                "model": self.llm.model_id,
                "market_id": market.market_id,
                "post_id": post.post_id,
                "market_text": {"question": market.question, "description": market.description},
                "post_text": post.text_normalized or post.text,
            }
        )
        cached = self.cache.get(key)
        if cached is not None:
            return LLMRelevanceVerdict.model_validate(cached)
        prompt = (
            "Decide only whether this social post is semantically relevant to the "
            "Polymarket market. Do not infer stance, sentiment, or outcome probability. "
            "Return JSON only matching the requested schema.\n"
            f"MARKET: {market.question}\nPOST: {(post.text_normalized or post.text)[:4000]}"
        )
        raw = self.llm.complete_structured(prompt, schema=LLMRelevanceVerdict)
        verdict = LLMRelevanceJudge._validate_response(raw)
        self.cache.set(key, verdict.model_dump(mode="json"))
        return verdict

    def to_link(
        self,
        post: SocialPost,
        market: Market,
        verdict: LLMRelevanceVerdict,
    ) -> Link | None:
        """Convert a positive relevance verdict to the legacy ``Link`` model."""
        if verdict.relevance_label == RelevanceLabel.UNRELATED:
            return None

        return Link(
            post_id=post.post_id,
            platform=post.platform,
            market_id=market.market_id,
            method=LinkMethod.LLM,
            confidence=verdict.confidence,
            created_at=datetime.now(tz=timezone.utc),
            evidence=[verdict.relevance_label.value, verdict.reason],
        )


# Backward-compatible name used by earlier imports.
RelevanceVerdict = LLMRelevanceVerdict


__all__ = [
    "PROMPT_VERSION",
    "LLMJudgePolicy",
    "LLMJudgeResult",
    "LLMJudgeStats",
    "LLMRelevanceJudge",
    "LLMRelevanceLinker",
    "LLMRelevanceVerdict",
    "RelevanceVerdict",
]

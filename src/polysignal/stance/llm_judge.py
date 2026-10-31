"""Optional cached LLM verifier for Branch 06B outcome-aware stance.

The judge mirrors :class:`polysignal.linking.llm_assisted.LLMRelevanceJudge`
(Branch 05C) for relevance: it wraps an :class:`LLMClient`, caches its
structured JSON output on disk, and merges the result with the
deterministic Branch 06A baseline only when the baseline is uncertain.

Eligibility rule
----------------

The judge runs only on baseline verdicts that are:

* labelled ``AMBIGUOUS`` or ``INSUFFICIENT_CONTEXT``, **or**
* labelled ``SUPPORTS_YES`` / ``SUPPORTS_NO`` / ``NEUTRAL`` with
  ``confidence < confidence_floor`` (default 0.5).

The CLI exposes ``--llm-confidence-floor`` so the operator can dial
this. ``unrelated`` rows are not scored at all (the deterministic
pipeline already strips them upstream).

Merge rule (conservative)
-------------------------

The LLM verdict replaces the deterministic verdict only if:

* the LLM returned a directional label (``SUPPORTS_YES`` /
  ``SUPPORTS_NO``), **and**
* the LLM confidence is at least 0.5 **and** ≥ the baseline confidence.

Otherwise the deterministic verdict is preserved; the LLM output is
still recorded in audit columns so the operator can see what the LLM
thought.

Branch 06E also sends conservative target aliases in target-aware
prompts and cache keys.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from polysignal.domain import StanceLabel
from polysignal.llm.client import LLMClient
from polysignal.llm.structured import parse_structured_response
from polysignal.stance.scorer import StanceVerdict
from polysignal.stance.target_aliases import generate_target_aliases
from polysignal.storage.cache import JSONCache, stable_key

PROMPT_VERSION = "branch06e-v1"
_MAX_RATIONALE_LENGTH = 240


class LLMStanceVerdict(BaseModel):
    """Validated stance response expected from the LLM.

    The schema mirrors :class:`polysignal.stance.scorer.StanceVerdict`
    so the merger can compare like-with-like, but ``rationale`` is
    bounded so the parquet column does not blow up.
    """

    model_config = ConfigDict(extra="forbid")

    stance_label: StanceLabel
    stance_score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: Any) -> float:
        number = float(value)
        return min(1.0, max(0.0, number))

    @field_validator("stance_score", mode="before")
    @classmethod
    def _clamp_score(cls, value: Any) -> float:
        number = float(value)
        return min(1.0, max(-1.0, number))

    @field_validator("rationale")
    @classmethod
    def _short_rationale(cls, value: str) -> str:
        return value.strip()[:_MAX_RATIONALE_LENGTH]


@dataclass(frozen=True)
class LLMStanceJudgePolicy:
    """Cost-control and merge thresholds for the stance judge."""

    confidence_floor: float = 0.5
    apply_min_confidence: float = 0.5

    def should_judge(self, baseline: StanceVerdict) -> bool:
        if baseline.label in {
            StanceLabel.AMBIGUOUS,
            StanceLabel.INSUFFICIENT_CONTEXT,
        }:
            return True
        if baseline.label is StanceLabel.UNRELATED:
            return False
        return baseline.confidence < self.confidence_floor


@dataclass
class LLMStanceJudgeStats:
    considered: int = 0
    selected: int = 0
    calls: int = 0
    cache_hits: int = 0
    invalid_outputs: int = 0
    applied: int = 0
    label_changes: int = 0


@dataclass(frozen=True)
class LLMStanceJudgeResult:
    """Result of running the judge on one baseline verdict."""

    selected: bool
    judged: bool
    cache_hit: bool
    invalid_output: bool
    applied: bool
    cache_key: str | None
    verdict: LLMStanceVerdict | None
    merged: StanceVerdict


class LLMStanceJudge:
    """Optional cached LLM stance judge with conservative merge logic."""

    def __init__(
        self,
        llm: LLMClient,
        cache: JSONCache,
        *,
        provider: str | None = None,
        model_id: str | None = None,
        policy: LLMStanceJudgePolicy | None = None,
        dry_run: bool = False,
    ) -> None:
        self.llm = llm
        self.cache = cache
        self.provider = provider or type(llm).__name__
        self.model_id = model_id or getattr(llm, "model_id", "unknown")
        self.policy = policy or LLMStanceJudgePolicy()
        self.dry_run = dry_run
        self.stats = LLMStanceJudgeStats()

    def judge(
        self,
        *,
        text: str,
        market_question: str,
        outcome_names: list[str],
        canonical_keywords: list[str] | None,
        baseline: StanceVerdict,
        target_outcome_name: str | None = None,
        target_aliases: list[str] | None = None,
    ) -> LLMStanceJudgeResult:
        """Judge one baseline verdict; otherwise return it unchanged."""
        self.stats.considered += 1
        if not self.policy.should_judge(baseline):
            return _baseline_result(baseline)
        self.stats.selected += 1
        if self.dry_run:
            return _baseline_result(baseline, selected=True)

        resolved_target_aliases = (
            list(target_aliases)
            if target_aliases is not None
            else generate_target_aliases(target_outcome_name)
        )
        key = self._cache_key(
            text=text,
            market_question=market_question,
            outcome_names=outcome_names,
            canonical_keywords=canonical_keywords or [],
            target_outcome_name=target_outcome_name,
            target_aliases=resolved_target_aliases,
        )
        cached = self.cache.get(key)
        if cached is not None:
            try:
                verdict = LLMStanceVerdict.model_validate(cached)
            except Exception:
                self.cache.delete(key)
                self.stats.invalid_outputs += 1
                return _baseline_result(
                    baseline, selected=True, cache_key=key, invalid_output=True
                )
            self.stats.cache_hits += 1
            merged, applied = self._merge(baseline, verdict)
            if applied:
                self.stats.applied += 1
            if merged.label is not baseline.label:
                self.stats.label_changes += 1
            return LLMStanceJudgeResult(
                selected=True,
                judged=True,
                cache_hit=True,
                invalid_output=False,
                applied=applied,
                cache_key=key,
                verdict=verdict,
                merged=merged,
            )

        prompt = self._prompt(
            text=text,
            market_question=market_question,
            outcome_names=outcome_names,
            canonical_keywords=canonical_keywords or [],
            baseline=baseline,
            target_outcome_name=target_outcome_name,
            target_aliases=resolved_target_aliases,
        )
        self.stats.calls += 1
        try:
            raw = self.llm.complete_structured(prompt, schema=LLMStanceVerdict)
            verdict = self._validate_response(raw)
        except Exception:
            self.stats.invalid_outputs += 1
            return _baseline_result(
                baseline, selected=True, cache_key=key, invalid_output=True
            )

        self.cache.set(key, verdict.model_dump(mode="json"))
        merged, applied = self._merge(baseline, verdict)
        if applied:
            self.stats.applied += 1
        if merged.label is not baseline.label:
            self.stats.label_changes += 1
        return LLMStanceJudgeResult(
            selected=True,
            judged=True,
            cache_hit=False,
            invalid_output=False,
            applied=applied,
            cache_key=key,
            verdict=verdict,
            merged=merged,
        )

    def _cache_key(
        self,
        *,
        text: str,
        market_question: str,
        outcome_names: list[str],
        canonical_keywords: list[str],
        target_outcome_name: str | None = None,
        target_aliases: list[str] | None = None,
    ) -> str:
        return stable_key(
            {
                "prompt_version": PROMPT_VERSION,
                "provider": self.provider,
                "model": self.model_id,
                "text": text,
                "market_question": market_question,
                "outcome_names": list(outcome_names),
                "target_outcome_name": target_outcome_name,
                "target_aliases": list(target_aliases or []),
                "canonical_keywords": list(canonical_keywords),
            }
        )

    def _prompt(
        self,
        *,
        text: str,
        market_question: str,
        outcome_names: list[str],
        canonical_keywords: list[str],
        baseline: StanceVerdict,
        target_outcome_name: str | None,
        target_aliases: list[str],
    ) -> str:
        context = {
            "market_question": market_question,
            "outcome_names": list(outcome_names),
            "target_outcome_name": target_outcome_name,
            "target_aliases": list(target_aliases),
            "canonical_keywords": list(canonical_keywords),
            "candidate_text": (text or "")[:4000],
            "baseline_verdict": {
                "label": baseline.label.value,
                "stance_score": baseline.stance_score,
                "confidence": baseline.confidence,
                "explanation": baseline.explanation,
            },
        }
        target_instruction = (
            f" Decide whether the text supports the specific target outcome "
            f"{target_outcome_name!r}. For this target-aware row, supports_yes "
            "means supports the target and supports_no means opposes the target."
            if target_outcome_name is not None
            else " Decide which outcome the text supports."
        )
        score_instruction = (
            "positive = supports target, negative = opposes target"
            if target_outcome_name is not None
            else "positive = YES, negative = NO"
        )
        return (
            "You are an outcome-aware stance classifier for a Polymarket "
            "market. The text was already judged relevant to this market. "
            + target_instruction
            + " Return JSON only with "
            "fields stance_label (one of supports_yes, supports_no, "
            "neutral, ambiguous, unrelated, insufficient_context), "
            f"stance_score in [-1, 1] ({score_instruction}), "
            "confidence in [0, 1], and a short rationale. Do not return "
            "anything else.\nCONTEXT:\n"
            + json.dumps(context, ensure_ascii=True, sort_keys=True)
        )

    @staticmethod
    def _validate_response(raw: Any) -> LLMStanceVerdict:
        if isinstance(raw, LLMStanceVerdict):
            return raw
        if isinstance(raw, str):
            return parse_structured_response(raw, LLMStanceVerdict)
        if isinstance(raw, BaseModel):
            raw = raw.model_dump(mode="json")
        return LLMStanceVerdict.model_validate(raw)

    def _merge(
        self,
        baseline: StanceVerdict,
        verdict: LLMStanceVerdict,
    ) -> tuple[StanceVerdict, bool]:
        directional = {StanceLabel.SUPPORTS_YES, StanceLabel.SUPPORTS_NO}
        if (
            verdict.stance_label in directional
            and verdict.confidence >= self.policy.apply_min_confidence
            and verdict.confidence >= baseline.confidence
        ):
            merged = StanceVerdict(
                label=verdict.stance_label,
                stance_score=verdict.stance_score,
                confidence=verdict.confidence,
                explanation=(
                    verdict.rationale or "llm structured stance judge"
                ),
                matched_yes_cues=list(baseline.matched_yes_cues),
                matched_no_cues=list(baseline.matched_no_cues),
                matched_uncertainty_cues=list(baseline.matched_uncertainty_cues),
                negated_cues=list(baseline.negated_cues),
            )
            return merged, True
        return baseline, False


def _baseline_result(
    baseline: StanceVerdict,
    *,
    selected: bool = False,
    cache_key: str | None = None,
    invalid_output: bool = False,
) -> LLMStanceJudgeResult:
    return LLMStanceJudgeResult(
        selected=selected,
        judged=False,
        cache_hit=False,
        invalid_output=invalid_output,
        applied=False,
        cache_key=cache_key,
        verdict=None,
        merged=baseline,
    )


__all__ = [
    "PROMPT_VERSION",
    "LLMStanceJudge",
    "LLMStanceJudgePolicy",
    "LLMStanceJudgeResult",
    "LLMStanceJudgeStats",
    "LLMStanceVerdict",
]

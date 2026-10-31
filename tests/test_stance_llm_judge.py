"""Tests for the optional Branch 06B LLM stance judge."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel
from typer.testing import CliRunner

import polysignal.cli.main as cli_main
from polysignal.cli.main import app
from polysignal.domain import LinkedPost, Outcome, Platform, StanceLabel
from polysignal.domain.enums import ConstraintMatch, LinkMethod, RelevanceLabel, StanceMethod
from polysignal.linking.linking_pipeline import write_linked_posts
from polysignal.stance import (
    LLMStanceJudge,
    LLMStanceJudgePolicy,
    LLMStanceVerdict,
    read_stance_records,
    run_stance_pipeline,
)
from polysignal.stance.scorer import StanceVerdict, score_stance
from polysignal.storage.cache import JSONCache
from polysignal.storage.paths import get_data_paths

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeStanceLLMClient:
    """Deterministic, network-free client used only by stance unit tests."""

    model_id = "fake-stance-judge"

    def __init__(
        self,
        verdict: LLMStanceVerdict | None = None,
        *,
        responder=None,
        responses: dict[str, LLMStanceVerdict] | None = None,
        raise_on_call: bool = False,
    ) -> None:
        self._verdict = verdict
        self._responder = responder
        self._responses = dict(responses) if responses else None
        self._raise_on_call = raise_on_call
        self.calls = 0
        self.last_prompt: str | None = None
        self.last_schema: type[BaseModel] | None = None

    def complete_structured(self, prompt: str, *, schema: type[BaseModel]) -> Any:
        self.calls += 1
        self.last_prompt = prompt
        self.last_schema = schema
        if self._raise_on_call:
            raise RuntimeError("fake LLM client configured to raise")
        if self._responses is not None:
            for needle, response in self._responses.items():
                if needle in prompt:
                    return response
        if self._responder is not None:
            return self._responder(prompt)
        if self._verdict is None:
            raise RuntimeError("FakeStanceLLMClient has no configured verdict")
        return self._verdict


def _verdict(
    *,
    label: StanceLabel,
    score: float = 0.0,
    confidence: float = 0.0,
    explanation: str = "",
) -> StanceVerdict:
    return StanceVerdict(
        label=label,
        stance_score=score,
        confidence=confidence,
        explanation=explanation,
    )


def _llm_verdict(
    *,
    label: StanceLabel = StanceLabel.SUPPORTS_YES,
    score: float = 0.7,
    confidence: float = 0.85,
    rationale: str = "fake reasoning",
) -> LLMStanceVerdict:
    return LLMStanceVerdict(
        stance_label=label,
        stance_score=score,
        confidence=confidence,
        rationale=rationale,
    )


def _judge(
    cache_dir: Path,
    *,
    verdict: LLMStanceVerdict | None = None,
    responder=None,
    confidence_floor: float = 0.5,
    raise_on_call: bool = False,
) -> tuple[LLMStanceJudge, FakeStanceLLMClient]:
    client = FakeStanceLLMClient(
        verdict=verdict, responder=responder, raise_on_call=raise_on_call
    )
    judge = LLMStanceJudge(
        client,
        JSONCache(cache_dir),
        policy=LLMStanceJudgePolicy(confidence_floor=confidence_floor),
    )
    return judge, client


def _linked(
    *,
    market_id: str = "m-btc",
    post_id: str = "p1",
    text_preview: str = "Bitcoin is set to reach $150k by 2026.",
    market_question: str = "Will Bitcoin hit $150k by December 2026?",
    matched_terms: list[str] | None = None,
    relevance_label: RelevanceLabel = RelevanceLabel.RELEVANT,
    confidence: float = 0.8,
) -> LinkedPost:
    return LinkedPost(
        link_id=f"link-{post_id}",
        market_id=market_id,
        post_id=post_id,
        platform=Platform.TELEGRAM,
        relevance_score=0.7,
        relevance_label=relevance_label,
        confidence=confidence,
        method=LinkMethod.DETERMINISTIC_BASELINE,
        explanation="seed",
        matched_entities=["Bitcoin"],
        matched_terms=matched_terms or ["Bitcoin", "150k"],
        constraint_match=ConstraintMatch.UNKNOWN,
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        post_created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        market_question=market_question,
        post_text_preview=text_preview,
        signal_count=2,
        anchor_score=0.6,
        constraint_score=0.0,
        domain_mismatch=False,
    )


# ---------------------------------------------------------------------------
# Eligibility rule
# ---------------------------------------------------------------------------


def test_judge_skips_high_confidence_directional_baseline(tmp_path: Path) -> None:
    judge, client = _judge(tmp_path / "cache", verdict=_llm_verdict())
    baseline = _verdict(label=StanceLabel.SUPPORTS_YES, score=0.6, confidence=0.65)
    result = judge.judge(
        text="Bitcoin will reach $150k by 2026.",
        market_question="Will Bitcoin hit $150k by 2026?",
        outcome_names=["Yes", "No"],
        canonical_keywords=["Bitcoin"],
        baseline=baseline,
    )
    assert result.selected is False
    assert result.judged is False
    assert client.calls == 0
    assert result.merged is baseline
    assert judge.stats.considered == 1
    assert judge.stats.selected == 0


def test_judge_selects_ambiguous_baseline(tmp_path: Path) -> None:
    judge, client = _judge(tmp_path / "cache", verdict=_llm_verdict())
    baseline = _verdict(label=StanceLabel.AMBIGUOUS, confidence=0.2)
    result = judge.judge(
        text="Bitcoin price news with mixed signals.",
        market_question="Will Bitcoin hit $150k by 2026?",
        outcome_names=["Yes", "No"],
        canonical_keywords=["Bitcoin"],
        baseline=baseline,
    )
    assert result.selected is True
    assert result.judged is True
    assert client.calls == 1
    assert result.merged.label is StanceLabel.SUPPORTS_YES
    assert result.applied is True


def test_judge_selects_insufficient_context_baseline(tmp_path: Path) -> None:
    judge, _ = _judge(tmp_path / "cache", verdict=_llm_verdict())
    baseline = _verdict(label=StanceLabel.INSUFFICIENT_CONTEXT, confidence=0.05)
    result = judge.judge(
        text="hint about something",
        market_question="Will Bitcoin hit $150k by 2026?",
        outcome_names=["Yes", "No"],
        canonical_keywords=["Bitcoin"],
        baseline=baseline,
    )
    assert result.selected is True


def test_judge_selects_low_confidence_directional_baseline(tmp_path: Path) -> None:
    judge, _ = _judge(
        tmp_path / "cache", verdict=_llm_verdict(), confidence_floor=0.6
    )
    baseline = _verdict(label=StanceLabel.SUPPORTS_NO, confidence=0.3, score=-0.3)
    result = judge.judge(
        text="Bitcoin is unlikely to reach $150k.",
        market_question="Will Bitcoin hit $150k by 2026?",
        outcome_names=["Yes", "No"],
        canonical_keywords=["Bitcoin"],
        baseline=baseline,
    )
    assert result.selected is True


def test_judge_never_runs_on_unrelated(tmp_path: Path) -> None:
    judge, client = _judge(tmp_path / "cache", verdict=_llm_verdict())
    baseline = _verdict(label=StanceLabel.UNRELATED, confidence=0.0)
    result = judge.judge(
        text="totally off topic",
        market_question="Will Bitcoin hit $150k by 2026?",
        outcome_names=["Yes", "No"],
        canonical_keywords=["Bitcoin"],
        baseline=baseline,
    )
    assert result.selected is False
    assert client.calls == 0


# ---------------------------------------------------------------------------
# Cache hit / miss
# ---------------------------------------------------------------------------


def test_judge_writes_to_cache_on_first_call_and_reuses_on_second(tmp_path: Path) -> None:
    judge, client = _judge(tmp_path / "cache", verdict=_llm_verdict())
    baseline = _verdict(label=StanceLabel.AMBIGUOUS, confidence=0.2)
    payload = dict(
        text="Bitcoin price news with mixed signals.",
        market_question="Will Bitcoin hit $150k by 2026?",
        outcome_names=["Yes", "No"],
        canonical_keywords=["Bitcoin"],
        baseline=baseline,
    )
    first = judge.judge(**payload)
    assert first.cache_hit is False
    assert client.calls == 1
    second = judge.judge(**payload)
    assert second.cache_hit is True
    # The cache hit must NOT trigger a second client call.
    assert client.calls == 1
    assert judge.stats.cache_hits == 1


def test_judge_recovers_from_corrupt_cache_entry(tmp_path: Path) -> None:
    judge, client = _judge(tmp_path / "cache", verdict=_llm_verdict())
    baseline = _verdict(label=StanceLabel.AMBIGUOUS, confidence=0.2)
    payload = dict(
        text="Bitcoin signals mixed.",
        market_question="Will Bitcoin hit $150k by 2026?",
        outcome_names=["Yes", "No"],
        canonical_keywords=["Bitcoin"],
        baseline=baseline,
    )
    cache_key = judge._cache_key(
        text=payload["text"],
        market_question=payload["market_question"],
        outcome_names=payload["outcome_names"],
        canonical_keywords=payload["canonical_keywords"],
    )
    judge.cache.set(cache_key, {"completely": "invalid"})
    result = judge.judge(**payload)
    # The invalid blob is dropped, no LLM call is forced this cycle,
    # and the baseline is preserved.
    assert result.invalid_output is True
    assert client.calls == 0
    assert result.merged is baseline


def test_judge_falls_back_when_client_raises(tmp_path: Path) -> None:
    judge, client = _judge(tmp_path / "cache", raise_on_call=True)
    baseline = _verdict(label=StanceLabel.AMBIGUOUS, confidence=0.2)
    result = judge.judge(
        text="Bitcoin signals mixed.",
        market_question="Will Bitcoin hit $150k by 2026?",
        outcome_names=["Yes", "No"],
        canonical_keywords=["Bitcoin"],
        baseline=baseline,
    )
    assert result.invalid_output is True
    assert client.calls == 1
    assert result.merged is baseline


# ---------------------------------------------------------------------------
# Merge rule
# ---------------------------------------------------------------------------


def test_merge_replaces_label_only_when_directional_and_confident(tmp_path: Path) -> None:
    judge, _ = _judge(
        tmp_path / "cache",
        verdict=_llm_verdict(label=StanceLabel.AMBIGUOUS, confidence=0.9),
    )
    baseline = _verdict(label=StanceLabel.AMBIGUOUS, confidence=0.2)
    result = judge.judge(
        text="mixed evidence",
        market_question="Will Bitcoin hit $150k by 2026?",
        outcome_names=["Yes", "No"],
        canonical_keywords=["Bitcoin"],
        baseline=baseline,
    )
    # LLM returned AMBIGUOUS with high confidence — non-directional →
    # baseline preserved, applied=False, but audit columns recorded.
    assert result.applied is False
    assert result.merged.label is StanceLabel.AMBIGUOUS
    assert result.verdict is not None


def test_merge_rejects_low_confidence_llm_verdict(tmp_path: Path) -> None:
    judge, _ = _judge(
        tmp_path / "cache",
        verdict=_llm_verdict(label=StanceLabel.SUPPORTS_YES, confidence=0.3),
    )
    baseline = _verdict(label=StanceLabel.AMBIGUOUS, confidence=0.2)
    result = judge.judge(
        text="mixed evidence",
        market_question="Will Bitcoin hit $150k by 2026?",
        outcome_names=["Yes", "No"],
        canonical_keywords=["Bitcoin"],
        baseline=baseline,
    )
    assert result.applied is False
    assert result.merged.label is StanceLabel.AMBIGUOUS


def test_merge_keeps_baseline_when_llm_confidence_below_baseline(tmp_path: Path) -> None:
    judge, _ = _judge(
        tmp_path / "cache",
        verdict=_llm_verdict(label=StanceLabel.SUPPORTS_NO, confidence=0.55),
        confidence_floor=0.7,
    )
    baseline = _verdict(label=StanceLabel.SUPPORTS_YES, confidence=0.6)
    result = judge.judge(
        text="conflicting signals",
        market_question="Will Bitcoin hit $150k by 2026?",
        outcome_names=["Yes", "No"],
        canonical_keywords=["Bitcoin"],
        baseline=baseline,
    )
    # confidence_floor (0.7) selects this row because baseline conf=0.6
    # < 0.7, but LLM confidence (0.55) < baseline (0.6) so we keep
    # the baseline.
    assert result.selected is True
    assert result.applied is False
    assert result.merged.label is StanceLabel.SUPPORTS_YES


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_judge_dry_run_selects_without_calling(tmp_path: Path) -> None:
    client = FakeStanceLLMClient(verdict=_llm_verdict())
    judge = LLMStanceJudge(client, JSONCache(tmp_path / "cache"), dry_run=True)
    baseline = _verdict(label=StanceLabel.AMBIGUOUS, confidence=0.2)
    result = judge.judge(
        text="mixed",
        market_question="Will Bitcoin hit $150k by 2026?",
        outcome_names=["Yes", "No"],
        canonical_keywords=["Bitcoin"],
        baseline=baseline,
    )
    assert result.selected is True
    assert result.judged is False
    assert client.calls == 0


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def test_pipeline_uses_judge_only_for_uncertain_rows(tmp_path: Path) -> None:
    """High-confidence directional rows must skip the judge; uncertain ones run it."""
    confident = _linked(
        post_id="p-confident",
        text_preview="Analysts expect Bitcoin to reach $150k before 2026 ends.",
    )
    uncertain = _linked(
        post_id="p-uncertain",
        # No directional cues — scorer will return NEUTRAL.
        text_preview="Bitcoin price chart updated this morning across exchanges.",
    )
    judge, client = _judge(
        tmp_path / "cache",
        verdict=_llm_verdict(label=StanceLabel.SUPPORTS_YES, confidence=0.85),
    )
    result = run_stance_pipeline(
        linked_posts=[confident, uncertain],
        llm_judge=judge,
        dry_run=True,
    )
    assert result.rows_scored == 2
    # The confident row produced SUPPORTS_YES deterministically; the
    # judge must not have been called on it.
    assert judge.stats.considered == 2
    # NEUTRAL falls below floor=0.5 → judged.
    assert judge.stats.selected == 1
    assert client.calls == 1


def test_pipeline_audit_columns_populated_on_judge_use(tmp_path: Path) -> None:
    paths = get_data_paths(root=tmp_path)
    row = _linked(post_id="p-uncertain", text_preview="Bitcoin price snapshot today.")
    write_linked_posts([row], paths.linked_posts_parquet)
    judge, _client = _judge(
        tmp_path / "cache",
        verdict=_llm_verdict(label=StanceLabel.SUPPORTS_YES, confidence=0.85),
    )
    result = run_stance_pipeline(paths=paths, llm_judge=judge)
    assert result.rows_scored == 1
    persisted = read_stance_records(paths.stance_labels_parquet)
    record = persisted[0]
    assert record.used_llm is True
    assert record.llm_label is StanceLabel.SUPPORTS_YES
    assert record.llm_confidence == pytest.approx(0.85)
    assert record.llm_rationale  # non-empty
    assert record.llm_cache_key  # populated for caching
    assert record.llm_applied is True
    assert record.method is StanceMethod.HYBRID


def test_pipeline_records_baseline_when_judge_does_not_apply(tmp_path: Path) -> None:
    row = _linked(post_id="p-uncertain", text_preview="Bitcoin snapshot today.")
    judge, _ = _judge(
        tmp_path / "cache",
        verdict=_llm_verdict(label=StanceLabel.AMBIGUOUS, confidence=0.9),
    )
    result = run_stance_pipeline(linked_posts=[row], llm_judge=judge, dry_run=True)
    assert result.rows_scored == 1
    record = result.records[0]
    # The LLM said AMBIGUOUS — non-directional → baseline preserved
    # (NEUTRAL or INSUFFICIENT_CONTEXT) but audit columns recorded.
    assert record.used_llm is True
    assert record.llm_applied is False
    assert record.llm_label is StanceLabel.AMBIGUOUS


def test_pipeline_without_judge_leaves_audit_columns_empty(tmp_path: Path) -> None:
    row = _linked(post_id="p1")
    result = run_stance_pipeline(linked_posts=[row], dry_run=True)
    record = result.records[0]
    assert record.used_llm is False
    assert record.llm_label is None
    assert record.llm_confidence is None
    assert record.llm_cache_key is None


def test_pipeline_llm_judge_receives_each_multi_outcome_target(tmp_path: Path) -> None:
    prompts: list[str] = []

    def _respond(prompt: str) -> LLMStanceVerdict:
        prompts.append(prompt)
        return _llm_verdict(label=StanceLabel.SUPPORTS_YES, confidence=0.85)

    row = _linked(
        market_id="m-election",
        post_id="p-election",
        market_question="Who will win the election?",
        text_preview="Election results remain unclear today.",
        matched_terms=["election"],
    )
    outcomes = [
        Outcome(market_id="m-election", outcome_id="a", label="Candidate A", index=0),
        Outcome(market_id="m-election", outcome_id="b", label="Candidate B", index=1),
        Outcome(market_id="m-election", outcome_id="other", label="Other", index=2),
    ]
    judge, client = _judge(tmp_path / "cache", responder=_respond)
    result = run_stance_pipeline(
        linked_posts=[row],
        outcomes=outcomes,
        llm_judge=judge,
        dry_run=True,
    )
    assert len(result.records) == 3
    assert client.calls == 3
    assert len(prompts) == 3
    for target in ("Candidate A", "Candidate B", "Other"):
        assert any(f'"target_outcome_name": "{target}"' in prompt for prompt in prompts)
    assert any('"target_aliases": ["candidate a"]' in prompt for prompt in prompts)
    assert all("supports_yes means supports the target" in prompt for prompt in prompts)


def test_score_stance_remains_pure_when_judge_not_passed() -> None:
    verdict = score_stance(
        text="Bitcoin is unlikely to reach $150k by 2026.",
        market_question="Will Bitcoin hit $150k by December 2026?",
        outcome_names=["Yes", "No"],
        canonical_keywords=["Bitcoin"],
    )
    assert verdict.label is StanceLabel.SUPPORTS_NO


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_use_llm_without_api_key_exits_with_clear_message(
    monkeypatch, tmp_path: Path
) -> None:
    paths = get_data_paths(root=tmp_path)
    paths.ensure()
    write_linked_posts([_linked()], paths.linked_posts_parquet)
    monkeypatch.setattr(
        cli_main,
        "get_settings",
        lambda: SimpleNamespace(
            llm=SimpleNamespace(
                provider="openai",
                model="gpt-4o-mini",
                api_key=None,
                openai_api_key=None,
                anthropic_api_key=None,
            )
        ),
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "stance",
            "score",
            "--output-dir",
            str(tmp_path),
            "--use-llm",
            "--llm-provider",
            "openai",
            "--llm-model",
            "gpt-4o-mini",
        ],
    )
    assert result.exit_code == 2
    assert "API key" in result.output


def test_cli_use_llm_runs_with_fake_client(monkeypatch, tmp_path: Path) -> None:
    paths = get_data_paths(root=tmp_path)
    paths.ensure()
    write_linked_posts(
        [_linked(text_preview="Bitcoin price chart updated this morning.")],
        paths.linked_posts_parquet,
    )
    judge, client = _judge(
        tmp_path / "cache",
        verdict=_llm_verdict(label=StanceLabel.SUPPORTS_YES, confidence=0.85),
    )

    def _fake_builder(**_kwargs):
        return judge

    monkeypatch.setattr(cli_main, "_build_cli_stance_judge", _fake_builder)
    result = CliRunner().invoke(
        app,
        ["stance", "score", "--output-dir", str(tmp_path), "--use-llm"],
    )
    assert result.exit_code == 0, result.output
    assert client.calls == 1
    record = read_stance_records(paths.stance_labels_parquet)[0]
    assert record.used_llm is True
    assert record.llm_applied is True
    assert record.method is StanceMethod.HYBRID


def test_cli_use_llm_dry_run_does_not_require_api_key(tmp_path: Path) -> None:
    paths = get_data_paths(root=tmp_path)
    paths.ensure()
    write_linked_posts(
        [
            # NEUTRAL baseline → selected for LLM judging.
            _linked(text_preview="Bitcoin price chart updated this morning."),
        ],
        paths.linked_posts_parquet,
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "stance",
            "score",
            "--output-dir",
            str(tmp_path),
            "--use-llm",
            "--llm-dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "llm considered" in result.output.lower()
    persisted = read_stance_records(paths.stance_labels_parquet)
    assert persisted
    # dry_run selects but does not judge → used_llm should be False
    record = persisted[0]
    assert record.used_llm is False

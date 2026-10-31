"""Offline tests for the optional Branch 05C LLM relevance verifier."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

import polysignal.cli.main as cli_main
from polysignal.domain import ConstraintMatch, RelevanceLabel, SocialPost
from polysignal.linking import (
    LLMJudgePolicy,
    LLMRelevanceJudge,
    evaluate_predictions,
    fixture_markets,
    fixture_posts,
    run_linking_cross_product,
    write_evaluation_report,
)
from polysignal.linking.fixtures import LabeledPair
from polysignal.linking.relevance import RelevanceJudgement
from polysignal.llm.client import LLMClient
from polysignal.storage.cache import JSONCache

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeLLMClient(LLMClient):
    model_id = "fake-model"

    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[str] = []

    def complete_structured(self, prompt: str, *, schema: type) -> Any:
        self.calls.append(prompt)
        return self.response


def _response(
    *,
    label: str = "relevant",
    confidence: float = 0.92,
    constraint: str = "partial",
) -> str:
    return json.dumps(
        {
            "relevance_label": label,
            "confidence": confidence,
            "constraint_match": constraint,
            "reason": "The post discusses the contract-specific product.",
            "matched_market_elements": ["foldable iPhone"],
            "missing_market_elements": ["WWDC deadline"],
            "is_stance": False,
        }
    )


def _baseline(
    *,
    label: RelevanceLabel = RelevanceLabel.WEAKLY_RELEVANT,
    score: float = 0.42,
    confidence: float = 0.55,
) -> RelevanceJudgement:
    return RelevanceJudgement(
        market_id="m-apple-foldable",
        post_id="p-apple-rumor",
        relevance_score=score,
        relevance_label=label,
        confidence=confidence,
        constraint_match=ConstraintMatch.MISSING,
    )


def _pair() -> tuple[SocialPost, Any]:
    market = next(m for m in fixture_markets() if m.market_id == "m-apple-foldable")
    post = SocialPost(
        post_id="p-apple-rumor",
        platform="reddit",
        created_at=_NOW,
        text="Rumors say Apple is testing a foldable iPhone prototype.",
    )
    return post, market


def _judge(
    tmp_path: Path,
    response: Any,
    *,
    policy: LLMJudgePolicy | None = None,
) -> tuple[LLMRelevanceJudge, FakeLLMClient]:
    client = FakeLLMClient(response)
    judge = LLMRelevanceJudge(client, JSONCache(tmp_path / "llm"), policy=policy)
    return judge, client


def test_default_linking_works_without_llm() -> None:
    rows = run_linking_cross_product(fixture_posts()[:1], fixture_markets()[:1])
    assert rows
    assert rows[0].llm_judged is False
    assert rows[0].final_relevance_label == rows[0].relevance_label


def test_borderline_rows_are_selected_by_default(tmp_path: Path) -> None:
    judge, client = _judge(tmp_path, _response())
    post, market = _pair()
    result = judge.judge(post, market, _baseline())
    assert result.selected is True
    assert len(client.calls) == 1


def test_prompt_serializes_missing_post_timestamp_as_null(tmp_path: Path) -> None:
    judge, client = _judge(tmp_path, _response())
    post, market = _pair()
    post.created_at = None
    judge.judge(post, market, _baseline())
    assert '"post_timestamp": null' in client.calls[0]


def test_clear_unrelated_and_relevant_rows_are_not_selected(tmp_path: Path) -> None:
    judge, client = _judge(tmp_path, _response())
    post, market = _pair()
    unrelated = _baseline(label=RelevanceLabel.UNRELATED, score=0.02, confidence=0.90)
    relevant = _baseline(label=RelevanceLabel.RELEVANT, score=0.92, confidence=0.90)
    assert judge.judge(post, market, unrelated).selected is False
    assert judge.judge(post, market, relevant).selected is False
    assert client.calls == []


def test_valid_json_is_parsed_and_confidence_is_clamped(tmp_path: Path) -> None:
    judge, _ = _judge(tmp_path, _response(confidence=1.4))
    post, market = _pair()
    result = judge.judge(post, market, _baseline())
    assert result.verdict is not None
    assert result.verdict.confidence == 1.0


def test_malformed_json_falls_back_safely(tmp_path: Path) -> None:
    judge, _ = _judge(tmp_path, "{not-json")
    post, market = _pair()
    baseline = _baseline()
    result = judge.judge(post, market, baseline)
    assert result.invalid_output is True
    assert result.final_relevance_label == baseline.relevance_label


def test_invalid_label_falls_back_safely(tmp_path: Path) -> None:
    judge, _ = _judge(tmp_path, _response(label="supports_yes"))
    post, market = _pair()
    baseline = _baseline()
    result = judge.judge(post, market, baseline)
    assert result.invalid_output is True
    assert result.final_relevance_label == baseline.relevance_label


def test_stance_output_is_rejected_safely(tmp_path: Path) -> None:
    raw = json.loads(_response())
    raw["is_stance"] = True
    judge, _ = _judge(tmp_path, json.dumps(raw))
    post, market = _pair()
    result = judge.judge(post, market, _baseline())
    assert result.invalid_output is True


def test_cache_prevents_duplicate_calls(tmp_path: Path) -> None:
    judge, client = _judge(tmp_path, _response())
    post, market = _pair()
    baseline = _baseline()
    judge.judge(post, market, baseline)
    second = judge.judge(post, market, baseline)
    assert len(client.calls) == 1
    assert second.cache_hit is True


def test_high_confidence_llm_can_adjust_borderline_label(tmp_path: Path) -> None:
    judge, _ = _judge(tmp_path, _response(label="relevant", confidence=0.92))
    post, market = _pair()
    result = judge.judge(post, market, _baseline())
    assert result.applied is True
    assert result.final_relevance_label == RelevanceLabel.RELEVANT


def test_low_confidence_llm_does_not_override_baseline(tmp_path: Path) -> None:
    judge, _ = _judge(tmp_path, _response(label="relevant", confidence=0.60))
    post, market = _pair()
    baseline = _baseline()
    result = judge.judge(post, market, baseline)
    assert result.applied is False
    assert result.final_relevance_label == baseline.relevance_label


def test_pipeline_keeps_baseline_and_adds_llm_final_fields(tmp_path: Path) -> None:
    judge, _ = _judge(tmp_path, _response(label="relevant", confidence=0.92))
    post, market = _pair()
    rows = run_linking_cross_product([post], [market], llm_judge=judge)
    row = rows[0]
    assert row.relevance_label == RelevanceLabel.WEAKLY_RELEVANT
    assert row.llm_relevance_label == RelevanceLabel.RELEVANT
    assert row.final_relevance_label == RelevanceLabel.RELEVANT
    assert row.llm_judged is True
    assert row.llm_applied is True


def test_evaluation_report_includes_llm_metrics(tmp_path: Path) -> None:
    judge, _ = _judge(tmp_path, _response(label="relevant", confidence=0.92))
    post, market = _pair()
    rows = run_linking_cross_product([post], [market], llm_judge=judge)
    labels = [
        LabeledPair(
            market_id=market.market_id,
            post_id=post.post_id,
            expected_label=RelevanceLabel.RELEVANT,
            expected_constraint_match=ConstraintMatch.PARTIAL,
            reason="LLM refinement fixture",
        )
    ]
    report = evaluate_predictions(rows, labels)
    assert report.llm_rows_sent == 1
    assert report.llm_rows_judged == 1
    assert report.llm_label_changes == 1
    output = write_evaluation_report(report, tmp_path / "report.json")
    assert '"llm_metrics"' in output.read_text(encoding="utf-8")


def test_cli_llm_dry_run_reports_selection() -> None:
    result = CliRunner().invoke(
        cli_main.app,
        [
            "linking",
            "score-candidates",
            "--use-fixtures",
            "--dry-run",
            "--use-llm-judge",
            "--llm-dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "dry-run selected=" in result.output


def test_cli_use_llm_judge_can_run_with_fake_client(monkeypatch, tmp_path: Path) -> None:
    judge, client = _judge(tmp_path, _response())

    def _fake_builder(**kwargs: Any) -> LLMRelevanceJudge:
        return judge

    monkeypatch.setattr(cli_main, "_build_cli_llm_judge", _fake_builder)
    result = CliRunner().invoke(
        cli_main.app,
        ["linking", "score-candidates", "--use-fixtures", "--dry-run", "--use-llm-judge"],
    )
    assert result.exit_code == 0, result.output
    assert client.calls

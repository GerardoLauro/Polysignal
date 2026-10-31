"""CLI smoke tests for the Branch 06 stance commands."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from polysignal.cli.main import app
from polysignal.domain import LinkedPost, Outcome, Platform, StanceLabel
from polysignal.domain.enums import ConstraintMatch, LinkMethod, RelevanceLabel
from polysignal.linking.linking_pipeline import write_linked_posts
from polysignal.stance import read_stance_records
from polysignal.storage.parquet_store import write_parquet
from polysignal.storage.paths import get_data_paths


def _seed_linked(root: Path) -> Path:
    paths = get_data_paths(root=root)
    rows = [
        LinkedPost(
            link_id="l1",
            market_id="m-btc",
            post_id="p1",
            platform=Platform.TELEGRAM,
            relevance_score=0.8,
            relevance_label=RelevanceLabel.RELEVANT,
            confidence=0.8,
            method=LinkMethod.DETERMINISTIC_BASELINE,
            explanation="seed",
            matched_entities=["Bitcoin"],
            matched_terms=["Bitcoin", "150k"],
            constraint_match=ConstraintMatch.UNKNOWN,
            created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            post_created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
            market_question="Will Bitcoin hit $150k by December 2026?",
            post_text_preview="Bitcoin is set to reach $150k by December 2026.",
            signal_count=2,
            anchor_score=0.6,
            constraint_score=0.0,
            domain_mismatch=False,
        ),
        LinkedPost(
            link_id="l2",
            market_id="m-gta",
            post_id="p2",
            platform=Platform.TELEGRAM,
            relevance_score=0.7,
            relevance_label=RelevanceLabel.RELEVANT,
            confidence=0.7,
            method=LinkMethod.DETERMINISTIC_BASELINE,
            explanation="seed",
            matched_entities=["GTA VI"],
            matched_terms=["GTA VI"],
            constraint_match=ConstraintMatch.UNKNOWN,
            created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            post_created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
            market_question="Will GTA VI be released before 2026?",
            post_text_preview="Rockstar may delay GTA VI until 2026.",
            signal_count=2,
            anchor_score=0.6,
            constraint_score=0.0,
            domain_mismatch=False,
        ),
        LinkedPost(
            link_id="l3",
            market_id="m-btc",
            post_id="p3",
            platform=Platform.TELEGRAM,
            relevance_score=0.1,
            relevance_label=RelevanceLabel.UNRELATED,
            confidence=0.2,
            method=LinkMethod.DETERMINISTIC_BASELINE,
            explanation="seed",
            matched_entities=[],
            matched_terms=[],
            constraint_match=ConstraintMatch.UNKNOWN,
            created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            post_created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
            market_question="Will Bitcoin hit $150k by December 2026?",
            post_text_preview="off topic",
            signal_count=0,
            anchor_score=0.0,
            constraint_score=0.0,
            domain_mismatch=False,
        ),
    ]
    write_linked_posts(rows, paths.linked_posts_parquet)
    return paths.linked_posts_parquet


def test_stance_score_writes_parquet_and_filters_unrelated(tmp_path: Path) -> None:
    _seed_linked(tmp_path)
    paths = get_data_paths(root=tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["stance", "score", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    persisted = read_stance_records(paths.stance_labels_parquet)
    # The unrelated row must be filtered out.
    assert {r.market_id for r in persisted} == {"m-btc", "m-gta"}
    labels = {r.stance_label for r in persisted}
    assert StanceLabel.SUPPORTS_YES in labels


def test_stance_score_joins_outcome_names_from_explicit_parquet(tmp_path: Path) -> None:
    _seed_linked(tmp_path)
    paths = get_data_paths(root=tmp_path)
    outcomes_path = tmp_path / "market-outcomes.parquet"
    write_parquet(
        [
            Outcome(market_id="m-btc", outcome_id="yes", label="Yes", index=0),
            Outcome(market_id="m-btc", outcome_id="no", label="No", index=1),
        ],
        outcomes_path,
    )
    result = CliRunner().invoke(
        app,
        [
            "stance",
            "score",
            "--output-dir",
            str(tmp_path),
            "--outcomes-path",
            str(outcomes_path),
        ],
    )
    assert result.exit_code == 0, result.output
    records = read_stance_records(paths.stance_labels_parquet)
    btc = next(record for record in records if record.market_id == "m-btc")
    gta = next(record for record in records if record.market_id == "m-gta")
    assert btc.outcome_names == ["Yes", "No"]
    assert gta.outcome_names == []


def test_stance_score_expands_multi_outcome_market_targets(tmp_path: Path) -> None:
    _seed_linked(tmp_path)
    paths = get_data_paths(root=tmp_path)
    outcomes_path = tmp_path / "multi-outcomes.parquet"
    write_parquet(
        [
            Outcome(market_id="m-btc", outcome_id="a", label="Candidate A", index=0),
            Outcome(market_id="m-btc", outcome_id="b", label="Candidate B", index=1),
            Outcome(market_id="m-btc", outcome_id="other", label="Other", index=2),
        ],
        outcomes_path,
    )
    result = CliRunner().invoke(
        app,
        [
            "stance",
            "score",
            "--output-dir",
            str(tmp_path),
            "--outcomes-path",
            str(outcomes_path),
        ],
    )
    assert result.exit_code == 0, result.output
    records = read_stance_records(paths.stance_labels_parquet)
    btc = [record for record in records if record.market_id == "m-btc"]
    assert len(btc) == 3
    assert {record.target_outcome_name for record in btc} == {
        "Candidate A",
        "Candidate B",
        "Other",
    }
    assert all(record.is_multi_outcome for record in btc)


def test_stance_score_missing_input_returns_error(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["stance", "score", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "linked_posts" in result.output.lower()


def test_stance_score_dry_run_does_not_write(tmp_path: Path) -> None:
    _seed_linked(tmp_path)
    paths = get_data_paths(root=tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["stance", "score", "--output-dir", str(tmp_path), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert not paths.stance_labels_parquet.exists()


def test_stance_inspect_filters_by_label(tmp_path: Path) -> None:
    _seed_linked(tmp_path)
    runner = CliRunner()
    runner.invoke(app, ["stance", "score", "--output-dir", str(tmp_path)])
    result = runner.invoke(
        app,
        [
            "stance",
            "inspect",
            "--input",
            str(get_data_paths(root=tmp_path).stance_labels_parquet),
            "--label",
            "supports_yes",
        ],
    )
    assert result.exit_code == 0, result.output


def test_stance_export_review_writes_csv(tmp_path: Path) -> None:
    _seed_linked(tmp_path)
    paths = get_data_paths(root=tmp_path)
    runner = CliRunner()
    runner.invoke(app, ["stance", "score", "--output-dir", str(tmp_path)])
    result = runner.invoke(
        app,
        [
            "stance",
            "export-review",
            "--input",
            str(paths.stance_labels_parquet),
            "--output",
            str(tmp_path / "review.csv"),
        ],
    )
    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader((tmp_path / "review.csv").open("r", encoding="utf-8-sig")))
    assert rows
    assert "predicted_stance_label" in rows[0]


def test_stance_evaluate_runs_without_human_labels(tmp_path: Path) -> None:
    _seed_linked(tmp_path)
    paths = get_data_paths(root=tmp_path)
    runner = CliRunner()
    runner.invoke(app, ["stance", "score", "--output-dir", str(tmp_path)])
    result = runner.invoke(
        app,
        [
            "stance",
            "evaluate",
            "--input",
            str(paths.stance_labels_parquet),
            "--output",
            str(tmp_path / "eval.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "eval.json").read_text(encoding="utf-8"))
    assert payload["total_predictions"] >= 2
    assert payload["accuracy"] is None  # no labels supplied
    assert any("no human labels" in w for w in payload["warnings"])


def test_stance_evaluate_with_human_labels_csv(tmp_path: Path) -> None:
    _seed_linked(tmp_path)
    paths = get_data_paths(root=tmp_path)
    runner = CliRunner()
    runner.invoke(app, ["stance", "score", "--output-dir", str(tmp_path)])
    records = read_stance_records(paths.stance_labels_parquet)
    label_csv = tmp_path / "human.csv"
    with label_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["stance_id", "human_stance_label", "review_status"]
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "stance_id": record.stance_id,
                    "human_stance_label": record.stance_label.value,
                    "review_status": "reviewed",
                }
            )
    result = runner.invoke(
        app,
        [
            "stance",
            "evaluate",
            "--input",
            str(paths.stance_labels_parquet),
            "--human-labels",
            str(label_csv),
            "--output",
            str(tmp_path / "eval.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "eval.json").read_text(encoding="utf-8"))
    assert payload["accuracy"] == 1.0

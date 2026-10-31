"""Branch 06A — outcome-aware deterministic stance baseline.

The subpackage implements the next layer on top of Branch 05 linking:

* :func:`score_stance` is a pure function that maps one (text, market,
  outcome_aliases) tuple to a :class:`StanceRecord`. The scorer is
  conservative by design — when in doubt it returns ``neutral``,
  ``ambiguous``, or ``insufficient_context`` rather than guessing a
  YES/NO direction.
* :func:`run_stance_pipeline` reads ``linked_posts.parquet`` (or a
  user-provided list), filters eligible rows, looks up the segment text
  when available, runs the scorer, and writes ``stance_labels.parquet``.
* :func:`evaluate_stance` compares the predictions against optional
  human labels and emits an evaluation report.
* :func:`export_manual_review` writes a CSV with the human label columns
  pre-seeded for review.

Branch 06B adds an optional, cached LLM stance judge — see
:mod:`polysignal.stance.llm_judge`. The judge is opt-in: passing
``llm_judge=None`` (the default) keeps the deterministic baseline.
Branch 06C joins normalized market outcome labels. Branch 06D expands
non-binary markets into explicit target-aware rows while keeping the
binary YES / NO path backward-compatible.
Branch 06E adds conservative target aliases for multi-outcome mention
detection and LLM audit context.
"""

from polysignal.stance.cues import (
    CueMatch,
    extract_cues,
    no_cues,
    yes_cues,
)
from polysignal.stance.evaluate import StanceEvaluation, evaluate_stance, write_evaluation_report
from polysignal.stance.llm_judge import (
    LLMStanceJudge,
    LLMStanceJudgePolicy,
    LLMStanceJudgeResult,
    LLMStanceVerdict,
)
from polysignal.stance.models import (
    HumanStanceLabel,
    StanceRecord,
    read_stance_records,
    write_stance_records,
)
from polysignal.stance.pipeline import (
    StancePipelineResult,
    run_stance_pipeline,
)
from polysignal.stance.review import write_manual_stance_review
from polysignal.stance.scorer import (
    ScorerConfig,
    StanceVerdict,
    score_stance,
)
from polysignal.stance.target_aliases import (
    generate_target_aliases,
    mentions_target_alias,
    normalize_target_alias,
)

__all__ = [
    "CueMatch",
    "HumanStanceLabel",
    "LLMStanceJudge",
    "LLMStanceJudgePolicy",
    "LLMStanceJudgeResult",
    "LLMStanceVerdict",
    "ScorerConfig",
    "StanceEvaluation",
    "StancePipelineResult",
    "StanceRecord",
    "StanceVerdict",
    "evaluate_stance",
    "extract_cues",
    "generate_target_aliases",
    "mentions_target_alias",
    "no_cues",
    "normalize_target_alias",
    "read_stance_records",
    "run_stance_pipeline",
    "score_stance",
    "write_evaluation_report",
    "write_manual_stance_review",
    "write_stance_records",
    "yes_cues",
]

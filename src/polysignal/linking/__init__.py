"""Contract ↔ post linking subsystem.

Branch 03/04 entry points (legacy):

* :class:`HeuristicLinker` — token-overlap candidate generator.
* :class:`LinkingPipeline` — heuristic + optional LLM review pipeline.

Branch 05 entry points (semantic relevance):

* :class:`RelevanceScorer` — deterministic per-pair scorer.
* :class:`RelevanceConfig` — tunable thresholds for the scorer.
* :class:`RelevanceJudgement` — per-pair verdict.
* :class:`CandidatePair` — (market, post) pair carrying retrieval context.
* :func:`run_linking_from_records` — score Branch 04 records → ``LinkedPost``.
* :func:`run_linking_cross_product` — score every (market, post) pair.
* :func:`write_linked_posts` / :func:`read_linked_posts` — parquet IO.
* :func:`evaluate_predictions` — offline metric computation.

The two halves coexist intentionally: Branch 05's scorer subsumes the
old heuristic linker for new code paths, but the legacy ``Link`` model
and ``LinkingPipeline`` are kept so older modules and tests do not
break. See ``docs/BRANCH_05_LINKING.md`` for the migration story.
"""

from polysignal.linking.candidates import (
    CandidateGenerationDiagnostics,
    CandidateGenerationResult,
    CandidateMode,
    CandidatePair,
    build_pairs_cross_product,
    build_pairs_from_records,
    build_pairs_from_records_with_diagnostics,
)
from polysignal.linking.constraints import (
    ConstraintMatchResult,
    MarketConstraint,
    extract_market_constraints,
    match_constraint,
)
from polysignal.linking.coverage import (
    LinkingCoverageReport,
    build_linking_coverage_report,
    write_linking_coverage_report,
)
from polysignal.linking.embeddings import (
    CachedEmbeddingProvider,
    EmbeddingProvider,
    NullEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    StaticMockEmbeddingProvider,
    build_default_provider,
)
from polysignal.linking.evaluate import (
    EvaluationReport,
    LabelMetrics,
    evaluate_predictions,
    write_evaluation_report,
)
from polysignal.linking.evidence import (
    EvidenceExportResult,
    export_evidence,
)
from polysignal.linking.fixtures import (
    LabeledPair,
    fixture_labels,
    fixture_markets,
    fixture_posts,
)
from polysignal.linking.heuristic import HeuristicLinker, LinkCandidate
from polysignal.linking.linking_pipeline import (
    LinkingRunResult,
    read_linked_posts,
    run_linking_cross_product,
    run_linking_from_records,
    score_pairs,
    summarize_by_label,
    write_linked_posts,
)
from polysignal.linking.llm_assisted import (
    LLMJudgePolicy,
    LLMJudgeResult,
    LLMJudgeStats,
    LLMRelevanceJudge,
    LLMRelevanceVerdict,
)
from polysignal.linking.pipeline import LinkingConfig, LinkingPipeline
from polysignal.linking.relevance import (
    MarketProfile,
    RelevanceConfig,
    RelevanceJudgement,
    RelevanceScorer,
    build_market_profile,
)

__all__ = [
    "CachedEmbeddingProvider",
    "CandidateGenerationDiagnostics",
    "CandidateGenerationResult",
    "CandidateMode",
    "CandidatePair",
    "ConstraintMatchResult",
    "EmbeddingProvider",
    "EvaluationReport",
    "EvidenceExportResult",
    "HeuristicLinker",
    "LLMJudgePolicy",
    "LLMJudgeResult",
    "LLMJudgeStats",
    "LLMRelevanceJudge",
    "LLMRelevanceVerdict",
    "LabelMetrics",
    "LabeledPair",
    "LinkCandidate",
    "LinkingConfig",
    "LinkingCoverageReport",
    "LinkingPipeline",
    "LinkingRunResult",
    "MarketConstraint",
    "MarketProfile",
    "NullEmbeddingProvider",
    "RelevanceConfig",
    "RelevanceJudgement",
    "RelevanceScorer",
    "SentenceTransformerEmbeddingProvider",
    "StaticMockEmbeddingProvider",
    "build_default_provider",
    "build_linking_coverage_report",
    "build_market_profile",
    "build_pairs_cross_product",
    "build_pairs_from_records",
    "build_pairs_from_records_with_diagnostics",
    "evaluate_predictions",
    "export_evidence",
    "extract_market_constraints",
    "fixture_labels",
    "fixture_markets",
    "fixture_posts",
    "match_constraint",
    "read_linked_posts",
    "run_linking_cross_product",
    "run_linking_from_records",
    "score_pairs",
    "summarize_by_label",
    "write_evaluation_report",
    "write_linked_posts",
    "write_linking_coverage_report",
]

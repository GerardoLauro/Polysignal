"""Tests for ``polysignal.linking.heuristic``."""

from __future__ import annotations

from polysignal.linking.heuristic import HeuristicLinker
from polysignal.linking.pipeline import LinkingConfig, LinkingPipeline


def test_score_unrelated_post_has_zero_confidence(binary_market, post_unrelated):
    linker = HeuristicLinker()
    candidate = linker.score(post_unrelated, binary_market)
    assert candidate.confidence == 0.0
    assert candidate.evidence == []


def test_score_relevant_post_has_positive_confidence(binary_market, post_about_market):
    linker = HeuristicLinker()
    candidate = linker.score(post_about_market, binary_market)
    # The post and question both mention "candidate" so confidence > 0.
    assert candidate.confidence > 0.0
    assert "candidate" in candidate.evidence


def test_pipeline_filters_below_threshold(binary_market, post_unrelated, post_about_market):
    pipeline = LinkingPipeline(config=LinkingConfig(accept_threshold=0.05))
    links = pipeline.run([post_unrelated, post_about_market], [binary_market])
    # The relevant post is linked, the unrelated one is not.
    assert len(links) == 1
    assert links[0].post_id == post_about_market.post_id
    assert links[0].market_id == binary_market.market_id


def test_pipeline_strict_threshold_drops_everything(binary_market, post_about_market):
    pipeline = LinkingPipeline(config=LinkingConfig(accept_threshold=0.99))
    assert pipeline.run([post_about_market], [binary_market]) == []

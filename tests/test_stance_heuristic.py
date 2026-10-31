"""Tests for ``polysignal.nlp.stance.heuristic``."""

from __future__ import annotations

from datetime import datetime, timezone

from polysignal.domain import (
    Platform,
    SocialPost,
    StanceLabel,
)
from polysignal.nlp.stance.heuristic import HeuristicStanceClassifier


def _post(text: str, post_id: str = "p") -> SocialPost:
    return SocialPost(
        post_id=post_id,
        platform=Platform.REDDIT,
        created_at=datetime(2025, 3, 1, tzinfo=timezone.utc),
        text=text,
    )


def test_unrelated_post_returns_unrelated(binary_market):
    clf = HeuristicStanceClassifier()
    stance = clf.classify(_post("I love sushi and ramen."), binary_market)
    assert stance.label is StanceLabel.UNRELATED
    assert stance.confidence == 0.0


def test_post_supporting_no_is_classified_as_no(binary_market, post_about_market):
    clf = HeuristicStanceClassifier()
    stance = clf.classify(post_about_market, binary_market)
    # The fixture says "candidate X is collapsing ... loss" — NO leaning.
    assert stance.label is StanceLabel.SUPPORTS_NO
    assert 0.0 < stance.confidence <= 0.6


def test_post_supporting_yes(binary_market):
    clf = HeuristicStanceClassifier()
    stance = clf.classify(
        _post("Candidate X will win the election; success is confirmed."),
        binary_market,
    )
    assert stance.label is StanceLabel.SUPPORTS_YES
    assert stance.confidence > 0.0


def test_post_mentioning_market_but_neutral(binary_market):
    clf = HeuristicStanceClassifier()
    stance = clf.classify(
        _post("Candidate election day is tomorrow according to the news."),
        binary_market,
    )
    assert stance.label is StanceLabel.NEUTRAL
    assert stance.confidence < 0.3

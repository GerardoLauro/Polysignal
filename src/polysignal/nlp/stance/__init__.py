"""Outcome-aware stance detection (heuristic baseline + LLM-assisted stub)."""

from polysignal.nlp.stance.base import StanceClassifier
from polysignal.nlp.stance.heuristic import HeuristicStanceClassifier

__all__ = ["HeuristicStanceClassifier", "StanceClassifier"]

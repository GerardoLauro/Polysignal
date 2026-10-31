"""Common interface for stance classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from polysignal.domain import Market, SocialPost, Stance


class StanceClassifier(ABC):
    """Maps a post + market pair to an outcome-aware stance.

    Implementations may use heuristics, classical ML, or an LLM. The
    rest of the pipeline only relies on this interface so they can be
    swapped via configuration.
    """

    @abstractmethod
    def classify(self, post: SocialPost, market: Market) -> Stance:
        """Return the stance of ``post`` toward ``market``.

        The implementation must always return a ``Stance``; if it has
        no signal, it should return one with label ``UNRELATED`` or
        ``AMBIGUOUS`` and a low confidence value.
        """


__all__ = ["StanceClassifier"]

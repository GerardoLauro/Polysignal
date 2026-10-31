"""Deterministic, transparent heuristic linker (post ↔ market).

The heuristic produces ``Link`` candidates with an explicit confidence
score derived from:

* Jaccard similarity between the token sets of the post and the
  market question;
* number of overlapping content tokens (stopwords already stripped);
* presence of the market slug (or close tokens) in the post.

It is **not** the only linker. The pipeline can combine it with an
LLM-assisted relevance check (see ``llm_assisted.py``).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone

from polysignal.domain import Link, LinkMethod, Market, SocialPost
from polysignal.nlp.normalize import jaccard, token_set


@dataclass(frozen=True)
class LinkCandidate:
    market_id: str
    post_id: str
    confidence: float
    evidence: list[str]


class HeuristicLinker:
    """Score candidate (post, market) pairs by token overlap."""

    def __init__(
        self,
        *,
        min_overlap: int = 1,
        min_jaccard: float = 0.05,
        confidence_floor: float = 0.0,
    ) -> None:
        self.min_overlap = min_overlap
        self.min_jaccard = min_jaccard
        self.confidence_floor = confidence_floor

    # ---------------------------------------------------------------- core
    def score(self, post: SocialPost, market: Market) -> LinkCandidate:
        post_tokens = token_set(post.text_normalized or post.text)
        market_tokens = token_set(market.question)
        slug_tokens = token_set(market.slug.replace("-", " "))
        market_tokens |= slug_tokens

        if not post_tokens or not market_tokens:
            return LinkCandidate(
                market_id=market.market_id,
                post_id=post.post_id,
                confidence=0.0,
                evidence=[],
            )

        overlap = post_tokens & market_tokens
        j = jaccard(post_tokens, market_tokens)
        overlap_count = len(overlap)

        # Confidence is a smooth function of overlap + Jaccard, capped
        # at 0.9 so heuristic links never look as confident as a
        # validated link.
        raw = 0.6 * j + 0.4 * min(1.0, overlap_count / 8.0)
        confidence = max(self.confidence_floor, min(0.9, raw))

        if overlap_count < self.min_overlap or j < self.min_jaccard:
            confidence = 0.0

        return LinkCandidate(
            market_id=market.market_id,
            post_id=post.post_id,
            confidence=round(confidence, 4),
            evidence=sorted(overlap),
        )

    # ----------------------------------------------------------- batch API
    def link(self, posts: Iterable[SocialPost], markets: Iterable[Market]) -> list[Link]:
        """Build a list of ``Link`` records above the confidence threshold."""
        materialized_markets = list(markets)
        now = datetime.now(tz=timezone.utc)
        links: list[Link] = []
        for post in posts:
            for market in materialized_markets:
                candidate = self.score(post, market)
                if candidate.confidence <= 0.0:
                    continue
                links.append(
                    Link(
                        post_id=post.post_id,
                        platform=post.platform,
                        market_id=market.market_id,
                        method=LinkMethod.HEURISTIC,
                        confidence=candidate.confidence,
                        created_at=now,
                        evidence=candidate.evidence,
                    )
                )
        return links


__all__ = ["HeuristicLinker", "LinkCandidate"]

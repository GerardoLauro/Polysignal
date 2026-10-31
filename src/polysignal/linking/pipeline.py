"""Linking pipeline — heuristic first, optionally validated by an LLM.

The pipeline is the only public entry point that callers should use.
It hides the policy of "when to ask the LLM" from the rest of the
codebase, which keeps the behavior centralized and changeable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from polysignal.domain import Link, LinkMethod, Market, SocialPost
from polysignal.linking.heuristic import HeuristicLinker
from polysignal.linking.llm_assisted import LLMRelevanceLinker


@dataclass(frozen=True)
class LinkingConfig:
    accept_threshold: float = 0.6
    review_threshold: float = 0.3  # below: reject; between: send to LLM if available
    use_llm: bool = False


class LinkingPipeline:
    """Two-stage linker: heuristic first, then optional LLM review."""

    def __init__(
        self,
        heuristic: HeuristicLinker | None = None,
        llm_linker: LLMRelevanceLinker | None = None,
        config: LinkingConfig | None = None,
    ) -> None:
        self.heuristic = heuristic or HeuristicLinker()
        self.llm_linker = llm_linker
        self.config = config or LinkingConfig()

    def run(self, posts: Iterable[SocialPost], markets: Iterable[Market]) -> list[Link]:
        materialized_markets = list(markets)
        accepted: list[Link] = []
        for post in posts:
            for market in materialized_markets:
                candidate = self.heuristic.score(post, market)
                if candidate.confidence >= self.config.accept_threshold:
                    accepted.append(_to_link(post, market, candidate, LinkMethod.HEURISTIC))
                    continue

                if (
                    self.config.use_llm
                    and self.llm_linker is not None
                    and candidate.confidence >= self.config.review_threshold
                ):
                    verdict = self.llm_linker.verdict(post, market)
                    link = self.llm_linker.to_link(post, market, verdict)
                    if link is not None:
                        # Blend confidences: average heuristic + LLM
                        blended = round((candidate.confidence + verdict.confidence) / 2.0, 4)
                        link = link.model_copy(
                            update={"confidence": blended, "method": LinkMethod.HYBRID}
                        )
                        accepted.append(link)
        return accepted


def _to_link(post: SocialPost, market: Market, candidate, method: LinkMethod) -> Link:
    from datetime import datetime, timezone

    return Link(
        post_id=post.post_id,
        platform=post.platform,
        market_id=market.market_id,
        method=method,
        confidence=candidate.confidence,
        created_at=datetime.now(tz=timezone.utc),
        evidence=candidate.evidence,
    )


__all__ = ["LinkingConfig", "LinkingPipeline"]

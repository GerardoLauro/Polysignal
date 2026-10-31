"""Cypher statements used by Graph Loader v1.

Every write is batched through ``UNWIND $rows`` and uses ``MERGE`` on stable
keys. Keeping Cypher in one module makes dry-run planning and offline tests
independent from a running Neo4j instance.
"""

from __future__ import annotations

UPSERT_CONTRACTS = """
UNWIND $rows AS row
MERGE (c:Contract {market_id: row.market_id})
SET c.slug = row.slug,
    c.question = row.question,
    c.description = row.description,
    c.domain = row.domain,
    c.created_at = datetime(row.created_at),
    c.close_time = CASE WHEN row.close_time IS NULL THEN NULL ELSE datetime(row.close_time) END,
    c.resolved_at = CASE WHEN row.resolved_at IS NULL THEN NULL ELSE datetime(row.resolved_at) END,
    c.is_resolved = row.is_resolved,
    c.is_closed = row.is_closed,
    c.resolution = row.resolution,
    c.tags = row.tags,
    c.source_url = row.source_url,
    c.event_id = row.event_id,
    c.condition_id = row.condition_id,
    c.volume = row.volume,
    c.liquidity = row.liquidity
""".strip()

UPSERT_DOMAINS = """
UNWIND $rows AS row
MERGE (d:Domain {name: row.name})
""".strip()

UPSERT_OUTCOMES = """
UNWIND $rows AS row
MERGE (o:Outcome {market_id: row.market_id, outcome_id: row.outcome_id})
SET o.label = row.label,
    o.index = row.index,
    o.current_price = row.current_price
""".strip()

UPSERT_PLATFORMS = """
UNWIND $rows AS row
MERGE (p:Platform {name: row.name})
""".strip()

UPSERT_POSTS = """
UNWIND $rows AS row
MERGE (p:Post {platform: row.platform, post_id: row.post_id})
SET p.source_id = row.source_id,
    p.url = row.url,
    p.author_id = row.author_id,
    p.author_name = row.author_name,
    p.created_at = CASE WHEN row.created_at IS NULL THEN NULL ELSE datetime(row.created_at) END,
    p.text = row.text,
    p.title = row.title,
    p.channel = row.channel,
    p.engagement_score = row.engagement_score,
    p.language = row.language,
    p.raw_path = row.raw_path,
    p.collected_at = datetime(row.collected_at)
""".strip()

UPSERT_AUTHORS = """
UNWIND $rows AS row
MERGE (a:Author {platform: row.platform, author_id: row.author_id})
SET a.handle = row.handle
""".strip()

LINK_CONTRACT_DOMAINS = """
UNWIND $rows AS row
MATCH (c:Contract {market_id: row.market_id})
MATCH (d:Domain {name: row.domain})
MERGE (c)-[:BELONGS_TO]->(d)
""".strip()

LINK_CONTRACT_OUTCOMES = """
UNWIND $rows AS row
MATCH (c:Contract {market_id: row.market_id})
MATCH (o:Outcome {market_id: row.market_id, outcome_id: row.outcome_id})
MERGE (c)-[:HAS_OUTCOME]->(o)
""".strip()

LINK_POST_PLATFORMS = """
UNWIND $rows AS row
MATCH (p:Post {platform: row.platform, post_id: row.post_id})
MATCH (platform:Platform {name: row.platform})
MERGE (p)-[:POSTED_ON]->(platform)
""".strip()

LINK_AUTHOR_POSTS = """
UNWIND $rows AS row
MATCH (a:Author {platform: row.platform, author_id: row.author_id})
MATCH (p:Post {platform: row.platform, post_id: row.post_id})
MERGE (a)-[:WROTE]->(p)
""".strip()

LINK_POST_CONTRACTS = """
UNWIND $rows AS row
MATCH (p:Post {platform: row.platform, post_id: row.parent_post_id})
MATCH (c:Contract {market_id: row.market_id})
MERGE (p)-[r:LINKED_TO {link_id: row.link_id}]->(c)
SET r.segment_id = row.segment_id,
    r.source_linked_post_id = row.source_linked_post_id,
    r.retrieval_query = row.retrieval_query,
    r.relevance_label = row.relevance_label,
    r.final_relevance_label = row.final_relevance_label,
    r.final_relevance_score = row.final_relevance_score,
    r.final_confidence = row.final_confidence,
    r.final_method = row.final_method,
    r.constraint_match = row.constraint_match,
    r.created_at = datetime(row.created_at),
    r.source_post_created_at = datetime(row.source_post_created_at),
    r.accepted = row.accepted,
    r.temporal_valid = row.temporal_valid,
    r.temporal_anchor = CASE WHEN row.temporal_anchor IS NULL THEN NULL ELSE datetime(row.temporal_anchor) END,
    r.temporal_anchor_kind = row.temporal_anchor_kind
""".strip()

LINK_POST_SUPPORTS_OUTCOME = """
UNWIND $rows AS row
MATCH (p:Post {platform: row.platform, post_id: row.parent_post_id})
MATCH (o:Outcome {market_id: row.market_id, outcome_id: row.outcome_id})
MERGE (p)-[r:SUPPORTS {stance_id: row.stance_id}]->(o)
SET r.stance_label = row.stance_label,
    r.confidence = row.confidence,
    r.method = row.method,
    r.rationale = row.rationale,
    r.created_at = datetime(row.created_at),
    r.source_post_created_at = datetime(row.source_post_created_at),
    r.temporal_valid = row.temporal_valid,
    r.temporal_anchor = CASE WHEN row.temporal_anchor IS NULL THEN NULL ELSE datetime(row.temporal_anchor) END,
    r.temporal_anchor_kind = row.temporal_anchor_kind,
    r.segment_id = row.segment_id,
    r.source_stance_post_id = row.source_stance_post_id
""".strip()

LINK_POST_OPPOSES_OUTCOME = """
UNWIND $rows AS row
MATCH (p:Post {platform: row.platform, post_id: row.parent_post_id})
MATCH (o:Outcome {market_id: row.market_id, outcome_id: row.outcome_id})
MERGE (p)-[r:OPPOSES {stance_id: row.stance_id}]->(o)
SET r.stance_label = row.stance_label,
    r.confidence = row.confidence,
    r.method = row.method,
    r.rationale = row.rationale,
    r.created_at = datetime(row.created_at),
    r.source_post_created_at = datetime(row.source_post_created_at),
    r.temporal_valid = row.temporal_valid,
    r.temporal_anchor = CASE WHEN row.temporal_anchor IS NULL THEN NULL ELSE datetime(row.temporal_anchor) END,
    r.temporal_anchor_kind = row.temporal_anchor_kind,
    r.segment_id = row.segment_id,
    r.source_stance_post_id = row.source_stance_post_id
""".strip()

LIVE_NODE_COUNTS = """
MATCH (n)
UNWIND labels(n) AS label
RETURN label, count(*) AS count
ORDER BY label
""".strip()

LIVE_RELATIONSHIP_COUNTS = """
MATCH ()-[r]->()
RETURN type(r) AS relationship, count(*) AS count
ORDER BY relationship
""".strip()


# ---------------------------------------------------------------------------
# Read-side analytical query builders (Branch 07 graph closure).
#
# Each builder returns a :class:`GraphQuery` carrying parameterized Cypher
# (user input always flows through ``$params`` — never string-interpolated)
# plus a default ``params`` dict. They are deterministic and unit-testable
# and require no live Neo4j connection to construct. The ``graph queries``
# CLI prints them in ``--dry-run`` mode and can execute them against a live
# database otherwise.
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field  # noqa: E402
from typing import Any  # noqa: E402


@dataclass(frozen=True)
class GraphQuery:
    """One named, parameterized read query."""

    name: str
    description: str
    cypher: str
    params: dict[str, Any] = field(default_factory=dict)


def contract_coverage(*, limit: int = 50) -> GraphQuery:
    """Per-contract linked-post coverage, author/channel breadth, and stance."""
    cypher = """
MATCH (c:Contract)
OPTIONAL MATCH (p:Post)-[r:LINKED_TO {accepted: true, temporal_valid: true}]->(c)
WHERE r IS NULL OR r.final_relevance_label IN ['relevant', 'weakly_relevant']
WITH c, count(DISTINCT p) AS linked_posts,
     avg(r.final_confidence) AS avg_link_confidence,
     count(DISTINCT p.channel) AS channels
OPTIONAL MATCH (a:Author)-[:WROTE]->(:Post)-[ar:LINKED_TO {accepted: true, temporal_valid: true}]->(c)
WHERE ar IS NULL OR ar.final_relevance_label IN ['relevant', 'weakly_relevant']
WITH c, linked_posts, avg_link_confidence, channels, count(DISTINCT a) AS authors
OPTIONAL MATCH (:Post)-[s:SUPPORTS {temporal_valid: true}]->(:Outcome {market_id: c.market_id})
OPTIONAL MATCH (:Post)-[o:OPPOSES {temporal_valid: true}]->(:Outcome {market_id: c.market_id})
RETURN c.market_id AS market_id, c.question AS question, c.domain AS domain,
       c.is_resolved AS is_resolved,
       linked_posts, avg_link_confidence, authors, channels,
       count(DISTINCT s) AS supports, count(DISTINCT o) AS opposes
ORDER BY linked_posts DESC, avg_link_confidence DESC
LIMIT $limit
""".strip()
    return GraphQuery(
        "contract-coverage",
        "Per-contract linked-post count, avg link confidence, author/channel breadth, stance counts.",
        cypher,
        {"limit": int(limit)},
    )


def top_contracts(*, limit: int = 20) -> GraphQuery:
    """Rank contracts by linked-post count weighted by average confidence."""
    cypher = """
MATCH (c:Contract)
OPTIONAL MATCH (p:Post)-[r:LINKED_TO {accepted: true, temporal_valid: true}]->(c)
WHERE r IS NULL OR r.final_relevance_label IN ['relevant', 'weakly_relevant']
WITH c, count(DISTINCT p) AS linked_posts,
     coalesce(avg(r.final_confidence), 0.0) AS avg_link_confidence
RETURN c.market_id AS market_id, c.question AS question, c.domain AS domain,
       linked_posts, avg_link_confidence,
       linked_posts * avg_link_confidence AS score
ORDER BY score DESC, linked_posts DESC
LIMIT $limit
""".strip()
    return GraphQuery(
        "top-contracts",
        "Top contracts ranked by linked-post count * average link confidence.",
        cypher,
        {"limit": int(limit)},
    )


def author_activity(*, limit: int = 50) -> GraphQuery:
    """Author/source activity: posts, distinct linked contracts, avg confidence."""
    cypher = """
MATCH (a:Author)-[:WROTE]->(p:Post)
WITH a, count(DISTINCT p) AS posts
OPTIONAL MATCH (a)-[:WROTE]->(:Post)-[r:LINKED_TO {accepted: true, temporal_valid: true}]->(c:Contract)
WHERE r IS NULL OR r.final_relevance_label IN ['relevant', 'weakly_relevant']
RETURN a.platform AS platform, a.author_id AS author_id, a.handle AS handle,
       posts, count(DISTINCT c) AS linked_contracts,
       avg(r.final_confidence) AS avg_link_confidence
ORDER BY linked_contracts DESC, posts DESC
LIMIT $limit
""".strip()
    return GraphQuery(
        "author-activity",
        "Per-author/source post count, distinct linked contracts, and avg link confidence.",
        cypher,
        {"limit": int(limit)},
    )


def platform_coverage(*, limit: int = 100) -> GraphQuery:
    """Platform + channel coverage: messages, linked messages, covered contracts.

    Pure Cypher (no APOC, which is not guaranteed on the Community image).
    """
    cypher = """
MATCH (p:Post)
OPTIONAL MATCH (p)-[r:LINKED_TO {accepted: true, temporal_valid: true}]->(c:Contract)
WHERE r IS NULL OR r.final_relevance_label IN ['relevant', 'weakly_relevant']
WITH p.platform AS platform, p.channel AS channel,
     count(DISTINCT p) AS posts,
     count(DISTINCT CASE WHEN c IS NOT NULL THEN p END) AS linked_posts,
     count(DISTINCT c) AS covered_contracts
RETURN platform, channel, posts, linked_posts, covered_contracts
ORDER BY covered_contracts DESC, posts DESC
LIMIT $limit
""".strip()
    return GraphQuery(
        "platform-coverage",
        "Per-platform/channel message count, linked-message count, and covered contracts.",
        cypher,
        {"limit": int(limit)},
    )


def contract_neighborhood(*, market_id: str, limit: int = 50) -> GraphQuery:
    """One contract's outcomes, linked posts/authors/channels, and stance edges."""
    cypher = """
MATCH (c:Contract {market_id: $market_id})
OPTIONAL MATCH (c)-[:HAS_OUTCOME]->(o:Outcome)
WITH c, collect(DISTINCT o.label) AS outcomes
OPTIONAL MATCH (p:Post)-[r:LINKED_TO {accepted: true, temporal_valid: true}]->(c)
WHERE r IS NULL OR r.final_relevance_label IN ['relevant', 'weakly_relevant']
WITH c, outcomes, p, r
ORDER BY r.final_confidence DESC
WITH c, outcomes,
     count(DISTINCT p) AS linked_posts,
     collect(DISTINCT p.channel)[0..$limit] AS channels,
     collect(DISTINCT p.author_name)[0..$limit] AS authors
OPTIONAL MATCH (:Post)-[s:SUPPORTS {temporal_valid: true}]->(so:Outcome {market_id: c.market_id})
OPTIONAL MATCH (:Post)-[op:OPPOSES {temporal_valid: true}]->(oo:Outcome {market_id: c.market_id})
RETURN c.market_id AS market_id, c.question AS question, c.domain AS domain,
       outcomes, linked_posts, channels, authors,
       count(DISTINCT s) AS supports, count(DISTINCT op) AS opposes
""".strip()
    return GraphQuery(
        "contract-neighborhood",
        "Outcomes, linked posts, authors, channels, and stance edges for one contract.",
        cypher,
        {"market_id": str(market_id), "limit": int(limit)},
    )


def stance_summary(*, limit: int = 100) -> GraphQuery:
    """SUPPORTS/OPPOSES counts per (contract, outcome); safe when no stance edges."""
    cypher = """
MATCH (c:Contract)-[:HAS_OUTCOME]->(o:Outcome)
OPTIONAL MATCH (:Post)-[s:SUPPORTS {temporal_valid: true}]->(o)
OPTIONAL MATCH (:Post)-[op:OPPOSES {temporal_valid: true}]->(o)
WITH c, o, count(s) AS supports, count(op) AS opposes,
     avg(s.confidence) AS avg_support_confidence
WHERE supports > 0 OR opposes > 0
RETURN c.market_id AS market_id, o.label AS outcome,
       supports, opposes, avg_support_confidence
ORDER BY supports DESC, opposes DESC
LIMIT $limit
""".strip()
    return GraphQuery(
        "stance-summary",
        "Directional stance (SUPPORTS/OPPOSES) counts per contract outcome; empty-safe.",
        cypher,
        {"limit": int(limit)},
    )


def graph_health() -> GraphQuery:
    """Integrity counters: orphan posts, contracts without outcomes/links, etc."""
    cypher = """
RETURN
  COUNT { MATCH (p:Post) WHERE NOT EXISTS { (:Author)-[:WROTE]->(p) } } AS posts_without_author,
  COUNT { MATCH (p:Post) WHERE NOT EXISTS { (p)-[:POSTED_ON]->(:Platform) } } AS posts_without_platform,
  COUNT { MATCH (c:Contract) WHERE NOT EXISTS { (c)-[:HAS_OUTCOME]->(:Outcome) } } AS contracts_without_outcomes,
  COUNT { MATCH (c:Contract) WHERE NOT EXISTS {
    (:Post)-[r:LINKED_TO {accepted: true, temporal_valid: true}]->(c)
    WHERE r.final_relevance_label IN ['relevant', 'weakly_relevant']
  } } AS contracts_without_links,
  COUNT { MATCH (o:Outcome) WHERE NOT EXISTS { (:Contract)-[:HAS_OUTCOME]->(o) } } AS orphan_outcomes,
  COUNT { MATCH (:Post)-[s:SUPPORTS]->(o:Outcome) WHERE o.market_id IS NULL } AS supports_without_outcome
""".strip()
    return GraphQuery(
        "health",
        "Graph integrity counters (orphan posts, contracts without outcomes/links, etc.).",
        cypher,
        {},
    )


# Registry for the `graph queries list` command and dispatch. Builders that
# need required arguments (e.g. contract-neighborhood) are handled explicitly
# by the CLI; the registry still advertises them for discoverability.
QUERY_BUILDERS: dict[str, Any] = {
    "contract-coverage": contract_coverage,
    "top-contracts": top_contracts,
    "author-activity": author_activity,
    "platform-coverage": platform_coverage,
    "contract-neighborhood": contract_neighborhood,
    "stance-summary": stance_summary,
    "health": graph_health,
}


def list_queries() -> list[tuple[str, str]]:
    """Return (name, description) for every available read query."""
    descriptions = {
        "contract-coverage": contract_coverage().description,
        "top-contracts": top_contracts().description,
        "author-activity": author_activity().description,
        "platform-coverage": platform_coverage().description,
        "contract-neighborhood": contract_neighborhood(market_id="<market_id>").description,
        "stance-summary": stance_summary().description,
        "health": graph_health().description,
    }
    return sorted(descriptions.items())


__all__ = [
    "LINK_AUTHOR_POSTS",
    "LINK_CONTRACT_DOMAINS",
    "LINK_CONTRACT_OUTCOMES",
    "LINK_POST_CONTRACTS",
    "LINK_POST_OPPOSES_OUTCOME",
    "LINK_POST_PLATFORMS",
    "LINK_POST_SUPPORTS_OUTCOME",
    "LIVE_NODE_COUNTS",
    "LIVE_RELATIONSHIP_COUNTS",
    "QUERY_BUILDERS",
    "UPSERT_AUTHORS",
    "UPSERT_CONTRACTS",
    "UPSERT_DOMAINS",
    "UPSERT_OUTCOMES",
    "UPSERT_PLATFORMS",
    "UPSERT_POSTS",
    "GraphQuery",
    "author_activity",
    "contract_coverage",
    "contract_neighborhood",
    "graph_health",
    "list_queries",
    "platform_coverage",
    "stance_summary",
    "top_contracts",
]

"""Neo4j schema as Python constants + Cypher generators.

The schema is described as Python constants instead of being scattered
across migration scripts so it can be:

* applied programmatically against any Neo4j instance
  (``schema.apply(driver)`` in the future);
* inspected from tests and the CLI without a live database;
* kept in lock-step with ``docs/GRAPH_MODEL.md``.
"""

from __future__ import annotations

from enum import Enum


class Label(str, Enum):
    """Node labels in the PolySignal graph."""

    CONTRACT = "Contract"
    OUTCOME = "Outcome"
    POST = "Post"
    COMMENT = "Comment"
    AUTHOR = "Author"
    PLATFORM = "Platform"
    ENTITY = "Entity"
    TOPIC = "Topic"
    DOMAIN = "Domain"


class Rel(str, Enum):
    """Relationship types in the PolySignal graph."""

    HAS_OUTCOME = "HAS_OUTCOME"
    BELONGS_TO = "BELONGS_TO"
    POSTED_ON = "POSTED_ON"
    WROTE = "WROTE"
    REPLIED_TO = "REPLIED_TO"
    MENTIONS = "MENTIONS"
    LINKED_TO = "LINKED_TO"
    SUPPORTS = "SUPPORTS"
    OPPOSES = "OPPOSES"
    DISCUSSES_TOPIC = "DISCUSSES_TOPIC"
    RELATED_TO = "RELATED_TO"


def constraint_statements() -> list[str]:
    """Cypher CREATE CONSTRAINT statements for the canonical schema.

    Idempotent thanks to ``IF NOT EXISTS``.
    """
    return [
        # Composite uniqueness constraints work on Neo4j Community. Loader
        # planning separately validates that required key fields are present.
        f"CREATE CONSTRAINT contract_key IF NOT EXISTS "
        f"FOR (c:{Label.CONTRACT.value}) REQUIRE c.market_id IS UNIQUE",
        # Outcomes: composite (market_id, outcome_id).
        f"CREATE CONSTRAINT outcome_key IF NOT EXISTS "
        f"FOR (o:{Label.OUTCOME.value}) REQUIRE (o.market_id, o.outcome_id) IS UNIQUE",
        # Posts: composite (platform, post_id).
        f"CREATE CONSTRAINT post_key IF NOT EXISTS "
        f"FOR (p:{Label.POST.value}) REQUIRE (p.platform, p.post_id) IS UNIQUE",
        # Authors: composite (platform, author_id).
        f"CREATE CONSTRAINT author_key IF NOT EXISTS "
        f"FOR (a:{Label.AUTHOR.value}) REQUIRE (a.platform, a.author_id) IS UNIQUE",
        # Platforms: unique name.
        f"CREATE CONSTRAINT platform_name_unique IF NOT EXISTS "
        f"FOR (p:{Label.PLATFORM.value}) REQUIRE p.name IS UNIQUE",
        # Entities: composite (type, text_norm).
        f"CREATE CONSTRAINT entity_key IF NOT EXISTS "
        f"FOR (e:{Label.ENTITY.value}) REQUIRE (e.type, e.text_norm) IS UNIQUE",
        # Topics: unique topic_id.
        f"CREATE CONSTRAINT topic_id_unique IF NOT EXISTS "
        f"FOR (t:{Label.TOPIC.value}) REQUIRE t.topic_id IS UNIQUE",
        # Domains: unique name.
        f"CREATE CONSTRAINT domain_name_unique IF NOT EXISTS "
        f"FOR (d:{Label.DOMAIN.value}) REQUIRE d.name IS UNIQUE",
    ]


def index_statements() -> list[str]:
    """Indexes that pay off for the typical analytic queries."""
    return [
        f"CREATE INDEX post_created_idx IF NOT EXISTS FOR (p:{Label.POST.value}) ON (p.created_at)",
        f"CREATE INDEX contract_resolved_idx IF NOT EXISTS "
        f"FOR (c:{Label.CONTRACT.value}) ON (c.is_resolved)",
        f"CREATE INDEX contract_domain_idx IF NOT EXISTS "
        f"FOR (c:{Label.CONTRACT.value}) ON (c.domain)",
    ]


def all_statements() -> list[str]:
    """All schema statements, in the order they should be applied."""
    return [*constraint_statements(), *index_statements()]


__all__ = [
    "Label",
    "Rel",
    "all_statements",
    "constraint_statements",
    "index_statements",
]

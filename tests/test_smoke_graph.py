"""Smoke tests for graph-layer behavior that must NOT require Neo4j."""

from __future__ import annotations

from polysignal.graph.client import Neo4jClient
from polysignal.graph.schema import all_statements


def test_apply_schema_dry_run_returns_all_statements_without_connecting():
    """Dry-run must short-circuit before opening a Bolt session."""
    client = Neo4jClient()
    statements = client.apply_schema(dry_run=True)
    assert statements == all_statements()


def test_dry_run_statements_include_unique_constraints_and_indexes():
    client = Neo4jClient()
    statements = client.apply_schema(dry_run=True)
    joined = "\n".join(statements)
    assert "CREATE CONSTRAINT" in joined
    assert "IS UNIQUE" in joined
    assert "NODE KEY" not in joined
    assert "CREATE INDEX" in joined


def test_dry_run_does_not_lazily_connect():
    """No private driver should be created when dry_run is True."""
    client = Neo4jClient()
    client.apply_schema(dry_run=True)
    # Intentional white-box check: dry-run must not lazily build a driver.
    assert client._driver is None

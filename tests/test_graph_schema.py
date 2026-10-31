"""Tests for ``polysignal.graph.schema``."""

from __future__ import annotations

from polysignal.graph.schema import (
    Label,
    Rel,
    all_statements,
    constraint_statements,
    index_statements,
)


def test_constraint_statements_are_idempotent_strings():
    stmts = constraint_statements()
    assert stmts, "expected at least one constraint statement"
    for stmt in stmts:
        assert isinstance(stmt, str)
        assert "IF NOT EXISTS" in stmt


def test_constraints_are_community_compatible_unique_constraints():
    joined = "\n".join(constraint_statements())
    assert "IS NODE KEY" not in joined
    assert "IS UNIQUE" in joined


def test_index_statements_are_idempotent_strings():
    stmts = index_statements()
    assert stmts, "expected at least one index statement"
    for stmt in stmts:
        assert "IF NOT EXISTS" in stmt
        assert "CREATE INDEX" in stmt


def test_all_statements_includes_constraints_then_indexes():
    stmts = all_statements()
    constraints = constraint_statements()
    assert stmts[: len(constraints)] == constraints


def test_label_enum_values_are_pascal_case():
    for label in Label:
        # First character capital, no underscores.
        assert label.value[0].isupper()
        assert "_" not in label.value


def test_rel_enum_values_are_screaming_snake_case():
    for rel in Rel:
        assert rel.value.isupper() or "_" in rel.value
        assert rel.value == rel.value.replace(" ", "_")

"""Offline tests for the Neo4j client wrapper."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from polysignal.graph.client import Neo4jClient


class _Session:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run(self, statement: str, params: dict[str, Any]):
        self.calls.append((statement, params))
        return [{"market_id": params.get("market_id")}]


def test_run_query_passes_parameter_mapping_to_session() -> None:
    client = Neo4jClient()
    session = _Session()

    @contextmanager
    def _session():
        yield session

    client.session = _session  # type: ignore[method-assign]
    rows = client.run_query("MATCH (c {market_id: $market_id}) RETURN c", {"market_id": "m1"})
    assert rows == [{"market_id": "m1"}]
    assert session.calls == [
        ("MATCH (c {market_id: $market_id}) RETURN c", {"market_id": "m1"})
    ]


def test_run_query_uses_empty_params_when_omitted() -> None:
    client = Neo4jClient()
    session = _Session()

    @contextmanager
    def _session():
        yield session

    client.session = _session  # type: ignore[method-assign]
    client.run_query("RETURN 1")
    assert session.calls == [("RETURN 1", {})]

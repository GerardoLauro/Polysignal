"""Neo4j client wrapper (interface + connection helper).

The Bolt driver is created lazily so importing this module does not
require a running database. The methods that actually execute Cypher
are intentionally light: the heavy graph construction belongs in
``polysignal.graph.loaders``.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from polysignal.config import Settings, get_settings
from polysignal.graph.schema import all_statements
from polysignal.storage.paths import DataPaths


class Neo4jClient:
    """Thin wrapper around the official Neo4j Python driver."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = (settings or get_settings()).neo4j
        self._driver: Any | None = None

    # ------------------------------------------------------------- driver
    def _connect(self) -> Any:
        if self._driver is not None:
            return self._driver
        try:
            from neo4j import GraphDatabase  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - missing optional dep
            raise RuntimeError(
                "The neo4j driver is not installed. Run 'pip install -e .' to add it."
            ) from exc

        self._driver = GraphDatabase.driver(
            self.settings.uri,
            auth=(self.settings.user, self.settings.password),
        )
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    @contextmanager
    def session(self):  # type: ignore[no-untyped-def]
        driver = self._connect()
        try:
            with driver.session(database=self.settings.database) as session:
                yield session
        except Exception as exc:
            raise RuntimeError(
                f"Neo4j is unavailable at {self.settings.uri!r} "
                f"(database {self.settings.database!r}). Start it with "
                "`docker compose up -d neo4j` and verify NEO4J_* settings."
            ) from exc

    # ----------------------------------------------------------- schema ops
    def apply_schema(self, *, dry_run: bool = False) -> list[str]:
        """Apply the canonical schema to the connected database.

        Returns the list of statements that were (or would have been)
        executed, in order. When ``dry_run`` is ``True`` nothing is sent
        to the database — useful for the CLI before a real driver is
        configured.
        """
        statements = all_statements()
        if dry_run:
            return statements
        with self.session() as session:
            for stmt in statements:
                session.run(stmt).consume()
        return statements

    # ----------------------------------------------------------- query ops
    def run_write(self, statement: str, rows: list[dict[str, Any]]) -> None:
        """Execute one batched write operation."""
        with self.session() as session:
            session.run(statement, rows=rows).consume()

    def run_query(
        self,
        statement: str,
        params: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a read-only query and return plain dictionaries."""
        with self.session() as session:
            return [dict(record) for record in session.run(statement, params or {})]

    # ----------------------------------------------------------- graph load
    def load_normalized(
        self,
        *,
        paths: DataPaths | None = None,
        segments_path: Path | None = None,
        stance_path: Path | None = None,
        dry_run: bool = False,
        batch_size: int = 500,
    ):  # type: ignore[no-untyped-def]
        """Plan or execute Graph Loader v1 against normalized Parquet outputs."""
        from polysignal.graph.loaders import build_load_plan, execute_load_plan

        plan = build_load_plan(
            paths=paths,
            segments_path=segments_path,
            stance_path=stance_path,
        )
        if dry_run:
            return plan
        return execute_load_plan(self, plan, batch_size=batch_size)


__all__ = ["Neo4jClient"]

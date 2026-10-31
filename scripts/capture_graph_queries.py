"""Run the canonical graph read-queries against the LIVE Neo4j and capture
results to ``data/graph_exports/`` (Traccia_2 Task E2 reproducibility artifact).

Prereq: Neo4j running (see docs/STORAGE_AND_SCALING.md runbook) and the graph
loaded via ``polysignal graph init --no-dry-run`` + ``graph load --no-dry-run``.
Credentials come from ``.env`` (``NEO4J_*``); defaults match the dev instance.

Usage:
    .venv\\Scripts\\python.exe scripts\\capture_graph_queries.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from polysignal.graph import queries
from polysignal.graph.client import Neo4jClient
from polysignal.storage.paths import get_data_paths

# The Task-E2 query set + the two live count summaries.
_NAMED = {
    "contract-coverage": queries.contract_coverage(limit=50),
    "top-contracts": queries.top_contracts(limit=20),
    "author-activity": queries.author_activity(limit=50),
    "platform-coverage": queries.platform_coverage(limit=100),
    "stance-summary": queries.stance_summary(limit=100),
    "health": queries.graph_health(),
}


def main() -> int:
    out_dir = get_data_paths().graph_exports
    out_dir.mkdir(parents=True, exist_ok=True)
    client = Neo4jClient()
    summary: dict[str, object] = {
        "captured_at": datetime.now(tz=timezone.utc).isoformat(),
        "uri": client.settings.uri,
        "database": client.settings.database,
        "queries": {},
    }
    try:
        node_counts = client.run_query(queries.LIVE_NODE_COUNTS)
        rel_counts = client.run_query(queries.LIVE_RELATIONSHIP_COUNTS)
        summary["node_counts"] = node_counts
        summary["relationship_counts"] = rel_counts
        (out_dir / "node_counts.json").write_text(
            json.dumps(node_counts, indent=2, default=str), encoding="utf-8"
        )
        (out_dir / "relationship_counts.json").write_text(
            json.dumps(rel_counts, indent=2, default=str), encoding="utf-8"
        )
        print(f"node labels: {len(node_counts)} | relationship types: {len(rel_counts)}")

        for name, gq in _NAMED.items():
            rows = client.run_query(gq.cypher, gq.params)
            path = out_dir / f"query_{name.replace('-', '_')}.json"
            path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
            summary["queries"][name] = {"rows": len(rows), "file": path.name}
            print(f"  {name:20s} -> {len(rows):5d} rows  ({path.name})")
    finally:
        client.close()

    (out_dir / "capture_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"OK captured graph query results -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

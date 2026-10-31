"""Offline checks for Graph Loader v1 Cypher."""

from __future__ import annotations

from polysignal.graph import queries


def test_write_queries_are_batched_idempotent_cypher():
    statements = [
        queries.UPSERT_CONTRACTS,
        queries.UPSERT_DOMAINS,
        queries.UPSERT_OUTCOMES,
        queries.UPSERT_PLATFORMS,
        queries.UPSERT_POSTS,
        queries.UPSERT_AUTHORS,
        queries.LINK_CONTRACT_DOMAINS,
        queries.LINK_CONTRACT_OUTCOMES,
        queries.LINK_POST_PLATFORMS,
        queries.LINK_AUTHOR_POSTS,
        queries.LINK_POST_CONTRACTS,
        queries.LINK_POST_SUPPORTS_OUTCOME,
        queries.LINK_POST_OPPOSES_OUTCOME,
    ]
    for statement in statements:
        assert "UNWIND $rows AS row" in statement
        assert "MERGE" in statement


def test_linked_to_query_preserves_segment_evidence():
    statement = queries.LINK_POST_CONTRACTS
    assert "link_id" in statement
    assert "segment_id" in statement
    assert "source_linked_post_id" in statement
    assert "retrieval_query" in statement


def test_live_queries_only_read_counts():
    assert "RETURN label" in queries.LIVE_NODE_COUNTS
    assert "RETURN type(r)" in queries.LIVE_RELATIONSHIP_COUNTS


# --------------------------------------------------------------------------- #
# Branch 07 — parameterized read-query builders
# --------------------------------------------------------------------------- #
def test_list_queries_advertises_all_builders():
    names = {name for name, _ in queries.list_queries()}
    assert names == {
        "contract-coverage",
        "top-contracts",
        "author-activity",
        "platform-coverage",
        "contract-neighborhood",
        "stance-summary",
        "health",
    }


def test_read_builders_are_parameterized_and_readonly():
    builders = [
        queries.contract_coverage(limit=10),
        queries.top_contracts(limit=5),
        queries.author_activity(limit=7),
        queries.platform_coverage(limit=9),
        queries.stance_summary(limit=3),
        queries.graph_health(),
    ]
    for gq in builders:
        # Read-only: no writes in analytical queries.
        assert "MERGE" not in gq.cypher
        assert "CREATE" not in gq.cypher
        assert "DELETE" not in gq.cypher
        # Limits flow through $params, never string-interpolated.
        if "limit" in gq.params:
            assert "$limit" in gq.cypher
            assert str(gq.params["limit"]) not in gq.cypher


def test_contract_neighborhood_uses_param_not_interpolation():
    # A market_id containing Cypher-breaking characters must never reach the
    # query text — it stays in params.
    malicious = "0xabc') DETACH DELETE n //"
    gq = queries.contract_neighborhood(market_id=malicious, limit=5)
    assert gq.params["market_id"] == malicious
    assert malicious not in gq.cypher
    assert "$market_id" in gq.cypher
    assert "DETACH DELETE" not in gq.cypher


def test_stance_summary_is_empty_safe():
    # OPTIONAL MATCH keeps the query valid even with zero stance edges.
    cypher = queries.stance_summary().cypher
    assert "OPTIONAL MATCH" in cypher
    assert ":SUPPORTS" in cypher
    assert ":OPPOSES" in cypher


def test_health_query_reports_integrity_counters():
    cypher = queries.graph_health().cypher
    for counter in (
        "posts_without_author",
        "posts_without_platform",
        "contracts_without_outcomes",
        "contracts_without_links",
        "orphan_outcomes",
    ):
        assert counter in cypher


def test_analytical_queries_filter_to_accepted_temporally_valid_links():
    for query in (
        queries.contract_coverage(),
        queries.top_contracts(),
        queries.author_activity(),
        queries.platform_coverage(),
        queries.contract_neighborhood(market_id="m1"),
        queries.graph_health(),
    ):
        assert "accepted: true" in query.cypher
        assert "temporal_valid: true" in query.cypher
        assert "weakly_relevant" in query.cypher

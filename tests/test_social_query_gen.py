"""Offline tests for branch 04 query generation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from polysignal.domain import Market, MarketDomain, Outcome
from polysignal.ingestion.social.query_gen import (
    decompose_question,
    generate_queries,
    generate_queries_for_market,
)


def _market(question: str, domain: MarketDomain = MarketDomain.POLITICS) -> Market:
    return Market(
        market_id="cond-1",
        slug="example",
        question=question,
        domain=domain,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        outcomes=[
            Outcome(outcome_id="yes", label="Yes", index=0),
            Outcome(outcome_id="no", label="No", index=1),
        ],
    )


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def test_decompose_preserves_event_core_and_constraint_before_gta():
    decomp = decompose_question("Will China invade Taiwan before GTA VI?")
    assert decomp.event_core == "China invade Taiwan"
    assert decomp.constraints == ("before GTA VI",)
    # Roman numeral / arabic variants are surfaced as anchors.
    assert "GTA VI" in decomp.anchors
    assert "GTA 6" in decomp.anchors


def test_decompose_handles_trump_out_before_gta_example():
    decomp = decompose_question("Will Trump be out as President before GTA VI?")
    assert "Trump" in decomp.event_core
    assert decomp.constraints == ("before GTA VI",)
    assert "GTA VI" in decomp.anchors


def test_decompose_strips_leading_will_and_trailing_question_mark():
    decomp = decompose_question("Will the Lakers win the NBA Finals?")
    assert decomp.event_core == "the Lakers win the NBA Finals"
    assert decomp.constraints == ()


def test_decompose_year_constraint_is_preserved():
    decomp = decompose_question("Bitcoin at 200K by end of 2026?")
    assert "Bitcoin" in decomp.event_core
    assert any("2026" in c for c in decomp.constraints)
    assert "2026" in decomp.anchors


def test_decompose_empty_string_returns_empty_decomposition():
    decomp = decompose_question("")
    assert decomp.event_core == ""
    assert decomp.constraints == ()
    assert decomp.anchors == ()


# ---------------------------------------------------------------------------
# Query family generation
# ---------------------------------------------------------------------------


def test_generate_queries_for_market_keeps_constraint_anchor():
    market = _market("Will China invade Taiwan before GTA VI?")
    queries = generate_queries_for_market(
        market,
        max_queries=8,
        now=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    assert queries, "generator must always emit something for a non-empty question"

    # The core query is short and does not include the constraint.
    core = next(q for q in queries if q.query_type == "core")
    assert "China invade Taiwan" in core.query
    assert "GTA" not in core.query

    # The constraint-aware query keeps the *full* relation phrase, not
    # just the anchor — this is the critical fix from the review.
    constraint_queries = [q for q in queries if q.query_type.startswith("constraint")]
    assert constraint_queries
    assert any("before GTA VI" in q.query for q in constraint_queries), (
        "constraint queries must preserve the relation phrase, not just the anchor"
    )

    # The Roman → arabic variant is emitted as an `anchor` query
    # (entity tokens + bare anchor, no relation word).
    anchor_queries = [q for q in queries if q.query_type == "anchor"]
    assert any("GTA 6" in q.query for q in anchor_queries)


def test_generate_queries_respects_max_queries():
    market = _market("Will China invade Taiwan before GTA VI?")
    queries = generate_queries_for_market(market, max_queries=2)
    assert len(queries) == 2
    assert queries[0].rank == 0
    assert queries[1].rank == 1


def test_generate_queries_marks_market_id_and_generated_at():
    market = _market("Will the Lakers win the NBA Finals?", domain=MarketDomain.SPORTS)
    moment = datetime(2026, 5, 31, 12, tzinfo=timezone.utc)
    queries = generate_queries_for_market(market, max_queries=4, now=moment)
    for q in queries:
        assert q.market_id == market.market_id
        assert q.generated_at == moment


def test_generate_queries_zero_queries_returns_empty():
    market = _market("Will the Lakers win the NBA Finals?")
    assert generate_queries_for_market(market, max_queries=0) == []


def test_generate_queries_avoids_overly_long_strings():
    long_question = (
        "Will the joint commission of the United Nations on Sustainable Development "
        "publish its final assessment regarding the Paris Climate Agreement compliance "
        "rates before December 31, 2027?"
    )
    market = _market(long_question, domain=MarketDomain.SCIENCE)
    queries = generate_queries_for_market(market, max_queries=6)
    assert queries
    # Reddit chokes on huge queries; the generator caps the length.
    for q in queries:
        assert len(q.query) <= 100, q


def test_bulk_generate_queries_limits_markets():
    markets = [_market(f"Will candidate {i} win the 2026 election?") for i in range(5)]
    queries = generate_queries(
        markets,
        queries_per_market=2,
        limit_markets=3,
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    # All markets in this fixture share market_id, so the unique
    # market_id set is small. What we care about is that limit_markets
    # caps the expansion: 3 markets * 2 queries per market = 6 rows.
    assert len(queries) == 3 * 2


@pytest.mark.parametrize(
    "question",
    [
        "Will Trump be out as President before GTA VI?",
        "Bitcoin at 200K by end of 2026?",
        "Will the Lakers win the NBA Finals?",
        "China invade Taiwan",
    ],
)
def test_generate_queries_is_deterministic(question):
    """Same input → same output. No timestamps, no randomness."""
    moment = datetime(2025, 1, 1, tzinfo=timezone.utc)
    market = _market(question)
    first = generate_queries_for_market(market, max_queries=6, now=moment)
    second = generate_queries_for_market(market, max_queries=6, now=moment)
    assert [q.model_dump() for q in first] == [q.model_dump() for q in second]


# ---------------------------------------------------------------------------
# Critical Fix 2 — relation phrases must be preserved verbatim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question,expected_phrase",
    [
        ("Will China invade Taiwan before GTA VI?", "before GTA VI"),
        ("Will Trump be out as President before GTA VI?", "before GTA VI"),
        ("GTA VI released before June 2026?", "before June 2026"),
        ("Bitcoin at 200K by end of 2026?", "by end of 2026"),
        ("Will Biden win by election day?", "by election day"),
        ("Will the bill pass by 2025?", "by 2025"),
        ("Will sanctions remain through 2025?", "through 2025"),
        ("Will the vote happen on election day?", "on election day"),
    ],
)
def test_constraint_queries_preserve_relation_phrase(question, expected_phrase):
    """Every example in the review must emit the relation phrase verbatim."""
    market = _market(question)
    queries = generate_queries_for_market(
        market, max_queries=8, now=datetime(2025, 1, 1, tzinfo=timezone.utc)
    )
    assert any(expected_phrase in q.query for q in queries), (
        f"no query preserved the relation phrase {expected_phrase!r}; "
        f"queries were: {[q.query for q in queries]}"
    )


@pytest.mark.parametrize(
    "question,core_phrase,expected_phrase",
    [
        (
            "Will a candidate drop out before election day?",
            "candidate drop out",
            "before election day",
        ),
        (
            "Will the CEO resign after the investigation report?",
            "CEO resign",
            "after the investigation report",
        ),
        ("Will Bitcoin hit $150k by December 2026?", "Bitcoin hit $150k", "by December 2026"),
        (
            "Will the product launch be delayed until next year?",
            "product launch",
            "until next year",
        ),
        ("Will sanctions remain through 2025?", "sanctions remain", "through 2025"),
        ("Will oil trade above $100 during the summer?", "oil trade", "during the summer"),
        ("Will a ceasefire happen within 30 days?", "ceasefire happen", "within 30 days"),
        (
            "Will Tesla launch robotaxi service in California in 2026?",
            "Tesla launch",
            "in California in 2026",
        ),
        ("Will Nvidia close above $200 on earnings day?", "Nvidia close", "on earnings day"),
        ("Will the announcement happen at WWDC?", "announcement happen", "at WWDC"),
    ],
)
def test_relation_matrix_emits_explicit_constraint_query(question, core_phrase, expected_phrase):
    """Every supported relation emits broad and explicit constraint-aware queries."""
    queries = generate_queries_for_market(
        _market(question),
        max_queries=8,
        now=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    core_queries = [q.query for q in queries if q.query_type == "core"]
    constraint_queries = [q.query for q in queries if q.query_type.startswith("constraint")]
    assert any(core_phrase in query for query in core_queries)
    assert any(expected_phrase in query for query in constraint_queries)


@pytest.mark.parametrize(
    "question,core_phrase,expected_phrase",
    [
        ("Will a ceasefire happen within 30 days?", "ceasefire happen", "within 30 days"),
        (
            "Will a room-temperature superconductor be verified within two years?",
            "room-temperature superconductor",
            "within two years",
        ),
        (
            "Will the strike end within one week of negotiations?",
            "strike end",
            "within one week of negotiations",
        ),
        ("Will the announcement happen at WWDC?", "announcement happen", "at WWDC"),
        ("Will the vote occur at market close?", "vote occur", "at market close"),
    ],
)
def test_within_and_at_constraints_emit_broad_and_constraint_aware_queries(
    question,
    core_phrase,
    expected_phrase,
):
    """Regression: within/at must not survive only by accident inside the core query."""
    queries = generate_queries_for_market(
        _market(question),
        max_queries=8,
        now=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    core_queries = [q.query for q in queries if q.query_type == "core"]
    constraint_queries = [q.query for q in queries if q.query_type.startswith("constraint")]
    assert any(core_phrase in query for query in core_queries)
    assert any(expected_phrase in query for query in constraint_queries)
    assert all(len(query.query) <= 96 for query in queries)


def test_constraint_query_for_gta_vi_release_includes_before_june_2026():
    """The reviewer's specific GTA VI example must work end-to-end."""
    market = _market("GTA VI released before June 2026?", domain=MarketDomain.CULTURE)
    queries = generate_queries_for_market(
        market, max_queries=8, now=datetime(2025, 1, 1, tzinfo=timezone.utc)
    )
    constraint_q = [q for q in queries if q.query_type.startswith("constraint")]
    assert constraint_q
    # The full relation phrase must appear in at least one constraint query.
    assert any("before June 2026" in q.query for q in constraint_q)
    # And the bare anchor variant (no relation) should also be available.
    anchor_q = [q for q in queries if q.query_type == "anchor"]
    assert any("June 2026" in q.query and "before" not in q.query for q in anchor_q)


def test_constraint_suffix_survives_core_token_trimming():
    market = _market("Will the next UK election be called by June 30, 2026?")
    queries = generate_queries_for_market(market, max_queries=6)
    core_constraint = next(q for q in queries if q.query_type == "constraint")
    assert "by June 30, 2026" in core_constraint.query


# ---------------------------------------------------------------------------
# Fix 7 — slug and description as fallback sources
# ---------------------------------------------------------------------------


def test_slug_tokens_recover_signal_missed_by_question():
    """When the question is terse, the slug should add tokens."""
    market = Market(
        market_id="m",
        slug="microstrategy-sells-any-bitcoin-by-december-31-2026",
        question="MicroStrategy?",  # intentionally terse
        domain=MarketDomain.CRYPTO,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        outcomes=[
            Outcome(outcome_id="yes", label="Yes", index=0),
            Outcome(outcome_id="no", label="No", index=1),
        ],
    )
    queries = generate_queries_for_market(
        market, max_queries=8, now=datetime(2025, 1, 1, tzinfo=timezone.utc)
    )
    # A slug-derived query should appear.
    slug_qs = [q for q in queries if q.query_type == "slug"]
    assert slug_qs


def test_description_anchor_query_uses_capitalized_terms():
    """Description-derived queries should pick capitalized proper nouns."""
    market = Market(
        market_id="m",
        slug="kraken-ipo",
        question="Kraken IPO?",
        description="This market resolves Yes if Kraken completes an IPO on NYSE.",
        domain=MarketDomain.CRYPTO,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        outcomes=[
            Outcome(outcome_id="yes", label="Yes", index=0),
            Outcome(outcome_id="no", label="No", index=1),
        ],
    )
    queries = generate_queries_for_market(
        market, max_queries=10, now=datetime(2025, 1, 1, tzinfo=timezone.utc)
    )
    description_qs = [q for q in queries if q.query_type == "description_anchor"]
    # The capitalized "NYSE" should be promoted into a description anchor.
    assert any("NYSE" in q.query for q in description_qs)


def test_description_is_not_injected_verbatim_in_a_long_query():
    """Even with a long description, no query exceeds the safe length cap."""
    long_desc = (
        "This market will resolve to Yes if and only if the relevant "
        "publicly available report from the U.S. Bureau of Economic Analysis "
        "is published before the specified ET deadline and clearly states "
        "the figure exceeds the threshold mentioned in the title above."
    )
    market = Market(
        market_id="m",
        slug="some-slug",
        question="GDP above 3 percent?",
        description=long_desc,
        domain=MarketDomain.ECONOMY,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        outcomes=[
            Outcome(outcome_id="yes", label="Yes", index=0),
            Outcome(outcome_id="no", label="No", index=1),
        ],
    )
    queries = generate_queries_for_market(market, max_queries=10)
    for q in queries:
        assert len(q.query) <= 100, q.query

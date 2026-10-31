"""Deterministic query generation from Polymarket markets.

The goal is to turn a market like

    "Will China invade Taiwan before GTA VI?"

into a small, *useful* set of search queries — without destroying the
contextual constraints that anchor the question. Branch 05 will decide
which retrieved posts are truly relevant; branch 04 is only responsible
for surfacing a sensible candidate set.

The generator runs in three steps:

1. **Decomposition.** The question is split into:
   - an *event core* (the action / topic — "China invade Taiwan");
   - one or more *temporal / contextual constraints*
     ("before GTA VI", "by 2026", "by election day");
   - *context anchors* extracted from the constraints
     ("GTA VI", "GTA 6", "election day").

2. **Query family generation.** From the decomposition we emit several
   query types — broad, entity, constraint-aware, domain-aware, and an
   optional "prediction market" variant. Each query is short (Reddit
   rejects very long search strings) and de-duplicated within the same
   market.

3. **Trace metadata.** Every emitted query carries the
   ``market_id`` that produced it, a ``query_type`` tag, a stable
   ``rank``, and a list of ``source_fields`` (which fields of the market
   contributed to the query). Branch 05 can join through this to
   reconstruct exactly *why* a candidate post was retrieved.

The module is intentionally pure (no I/O), deterministic, and easy to
unit-test.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from polysignal.domain import Market, MarketDomain, MarketQuery

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants & regexes
# ---------------------------------------------------------------------------

# Phrases that introduce a temporal/contextual constraint. The order
# matters — we look for the most specific phrase first so that "by the
# end of 2026" is captured as one unit rather than split into "by" +
# "the end of 2026".
_CONSTRAINT_PHRASES: tuple[str, ...] = (
    "before the end of",
    "by the end of",
    "by end of",
    "on or before",
    "ahead of",
    "no later than",
    "prior to",
    "before",
    "after",
    "through",
    "during",
    "until",
    "within",
    "by",
    "in",
    "on",
    "at",
)


def _constraint_phrase_pattern(phrase: str) -> str:
    """Return the regex fragment for one contextual relation phrase.

    ``at`` also appears in threshold-style event cores (for example,
    ``"Bitcoin at 200K by 2026"``). Keep numeric ``at`` clauses in the
    core so a later deadline can still be extracted independently.
    """
    escaped = re.escape(phrase)
    if phrase == "at":
        return rf"{escaped}(?!\s+(?:[$]\s*)?\d)"
    return escaped


_CONSTRAINT_PATTERN = re.compile(
    r"\b("
    + "|".join(_constraint_phrase_pattern(p) for p in _CONSTRAINT_PHRASES)
    + r")\s+(?P<anchor>.+?)$",
    re.IGNORECASE,
)

# "Will X ...?" / "Will X ..." — drop the leading interrogative.
_LEADING_QUESTION = re.compile(
    r"^\s*(will|does|do|did|is|are|was|were|has|have|had|can|could|should|would|may|might)\s+",
    re.IGNORECASE,
)

# Trailing question mark / punctuation.
_TRAILING_PUNCT = re.compile(r"[?!.,;:]+\s*$")

# A small stopword list used only for ranking entity tokens. The full
# NLP-normalization stopwords are too aggressive for query generation
# (they drop "Trump" because it's title-cased only — but here we keep
# original casing and only filter very common English words).
_ENTITY_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "from",
        "with",
        "in",
        "on",
        "at",
        "for",
        "by",
        "be",
        "is",
        "are",
        "was",
        "were",
        "as",
        "this",
        "that",
        "it",
        "its",
        "any",
        "more",
        "than",
        "before",
        "after",
        "during",
        "until",
    }
)

# Maximum length we feed to a Reddit search query — Reddit truncates
# long queries and PRAW occasionally returns 400s on >256-char strings.
_REDDIT_SAFE_LEN = 96

# Maximum tokens per generated query. Most useful Reddit searches are
# short noun phrases. We cap at 8 tokens so the generator never emits
# the literal long-form question.
_MAX_QUERY_TOKENS = 8


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Decomposition:
    """Internal: structured view of a market question."""

    event_core: str
    constraints: tuple[str, ...]
    anchors: tuple[str, ...]


def _strip_question_form(text: str) -> str:
    """Drop the leading interrogative and the trailing punctuation."""
    if not text:
        return ""
    out = _LEADING_QUESTION.sub("", text.strip())
    out = _TRAILING_PUNCT.sub("", out)
    return out.strip()


def _split_on_constraint(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Recursively peel off ``<phrase> <anchor>`` clauses from the end.

    Returns the residual event core and the list of
    ``(phrase, anchor)`` pairs in the order they were stripped (from the
    end backwards). Each match also strips the anchor in the same pass —
    constraints may chain (e.g. "X before Y and after Z").
    """
    core = text
    matches: list[tuple[str, str]] = []
    # Loop with a hard upper bound so we never trip on a pathological
    # regex match.
    for _ in range(6):
        match = _CONSTRAINT_PATTERN.search(core)
        if match is None:
            break
        phrase = match.group(1).strip()
        anchor = match.group("anchor").strip()
        if not anchor:
            break
        # Defensive: anchors longer than half the original sentence are
        # almost always the whole question — refuse to strip them, the
        # question is probably just declarative.
        if len(anchor) > max(40, len(text) // 2 + 8):
            break
        core = core[: match.start()].rstrip(" ,;:-")
        matches.append((phrase, anchor))
        if not core:
            break
    return core.strip(), matches


def _anchor_variants(anchor: str) -> list[str]:
    """Expand an anchor like 'GTA VI' into searchable surface variants.

    Roman numerals → arabic and vice-versa; "year 2026" stays "2026";
    "election day" stays "election day". The variants are returned in a
    stable order so the test fixture is deterministic.
    """
    anchor = anchor.strip()
    if not anchor:
        return []
    seen: list[str] = []

    def _add(value: str) -> None:
        value = value.strip()
        if value and value not in seen:
            seen.append(value)

    _add(anchor)
    # Roman numerals → arabic.
    roman_to_arabic = {
        "i": "1",
        "ii": "2",
        "iii": "3",
        "iv": "4",
        "v": "5",
        "vi": "6",
        "vii": "7",
        "viii": "8",
        "ix": "9",
        "x": "10",
    }
    arabic_to_roman = {v: k.upper() for k, v in roman_to_arabic.items()}
    tokens = anchor.split()
    rewritten_to_arabic: list[str] = []
    rewritten_to_roman: list[str] = []
    for tok in tokens:
        low = tok.lower().strip(",.;:")
        if low in roman_to_arabic:
            rewritten_to_arabic.append(roman_to_arabic[low])
        else:
            rewritten_to_arabic.append(tok)
        if low in arabic_to_roman:
            rewritten_to_roman.append(arabic_to_roman[low])
        else:
            rewritten_to_roman.append(tok)
    if rewritten_to_arabic != tokens:
        _add(" ".join(rewritten_to_arabic))
    if rewritten_to_roman != tokens:
        _add(" ".join(rewritten_to_roman))
    return seen


def decompose_question(question: str) -> _Decomposition:
    """Split a market question into ``(event_core, constraints, anchors)``."""
    stripped = _strip_question_form(question)
    if not stripped:
        return _Decomposition(event_core="", constraints=(), anchors=())
    core, matches = _split_on_constraint(stripped)
    if not core:
        # Every word was a constraint — fall back to the original
        # stripped question so callers always get something.
        core = stripped
        matches = []
    constraints: list[str] = []
    anchors: list[str] = []
    for phrase, anchor in matches:
        constraints.append(f"{phrase} {anchor}".strip())
        for variant in _anchor_variants(anchor):
            if variant.lower() not in {a.lower() for a in anchors}:
                anchors.append(variant)
    return _Decomposition(
        event_core=core,
        constraints=tuple(constraints),
        anchors=tuple(anchors),
    )


# ---------------------------------------------------------------------------
# Query family generation
# ---------------------------------------------------------------------------


def _trim_tokens(text: str, max_tokens: int = _MAX_QUERY_TOKENS) -> str:
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return " ".join(tokens)
    return " ".join(tokens[:max_tokens])


def _safe_query(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > _REDDIT_SAFE_LEN:
        text = text[:_REDDIT_SAFE_LEN].rstrip()
    return text


def _entity_tokens(event_core: str) -> list[str]:
    """Pull noun-like tokens from the event core, preserving case.

    We keep tokens that are either capitalized, all-uppercase, or
    numeric, plus a handful of action verbs that are critical to the
    semantic of binary markets ("invade", "win", "resign"). Common
    English words are dropped.
    """
    out: list[str] = []
    for raw in event_core.split():
        token = raw.strip(",.;:!?()[]{}\"'")
        if not token:
            continue
        low = token.lower()
        if low in _ENTITY_STOPWORDS:
            continue
        # Always keep numeric tokens (years, version numbers).
        if any(ch.isdigit() for ch in token):
            out.append(token)
            continue
        # Keep capitalized tokens (proper nouns / domain terms).
        if token[0].isupper() or token.isupper():
            out.append(token)
            continue
        # Keep a few action verbs that are load-bearing in binary
        # markets — without them queries become indistinguishable.
        if low in {
            "invade",
            "win",
            "lose",
            "resign",
            "removed",
            "remove",
            "approve",
            "ban",
            "launch",
            "release",
            "out",
            "elected",
            "default",
            "merge",
            "ipo",
            "halve",
            "halving",
            "sells",
            "sell",
            "sold",
            "drop",
            "drops",
            "rises",
            "rise",
            "fall",
            "falls",
        }:
            out.append(token)
    return out


def _domain_keywords(market: Market) -> list[str]:
    """Map ``MarketDomain`` to a few generic search keywords."""
    return {
        MarketDomain.POLITICS: ["politics", "election"],
        MarketDomain.SPORTS: ["sports"],
        MarketDomain.CRYPTO: ["crypto", "cryptocurrency"],
        MarketDomain.ECONOMY: ["economy", "markets"],
        MarketDomain.CULTURE: ["culture"],
        MarketDomain.SCIENCE: ["science"],
        MarketDomain.OTHER: [],
    }[market.domain]


@dataclass(frozen=True)
class _Candidate:
    """A single query plus enough metadata to score and label it."""

    text: str
    query_type: str
    source_fields: tuple[str, ...]


def _emit(text: str, query_type: str, source_fields: Sequence[str]) -> _Candidate | None:
    cleaned = _safe_query(_trim_tokens(text))
    if not cleaned:
        return None
    return _Candidate(text=cleaned, query_type=query_type, source_fields=tuple(source_fields))


def _emit_with_suffix(
    prefix: str,
    suffix: str,
    query_type: str,
    source_fields: Sequence[str],
) -> _Candidate | None:
    """Emit a short query while preserving a load-bearing suffix.

    Constraint clauses belong at the end of the query. A normal
    left-to-right token trim can turn ``"by June 30, 2026"`` into
    ``"by June"``, changing the meaning. Reserve room for the complete
    suffix first and trim only the event-core prefix whenever possible.
    """
    prefix_tokens = prefix.split()
    suffix = re.sub(r"\s+", " ", suffix).strip()
    if not suffix:
        return _emit(prefix, query_type, source_fields)
    suffix_tokens = suffix.split()
    available = max(0, _MAX_QUERY_TOKENS - len(suffix_tokens))
    kept_prefix = prefix_tokens[:available]
    text = " ".join([*kept_prefix, *suffix_tokens])
    while kept_prefix and len(text) > _REDDIT_SAFE_LEN:
        kept_prefix.pop()
        text = " ".join([*kept_prefix, *suffix_tokens])
    return _emit(text, query_type, source_fields)


_SLUG_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "from",
        "with",
        "in",
        "on",
        "at",
        "for",
        "by",
        "be",
        "is",
        "will",
        "this",
        "that",
        "any",
    }
)


def _slug_tokens(slug: str) -> list[str]:
    """Extract content-bearing tokens from a Polymarket slug.

    Slugs are kebab-cased (e.g. ``microstrategy-sells-any-bitcoin-by-december-31-2026``).
    We split on hyphens, drop a small set of stopwords plus very short
    fragments (single letters, numbers shorter than 4 digits), and keep
    the rest preserving order. The result is **not** used as a
    standalone query (slugs are often noisy) — it is used to recover
    tokens that were missed by the entity extractor.
    """
    if not slug:
        return []
    cleaned: list[str] = []
    for raw in slug.replace("_", "-").split("-"):
        part = raw.strip()
        if not part or len(part) < 3:
            continue
        low = part.lower()
        if low in _SLUG_STOPWORDS:
            continue
        cleaned.append(part)
    return cleaned


_DESCRIPTION_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "this",
        "that",
        "those",
        "these",
        "yes",
        "no",
        "et",
        "the market",
        "this market",
        "if",
        "otherwise",
        "yes if",
        "no if",
        # Polymarket's resolution boilerplate phrases that appear in
        # virtually every market description and add no search signal.
        "the resolution source",
        "the primary resolution source",
        "the primary resolution",
        "resolution source",
        "this market will resolve",
        "this market resolves",
    }
)


def _description_anchors(description: str | None) -> list[str]:
    """Pull a handful of capitalized phrases out of a description, capped.

    We do *not* feed the description in verbatim — it is usually long
    boilerplate resolution language. Instead we extract sequences of
    capitalized tokens (proper nouns like "MicroStrategy", "NYSE",
    "Bitcoin") and return at most three. This is the lightest possible
    "signal boost" — it never produces a long or risky query.
    """
    if not description:
        return []
    # Sequence of consecutive capitalized words / numeric tokens. We
    # accept single-token uppercase acronyms too (e.g. "IPO", "NYSE").
    pattern = re.compile(r"\b(?:[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,2})\b")
    found: list[str] = []
    seen: set[str] = set()
    for match in pattern.finditer(description):
        candidate = match.group(0).strip()
        low = candidate.lower()
        # Skip generic stop-phrases that appear in every resolution blurb.
        if low in _DESCRIPTION_STOPWORDS:
            continue
        if low in seen:
            continue
        seen.add(low)
        found.append(candidate)
        if len(found) >= 6:
            # Soft cap: we filter further at the call site against
            # entity_tokens, so collect a small surplus to survive that.
            break
    return found


def _build_candidates(market: Market) -> list[_Candidate]:
    """Produce a deterministic ordered list of query candidates for one market.

    Output families (in emission order):

    1. **core** — the event core (the question minus interrogative and
       constraint clauses).
    2. **entity** — the salient nouns / action verbs of the core.
    3. **constraint** — the event core paired with the *full*
       constraint phrase (e.g. ``"Trump out before GTA VI"``). This is
       the most important family: it preserves the relational word
       ("before", "by", "until", ...) and is the difference between
       a useful query and a misleading one.
    4. **constraint_entity** — same as ``constraint`` but anchored on
       the entity tokens, so the query is still short when the event
       core is long.
    5. **anchor** — entity tokens + bare anchor (no relation word) for
       contexts where the relation is implicit in the anchor itself.
    6. **slug** — content-bearing tokens recovered from the market
       slug, in case the question normalization dropped something
       (e.g. a numeric build version that the regex stripped).
    7. **domain** / **tag** — entity tokens plus a domain keyword
       or one of the first two tags from the Gamma payload.
    8. **polymarket** — entity tokens plus ``"prediction market"`` for
       capturing meta-discussion of the contract itself.
    9. **description_anchor** — entity tokens plus one capitalized
       phrase pulled from the description (capped at three). This is
       the modest fallback when the question doesn't expose enough
       proper nouns.

    The list is deduplicated case-insensitively before being returned.
    """
    decomposition = decompose_question(market.question)
    event_core = decomposition.event_core
    entity_tokens = _entity_tokens(event_core)
    candidates: list[_Candidate] = []

    # 1) Core topic — the event core verbatim (trimmed).
    if event_core:
        c = _emit(event_core, "core", ("question",))
        if c is not None:
            candidates.append(c)

    # 2) Entity / action — the salient nouns and verbs from the core.
    if entity_tokens:
        c = _emit(" ".join(entity_tokens), "entity", ("question",))
        if c is not None:
            candidates.append(c)

    # 3) Constraint-aware queries.
    # 3a) Event core + full constraint phrase (keeps "before", "by", etc.).
    # 3b) Entity tokens + full constraint phrase (shorter variant).
    for constraint in decomposition.constraints:
        if event_core:
            c = _emit_with_suffix(
                event_core,
                constraint,
                "constraint",
                ("question", "constraint"),
            )
            if c is not None:
                candidates.append(c)
        if entity_tokens:
            c = _emit_with_suffix(
                " ".join(entity_tokens),
                constraint,
                "constraint_entity",
                ("question", "constraint"),
            )
            if c is not None:
                candidates.append(c)

    # 4) Anchor-only — entity tokens + bare anchor (no relation word).
    #    Useful when the anchor itself carries the temporal meaning
    #    (e.g. just "GTA VI" is enough; "by GTA VI" adds noise on Reddit).
    for anchor in decomposition.anchors:
        if entity_tokens:
            c = _emit(
                " ".join(entity_tokens) + " " + anchor,
                "anchor",
                ("question", "constraint"),
            )
            if c is not None:
                candidates.append(c)

    # 5) Slug fallback — recover tokens the question stripped (rare but
    #    valuable when a market is identified by a build number / date
    #    only in the slug).
    extra_slug_tokens = [
        t
        for t in _slug_tokens(market.slug)
        if t.lower() not in {tok.lower() for tok in entity_tokens}
    ]
    if entity_tokens and extra_slug_tokens:
        c = _emit(
            " ".join(entity_tokens + extra_slug_tokens[:2]),
            "slug",
            ("question", "slug"),
        )
        if c is not None:
            candidates.append(c)

    # 6) Domain / tag-enhanced — entities plus a domain keyword.
    domain_keywords = _domain_keywords(market)
    if entity_tokens and domain_keywords:
        for kw in domain_keywords:
            c = _emit(
                " ".join(entity_tokens) + " " + kw,
                "domain",
                ("question", "domain"),
            )
            if c is not None:
                candidates.append(c)
    # Tags from the market payload, when present, are mixed in too.
    for tag in market.tags[:2]:
        if entity_tokens:
            c = _emit(
                " ".join(entity_tokens) + " " + tag,
                "tag",
                ("question", "tags"),
            )
            if c is not None:
                candidates.append(c)

    # 7) Prediction-market variant — useful for capturing meta-discussion.
    if entity_tokens:
        c = _emit(
            " ".join(entity_tokens) + " prediction market",
            "polymarket",
            ("question", "polymarket"),
        )
        if c is not None:
            candidates.append(c)

    # 8) Description anchors — small, capped, capitalized phrases only.
    if entity_tokens:
        entity_set = {t.lower() for t in entity_tokens}
        for anchor in _description_anchors(market.description):
            if anchor.lower() in entity_set:
                continue
            c = _emit(
                " ".join(entity_tokens[:3]) + " " + anchor,
                "description_anchor",
                ("question", "description"),
            )
            if c is not None:
                candidates.append(c)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[_Candidate] = []
    for cand in candidates:
        key = cand.text.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
    return unique


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_queries_for_market(
    market: Market,
    *,
    max_queries: int = 6,
    now: datetime | None = None,
) -> list[MarketQuery]:
    """Return up to ``max_queries`` :class:`MarketQuery` rows for ``market``.

    The order is deterministic: core first, then entity, then
    constraint-aware queries, then domain-enhanced, then the optional
    prediction-market variant. Each :class:`MarketQuery` carries the
    market id, query type, rank (its position in this list), and the
    list of source fields that produced it.
    """
    if max_queries <= 0:
        return []
    generated_at = (now or datetime.now(tz=timezone.utc)).astimezone(timezone.utc)
    candidates = _build_candidates(market)
    out: list[MarketQuery] = []
    for rank, cand in enumerate(candidates[:max_queries]):
        out.append(
            MarketQuery(
                market_id=market.market_id,
                query=cand.text,
                query_type=cand.query_type,
                rank=rank,
                source_fields=list(cand.source_fields),
                generated_at=generated_at,
            )
        )
    return out


def generate_queries(
    markets: Iterable[Market],
    *,
    queries_per_market: int = 6,
    limit_markets: int | None = None,
    now: datetime | None = None,
) -> list[MarketQuery]:
    """Bulk variant of :func:`generate_queries_for_market`.

    ``markets`` is consumed once; ``limit_markets`` caps how many of
    them we expand. The function is pure — no I/O, no settings reads.
    """
    out: list[MarketQuery] = []
    for count, market in enumerate(markets):
        if limit_markets is not None and count >= limit_markets:
            break
        out.extend(generate_queries_for_market(market, max_queries=queries_per_market, now=now))
    return out


__all__ = [
    "MarketQuery",
    "decompose_question",
    "generate_queries",
    "generate_queries_for_market",
]

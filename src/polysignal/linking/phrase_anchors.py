"""Deterministic phrase-level anchor extraction (Branch 05G).

Token-level anchors are noisy on multi-topic Telegram content:

* ``Stanley`` alone is not enough for a *Stanley Cup* market.
* ``IPO`` alone is not enough for a *Kraken IPO* market.
* ``Trump`` alone is not enough for a *capital gains tax on crypto* market.
* ``2026`` / ``2027`` are constraints, never anchors.

This module extracts **phrase anchors** (multi-word entities and events)
from a market and tells the scorer whether a post text contains the
*whole* phrase. When a market has at least one phrase anchor, the scorer
treats isolated tokens of that phrase as insufficient evidence on their
own — the post must mention the phrase (or a registered alias) to earn
phrase credit.

The logic is intentionally minimal and transparent. No NER model is
loaded; the rules are documented inline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from polysignal.domain import Market

# Multi-word phrases worth treating as a single anchor even when extracted
# from a slug-cased question. Lower-cased on both sides of the comparison.
_KNOWN_PHRASES: frozenset[str] = frozenset(
    {
        "capital gains tax",
        "crypto tax",
        "stanley cup",
        "nba finals",
        "world cup",
        "world series",
        "super bowl",
        "donald trump",
        "joe biden",
        "kamala harris",
        "xi jinping",
        "vladimir putin",
        "taylor swift",
        "harvey weinstein",
        "elon musk",
        "federal reserve",
        "supreme court",
        "white house",
        "wall street",
        "interest rate",
        "rate cut",
        "rate hike",
        "kraken ipo",
        "openai ipo",
        "spacex ipo",
        "anthropic ipo",
        "gta vi",
        "gta 6",
        "grand theft auto",
        "north korea",
        "south korea",
        "saudi arabia",
        "european union",
        "new york knicks",
        "new york yankees",
        "los angeles lakers",
        "los angeles dodgers",
        "oklahoma city thunder",
        "vegas golden knights",
        "carolina hurricanes",
        "boston celtics",
        "golden state warriors",
        "manchester united",
        "real madrid",
        "fc barcelona",
        "premier league",
        "champions league",
        "invade taiwan",
        "cut rates",
        "raise rates",
        "abraham accords",
        "model y",
        "model 3",
        "foldable iphone",
        "vision pro",
    }
)

# Bidirectional alias map: every variant resolves to the same canonical
# form (the *first* element of each tuple).
_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("gta vi", "gta 6", "grand theft auto vi", "grand theft auto 6"),
    ("federal reserve", "fed"),
    ("bitcoin", "btc"),
    ("ethereum", "eth"),
    ("donald trump", "trump"),
    ("joe biden", "biden"),
    ("kamala harris", "harris"),
    ("xi jinping", "xi"),
    ("vladimir putin", "putin"),
    ("taylor swift", "swift"),
    ("elon musk", "musk"),
    ("openai", "open ai"),
    ("supreme court", "scotus"),
    ("kraken ipo", "kraken initial public offering"),
)


@lru_cache(maxsize=1)
def _alias_to_canonical() -> dict[str, str]:
    """Lazy alias dict; lowercase keys → canonical phrase."""
    out: dict[str, str] = {}
    for group in _ALIAS_GROUPS:
        canonical = group[0]
        for variant in group:
            out[variant] = canonical
    return out


_GENERIC_STOPWORDS: frozenset[str] = frozenset(
    {
        "will",
        "be",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "for",
        "to",
        "with",
        "from",
        "by",
        "on",
        "at",
        "in",
        "this",
        "that",
        "next",
        "before",
        "after",
        "until",
        "into",
        "out",
        "over",
        "under",
        "any",
        "all",
        "new",
        "old",
        "day",
        "days",
        "year",
        "years",
        "hit",
        "hits",
        "get",
        "gets",
        "got",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "make",
        "makes",
        "made",
        "say",
        "says",
        "said",
        "see",
        "sees",
        "saw",
        "more",
        "less",
        "than",
        "very",
        "much",
        "us",
        "u.s.",
    }
)


def _is_stopwordy(token: str) -> bool:
    return token.lower() in _GENERIC_STOPWORDS


def _slug_phrases(slug: str) -> set[str]:
    """Extract content bigrams/trigrams from a market slug.

    Stopwords (``will``, ``be``, ``the``, ...) at the edges are stripped,
    and a phrase needs at least one token >= 4 chars *and* at least one
    real content word remaining after stopword removal.
    """
    tokens = [tok for tok in slug.replace("_", "-").split("-") if tok]
    out: set[str] = set()
    for size in (3, 2):
        for i in range(len(tokens) - size + 1):
            window = tokens[i : i + size]
            if any(tok.isdigit() for tok in window):
                continue
            if any(_is_stopwordy(tok) for tok in window):
                continue
            if not all(len(tok) >= 2 for tok in window):
                continue
            if not any(len(tok) >= 4 for tok in window):
                continue
            out.add(" ".join(window).lower())
    return out


def _question_phrases(question: str) -> set[str]:
    """Extract capitalized bigrams/trigrams from the market question.

    A capitalized run of 2+ adjacent words (``Stanley Cup``, ``Donald
    Trump``, ``Federal Reserve``) is taken as a candidate phrase anchor.
    The first word of the sentence is excluded so ``Will Bitcoin`` does
    not become a phrase.
    """
    out: set[str] = set()
    pattern = re.compile(
        r"\b([A-Z][\w']*"
        r"(?:\s+(?:of|the|and|for|to|on|at|in)\s+[A-Z][\w']*|\s+[A-Z][\w']*)+)\b"
    )
    for match in pattern.finditer(question or ""):
        phrase = match.group(1).strip()
        tokens = phrase.split()
        # Strip a leading sentence-starter (Will/What/Is/...).
        if tokens and tokens[0].lower() in _GENERIC_STOPWORDS:
            tokens = tokens[1:]
        if len(tokens) < 2:
            continue
        normalized = " ".join(tokens).lower()
        if not any(len(tok) >= 4 for tok in tokens):
            continue
        out.add(normalized)
    return out


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


@dataclass(frozen=True)
class MarketPhraseProfile:
    """Phrase-level anchor profile for a single market.

    ``hard_phrases`` are curated phrases from :data:`_KNOWN_PHRASES` and
    capitalized question runs — these are high-precision and trigger the
    scorer's promotion gate. ``soft_phrases`` are slug-derived bigrams
    used only for diagnostics.
    """

    market_id: str
    hard_phrases: frozenset[str] = field(default_factory=frozenset)
    soft_phrases: frozenset[str] = field(default_factory=frozenset)

    @property
    def phrases(self) -> frozenset[str]:
        return self.hard_phrases | self.soft_phrases

    @property
    def canonical_phrases(self) -> frozenset[str]:
        return self.hard_phrases

    @property
    def has_phrase(self) -> bool:
        return bool(self.hard_phrases)


_PHRASE_PROFILE_CACHE: dict[tuple[str, str, str], MarketPhraseProfile] = {}


def build_phrase_profile(market: Market) -> MarketPhraseProfile:
    """Return the phrase anchors derived from this market (memoized per market).

    ``MarketPhraseProfile`` is immutable (frozensets) and depends only on the
    market, so caching it avoids rebuilding the same phrase profile for every
    (post, market) pair the scorer evaluates.
    """
    key = (market.market_id, market.question or "", market.slug or "")
    cached = _PHRASE_PROFILE_CACHE.get(key)
    if cached is not None:
        return cached
    profile = _build_phrase_profile_uncached(market)
    _PHRASE_PROFILE_CACHE[key] = profile
    return profile


def _build_phrase_profile_uncached(market: Market) -> MarketPhraseProfile:
    """Return the phrase anchors derived from this market."""
    question = market.question or ""
    description = market.description or ""
    slug = market.slug or ""
    tags = " ".join(market.tags or [])

    alias_map = _alias_to_canonical()

    hard: set[str] = set()
    for phrase in _question_phrases(question):
        hard.add(alias_map.get(phrase, phrase))

    haystack = _normalize(f"{question} {description} {slug.replace('-', ' ')} {tags}")
    for phrase in _KNOWN_PHRASES:
        if phrase in haystack:
            hard.add(alias_map.get(phrase, phrase))

    soft: set[str] = set()
    for phrase in _slug_phrases(slug):
        canonical = alias_map.get(phrase, phrase)
        if canonical not in hard:
            soft.add(canonical)

    return MarketPhraseProfile(
        market_id=market.market_id,
        hard_phrases=frozenset(hard),
        soft_phrases=frozenset(soft),
    )


def _phrase_matches(phrase: str, text: str) -> bool:
    """Whole-phrase match: every token of ``phrase`` appears as a
    word-bounded run inside ``text``."""
    pattern = r"\b" + r"\s+".join(re.escape(tok) for tok in phrase.split()) + r"\b"
    return re.search(pattern, text) is not None


def phrase_matches(profile: MarketPhraseProfile, text: str) -> list[str]:
    """Return the subset of profile phrases (canonical) present in ``text``.

    Aliases are checked so that ``Fed`` matches a *Federal Reserve* market
    profile and ``GTA 6`` matches a *GTA VI* profile, and vice versa.
    """
    if not profile.canonical_phrases or not text:
        return []
    norm = _normalize(text)
    alias_map = _alias_to_canonical()
    hits: set[str] = set()
    for canonical in profile.canonical_phrases:
        candidates = {canonical}
        for variant, target in alias_map.items():
            if target == canonical:
                candidates.add(variant)
        for phrase in candidates:
            if _phrase_matches(phrase, norm):
                hits.add(canonical)
                break
    return sorted(hits)


__all__ = [
    "MarketPhraseProfile",
    "build_phrase_profile",
    "phrase_matches",
]

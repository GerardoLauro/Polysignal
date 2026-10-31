"""Deterministic text normalization utilities.

The same normalization is applied in two places:

* on every ingested ``SocialPost.text`` (stored as ``text_normalized``);
* on every ``Market.question`` and the candidate keywords used by the
  heuristic linker.

Keeping the rules in one place guarantees that the two halves of the
linking heuristic see exactly the same surface form.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"(?<![A-Za-z0-9_])@([A-Za-z0-9_]{1,30})")
_HASHTAG_RE = re.compile(r"(?<![A-Za-z0-9_])#([A-Za-z0-9_]{1,50})")
_NON_WORD_RE = re.compile(r"[^a-z0-9\s]+")

# A small, conservative stopword list. We deliberately do not import a
# heavy NLP library; the cost / opacity is not justified at this layer.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "then",
        "so",
        "of",
        "in",
        "on",
        "at",
        "by",
        "for",
        "to",
        "from",
        "with",
        "without",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "do",
        "does",
        "did",
        "doing",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "not",
        "no",
        "yes",
    }
)


def normalize_text(text: str) -> str:
    """Return a canonical, lower-cased, URL-free form of ``text``.

    Steps:
      1. Unicode NFKC normalization (full-width → half-width, ligatures).
      2. URL removal.
      3. Lower-casing.
      4. Mention / hashtag handles are kept (without the ``@``/``#``)
         so they remain searchable.
      5. Non-word characters collapsed to whitespace.
      6. Whitespace collapsed.
    """
    if text is None:
        return ""

    out = unicodedata.normalize("NFKC", text)
    out = _URL_RE.sub(" ", out)
    out = out.lower()
    out = _MENTION_RE.sub(lambda m: f" {m.group(1)} ", out)
    out = _HASHTAG_RE.sub(lambda m: f" {m.group(1)} ", out)
    out = _NON_WORD_RE.sub(" ", out)
    out = _WHITESPACE_RE.sub(" ", out).strip()
    return out


def tokenize(text: str, *, drop_stopwords: bool = True, min_length: int = 2) -> list[str]:
    """Tokenize text, dropping stopwords and very short tokens by default."""
    if not text:
        return []
    normalized = normalize_text(text)
    tokens = [t for t in normalized.split(" ") if len(t) >= min_length]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens


def extract_hashtags(text: str) -> list[str]:
    """Return hashtags from raw text, lowercased, without leading ``#``."""
    if not text:
        return []
    return [m.group(1).lower() for m in _HASHTAG_RE.finditer(text)]


def extract_mentions(text: str) -> list[str]:
    """Return mention handles from raw text, lowercased, without leading ``@``."""
    if not text:
        return []
    return [m.group(1).lower() for m in _MENTION_RE.finditer(text)]


def token_set(text: str) -> set[str]:
    """Convenience: ``set(tokenize(text))``."""
    return set(tokenize(text))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard similarity over two token sets. Empty inputs → ``0.0``."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


__all__ = [
    "extract_hashtags",
    "extract_mentions",
    "jaccard",
    "normalize_text",
    "token_set",
    "tokenize",
]

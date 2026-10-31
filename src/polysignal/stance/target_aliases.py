"""Conservative target-outcome alias generation for multi-outcome stance."""

from __future__ import annotations

import re
import unicodedata

_PAREN_RE = re.compile(r"\(([^()]*)\)")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")
_COMMON_SINGLE_TOKENS = {
    "candidate",
    "city",
    "club",
    "country",
    "market",
    "none",
    "other",
    "party",
    "state",
    "team",
    "united",
}
_NON_PERSON_PREFIXES = {
    "candidate",
    "city",
    "club",
    "country",
    "east",
    "los",
    "new",
    "north",
    "party",
    "san",
    "south",
    "state",
    "team",
    "the",
    "united",
    "west",
}
_NAME_SUFFIXES = {"ii", "iii", "iv", "jr", "sr"}


def normalize_target_alias(value: str) -> str:
    """Lowercase ``value``, strip punctuation, and collapse whitespace."""
    normalized = unicodedata.normalize("NFKC", value or "").lower()
    return _WHITESPACE_RE.sub(" ", " ".join(_TOKEN_RE.findall(normalized))).strip()


def _safe_derived_alias(value: str) -> str | None:
    alias = normalize_target_alias(value)
    if len(alias) < 3:
        return None
    tokens = alias.split()
    if len(tokens) == 1 and alias in _COMMON_SINGLE_TOKENS:
        return None
    return alias


def _surname_alias(value: str) -> str | None:
    """Return a surname-like alias only for conservative person-shaped labels."""
    outside = _PAREN_RE.sub("", value).strip()
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z'.-]*", outside)
    while raw_tokens and normalize_target_alias(raw_tokens[-1]) in _NAME_SUFFIXES:
        raw_tokens.pop()
    if not 2 <= len(raw_tokens) <= 4:
        return None
    normalized_tokens = [normalize_target_alias(token) for token in raw_tokens]
    if normalized_tokens[0] in _NON_PERSON_PREFIXES:
        return None
    if not all(token and token[0].isupper() for token in raw_tokens):
        return None
    surname = normalized_tokens[-1]
    if len(surname) < 4 or surname in _COMMON_SINGLE_TOKENS:
        return None
    return surname


def _parenthetical_short_form(value: str) -> str | None:
    """Return acronym-shaped parenthetical text, rejecting broad descriptors."""
    alias = _safe_derived_alias(value)
    if alias is None or len(alias) > 12:
        return None
    raw_tokens = re.findall(r"[A-Za-z0-9]+", value)
    if not raw_tokens:
        return None
    if len(raw_tokens) == 1 and raw_tokens[0].isupper():
        return alias
    if (
        raw_tokens[0].isupper()
        and len(raw_tokens[0]) >= 2
        and all(normalize_target_alias(token) in _NAME_SUFFIXES for token in raw_tokens[1:])
    ):
        return alias
    return None


def generate_target_aliases(target_outcome_name: str | None) -> list[str]:
    """Return stable, conservative aliases for one target outcome label."""
    if not target_outcome_name:
        return []

    aliases: set[str] = set()
    normalized = normalize_target_alias(target_outcome_name)
    if normalized:
        aliases.add(normalized)

    outside = _PAREN_RE.sub("", target_outcome_name).strip()
    outside_alias = _safe_derived_alias(outside)
    if outside_alias:
        aliases.add(outside_alias)

    for parenthetical in _PAREN_RE.findall(target_outcome_name):
        parenthetical_alias = _parenthetical_short_form(parenthetical)
        if parenthetical_alias:
            aliases.add(parenthetical_alias)

    surname = _surname_alias(outside)
    if surname:
        aliases.add(surname)

    return sorted(aliases, key=lambda alias: (-len(alias.split()), -len(alias), alias))


def mentions_target_alias(text: str, aliases: list[str]) -> bool:
    """Return whether normalized ``text`` contains any complete alias phrase."""
    normalized_text = f" {normalize_target_alias(text)} "
    normalized_aliases = [normalize_target_alias(alias) for alias in aliases]
    return any(alias and f" {alias} " in normalized_text for alias in normalized_aliases)


__all__ = [
    "generate_target_aliases",
    "mentions_target_alias",
    "normalize_target_alias",
]

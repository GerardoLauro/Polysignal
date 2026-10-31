"""Hashtag, mention, and named-entity extraction (Traccia_2 §2 post fields).

Section 2 of the assignment requires ``hashtags`` and ``mentioned_entities``
(persons / organizations / locations via NER) on each collected post, plus
``language``. This module materializes those fields. Like the sentiment
layer it is tiered so it always runs offline:

* **entities** — uses spaCy (``en_core_web_sm``) when installed; otherwise a
  zero-dependency heuristic extractor that combines a domain gazetteer
  (politicians, central banks, crypto assets, leagues, …) with capitalized
  n-gram detection. The heuristic is deliberately conservative to keep
  precision reasonable on noisy social text.
* **hashtags / mentions** — deterministic regex, identical across backends.
* **language** — uses ``langdetect`` when installed; otherwise a tiny
  stop-word heuristic that distinguishes the corpus's dominant languages
  (en / ru / it / es / fr) well enough for filtering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from polysignal.domain.enums import EntityType

_HASHTAG_RE = re.compile(r"(?:^|(?<=\s))#(\w{2,50})")
_MENTION_RE = re.compile(r"(?:^|(?<=\s))@(\w{2,50})")
_CAP_SEQ_RE = re.compile(r"\b([A-Z][a-zA-Z0-9.&'-]+(?:\s+[A-Z][a-zA-Z0-9.&'-]+){0,3})\b")
_WORD_RE = re.compile(r"[a-zA-Zа-яА-Я']+")

# Words that are capitalized for non-entity reasons (sentence start, common
# nouns) — filtered out of the heuristic capitalized-sequence pass.
_CAP_STOPWORDS = frozenset(
    """The A An And But Or So Yes No Will Is Are Was Were Be This That These Those It
    He She They We You I If When Then Why How What Who Where Which Just Now Today
    Tomorrow Yesterday New Big Good Great Best More Most Very Really Breaking Update
    Watch Live Read Click Follow Like Share Comment Thread Source Via RT""".split()
)

# Domain gazetteer: (surface form lower-cased) -> EntityType. Kept compact and
# focused on the three thematic domains so the heuristic backend surfaces the
# entities that actually matter for these markets.
_GAZETTEER: dict[str, EntityType] = {}
_PERSONS = [
    "trump", "biden", "harris", "putin", "zelensky", "xi jinping", "netanyahu",
    "musk", "elon musk", "powell", "jerome powell", "modi", "macron", "starmer",
    "milei", "lula", "erdogan", "kim jong un",
]
_ORGS = [
    "fed", "federal reserve", "ecb", "sec", "nato", "opec", "white house",
    "congress", "senate", "supreme court", "binance", "coinbase", "tesla",
    "microstrategy", "blackrock", "nasa", "spacex", "openai", "fifa", "uefa",
    "nfl", "nba", "mlb", "nhl", "ufc", "ncaa", "premier league", "champions league",
    "republican party", "democratic party",
]
_LOCS = [
    "ukraine", "russia", "china", "iran", "israel", "gaza", "taiwan", "united states",
    "usa", "us", "uk", "europe", "india", "north korea", "south korea", "venezuela",
    "argentina", "france", "germany", "japan",
]
_PRODUCTS = [
    "bitcoin", "btc", "ethereum", "eth", "solana", "xrp", "dogecoin", "gta vi",
    "gta 6", "nvidia", "s&p 500", "nasdaq",
]
for _w in _PERSONS:
    _GAZETTEER[_w] = EntityType.PERSON
for _w in _ORGS:
    _GAZETTEER[_w] = EntityType.ORG
for _w in _LOCS:
    _GAZETTEER[_w] = EntityType.LOC
for _w in _PRODUCTS:
    _GAZETTEER[_w] = EntityType.PRODUCT

# One compiled alternation over the whole gazetteer (longest-first so multi-word
# surfaces win), so each post costs a single regex pass instead of ~80 — this is
# what makes enrichment tractable over the >1M-row corpus.
_GAZETTEER_RE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in sorted(_GAZETTEER, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExtractedEntity:
    text: str
    label: str  # EntityType value

    def as_dict(self) -> dict[str, str]:
        return {"text": self.text, "label": self.label}


def extract_hashtags(text: str) -> list[str]:
    if not text:
        return []
    seen: dict[str, None] = {}
    for m in _HASHTAG_RE.finditer(text):
        seen.setdefault("#" + m.group(1), None)
    return list(seen)


def extract_mentions(text: str) -> list[str]:
    if not text:
        return []
    seen: dict[str, None] = {}
    for m in _MENTION_RE.finditer(text):
        seen.setdefault("@" + m.group(1), None)
    return list(seen)


class EntityBackend:
    name = "base"

    def entities(self, text: str) -> list[ExtractedEntity]:  # pragma: no cover
        raise NotImplementedError


class HeuristicEntityBackend(EntityBackend):
    """Gazetteer + capitalized-sequence NER (no third-party dependency)."""

    name = "heuristic"

    def entities(self, text: str) -> list[ExtractedEntity]:
        if not text or not text.strip():
            return []
        found: dict[str, str] = {}
        # 1) gazetteer hits (highest precision, carries a real label) — single pass
        for m in _GAZETTEER_RE.finditer(text):
            surface = m.group(1).lower()
            etype = _GAZETTEER.get(surface)
            if etype is None:
                continue
            canon = surface.upper() if len(surface) <= 4 else surface.title()
            found.setdefault(canon, etype.value)
        # 2) capitalized multi-word sequences not already covered
        for m in _CAP_SEQ_RE.finditer(text):
            phrase = m.group(1).strip()
            words = phrase.split()
            kept = [w for w in words if w not in _CAP_STOPWORDS]
            if not kept:
                continue
            phrase = " ".join(kept)
            if len(phrase) < 3 or phrase.lower() in _GAZETTEER:
                continue
            if phrase in found:
                continue
            # multi-word capitalized phrase, or a single non-stopword proper noun
            if len(kept) >= 2 or (len(kept) == 1 and kept[0] not in _CAP_STOPWORDS):
                found.setdefault(phrase, EntityType.MISC.value)
        # bound the per-post entity list
        return [ExtractedEntity(t, lbl) for t, lbl in list(found.items())[:25]]


class SpacyEntityBackend(EntityBackend):
    """Optional spaCy NER backend (``en_core_web_sm``)."""

    name = "spacy"
    _LABEL_MAP = {
        "PERSON": EntityType.PERSON,
        "ORG": EntityType.ORG,
        "GPE": EntityType.LOC,
        "LOC": EntityType.LOC,
        "NORP": EntityType.ORG,
        "EVENT": EntityType.EVENT,
        "WORK_OF_ART": EntityType.WORK_OF_ART,
        "PRODUCT": EntityType.PRODUCT,
        "FAC": EntityType.LOC,
    }

    def __init__(self, model: str = "en_core_web_sm") -> None:
        import spacy  # type: ignore

        self._nlp = spacy.load(model, disable=["lemmatizer", "tagger", "parser"])

    def entities(self, text: str) -> list[ExtractedEntity]:
        if not text or not text.strip():
            return []
        doc = self._nlp(text[:2000])
        out: dict[str, str] = {}
        for ent in doc.ents:
            etype = self._LABEL_MAP.get(ent.label_)
            if etype is None:
                continue
            out.setdefault(ent.text.strip(), etype.value)
        return [ExtractedEntity(t, lbl) for t, lbl in out.items()]


def get_entity_backend(prefer: str = "auto") -> EntityBackend:
    """Return the best available entity backend (spaCy if present, else heuristic)."""
    if prefer in ("auto", "spacy"):
        try:
            return SpacyEntityBackend()
        except Exception:
            if prefer == "spacy":
                pass  # fall through to heuristic
    return HeuristicEntityBackend()


# ---- lightweight language detection -------------------------------------

_LANG_STOPWORDS = {
    "ru": "и в не на что с по как это все она так его но да ты к у же вы за",
    "it": "che di la il un per non con una sono come anche più ma se nel alla",
    "es": "que de la el en los una por con para como pero más este está",
    "fr": "que de la le les des une pour dans est pas avec sur plus mais",
    "en": "the and to of a in is that it for on with as are this be at by",
}
_LANG_SETS = {k: frozenset(v.split()) for k, v in _LANG_STOPWORDS.items()}
_CYRILLIC_RE = re.compile(r"[а-яА-Я]")


def detect_language(text: str) -> str | None:
    """Best-effort language code; ``langdetect`` when available, else heuristic."""
    if not text or len(text.strip()) < 3:
        return None
    try:  # pragma: no cover - optional dependency
        from langdetect import detect  # type: ignore

        return detect(text)
    except Exception:
        pass
    if _CYRILLIC_RE.search(text):
        return "ru"
    tokens = set(_WORD_RE.findall(text.lower()))
    if not tokens:
        return None
    best, best_hits = None, 0
    for lang, words in _LANG_SETS.items():
        hits = len(tokens & words)
        if hits > best_hits:
            best, best_hits = lang, hits
    return best if best_hits >= 1 else "en"


__all__ = [
    "EntityBackend",
    "ExtractedEntity",
    "HeuristicEntityBackend",
    "SpacyEntityBackend",
    "detect_language",
    "extract_hashtags",
    "extract_mentions",
    "get_entity_backend",
]

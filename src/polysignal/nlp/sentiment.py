"""Post-text sentiment analysis (Traccia_2 Task 2 / Task 3).

Sentiment here means **polarity of the language** (positive / negative /
neutral) and is deliberately kept distinct from the outcome-aware
*stance* layer (:mod:`polysignal.stance`): stance is interpreted relative
to a market's YES/NO outcomes, whereas sentiment is a property of the
text alone. The assignment requires both — sentiment analysis as a
descriptive technique (Task 2.2) and sentiment features (mean/variance,
positive/negative/neutral fractions, trend) for the optional predictor
(Task 3).

Backends, in order of preference (the analyzer degrades gracefully so
the pipeline always runs offline):

1. ``transformer`` — a HuggingFace text-classification pipeline
   (default model: ``cardiffnlp/twitter-roberta-base-sentiment-latest``),
   used only when ``transformers`` is installed and explicitly selected.
2. ``vader`` — NLTK's VADER (``vaderSentiment``), a rule/lexicon model
   tuned for social-media text (emoji, slang, caps, punctuation). This
   is the default when the dependency is present.
3. ``lexicon`` — a small built-in opinion-lexicon fallback with zero
   third-party dependencies, so sentiment is always available even in a
   minimal environment.

All backends return the same :class:`SentimentScore` so downstream code
never branches on the backend.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from polysignal.domain.enums import SentimentLabel

# VADER-style decision band on the compound score.
_POS_THRESHOLD = 0.05
_NEG_THRESHOLD = -0.05


@dataclass(frozen=True)
class SentimentScore:
    """Polarity scores for one piece of text.

    ``compound`` is a single normalized polarity in ``[-1, 1]``; ``pos``,
    ``neu``, ``neg`` are the proportional intensities (roughly summing to
    1.0). ``label`` is the thresholded verdict. ``method`` records which
    backend produced the score so mixed-provenance datasets stay
    auditable.
    """

    label: SentimentLabel
    compound: float
    pos: float
    neu: float
    neg: float
    method: str

    def as_row(self) -> dict[str, object]:
        return {
            "sentiment_label": self.label.value,
            "sentiment_compound": round(self.compound, 4),
            "sentiment_pos": round(self.pos, 4),
            "sentiment_neu": round(self.neu, 4),
            "sentiment_neg": round(self.neg, 4),
            "sentiment_method": self.method,
        }


def _label_from_compound(compound: float) -> SentimentLabel:
    if compound >= _POS_THRESHOLD:
        return SentimentLabel.POSITIVE
    if compound <= _NEG_THRESHOLD:
        return SentimentLabel.NEGATIVE
    return SentimentLabel.NEUTRAL


_NEUTRAL = SentimentScore(SentimentLabel.NEUTRAL, 0.0, 0.0, 1.0, 0.0, "empty")


class SentimentBackend:
    """Common interface; subclasses implement :meth:`score`."""

    name = "base"

    def score(self, text: str) -> SentimentScore:  # pragma: no cover - interface
        raise NotImplementedError


class VaderBackend(SentimentBackend):
    """VADER lexicon backend (``vaderSentiment``)."""

    name = "vader"

    def __init__(self) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._analyzer = SentimentIntensityAnalyzer()

    def score(self, text: str) -> SentimentScore:
        if not text or not text.strip():
            return _NEUTRAL
        s = self._analyzer.polarity_scores(text)
        compound = float(s["compound"])
        return SentimentScore(
            label=_label_from_compound(compound),
            compound=compound,
            pos=float(s["pos"]),
            neu=float(s["neu"]),
            neg=float(s["neg"]),
            method="vader",
        )


class TransformerBackend(SentimentBackend):
    """Optional HuggingFace transformer backend (opt-in, heavy)."""

    name = "transformer"

    def __init__(
        self, model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ) -> None:
        from transformers import pipeline  # type: ignore

        self._pipe = pipeline("sentiment-analysis", model=model, truncation=True)
        self._model = model

    def score(self, text: str) -> SentimentScore:
        if not text or not text.strip():
            return _NEUTRAL
        out = self._pipe(text[:512])[0]
        raw = str(out["label"]).lower()
        conf = float(out["score"])
        if raw in ("positive", "label_2", "pos"):
            label, compound, pos, neg = SentimentLabel.POSITIVE, conf, conf, 0.0
        elif raw in ("negative", "label_0", "neg"):
            label, compound, pos, neg = SentimentLabel.NEGATIVE, -conf, 0.0, conf
        else:
            label, compound, pos, neg = SentimentLabel.NEUTRAL, 0.0, 0.0, 0.0
        return SentimentScore(
            label=label,
            compound=compound,
            pos=pos,
            neu=max(0.0, 1.0 - pos - neg),
            neg=neg,
            method=f"transformer:{self._model.split('/')[-1]}",
        )


# A compact opinion lexicon for the zero-dependency fallback. It is not a
# replacement for VADER/transformers — it exists so the pipeline never has
# to skip sentiment in a bare environment.
_POS_WORDS = frozenset(
    """good great excellent amazing awesome positive win winning won bullish up surge
    rally gain gains profit success successful strong beat beats record high happy love
    confident optimistic likely sure agree support yes correct right best better boom
    breakthrough soar soared rise rising rises hike approve approved""".split()
)
_NEG_WORDS = frozenset(
    """bad terrible awful poor negative lose losing lost bearish down crash dump drop
    plunge fall falling fell loss fail failed failure weak miss missed low sad hate fear
    worried doubt doubtful unlikely no wrong worse worst disaster collapse reject rejected
    scam dead dying recession crisis selloff""".split()
)
_NEGATORS = frozenset("not no never n't without cannot can't won't don't".split())
_TOKEN_RE = re.compile(r"[a-zA-Z']+")


class LexiconBackend(SentimentBackend):
    """Zero-dependency opinion-lexicon fallback with simple negation."""

    name = "lexicon"

    def score(self, text: str) -> SentimentScore:
        if not text or not text.strip():
            return _NEUTRAL
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return _NEUTRAL
        pos = neg = 0
        for i, tok in enumerate(tokens):
            negated = any(t in _NEGATORS for t in tokens[max(0, i - 3) : i])
            if tok in _POS_WORDS:
                neg += 1 if negated else 0
                pos += 0 if negated else 1
            elif tok in _NEG_WORDS:
                pos += 1 if negated else 0
                neg += 0 if negated else 1
        total = pos + neg
        if total == 0:
            return SentimentScore(SentimentLabel.NEUTRAL, 0.0, 0.0, 1.0, 0.0, "lexicon")
        compound = (pos - neg) / total
        n = len(tokens)
        return SentimentScore(
            label=_label_from_compound(compound),
            compound=compound,
            pos=pos / n,
            neu=max(0.0, (n - total) / n),
            neg=neg / n,
            method="lexicon",
        )


class SentimentAnalyzer:
    """Front door over a single :class:`SentimentBackend`."""

    def __init__(self, backend: SentimentBackend) -> None:
        self._backend = backend

    @property
    def method(self) -> str:
        return self._backend.name

    def score(self, text: str) -> SentimentScore:
        return self._backend.score(text)

    def score_many(self, texts: list[str]) -> list[SentimentScore]:
        return [self._backend.score(t) for t in texts]


def get_sentiment_analyzer(prefer: str = "auto") -> SentimentAnalyzer:
    """Build the best available analyzer.

    ``prefer`` is one of ``auto`` (transformer? → vader → lexicon),
    ``vader``, ``transformer``, or ``lexicon``. Selecting a backend whose
    dependency is missing falls back to the next best option rather than
    raising, so callers can always obtain a working analyzer.
    """
    order = {
        "auto": ["vader", "lexicon"],
        "vader": ["vader", "lexicon"],
        "transformer": ["transformer", "vader", "lexicon"],
        "lexicon": ["lexicon"],
    }.get(prefer, ["vader", "lexicon"])

    for choice in order:
        try:
            if choice == "transformer":
                return SentimentAnalyzer(TransformerBackend())
            if choice == "vader":
                return SentimentAnalyzer(VaderBackend())
            if choice == "lexicon":
                return SentimentAnalyzer(LexiconBackend())
        except Exception:
            continue
    return SentimentAnalyzer(LexiconBackend())


__all__ = [
    "LexiconBackend",
    "SentimentAnalyzer",
    "SentimentBackend",
    "SentimentScore",
    "TransformerBackend",
    "VaderBackend",
    "get_sentiment_analyzer",
]

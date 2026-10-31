"""Topic modeling over social-post text (Traccia_2 Task 2.2).

A deterministic Latent Dirichlet Allocation (LDA) topic model built on
scikit-learn (already a core dependency, so no new runtime requirement).
It is intended to run over the *linked / relevant* posts for the collected
markets — the discourse that actually concerns the contracts — and to
surface the dominant themes per thematic domain.

The output is two small tables:

* ``topics.parquet`` — one row per topic with its top terms and weight;
* an optional per-document assignment (returned in-memory) mapping each
  post to its dominant topic, used by the dashboard / feature layer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Social/markety stop terms layered on top of sklearn's English list.
_EXTRA_STOPWORDS = frozenset(
    """https http www com t co amp rt via just like will market markets polymarket bet
    bets odds today tonight day week month year people think know going get got really
    new said say says according report reports breaking update""".split()
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NONWORD_RE = re.compile(r"[^a-zA-Z\s]")


@dataclass
class Topic:
    topic_id: int
    terms: list[str]
    weight: float


@dataclass
class TopicModelResult:
    topics: list[Topic] = field(default_factory=list)
    doc_topic: list[int] = field(default_factory=list)
    doc_topic_weight: list[float] = field(default_factory=list)
    n_docs: int = 0
    method: str = "sklearn-lda"

    def topics_as_rows(self) -> list[dict[str, object]]:
        return [
            {
                "topic_id": t.topic_id,
                "top_terms": t.terms,
                "label": ", ".join(t.terms[:4]),
                "weight": round(t.weight, 5),
            }
            for t in self.topics
        ]


def _clean(text: str) -> str:
    text = _URL_RE.sub(" ", text or "")
    text = _NONWORD_RE.sub(" ", text)
    return text.lower()


def fit_topic_model(
    texts: list[str],
    *,
    n_topics: int = 10,
    max_features: int = 2000,
    n_top_words: int = 12,
    min_df: int = 3,
    random_state: int = 42,
) -> TopicModelResult:
    """Fit an LDA topic model over ``texts`` and return topics + assignments."""
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    cleaned = [_clean(t) for t in texts if t and t.strip()]
    if len(cleaned) < max(2 * n_topics, 20):
        logger.warning("topics: only %d usable docs; reducing topic count", len(cleaned))
        n_topics = max(2, min(n_topics, max(2, len(cleaned) // 4)))

    stop = list(ENGLISH_STOP_WORDS | _EXTRA_STOPWORDS)
    vec = CountVectorizer(
        max_features=max_features,
        min_df=min(min_df, max(1, len(cleaned) // 10)),
        max_df=0.6,
        stop_words=stop,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    try:
        dtm = vec.fit_transform(cleaned)
    except ValueError as exc:  # empty vocabulary
        logger.warning("topics: vectorizer produced empty vocabulary (%s)", exc)
        return TopicModelResult(n_docs=len(cleaned))
    if dtm.shape[1] == 0:
        return TopicModelResult(n_docs=len(cleaned))

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        max_iter=20,
        random_state=random_state,
    )
    doc_topic = lda.fit_transform(dtm)
    vocab = np.array(vec.get_feature_names_out())

    topics: list[Topic] = []
    topic_mass = doc_topic.sum(axis=0)
    topic_mass = topic_mass / topic_mass.sum() if topic_mass.sum() else topic_mass
    for k, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[::-1][:n_top_words]
        topics.append(Topic(topic_id=k, terms=vocab[top_idx].tolist(), weight=float(topic_mass[k])))

    assignments = doc_topic.argmax(axis=1).tolist()
    weights = doc_topic.max(axis=1).tolist()
    return TopicModelResult(
        topics=topics,
        doc_topic=assignments,
        doc_topic_weight=[round(float(w), 5) for w in weights],
        n_docs=len(cleaned),
    )


__all__ = ["Topic", "TopicModelResult", "fit_topic_model"]

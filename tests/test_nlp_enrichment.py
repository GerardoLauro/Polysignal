"""Tests for the NLP enrichment layer: sentiment, extraction, topics."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from polysignal.domain.enums import MarketDomain, SentimentLabel, ThematicCategory, thematic_category
from polysignal.nlp.extract import (
    detect_language,
    extract_hashtags,
    extract_mentions,
    get_entity_backend,
)
from polysignal.nlp.sentiment import get_sentiment_analyzer
from polysignal.nlp.topics import fit_topic_model


# --------------------------------------------------------------- sentiment
def test_sentiment_polarity_directions():
    an = get_sentiment_analyzer("auto")
    assert an.score("This is great, wonderful, excellent news!").label == SentimentLabel.POSITIVE
    assert an.score("This is terrible, awful, a total disaster.").label == SentimentLabel.NEGATIVE


def test_sentiment_empty_is_neutral():
    an = get_sentiment_analyzer("auto")
    s = an.score("")
    assert s.label == SentimentLabel.NEUTRAL
    assert s.compound == 0.0


def test_lexicon_backend_always_available():
    an = get_sentiment_analyzer("lexicon")
    assert an.method == "lexicon"
    assert an.score("win profit success bullish").label == SentimentLabel.POSITIVE
    assert an.score("loss crash disaster bearish").label == SentimentLabel.NEGATIVE


def test_lexicon_negation_flips_polarity():
    an = get_sentiment_analyzer("lexicon")
    assert an.score("not good").label != SentimentLabel.POSITIVE


def test_sentiment_row_serialization():
    an = get_sentiment_analyzer("auto")
    row = an.score("great").as_row()
    assert set(row) == {
        "sentiment_label",
        "sentiment_compound",
        "sentiment_pos",
        "sentiment_neu",
        "sentiment_neg",
        "sentiment_method",
    }


# ---------------------------------------------------------------- extract
def test_extract_hashtags_and_mentions():
    text = "Big news on #Bitcoin and #ETH from @elonmusk and @saylor!"
    assert extract_hashtags(text) == ["#Bitcoin", "#ETH"]
    assert extract_mentions(text) == ["@elonmusk", "@saylor"]


def test_entity_extraction_gazetteer_labels():
    eb = get_entity_backend("heuristic")
    ents = {e.text: e.label for e in eb.entities("Trump met Putin about Ukraine and Bitcoin.")}
    assert ents.get("Trump") == "person"
    assert ents.get("Putin") == "person"
    assert ents.get("Ukraine") == "loc"
    assert ents.get("Bitcoin") == "product"


def test_entity_extraction_handles_empty():
    eb = get_entity_backend("heuristic")
    assert eb.entities("") == []


def test_language_detection_basic():
    assert detect_language("the market is going to resolve yes tomorrow") == "en"
    assert detect_language("это очень хорошая новость для рынка") == "ru"


# ----------------------------------------------------------------- topics
def test_topic_model_finds_themes():
    docs = (
        ["bitcoin ethereum crypto price rally bull token blockchain"] * 15
        + ["election president vote senate congress political campaign"] * 15
        + ["team game score playoff championship league season win"] * 15
    )
    result = fit_topic_model(docs, n_topics=3, min_df=1)
    assert result.topics, "expected at least one topic"
    assert result.n_docs == len(docs)
    # every doc receives a topic assignment
    assert len(result.doc_topic) == len(docs)
    all_terms = {term for t in result.topics for term in t.terms}
    assert all_terms & {"bitcoin", "election", "team", "crypto", "president", "game"}


def test_topic_model_tiny_corpus_degrades_gracefully():
    result = fit_topic_model(["short text", "another short text"], n_topics=10)
    # must not raise; may legitimately return few/no topics
    assert result.n_docs >= 0


# ----------------------------------------------------------- thematic map
def test_thematic_category_mapping():
    assert thematic_category(MarketDomain.POLITICS) == ThematicCategory.POLITICS
    assert thematic_category(MarketDomain.CRYPTO) == ThematicCategory.FINANCE
    assert thematic_category(MarketDomain.ECONOMY) == ThematicCategory.FINANCE
    assert thematic_category(MarketDomain.SPORTS) == ThematicCategory.SPORTS
    assert thematic_category(MarketDomain.CULTURE) == ThematicCategory.SPORTS
    assert thematic_category(MarketDomain.SCIENCE) is None
    assert thematic_category(MarketDomain.OTHER) is None


# -------------------------------------------------------- enrichment run
def test_run_enrichment_writes_table(tmp_path):
    from polysignal.nlp.enrichment_runner import run_enrichment

    posts = pd.DataFrame(
        {
            "post_id": ["p1", "p2", "p3"],
            "platform": ["telegram", "reddit", "telegram"],
            "text": [
                "Bitcoin is surging, bullish! #crypto @saylor",
                "Terrible loss for the team, awful game.",
                float("nan"),  # exercises NaN-safe coercion
            ],
            "title": [None, "Match recap", None],
            "language": [None, None, None],
        }
    )
    src = tmp_path / "social_posts.parquet"
    posts.to_parquet(src, index=False)
    out = tmp_path / "post_enrichment.parquet"

    summary = run_enrichment(src, out, sentiment_prefer="auto", entity_prefer="heuristic")
    assert summary.posts_enriched == 3
    df = pd.read_parquet(out)
    assert {"post_id", "sentiment_label", "hashtags", "mentioned_entities", "language"} <= set(
        df.columns
    )
    p1 = df[df["post_id"] == "p1"].iloc[0]
    assert list(p1["hashtags"]) == ["#crypto"]
    assert isinstance(p1["enriched_at"], str) or hasattr(p1["enriched_at"], "isoformat")
    # timestamp present
    assert datetime.now(tz=timezone.utc).year >= 2026

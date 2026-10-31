"""Tests for ``polysignal.nlp.normalize``."""

from __future__ import annotations

import pytest

from polysignal.nlp.normalize import (
    extract_hashtags,
    extract_mentions,
    jaccard,
    normalize_text,
    token_set,
    tokenize,
)


class TestNormalizeText:
    def test_empty(self):
        assert normalize_text("") == ""
        assert normalize_text(None) == ""  # type: ignore[arg-type]

    def test_strips_urls(self):
        assert normalize_text("see https://example.com today") == "see today"

    def test_lowercases(self):
        assert normalize_text("Polymarket WINS") == "polymarket wins"

    def test_preserves_mention_handle_without_at(self):
        out = normalize_text("hello @alice how are you")
        assert "alice" in out.split()
        assert "@alice" not in out

    def test_preserves_hashtag_text_without_hash(self):
        out = normalize_text("breaking #election2024 news")
        assert "election2024" in out.split()
        assert "#election2024" not in out

    def test_unicode_nfkc(self):
        # full-width digits and letters fold to ASCII
        out = normalize_text("Ｐｏｌｙ１２３")
        assert out == "poly123"

    def test_collapses_whitespace(self):
        assert normalize_text("a    b\tc\nd") == "a b c d"


class TestTokenize:
    def test_drops_stopwords_by_default(self):
        assert "the" not in tokenize("the candidate wins the election")

    def test_keeps_stopwords_when_disabled(self):
        tokens = tokenize("the candidate", drop_stopwords=False)
        assert "the" in tokens

    def test_drops_short_tokens(self):
        tokens = tokenize("a be the candidate", min_length=3)
        assert "be" not in tokens
        assert "candidate" in tokens


class TestHashtagsAndMentions:
    def test_hashtags(self):
        assert extract_hashtags("#A #b #C2") == ["a", "b", "c2"]

    def test_mentions(self):
        assert extract_mentions("@alice and @BOB greet @charlie") == ["alice", "bob", "charlie"]


class TestJaccard:
    def test_identical(self):
        assert jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint(self):
        assert jaccard({"a"}, {"b"}) == 0.0

    def test_partial(self):
        assert jaccard({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(0.5)

    def test_empty(self):
        assert jaccard(set(), {"a"}) == 0.0
        assert jaccard({"a"}, set()) == 0.0


def test_token_set_dedupes():
    assert token_set("hello hello world") == {"hello", "world"}

"""Unit tests for stance cue extraction (Branch 06A)."""

from __future__ import annotations

from polysignal.stance.cues import extract_cues, no_cues, yes_cues


def _phrases(cues):
    return [cue.phrase for cue in cues]


def test_extract_cues_finds_yes_phrases_only_at_word_boundaries() -> None:
    matches = extract_cues("Bitcoin is set to reach $150k by 2026.")
    yes_phrases = _phrases(yes_cues(matches))
    # Longest phrase wins — "is set to" beats "set to" — and "set to"
    # must not double-count.
    assert "is set to" in yes_phrases
    assert "set to" not in yes_phrases


def test_extract_cues_picks_no_phrases() -> None:
    matches = extract_cues("Rockstar may delay GTA VI until 2026.")
    no_phrases = _phrases(no_cues(matches))
    assert any("delay" in p for p in no_phrases)


def test_extract_cues_detects_uncertainty_cues() -> None:
    matches = extract_cues("Rumor says the deal could close next month.")
    categories = {cue.category for cue in matches}
    assert "uncertainty" in categories


def test_negated_yes_cue_is_reported_as_no() -> None:
    matches = extract_cues("Bitcoin will not reach $150k by 2026.")
    # "will not" is itself a NO phrase; ensure NO bucket is non-empty.
    no_phrases = _phrases(no_cues(matches))
    assert no_phrases
    # And no YES phrase should survive un-negated.
    surviving_yes = [c for c in matches if c.category == "yes" and not c.negated]
    assert not surviving_yes


def test_longest_phrase_wins_over_shorter_substring() -> None:
    matches = extract_cues("Analysts expect Bitcoin to reach $150k.")
    # "analysts expect" should be matched as a single phrase
    yes_phrases = _phrases(yes_cues(matches))
    assert "analysts expect" in yes_phrases


def test_empty_text_returns_no_cues() -> None:
    assert extract_cues("") == []
    assert extract_cues("   ") == []

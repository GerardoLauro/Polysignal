"""Tests for ``polysignal.storage.cache``."""

from __future__ import annotations

from polysignal.storage.cache import JSONCache, stable_key


def test_stable_key_is_deterministic():
    a = stable_key({"prompt": "hi", "model": "foo"})
    b = stable_key({"model": "foo", "prompt": "hi"})
    assert a == b


def test_stable_key_changes_with_payload():
    a = stable_key({"prompt": "hi"})
    b = stable_key({"prompt": "hello"})
    assert a != b


def test_json_cache_roundtrip(tmp_path):
    cache = JSONCache(tmp_path / "llm")
    key = stable_key({"q": "test"})
    assert cache.has(key) is False
    cache.set(key, {"answer": 42})
    assert cache.has(key) is True
    assert cache.get(key) == {"answer": 42}


def test_get_or_compute_only_runs_when_missing(tmp_path):
    cache = JSONCache(tmp_path / "llm")
    key = stable_key({"q": "test"})
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"answer": "computed"}

    out_a = cache.get_or_compute(key, compute)
    out_b = cache.get_or_compute(key, compute)
    assert out_a == out_b == {"answer": "computed"}
    assert calls["n"] == 1


def test_delete(tmp_path):
    cache = JSONCache(tmp_path / "llm")
    cache.set("k", {"v": 1})
    assert cache.has("k") is True
    cache.delete("k")
    assert cache.has("k") is False

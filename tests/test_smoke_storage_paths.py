"""Smoke tests for the storage path layout."""

from __future__ import annotations

from polysignal.storage.paths import DataPaths, get_data_paths


def test_get_data_paths_returns_typed_layout(tmp_path):
    paths = get_data_paths(root=tmp_path)
    assert isinstance(paths, DataPaths)
    assert paths.root == tmp_path.resolve()


def test_ensure_creates_full_layout(tmp_path):
    paths = get_data_paths(root=tmp_path)
    # Before ensure(): nothing exists under tmp_path.
    assert not paths.raw.exists()
    paths.ensure()
    # All declared subdirectories must exist after a single call.
    for directory in (
        paths.raw,
        paths.normalized,
        paths.analytical,
        paths.cache,
        paths.graph_exports,
        paths.llm_cache,
        paths.feature_cache,
    ):
        assert directory.exists(), f"{directory} was not created"
        assert directory.is_dir()


def test_ensure_is_idempotent(tmp_path):
    paths = get_data_paths(root=tmp_path)
    paths.ensure()
    # Re-running must not raise and must leave the layout intact.
    paths.ensure()
    assert paths.cache.exists()


def test_known_subpaths_resolve_relative_to_root(tmp_path):
    paths = get_data_paths(root=tmp_path)
    assert paths.markets_parquet.parent == paths.normalized
    assert paths.raw_polymarket_events.parent == paths.raw_polymarket
    assert paths.posts_parquet.parent == paths.normalized
    assert paths.social_queries_parquet.name == "social_queries.parquet"
    assert paths.social_queries_parquet.parent == paths.normalized_social
    assert paths.social_pressure_parquet.parent == paths.analytical
    assert paths.llm_cache.parent == paths.cache

"""Smoke tests for configuration loading."""

from __future__ import annotations

from polysignal.config import Settings, get_settings
from polysignal.config.settings import LLMSettings


def test_get_settings_returns_settings_instance():
    settings = get_settings()
    assert isinstance(settings, Settings)


def test_default_settings_are_internally_consistent(monkeypatch):
    settings = get_settings()
    # Defaults must always resolve to something sane.
    assert settings.env in {"local", "ci", "prod"}
    assert settings.polymarket.gamma_api_base.startswith("https://")
    assert settings.polymarket.clob_api_base.startswith("https://")
    assert settings.neo4j.uri.startswith(("bolt://", "neo4j://"))
    # Clean package defaults keep LLM use opt-in even if local .env enables it.
    for name in ("POLYSIGNAL_LLM_PROVIDER", "LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    defaults = LLMSettings(_env_file=None)
    assert defaults.provider == "disabled"
    assert defaults.enabled is False


def test_unconfigured_social_platforms_report_unavailable():
    settings = get_settings()
    # The .env.example contains empty credentials; a developer who has
    # not yet filled them in must see availability=False.
    if not (settings.reddit.client_id and settings.reddit.client_secret):
        assert settings.reddit.configured is False
    if not settings.twitter.bearer_token:
        assert settings.twitter.configured is False
    if not (settings.telegram.api_id and settings.telegram.api_hash):
        assert settings.telegram.configured is False

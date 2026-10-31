"""Application settings, loaded from environment variables / .env file.

We use a single ``Settings`` object so that the rest of the codebase
never reads environment variables directly. This keeps tests easy
(swap settings in fixtures) and makes the config surface auditable.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PolymarketSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POLYMARKET_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gamma_api_base: str = "https://gamma-api.polymarket.com"
    clob_api_base: str = "https://clob.polymarket.com"
    request_timeout_seconds: float = 30.0
    user_agent: str = "PolySignal-Research/0.1"


class RedditSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDDIT_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    client_id: str = ""
    client_secret: str = ""
    user_agent: str = "PolySignal-Research/0.1"
    username: str = ""
    password: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)


class TwitterSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    bearer_token: str = Field(default="", validation_alias=AliasChoices("TWITTER_BEARER_TOKEN"))
    # X/Twitter serves profile timelines only to an authenticated session. The
    # public-web adapter is gated on these two browser-session cookies, read
    # from the untracked .env (never versioned). Accept the canonical X_* names
    # as well as TWITTER_* for symmetry with the other platforms.
    auth_token: str = Field(
        default="",
        validation_alias=AliasChoices("X_AUTH_TOKEN", "TWITTER_AUTH_TOKEN"),
    )
    csrf_token: str = Field(
        default="",
        validation_alias=AliasChoices("X_CT0", "TWITTER_CT0"),
    )

    @property
    def configured(self) -> bool:
        return bool(self.bearer_token)

    @property
    def web_configured(self) -> bool:
        """True when both X session cookies are present for the web adapter."""
        return bool(self.auth_token and self.csrf_token)


class TelegramSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    api_id: str = Field(
        default="",
        validation_alias=AliasChoices("POLYSIGNAL_TELEGRAM_API_ID", "TELEGRAM_API_ID"),
    )
    api_hash: str = Field(
        default="",
        validation_alias=AliasChoices("POLYSIGNAL_TELEGRAM_API_HASH", "TELEGRAM_API_HASH"),
    )
    session_name: str = Field(
        default="polysignal_telegram",
        validation_alias=AliasChoices(
            "POLYSIGNAL_TELEGRAM_SESSION_NAME",
            "TELEGRAM_SESSION_NAME",
        ),
    )
    session_dir: Path = Field(
        default=Path("./data/cache/telegram"),
        validation_alias=AliasChoices(
            "POLYSIGNAL_TELEGRAM_SESSION_DIR",
            "TELEGRAM_SESSION_DIR",
        ),
    )

    @property
    def configured(self) -> bool:
        return bool(self.api_id and self.api_hash)


class Neo4jSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEO4J_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "polysignal-dev"
    database: str = "neo4j"


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    provider: Literal["disabled", "openai", "anthropic"] = Field(
        default="disabled",
        validation_alias=AliasChoices("POLYSIGNAL_LLM_PROVIDER", "LLM_PROVIDER"),
    )
    model: str = Field(
        default="",
        validation_alias=AliasChoices("POLYSIGNAL_LLM_MODEL", "LLM_MODEL"),
    )
    api_key: str = Field(
        default="",
        validation_alias=AliasChoices("POLYSIGNAL_LLM_API_KEY", "LLM_API_KEY"),
    )
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("POLYSIGNAL_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    anthropic_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("POLYSIGNAL_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    )
    temperature: float = Field(
        default=0.0,
        validation_alias=AliasChoices("POLYSIGNAL_LLM_TEMPERATURE", "LLM_TEMPERATURE"),
    )
    timeout_seconds: float = Field(
        default=30.0,
        validation_alias=AliasChoices(
            "POLYSIGNAL_LLM_TIMEOUT_SECONDS",
            "LLM_TIMEOUT_SECONDS",
        ),
    )
    max_tokens: int = Field(
        default=512,
        validation_alias=AliasChoices("POLYSIGNAL_LLM_MAX_TOKENS", "LLM_MAX_TOKENS"),
    )
    cache_dir: Path = Field(
        default=Path("./data/cache/llm"),
        validation_alias=AliasChoices("POLYSIGNAL_LLM_CACHE_DIR", "LLM_CACHE_DIR"),
    )

    @property
    def resolved_api_key(self) -> str:
        if self.provider == "openai":
            return self.openai_api_key or self.api_key
        if self.provider == "anthropic":
            return self.anthropic_api_key or self.api_key
        return ""

    @property
    def enabled(self) -> bool:
        return self.provider != "disabled" and bool(self.resolved_api_key)


class Settings(BaseSettings):
    """Top-level settings container, loaded from `.env` and the environment."""

    model_config = SettingsConfigDict(
        env_prefix="POLYSIGNAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["local", "ci", "prod"] = "local"
    log_level: str = "INFO"
    data_dir: Path = Path("./data")

    polymarket: PolymarketSettings = Field(default_factory=PolymarketSettings)
    reddit: RedditSettings = Field(default_factory=RedditSettings)
    twitter: TwitterSettings = Field(default_factory=TwitterSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton settings object.

    Cached for the lifetime of the process. Tests can clear the cache
    with ``get_settings.cache_clear()`` if they need a fresh instance.
    """

    return Settings()

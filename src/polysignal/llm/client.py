"""Optional provider adapters for cached structured LLM calls."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

from polysignal.llm.structured import parse_structured_response, pydantic_to_json_schema

T = TypeVar("T", bound=BaseModel)


class LLMClient(ABC):
    """Minimal interface every LLM provider wrapper must implement."""

    model_id: str

    @abstractmethod
    def complete_structured(self, prompt: str, *, schema: type[T]) -> T:
        """Call the model and return a validated ``schema`` instance."""


class NullLLMClient(LLMClient):
    """Default no-op client used when no LLM provider is configured."""

    model_id = "null"

    def complete_structured(self, prompt: str, *, schema: type[T]) -> T:
        raise RuntimeError(
            "LLM features are disabled. Configure POLYSIGNAL_LLM_PROVIDER and "
            "the matching provider API key in your .env file."
        )


class OpenAIClient(LLMClient):
    """OpenAI JSON-mode adapter. The SDK is imported only when enabled."""

    def __init__(
        self,
        *,
        api_key: str,
        model_id: str,
        temperature: float = 0.0,
        timeout_seconds: float = 30.0,
        max_tokens: int = 512,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise RuntimeError("OpenAI LLM provider requires POLYSIGNAL_OPENAI_API_KEY.")
        if not model_id:
            raise RuntimeError("OpenAI LLM provider requires POLYSIGNAL_LLM_MODEL.")
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    'OpenAI SDK is required. Install it with: pip install -e ".[llm]"'
                ) from exc
            client = OpenAI(api_key=api_key, timeout=timeout_seconds)
        self._client = client

    def complete_structured(self, prompt: str, *, schema: type[T]) -> T:
        schema_json = json.dumps(pydantic_to_json_schema(schema), ensure_ascii=True)
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return one JSON object only. It must satisfy this JSON schema: "
                        f"{schema_json}"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("OpenAI returned an empty structured response.")
        return parse_structured_response(content, schema)


class AnthropicClient(LLMClient):
    """Anthropic JSON-text adapter. The SDK is imported only when enabled."""

    def __init__(
        self,
        *,
        api_key: str,
        model_id: str,
        temperature: float = 0.0,
        timeout_seconds: float = 30.0,
        max_tokens: int = 512,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise RuntimeError("Anthropic LLM provider requires POLYSIGNAL_ANTHROPIC_API_KEY.")
        if not model_id:
            raise RuntimeError("Anthropic LLM provider requires POLYSIGNAL_LLM_MODEL.")
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise RuntimeError(
                    'Anthropic SDK is required. Install it with: pip install -e ".[llm]"'
                ) from exc
            client = Anthropic(api_key=api_key, timeout=timeout_seconds)
        self._client = client

    def complete_structured(self, prompt: str, *, schema: type[T]) -> T:
        schema_json = json.dumps(pydantic_to_json_schema(schema), ensure_ascii=True)
        response = self._client.messages.create(
            model=self.model_id,
            system=(
                f"Return one JSON object only. It must satisfy this JSON schema: {schema_json}"
            ),
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        if not content.strip():
            raise ValueError("Anthropic returned an empty structured response.")
        return parse_structured_response(content, schema)


def build_llm_client(
    *,
    provider: str,
    api_key: str,
    model_id: str,
    temperature: float = 0.0,
    timeout_seconds: float = 30.0,
    max_tokens: int = 512,
) -> LLMClient:
    """Build one explicitly enabled live adapter."""
    kwargs = {
        "api_key": api_key,
        "model_id": model_id,
        "temperature": temperature,
        "timeout_seconds": timeout_seconds,
        "max_tokens": max_tokens,
    }
    if provider == "openai":
        return OpenAIClient(**kwargs)
    if provider == "anthropic":
        return AnthropicClient(**kwargs)
    if provider == "disabled":
        raise RuntimeError(
            "LLM judge is disabled. Set POLYSIGNAL_LLM_PROVIDER=openai or anthropic."
        )
    raise RuntimeError(f"Unsupported LLM provider: {provider!r}.")


__all__ = [
    "AnthropicClient",
    "LLMClient",
    "NullLLMClient",
    "OpenAIClient",
    "build_llm_client",
]

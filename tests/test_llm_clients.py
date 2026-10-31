"""Offline tests for optional live LLM adapters."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from polysignal.llm.client import OpenAIClient, build_llm_client


class _Reply(BaseModel):
    answer: str


class _FakeCompletions:
    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"ok"}'))]
        )


def test_openai_adapter_validates_fake_structured_response() -> None:
    completions = _FakeCompletions()
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = OpenAIClient(api_key="fake", model_id="fake-model", client=sdk)
    assert client.complete_structured("prompt", schema=_Reply).answer == "ok"
    assert completions.kwargs["max_completion_tokens"] == 512
    assert completions.kwargs["response_format"] == {"type": "json_object"}


def test_live_llm_factory_rejects_disabled_provider() -> None:
    with pytest.raises(RuntimeError, match="disabled"):
        build_llm_client(provider="disabled", api_key="", model_id="")

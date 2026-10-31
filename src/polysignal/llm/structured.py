"""Helpers for working with structured-output LLM calls.

These helpers are provider-agnostic. When the time comes to plug in an
OpenAI/Anthropic client, the schema-to-JSON-schema conversion and the
parsing helpers below will be reused.
"""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def pydantic_to_json_schema(model_cls: type[BaseModel]) -> dict:
    """Return the JSON schema for a Pydantic model.

    This is a thin wrapper to give callers a single import to use.
    """
    return model_cls.model_json_schema()


def parse_structured_response(raw: str, schema: type[T]) -> T:
    """Parse ``raw`` (expected JSON) and validate against ``schema``.

    Raises ``ValueError`` with a clear message if the payload is not
    JSON or does not match the schema. This is intentionally strict:
    silently accepting partially valid outputs would erode the
    "structured" guarantee.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {exc}") from exc
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"LLM response does not match schema {schema.__name__}: {exc}") from exc


__all__ = ["parse_structured_response", "pydantic_to_json_schema"]

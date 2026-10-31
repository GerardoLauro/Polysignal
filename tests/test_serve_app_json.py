"""Regression tests for the dashboard JSON serialization (NaN -> valid JSON).

The endpoint ``GET /api/social-enriched/ps-data`` must return spec-compliant JSON.
Python's ``json.dumps`` emits bare ``NaN``/``Infinity`` tokens by default, which the
browser's ``JSON.parse`` rejects ("Response was not valid JSON"). These tests lock in
the two-layer fix: ``_s`` cleans missing text at the source and ``_json_safe`` sanitises
the whole payload before serialisation with ``allow_nan=False``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _strict_loads(text: str):
    """Parse like the browser: reject NaN/Infinity constants."""
    return json.loads(
        text,
        parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"bad token {c}")),
    )


class TestJsonSafe:
    def test_replaces_nan_and_inf_recursively(self):
        from serve_app import _json_safe

        payload = {
            "a": float("nan"),
            "b": [1.0, float("inf"), {"c": float("-inf")}],
            "d": "ok",
            "e": 3,
        }
        safe = _json_safe(payload)
        assert safe["a"] is None
        assert safe["b"][1] is None
        assert safe["b"][2]["c"] is None
        assert safe["d"] == "ok" and safe["e"] == 3

    def test_output_serialises_under_strict_json(self):
        from serve_app import _json_safe

        body = json.dumps(_json_safe({"x": float("nan")}), allow_nan=False)
        assert _strict_loads(body) == {"x": None}

    def test_raw_dumps_would_fail_strict_parse(self):
        # Guards the premise: default dumps emits an unparseable NaN token.
        bad = json.dumps({"x": float("nan")})  # allow_nan=True default
        with pytest.raises(ValueError):
            _strict_loads(bad)


class TestStringCoercion:
    def test_nan_and_blank_become_none(self):
        from build_ps_data import _s

        assert _s(float("nan")) is None
        assert _s(None) is None
        assert _s("   ") is None

    def test_real_text_is_stripped(self):
        from build_ps_data import _s

        assert _s("  en ") == "en"
        assert _s("english") == "english"

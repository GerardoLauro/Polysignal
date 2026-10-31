#!/usr/bin/env bash
# PolySignal — launch the local dashboard (UI + live data API) on macOS / Linux.
# Windows users: run run_app.cmd instead. Requires Python >= 3.11.
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3 || command -v python || true)"
  if [ -z "$PY" ]; then
    echo "[PolySignal] No Python found. Create a venv first: python3 -m venv .venv" >&2
    exit 1
  fi
  echo "[PolySignal] .venv not found; using system: $PY"
fi

echo "[PolySignal] Starting local app server..."
exec "$PY" scripts/serve_app.py "$@"

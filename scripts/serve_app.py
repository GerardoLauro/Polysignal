"""PolySignal — one-command local app server.

Serves the vendored frontend (``webapp/``) and the live data endpoint
``GET /api/social-enriched/ps-data`` from the real pipeline artifacts, on a single
origin (so the browser needs no CORS and no internet). Opens the browser on start.

    .venv/Scripts/python.exe scripts/serve_app.py
    # or simply:  run_app.cmd
"""

from __future__ import annotations

import json
import argparse
import math
import os
import socket
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:  # keep console output safe on Windows cp1252 terminals
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
WEBAPP = ROOT / "webapp"
sys.path.insert(0, str(ROOT / "scripts"))

from build_ps_data import build_ps_data, _validate  # noqa: E402

API_PATH = "/api/social-enriched/ps-data"
DEFAULT_PORT = 8000


def _json_safe(obj):
    """Recursively replace non-finite floats (NaN/Inf) with ``None``.

    Python's ``json.dumps`` emits bare ``NaN``/``Infinity`` tokens by default,
    which are valid for Python but invalid JSON — the browser's ``JSON.parse``
    rejects them ("Response was not valid JSON"). Sanitising once at startup keeps
    the served payload spec-compliant regardless of stray nulls in the artifacts.
    """
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".jsx": "text/javascript",
        ".json": "application/json",
        ".html": "text/html",
        ".css": "text/css",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEBAPP), **kwargs)

    def do_GET(self):  # noqa: N802
        if self.path.split("?")[0] == API_PATH:
            return self._serve_api()
        if self.path in ("", "/"):
            self.path = "/index.html"
        return super().do_GET()

    def _serve_api(self):
        if self.server.ps_data is None:
            body = json.dumps({"error": self.server.data_error}, ensure_ascii=False).encode("utf-8")
            self.send_response(503)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        try:
            body = json.dumps(
                self.server.ps_data, ensure_ascii=False, allow_nan=False
            ).encode("utf-8")
        except Exception as exc:  # pragma: no cover
            self.send_error(500, f"data build failed: {exc}")
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        if API_PATH in (self.path or ""):
            sys.stdout.write(f"  · served {self.path}\n")


def _pick_port(preferred: int) -> int:
    for port in [preferred, *range(preferred + 1, preferred + 10)]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


def load_dashboard_data():
    """Return real data or an explicit setup message; never fabricate results."""
    try:
        data = _json_safe(build_ps_data(verbose=False))
        _validate(data)
        return data, None
    except FileNotFoundError:
        return None, (
            "Dashboard data is not installed. The complete dataset is intentionally kept outside GitHub. "
            "Restore your private data archives with scripts/restore_bundle_data.py "
            "or set POLYSIGNAL_DATA_DIR, then restart the app. See docs/RUN_APP.md."
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true", help="Validate data and exit without starting a server")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    print("PolySignal - building dataset from pipeline artifacts...")
    ps_data, data_error = load_dashboard_data()
    if data_error:
        print(f"  setup required: {data_error}")
    else:
        n_posts = sum(len(m["posts"]) for m in ps_data["markets"])
        print(f"  ready: {len(ps_data['markets'])} markets, {n_posts} evidence cards, "
              f"{ps_data['system']['postsLoaded']:,} posts in corpus")
    if args.check:
        return 2 if data_error else 0

    port = _pick_port(args.port)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.ps_data = ps_data
    httpd.data_error = data_error
    url = f"http://127.0.0.1:{httpd.server_port}/"

    print("-" * 60)
    print(f"  PolySignal app  ->  {url}")
    print(f"  API endpoint    ->  {url.rstrip('/')}{API_PATH}")
    print("  Press Ctrl+C to stop.")
    print("-" * 60)

    if not args.no_browser and not os.environ.get("PS_NO_BROWSER"):
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping PolySignal…")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Allow ``python -m polysignal`` to invoke the CLI.

This is the canonical alternative to the installed ``polysignal``
console script. It delegates to the Typer application defined in
``polysignal.cli.main`` without re-importing it through the
``polysignal.cli`` package, which would otherwise trigger a runpy
RuntimeWarning when the same module is also addressed by
``python -m polysignal.cli.main``.
"""

from __future__ import annotations

from polysignal.cli.main import app


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover - thin entry point
    main()

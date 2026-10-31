"""Typer CLI for PolySignal.

This package intentionally does NOT re-export ``app`` at the package
level. Doing so would import ``polysignal.cli.main`` eagerly, which
causes ``python -m polysignal.cli.main`` to trigger a runpy
RuntimeWarning ("module found in sys.modules after import of package").

Use one of the supported entry points instead:

* ``polysignal ...`` (via the installed console script);
* ``python -m polysignal ...`` (via ``polysignal.__main__``);
* ``from polysignal.cli.main import app`` for programmatic access.
"""

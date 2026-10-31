"""Smoke tests: package and main module import without side effects."""

from __future__ import annotations

import importlib


def test_package_imports_and_exposes_version():
    polysignal = importlib.import_module("polysignal")
    assert isinstance(polysignal.__version__, str)
    assert polysignal.__version__.count(".") >= 1


def test_main_module_imports_without_error():
    # ``python -m polysignal`` resolves to ``polysignal.__main__``.
    # Importing the module must not invoke the CLI nor emit warnings.
    main_module = importlib.import_module("polysignal.__main__")
    assert callable(main_module.main)
    assert hasattr(main_module, "app")


def test_cli_subpackage_does_not_eager_import_main():
    # The runpy RuntimeWarning fix relies on ``polysignal.cli`` NOT
    # importing ``polysignal.cli.main`` at package load time. Verify
    # that ``app`` is not exposed at the subpackage level (the docstring
    # in ``polysignal.cli.__init__`` documents this intentional choice).
    import polysignal.cli as cli_pkg

    assert not hasattr(cli_pkg, "app"), (
        "polysignal.cli must not re-export `app` — see the package docstring"
    )

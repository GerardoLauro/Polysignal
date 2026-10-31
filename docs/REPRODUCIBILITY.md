# Reproducing the technical checks

Run commands from the repository root. The publication audit uses Python 3.12 on Windows. The CI workflow repeats the same checks on Windows and Ubuntu with Python 3.12. The declared minimum Python 3.11 has a syntax check; it is not a claim that every optional integration was tested on every supported version.

## Install the verified environment

Create a virtual environment with Python 3.12. On Windows:

```text
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.lock
.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.venv\Scripts\python.exe -m pip check
```

On Linux/macOS replace `.venv\Scripts\python.exe` with `.venv/bin/python`. The runtime lock contains core dependencies and the lightweight VADER extra; the development lock adds test and build tools, including lxml for the HTML parser fixtures. Locks record the environment used for this audit, rather than promising that newer unpinned releases behave identically. Heavy optional collectors, models, Spark and paid LLM integrations are not installed by these locks.

For an ordinary runtime installation use `requirements-runtime.lock`, followed by `pip install --no-deps --no-build-isolation -e .` after installing the build requirements from `pyproject.toml`. Alternatively, `pip install -e ".[nlp]"` resolves a current environment without the exact audit pins.

## Checks

```text
python scripts/check_repository.py
python -m ruff check --select F821,F822,F823,E9 src tests scripts
node scripts/check_frontend.cjs
python -m pytest -q
python -m build --no-isolation
python scripts/check_wheel.py
```

Use the virtual environment's Python for these commands. Node is needed only for the frontend syntax check; it does not add an npm dependency tree. `bash -n run_app.sh` and PowerShell's parser check the platform launchers without running collectors.

Tests block outgoing non-loopback socket connections by default. Live Polymarket and Reddit checks are opt-in; do not enable them merely to reproduce this audit. Dataset-dependent dashboard metrics and the Spark integration are skipped when their private artifacts or JVM/PySpark are unavailable. A passing offline suite does not validate live credentials, service availability, the complete dashboard dataset or scientific conclusions.

On Windows, clone to a reasonably short directory when generating nested raw data. New long raw filenames are shortened deterministically and mapped back to logical names in each run manifest. Existing deep legacy paths may still depend on the machine's path-length support.

The wheel installs the Python package and CLI. The dashboard, report, examples and scripts are distributed in the repository/source archive, so use a clone or source archive to run `run_app.cmd` or `run_app.sh`.

## Data and generated outputs

Only `data/examples/` and its documentation belong to Git. Full data, `.env`, sessions, tool downloads, raw runs, build products and test reports remain ignored. The original archive manifest in `BUNDLE_DATA.json` describes private companion archives; those archives are deliberately absent from GitHub. See [dashboard setup](RUN_APP.md).

`python scripts/make_release.py` creates a source-only ZIP by default. `--include-data` is an explicit private export option and should not be used for normal GitHub uploads. The original report PDFs, figures, bibliography and scientific code are preserved; publication checks do not rerun or validate the scientific experiments.

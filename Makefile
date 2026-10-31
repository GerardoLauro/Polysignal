# PolySignal — convenience commands.
# Works on Unix-like shells. On Windows PowerShell, run the underlying commands
# directly (they are all simple one-liners).

PY ?= python

.PHONY: help venv install install-dev install-llm install-social lint format test test-cov \
        infra-up infra-down dashboard cli clean

help:
	@echo "PolySignal — make targets"
	@echo "  install         Install runtime dependencies in editable mode."
	@echo "  install-dev     Install dev/test dependencies."
	@echo "  install-llm     Install optional LLM dependencies."
	@echo "  install-social  Install optional legacy Reddit API dependencies."
	@echo "  lint            Run ruff."
	@echo "  format          Format with ruff."
	@echo "  test            Run pytest."
	@echo "  test-cov        Run pytest with coverage."
	@echo "  infra-up        Start local Neo4j via docker compose."
	@echo "  infra-down      Stop local infrastructure."
	@echo "  dashboard       Launch the local React dashboard + data API (serve_app.py)."
	@echo "  cli             Show available CLI commands."

venv:
	$(PY) -m venv .venv

install:
	$(PY) -m pip install -e .

install-dev:
	$(PY) -m pip install -e ".[dev]"

install-llm:
	$(PY) -m pip install -e ".[llm]"

install-social:
	$(PY) -m pip install -e ".[social]"

lint:
	ruff check src tests

format:
	ruff format src tests
	ruff check --fix src tests

test:
	pytest

test-cov:
	pytest --cov=polysignal --cov-report=term-missing

infra-up:
	docker compose up -d neo4j

infra-down:
	docker compose down

dashboard:
	$(PY) scripts/serve_app.py

cli:
	polysignal --help

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache

.PHONY: install test build lint docs-dev docs-build clean

# ── Install ──────────────────────────────────────────────

install:
	cd sdks/python && pip install -e ".[dev]"
	cd sdks/typescript && npm install
	cd docs && npm install

# ── Test ─────────────────────────────────────────────────

test:
	cd sdks/python && pytest
	cd sdks/typescript && npm test

test-python:
	cd sdks/python && pytest

test-typescript:
	cd sdks/typescript && npm test

# ── Build ────────────────────────────────────────────────

build:
	cd sdks/python && python -m build
	cd sdks/typescript && npm run build

build-python:
	cd sdks/python && python -m build

build-typescript:
	cd sdks/typescript && npm run build

# ── Lint ─────────────────────────────────────────────────

lint:
	cd sdks/python && ruff check . && ruff format --check .
	cd sdks/typescript && npm run lint

lint-fix:
	cd sdks/python && ruff check --fix . && ruff format .
	cd sdks/typescript && npm run lint -- --fix

# ── Docs ─────────────────────────────────────────────────

docs-dev:
	cd docs && npm run dev

docs-build:
	cd docs && npm run build

# ── Clean ────────────────────────────────────────────────

clean:
	rm -rf sdks/python/dist sdks/python/build sdks/python/*.egg-info
	rm -rf sdks/python/.pytest_cache sdks/python/.mypy_cache sdks/python/.ruff_cache
	rm -rf sdks/python/__pycache__ sdks/python/**/__pycache__
	rm -rf sdks/python/htmlcov sdks/python/.coverage
	rm -rf sdks/typescript/dist sdks/typescript/node_modules/.cache
	rm -rf sdks/typescript/coverage
	rm -rf docs/dist docs/.astro

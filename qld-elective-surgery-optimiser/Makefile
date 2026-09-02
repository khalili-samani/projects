PYTHON ?= python
PACKAGE := qld_surgery_optimiser

.PHONY: help install install-all doctor show-config discover ingest ingest-latest validate warehouse format lint typecheck test coverage clean

help:
	@echo "Available commands:"
	@echo "  make install        Install core and development dependencies"
	@echo "  make install-all    Install all optional dependencies"
	@echo "  make doctor         Validate configuration"
	@echo "  make show-config    Print resolved configuration"
	@echo "  make discover       Discover source resources"
	@echo "  make ingest         Download historical resources"
	@echo "  make ingest-latest  Download latest resources"
	@echo "  make validate       Validate and quarantine source records"
	@echo "  make warehouse      Build canonical data and DuckDB warehouse"
	@echo "  make format         Format Python files"
	@echo "  make lint           Run Ruff"
	@echo "  make typecheck      Run mypy"
	@echo "  make test           Run tests"
	@echo "  make coverage       Run tests with coverage"
	@echo "  make clean          Remove development artefacts"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

install-all:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[api,app,dev]"

doctor:
	$(PYTHON) -m $(PACKAGE).cli doctor

show-config:
	$(PYTHON) -m $(PACKAGE).cli show-config

discover:
	$(PYTHON) -m $(PACKAGE).cli discover

ingest:
	$(PYTHON) -m $(PACKAGE).cli ingest

ingest-latest:
	$(PYTHON) -m $(PACKAGE).cli ingest --latest-only

validate:
	$(PYTHON) -m $(PACKAGE).cli validate

warehouse:
	$(PYTHON) -m $(PACKAGE).cli warehouse

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check . --fix

lint:
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy src tests

test:
	$(PYTHON) -m pytest

coverage:
	$(PYTHON) -m pytest --cov=$(PACKAGE) --cov-report=term-missing

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in [Path('.pytest_cache'), Path('.mypy_cache'), Path('.ruff_cache'), Path('htmlcov'), Path('build'), Path('dist')]]"
	$(PYTHON) -c "from pathlib import Path; [p.unlink(missing_ok=True) for p in Path('.').glob('.coverage*')]"
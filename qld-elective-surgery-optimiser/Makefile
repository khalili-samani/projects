PYTHON ?= python
PACKAGE := qld_surgery_optimiser

.PHONY: help install install-all doctor show-config format lint typecheck test coverage clean

help:
	@echo "Available commands:"
	@echo "  make install       Install core and development dependencies"
	@echo "  make install-all   Install API, application and development dependencies"
	@echo "  make doctor        Validate configuration and create required directories"
	@echo "  make show-config   Print resolved non-secret configuration"
	@echo "  make format        Format Python files with Ruff"
	@echo "  make lint          Run Ruff linting and formatting checks"
	@echo "  make typecheck     Run mypy"
	@echo "  make test          Run the test suite"
	@echo "  make coverage      Run tests with coverage"
	@echo "  make clean         Remove generated development artefacts"

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
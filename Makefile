PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
START ?= 2025-07
END ?= 2026-09

.PHONY: help setup test lint type check demo real walkforward example build docker clean

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup: ## create .venv and install the package with dev tools
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e '.[dev]'
	$(BIN)/pre-commit install 2>/dev/null || true

test: ## run the test suite with coverage
	$(BIN)/pytest -q

lint: ## ruff
	$(BIN)/ruff check .

type: ## mypy
	$(BIN)/mypy

check: lint type test ## everything CI runs, locally

demo: ## synthetic end-to-end run (no internet)
	$(BIN)/crypto-stat-arb --demo --output results/demo

walkforward: ## synthetic walk-forward run
	$(BIN)/crypto-stat-arb --demo --walk-forward --fold-bars 480 --output results/demo-walkforward

real: ## download Binance spot history and run the preregistered candidates
	$(BIN)/crypto-stat-arb --market spot --start $(START) --end $(END) --walk-forward \
		--output results/spot-$(START)_$(END)

example: ## refresh examples/demo_run from the current code
	$(BIN)/crypto-stat-arb --demo --output /tmp/csal-example --quiet
	cp /tmp/csal-example/RESEARCH_REPORT.md examples/demo_run/RESEARCH_REPORT_SYNTHETIC.md
	cp /tmp/csal-example/summary.json examples/demo_run/summary_SYNTHETIC.json
	cp /tmp/csal-example/backtest.png examples/demo_run/backtest_SYNTHETIC.png

build: ## build sdist and wheel into dist/
	$(BIN)/python -m build
	$(BIN)/twine check dist/*

docker: ## build the image and run the demo inside it
	docker build -t crypto-stat-arb-lab .
	docker run --rm -v "$(PWD)/results:/work/results" crypto-stat-arb-lab --demo --output results/docker-demo

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache .coverage coverage.xml
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: help install install-dev lint format test test-cov typecheck precommit pretrain finetune evaluate generate preprocess clean

PYTHON ?= python
SRC = src/riedfm
TESTS = tests

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ─── Setup ────────────────────────────────────────────────────────────
install: ## Install package
	pip install -e .

install-dev: ## Install with dev dependencies
	pip install -e ".[dev]"
	pre-commit install

# ─── Code Quality ─────────────────────────────────────────────────────
lint: ## Run ruff linter
	ruff check $(SRC) $(TESTS)

format: ## Run ruff formatter
	ruff format $(SRC) $(TESTS)
	ruff check --fix $(SRC) $(TESTS)

typecheck: ## Run mypy type checker
	mypy $(SRC)

precommit: ## Run all pre-commit hooks
	pre-commit run --all-files

# ─── Tests ────────────────────────────────────────────────────────────
test: ## Run tests
	pytest $(TESTS) -v

test-cov: ## Run tests with coverage
	pytest $(TESTS) -v --cov=$(SRC) --cov-report=term-missing --cov-report=html

# ─── Training ─────────────────────────────────────────────────────────
pretrain: ## Run pretraining
	$(PYTHON) -m riedfm.cli.pretrain $(ARGS)

finetune: ## Run fine-tuning
	$(PYTHON) -m riedfm.cli.finetune $(ARGS)

evaluate: ## Run evaluation
	$(PYTHON) -m riedfm.cli.evaluate $(ARGS)

generate: ## Run graph generation
	$(PYTHON) -m riedfm.cli.generate $(ARGS)

preprocess: ## Run data preprocessing
	$(PYTHON) -m riedfm.cli.preprocess $(ARGS)

# ─── Cleanup ──────────────────────────────────────────────────────────
clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage

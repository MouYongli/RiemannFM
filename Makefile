.PHONY: help install install-dev lint format test test-cov typecheck precommit download pretrain finetune evaluate generate preprocess clean

SRC = src/riemannfm
TESTS = tests

# ─── Environment detection ───────────────────────────────────────────
# Use uv if available, otherwise fall back to bare commands (conda / venv)
UV := $(shell command -v uv 2>/dev/null)
ifdef UV
  RUN := uv run
  SYNC := uv sync
  SYNC_DEV := uv sync --group dev
else
  RUN :=
  SYNC := pip install -e .
  SYNC_DEV := pip install -e ".[dev]" && pre-commit install
endif

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ─── Setup ────────────────────────────────────────────────────────────
install: ## Install package
	$(SYNC)

install-dev: ## Install with dev dependencies
	$(SYNC_DEV)
	$(RUN) pre-commit install

# ─── Code Quality ─────────────────────────────────────────────────────
lint: ## Run ruff linter
	$(RUN) ruff check $(SRC) $(TESTS)

format: ## Run ruff formatter
	$(RUN) ruff format $(SRC) $(TESTS)
	$(RUN) ruff check --fix $(SRC) $(TESTS)

typecheck: ## Run mypy type checker
	$(RUN) mypy $(SRC)

precommit: ## Run all pre-commit hooks
	$(RUN) pre-commit run --all-files

# ─── Tests ────────────────────────────────────────────────────────────
test: ## Run tests
	$(RUN) pytest $(TESTS) -v

test-cov: ## Run tests with coverage
	$(RUN) pytest $(TESTS) -v --cov=$(SRC) --cov-report=term-missing --cov-report=html

# ─── Data ─────────────────────────────────────────────────────────────
download: ## Download datasets and precompute text embeddings
	$(RUN) python -m riemannfm.cli.download $(ARGS)

# ─── Training ─────────────────────────────────────────────────────────
pretrain: ## Run pretraining
	$(RUN) python -m riemannfm.cli.pretrain $(ARGS)

finetune: ## Run fine-tuning
	$(RUN) python -m riemannfm.cli.finetune $(ARGS)

evaluate: ## Run evaluation
	$(RUN) python -m riemannfm.cli.evaluate $(ARGS)

generate: ## Run graph generation
	$(RUN) python -m riemannfm.cli.generate $(ARGS)

preprocess: ## Run data preprocessing
	$(RUN) python -m riemannfm.cli.preprocess $(ARGS)

# ─── Cleanup ──────────────────────────────────────────────────────────
clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage

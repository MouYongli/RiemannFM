.PHONY: help install install-dev lint format test test-cov typecheck precommit pretrain finetune evaluate generate preprocess clean

SRC = src/riemannfm
TESTS = tests

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ─── Setup ────────────────────────────────────────────────────────────
install: ## Install package
	uv sync

install-dev: ## Install with dev dependencies
	uv sync --group dev
	uv run pre-commit install

# ─── Code Quality ─────────────────────────────────────────────────────
lint: ## Run ruff linter
	uv run ruff check $(SRC) $(TESTS)

format: ## Run ruff formatter
	uv run ruff format $(SRC) $(TESTS)
	uv run ruff check --fix $(SRC) $(TESTS)

typecheck: ## Run mypy type checker
	uv run mypy $(SRC)

precommit: ## Run all pre-commit hooks
	uv run pre-commit run --all-files

# ─── Tests ────────────────────────────────────────────────────────────
test: ## Run tests
	uv run pytest $(TESTS) -v

test-cov: ## Run tests with coverage
	uv run pytest $(TESTS) -v --cov=$(SRC) --cov-report=term-missing --cov-report=html

# ─── Training ─────────────────────────────────────────────────────────
pretrain: ## Run pretraining
	uv run python -m riemannfm.cli.pretrain $(ARGS)

finetune: ## Run fine-tuning
	uv run python -m riemannfm.cli.finetune $(ARGS)

evaluate: ## Run evaluation
	uv run python -m riemannfm.cli.evaluate $(ARGS)

generate: ## Run graph generation
	uv run python -m riemannfm.cli.generate $(ARGS)

preprocess: ## Run data preprocessing
	uv run python -m riemannfm.cli.preprocess $(ARGS)

# ─── Cleanup ──────────────────────────────────────────────────────────
clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage

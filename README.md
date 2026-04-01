# RiemannFM

> Geometry- and Text-Aware Flow Matching on Product Manifolds with Joint Continuous-Discrete Dynamics for Knowledge Graph Generation

[![CI](https://github.com/MouYongli/RiemannFM/actions/workflows/ci.yml/badge.svg)](https://github.com/MouYongli/RiemannFM/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A deep graph generative foundation model that performs joint continuous-discrete flow matching on product Riemannian manifolds (H × S × E) for knowledge graph generation and downstream tasks.

## Key Features

- **Product Riemannian Manifolds** — Joint embedding on Hyperbolic × Spherical × Euclidean spaces with learnable curvatures
- **Joint Flow Matching** — Continuous flow for node coordinates (via Riemannian exp map) + discrete CTMC flow for edge types
- **RieFormer Backbone** — Riemannian Equivariant Dual-Stream Transformer with geodesic attention and manifold RoPE
- **Text-Conditioned Generation** — Multi-backend text encoder (SBERT / Qwen3-Embedding / XLM-RoBERTa) with cross-attention for text-guided KG generation
- **Multi-Task Support** — Knowledge graph completion, text-conditional subgraph generation, zero-shot anomaly detection

## Installation

Requires Python 3.12+. Supports [uv](https://docs.astral.sh/uv/) (recommended) or conda.

```bash
git clone https://github.com/MouYongli/RiemannFM.git
cd RiemannFM
```

### Option A — uv (recommended)

```bash
uv sync              # production deps
uv sync --group dev  # + dev tools (ruff, mypy, pytest)
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate riemannfm
pip install -e ".[dev]"   # editable install with dev tools
```

> Makefile and shell scripts auto-detect `uv`. With conda, activate the environment first — all `make` commands then work without changes.

## Data

### Download WikiData5M

```bash
uv run python -m riemannfm.cli.download data=wikidata_5m
```

Downloads raw triples and entity/relation texts to `data/wikidata_5m/raw/`.

### Build mini validation dataset

A small subset (~7.8K entities, 822 relations, ~22K triples) for fast engineering validation:

```bash
uv run python -m riemannfm.cli.preprocess data=wikidata_5m preprocess.build_mini=true
```

### Preprocess (ID mapping + text embeddings)

```bash
uv run python -m riemannfm.cli.preprocess data=wikidata_5m_mini embedding=sbert
```

See [docs/data.md](docs/data.md) for full documentation (all datasets, embedding options, `.env` configuration).

## Pretraining

All commands use [Hydra](https://hydra.cc/) for configuration. Configs live in `configs/`.

### Quick start (mini dataset)

```bash
# 1. Download + preprocess WikiData5M mini (~22K triples, fast iteration)
uv run python -m riemannfm.cli.download data=wikidata_5m
uv run python -m riemannfm.cli.preprocess data=wikidata_5m preprocess.build_mini=true
uv run python -m riemannfm.cli.preprocess data=wikidata_5m_mini embedding=sbert

# 2. Pretrain (small model, mini data)
uv run python -m riemannfm.cli.pretrain \
    model=rieformer_small \
    data=wikidata_5m_mini \
    training.max_steps=1000
```

Or via Makefile:

```bash
make pretrain ARGS="model=rieformer_small data=wikidata_5m_mini training.max_steps=1000"
```

### Full pretraining

```bash
uv run python -m riemannfm.cli.pretrain \
    model=rieformer_base \
    data=wikidata_5m \
    manifold=product_h_s_e \
    training=pretrain
```

### Key Hydra overrides

| Override | Example | Description |
|----------|---------|-------------|
| `model=` | `rieformer_small`, `rieformer_base`, `rieformer_large` | Model size |
| `data=` | `wikidata_5m_mini`, `wikidata_5m`, `fb15k237` | Dataset |
| `manifold=` | `product_h_s_e`, `h_only`, `e_only` | Manifold composition |
| `training.max_steps=` | `1000`, `500000` | Total training steps |
| `training.lr=` | `1e-4` | Learning rate |
| `training.mixed_precision=` | `bf16`, `fp16`, `null` | Mixed precision |
| `~logger.wandb` | | Disable W&B (keep CSV only) |
| `~logger.csv` | | Disable CSV (keep W&B only) |

### Logging

By default both W&B and CSV loggers are active. CSV logs are written to `outputs/<run_name>/csv_logs/` as a local fallback.

```bash
# W&B + CSV (default)
make pretrain

# Offline development (CSV only, no W&B)
make pretrain ARGS="~logger.wandb"

# W&B only
make pretrain ARGS="~logger.csv"
```

### Downstream tasks (not yet implemented)

```bash
# Fine-tuning on FB15k-237
uv run python -m riemannfm.cli.finetune data=fb15k237 training=finetune

# Evaluation
uv run python -m riemannfm.cli.evaluate data=fb15k237 eval=default

# Graph generation
uv run python -m riemannfm.cli.generate data=fb15k237
```

## Architecture

```text
configs/                        Hydra config groups (model, data, training, manifold, flow, etc.)
src/riemannfm/
├── manifolds/      Riemannian manifold implementations (H, S, E, Product)
├── flow/           Flow matching (continuous, discrete, joint, ODE solver)
├── models/         RieFormer backbone, RiemannFM top-level model, attention, normalization
├── tasks/          Downstream task heads (KGC link/relation prediction, T2G, GAD)
├── data/           KG benchmarks, datasets, transforms, collator
├── losses/         Flow matching loss, contrastive loss, combined multi-task loss
├── optim/          Riemannian optimizer and LR scheduler
├── utils/          Distributed training, checkpointing, logging, seed, metrics
└── cli/            CLI entry points (download, pretrain, finetune, evaluate, generate, preprocess)
```

## Development

```bash
make lint        # ruff linter
make format      # ruff formatter + auto-fix
make typecheck   # mypy
make test        # pytest
make test-cov    # pytest with coverage
make precommit   # all pre-commit hooks
```

See [CLAUDE.md](CLAUDE.md) for detailed conventions (git workflow, code style, experiment discipline).

## Citation

```bibtex
@software{riemannfm,
  title   = {RiemannFM: Geometry- and Text-Aware Flow Matching on Product Manifolds},
  author  = {Mou, Yongli},
  year    = {2025},
  url     = {https://github.com/MouYongli/RiemannFM}
}
```

## License

MIT

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

## Quick Start

All training commands use Hydra for configuration:

```bash
# Pretraining on WikiData5M
uv run python -m riemannfm.cli.pretrain data=wikidata_5m model=rieformer_base training=pretrain

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

# RieDFM-G

> Riemannian Discrete Flow Matching on Graphs

[![CI](https://github.com/MouYongli/RieDFM-G/actions/workflows/ci.yml/badge.svg)](https://github.com/MouYongli/RieDFM-G/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A deep graph generative foundation model that performs joint continuous-discrete flow matching on product Riemannian manifolds (H × S × E) for knowledge graph generation and downstream tasks.

## Key Features

- **Product Riemannian Manifolds** — Joint embedding on Hyperbolic × Spherical × Euclidean spaces with learnable curvatures
- **Joint Flow Matching** — Continuous flow for node coordinates (via Riemannian exp map) + discrete CTMC flow for edge types
- **RED-Former Backbone** — Riemannian Equivariant Dual-Stream Transformer with geodesic attention and manifold RoPE
- **Text-Conditioned Generation** — Frozen Sentence-BERT encoder with cross-attention for text-guided KG generation
- **Multi-Task Support** — Knowledge graph completion, text-conditional subgraph generation, zero-shot anomaly detection

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone
git clone https://github.com/MouYongli/RieDFM-G.git
cd RieDFM-G

# Install
uv sync              # production deps
uv sync --group dev  # + dev tools (ruff, mypy, pytest)
```

## Quick Start

All training commands use Hydra for configuration:

```bash
# Pretraining on WikiData5M
make pretrain ARGS="data=wikidata_5m model=red_former_base training=pretrain"

# Fine-tuning on FB15k-237
make finetune ARGS="data=fb15k237 training=finetune"

# Evaluation
make evaluate ARGS="data=fb15k237 eval=default"

# Graph generation
make generate ARGS="data=fb15k237"
```

## Architecture

```text
src/riedfm/
├── manifolds/      Riemannian manifold implementations (H, S, E, Product)
├── flow/           Flow matching (continuous, discrete, joint, ODE solver)
├── models/         RED-Former backbone, RieDFMG top-level model, downstream heads
├── layers/         Geodesic attention, manifold RoPE, ATH-Norm, vector field heads
├── data/           KG benchmarks (FB15k-237, WN18RR), WikiData, molecular datasets
├── losses/         Flow matching loss, contrastive loss, combined multi-task loss
├── utils/          Distributed training, checkpointing, Riemannian optimizers, metrics
├── cli/            CLI entry points (pretrain, finetune, evaluate, generate, preprocess)
└── configs/        Hydra YAML configs (model, data, training, manifold, flow, eval)
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
@software{riedfm_g,
  title   = {RieDFM-G: Riemannian Discrete Flow Matching on Graphs},
  author  = {Mou, Yongli},
  year    = {2025},
  url     = {https://github.com/MouYongli/RieDFM-G}
}
```

## License

MIT

#!/usr/bin/env bash
# RiemannFM pretraining launcher
set -euo pipefail

if command -v uv &>/dev/null; then
    uv run python -m riemannfm.cli.pretrain "$@"
else
    python -m riemannfm.cli.pretrain "$@"
fi

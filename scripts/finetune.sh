#!/usr/bin/env bash
# RiemannFM fine-tuning launcher
set -euo pipefail

if command -v uv &>/dev/null; then
    uv run python -m riemannfm.cli.finetune "$@"
else
    python -m riemannfm.cli.finetune "$@"
fi

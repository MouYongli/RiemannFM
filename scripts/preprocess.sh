#!/usr/bin/env bash
# RiemannFM data preprocessing launcher
set -euo pipefail

if command -v uv &>/dev/null; then
    uv run python -m riemannfm.cli.preprocess "$@"
else
    python -m riemannfm.cli.preprocess "$@"
fi

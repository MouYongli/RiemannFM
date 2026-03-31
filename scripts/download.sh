#!/usr/bin/env bash
# RiemannFM dataset download and preprocessing launcher
set -euo pipefail

if command -v uv &>/dev/null; then
    uv run python -m riemannfm.cli.download "$@"
else
    python -m riemannfm.cli.download "$@"
fi

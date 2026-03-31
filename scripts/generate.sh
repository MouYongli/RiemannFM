#!/usr/bin/env bash
# RiemannFM graph generation launcher
set -euo pipefail

if command -v uv &>/dev/null; then
    uv run python -m riemannfm.cli.generate "$@"
else
    python -m riemannfm.cli.generate "$@"
fi

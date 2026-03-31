#!/usr/bin/env bash
# RiemannFM evaluation launcher
set -euo pipefail

if command -v uv &>/dev/null; then
    uv run python -m riemannfm.cli.evaluate "$@"
else
    python -m riemannfm.cli.evaluate "$@"
fi

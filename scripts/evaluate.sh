#!/usr/bin/env bash
# RieDFM-G evaluation launcher
set -euo pipefail

python -m riedfm.cli.evaluate "$@"

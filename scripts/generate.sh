#!/usr/bin/env bash
# RieDFM-G graph generation launcher
set -euo pipefail

python -m riedfm.cli.generate "$@"

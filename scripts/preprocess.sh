#!/usr/bin/env bash
# RieDFM-G data preprocessing launcher
set -euo pipefail

python -m riedfm.cli.preprocess "$@"

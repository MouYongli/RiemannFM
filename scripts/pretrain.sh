#!/usr/bin/env bash
# RieDFM-G pretraining launcher
set -euo pipefail

python -m riedfm.cli.pretrain "$@"

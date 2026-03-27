#!/usr/bin/env bash
# RieDFM-G fine-tuning launcher
set -euo pipefail

python -m riedfm.cli.finetune "$@"

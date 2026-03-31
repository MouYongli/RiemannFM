#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
activate_env
ensure_dirs
log_header "Fine-tune" "$*"

run_distributed riemannfm.cli.finetune "$@"

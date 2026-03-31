#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
activate_env
ensure_dirs
log_header "Download & Preprocess" "$*"

run_module riemannfm.cli.download "$@"

#!/usr/bin/env bash
# Common functions for all RiemannFM scripts.
# Source from scripts/: source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────
_resolve_root() {
    local dir="$1"
    while [ "$dir" != "/" ]; do
        [ -f "${dir}/pyproject.toml" ] && echo "$dir" && return
        dir="$(dirname "$dir")"
    done
    echo "ERROR: Cannot find project root" >&2; exit 1
}

_CALLER_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
export PROJECT_ROOT="$(_resolve_root "$_CALLER_DIR")"
export DATA_DIR="${PROJECT_ROOT}/data"

# ── Logging ────────────────────────────────────────────────────────────
log_info()  { echo "[$(date '+%H:%M:%S')] [INFO]  $*"; }
log_warn()  { echo "[$(date '+%H:%M:%S')] [WARN]  $*" >&2; }
log_error() { echo "[$(date '+%H:%M:%S')] [ERROR] $*" >&2; }

log_header() {
    local stage="$1"; shift
    echo "=============================================="
    echo "  RiemannFM — ${stage}"
    echo "  Host: $(hostname)"
    echo "  Time: $(date -Iseconds)"
    [ $# -gt 0 ] && echo "  Args: $*"
    echo "=============================================="
}

# ── Environment ────────────────────────────────────────────────────────
# Detect and activate the Python environment: uv > conda > venv
activate_env() {
    if command -v uv &>/dev/null; then
        # uv manages the venv transparently via `uv run`
        RUN="uv run"
        log_info "Using uv"
    elif [ -n "${CONDA_PREFIX:-}" ]; then
        RUN=""
        log_info "Using conda env: ${CONDA_PREFIX}"
    elif [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "${PROJECT_ROOT}/.venv/bin/activate"
        RUN=""
        log_info "Using venv: ${PROJECT_ROOT}/.venv"
    else
        log_error "No Python environment found."
        log_error "Install with: uv sync  OR  conda env create -f environment.yml"
        exit 1
    fi
    export RUN
    export TOKENIZERS_PARALLELISM=false
}

# Helper: run a python module through the detected environment
run_module() {
    local module="$1"; shift
    ${RUN} python -m "$module" "$@"
}

# ── GPU ────────────────────────────────────────────────────────────────
# Relies on $RUN being set by activate_env() — otherwise a bare `python`
# call would probe the system interpreter (usually no torch) and silently
# report 0 GPUs, forcing single-GPU fall-through in run_distributed.
detect_gpus() {
    ${RUN:-} python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0
}

# ── Local distributed ─────────────────────────────────────────────────
run_distributed() {
    local module="$1"; shift
    local num_gpus
    num_gpus=$(detect_gpus)
    log_info "Detected ${num_gpus} GPU(s)"
    if [ "$num_gpus" -gt 1 ]; then
        ${RUN} torchrun --nproc_per_node="$num_gpus" \
                        --master_port="${MASTER_PORT:-29500}" \
                        -m "$module" "$@"
    else
        run_module "$module" "$@"
    fi
}

# ── SLURM ──────────────────────────────────────────────────────────────
setup_slurm_distributed() {
    export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
    export MASTER_PORT="${MASTER_PORT:-29500}"
    export WORLD_SIZE=$((SLURM_NNODES * SLURM_GPUS_ON_NODE))
    log_info "SLURM distributed: nodes=${SLURM_NNODES}, gpus/node=${SLURM_GPUS_ON_NODE}, world=${WORLD_SIZE}"
    log_info "Master: ${MASTER_ADDR}:${MASTER_PORT}"
}

# ── Directories ────────────────────────────────────────────────────────
ensure_dirs() {
    mkdir -p "${DATA_DIR}" "${PROJECT_ROOT}/outputs" "${PROJECT_ROOT}/logs"
}

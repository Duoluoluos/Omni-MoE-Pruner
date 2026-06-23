#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner}"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

exec "${PROJECT_ROOT}/.venv/bin/python" \
  "${PROJECT_ROOT}/profile_omnivideobench_qwen3/run_keepalive.py" "$@"

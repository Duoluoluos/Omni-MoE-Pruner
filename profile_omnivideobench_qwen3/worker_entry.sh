#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/bn/isp-traindata-lf-1/wq/OmniZip}"
cd "${PROJECT_ROOT}"

export HF_HOME="${PROJECT_ROOT}/eval_data/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/qwen-omni-utils/src:${PROJECT_ROOT}/lmms-eval:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

unset OMNIZIP_RHO_AUDIO OMNIZIP_RHO_VIDEO OMNIZIP_G OMNIZIP_CONTEXTUAL_RATIO
export WRAPPER=None

mkdir -p "${HF_HOME}"

exec "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/profile_omnivideobench_qwen3/run_omnivideobench_profile.py" "$@"

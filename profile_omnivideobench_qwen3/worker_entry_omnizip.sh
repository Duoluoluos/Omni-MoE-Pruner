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
export WRAPPER="${WRAPPER:-OmniZip}"

RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${PROJECT_ROOT}/logs/omnivideobench_qwen3_omnizip/$(date +%Y%m%d_%H%M%S)}"
KEEPALIVE_LOG="${KEEPALIVE_LOG:-${RUN_OUTPUT_DIR}/keepalive.log}"
START_KEEPALIVE="${START_KEEPALIVE:-1}"
mkdir -p "${HF_HOME}" "${RUN_OUTPUT_DIR}"

if [[ "${START_KEEPALIVE}" == "1" ]]; then
    if "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/scripts/run_keepalive.py" \
        --project-root "${PROJECT_ROOT}" \
        --target-util "${TARGET_UTIL:-0.3}" \
        --mem-gb "${MEM_GB:-0.6}" \
        --device "${DEVICE:-0}" \
        --log "${KEEPALIVE_LOG}" \
        > "${RUN_OUTPUT_DIR}/keepalive.pid" \
        2> "${RUN_OUTPUT_DIR}/keepalive_start.err"; then
        echo "[setup] keepalive_pid=$(tr -d '\n' < "${RUN_OUTPUT_DIR}/keepalive.pid") log=${KEEPALIVE_LOG}" >&2
    else
        echo "[warn] keepalive failed to start; see ${RUN_OUTPUT_DIR}/keepalive_start.err" >&2
    fi
fi

exec "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/profile_omnivideobench_qwen3/run_omnivideobench_profile.py" "$@"

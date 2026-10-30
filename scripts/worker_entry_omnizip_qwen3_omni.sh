#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/bn/isp-traindata-lf-1/wq/OmniZip}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${PROJECT_ROOT}/logs/omnizip_qwen3_omni_videomme_$(date +%Y%m%d_%H%M%S)}"
EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH:-${RUN_OUTPUT_DIR}/lmms_eval}"
KEEPALIVE_LOG="${KEEPALIVE_LOG:-${RUN_OUTPUT_DIR}/keepalive.log}"
START_KEEPALIVE="${START_KEEPALIVE:-1}"

cd "${PROJECT_ROOT}"
mkdir -p "${RUN_OUTPUT_DIR}" "${EVAL_OUTPUT_PATH}"

export PROJECT_ROOT
export EVAL_OUTPUT_PATH
export PRETRAINED="${PRETRAINED:-${PROJECT_ROOT}/models/Qwen3-Omni-30B-A3B-Instruct}"
export WRAPPER="${WRAPPER:-OmniZip}"
export OMNIZIP_RHO_AUDIO="${OMNIZIP_RHO_AUDIO:-0.3}"
export OMNIZIP_RHO_VIDEO="${OMNIZIP_RHO_VIDEO:-0.6}"
export OMNIZIP_G="${OMNIZIP_G:-3}"
export OMNIZIP_CONTEXTUAL_RATIO="${OMNIZIP_CONTEXTUAL_RATIO:-0.05}"
export MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-768}"
export TASKS="${TASKS:-videomme}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12347}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export KEEPALIVE_LOG

{
    echo "[setup] project_root=${PROJECT_ROOT}"
    echo "[setup] run_output_dir=${RUN_OUTPUT_DIR}"
    echo "[setup] eval_output_path=${EVAL_OUTPUT_PATH}"
    echo "[setup] wrapper=${WRAPPER}"
    echo "[setup] tasks=${TASKS}"
    echo "[setup] pretrained=${PRETRAINED}"
    echo "[setup] max_num_frames=${MAX_NUM_FRAMES}"
    echo "[setup] omnizip rho_audio=${OMNIZIP_RHO_AUDIO} rho_video=${OMNIZIP_RHO_VIDEO} g=${OMNIZIP_G} contextual_ratio=${OMNIZIP_CONTEXTUAL_RATIO}"
    echo "[setup] cuda_visible_devices=${CUDA_VISIBLE_DEVICES} main_process_port=${MAIN_PROCESS_PORT}"
} | tee "${RUN_OUTPUT_DIR}/worker_entry_setup.log"

if [[ "${START_KEEPALIVE}" == "1" ]]; then
    if "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/scripts/run_keepalive.py" \
        --project-root "${PROJECT_ROOT}" \
        --target-util "${TARGET_UTIL:-0.3}" \
        --mem-gb "${MEM_GB:-0.6}" \
        --device "${DEVICE:-0}" \
        --log "${KEEPALIVE_LOG}" \
        > "${RUN_OUTPUT_DIR}/keepalive.pid" \
        2> "${RUN_OUTPUT_DIR}/keepalive_start.err"; then
        echo "[setup] keepalive_pid=$(tr -d '\n' < "${RUN_OUTPUT_DIR}/keepalive.pid") log=${KEEPALIVE_LOG}"
    else
        echo "[warn] keepalive failed to start; see ${RUN_OUTPUT_DIR}/keepalive_start.err" >&2
    fi
fi

exec bash "${PROJECT_ROOT}/eval.sh"

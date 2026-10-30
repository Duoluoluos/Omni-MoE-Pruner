#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/bn/isp-traindata-lf-1/wq/OmniZip}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${PROJECT_ROOT}/logs/omnizip_qwen3_omni_videomme_$(date +%Y%m%d_%H%M%S)}"
EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH:-${RUN_OUTPUT_DIR}/lmms_eval}"
LOG="${LOG:-${RUN_OUTPUT_DIR}/worker.log}"

mkdir -p "${RUN_OUTPUT_DIR}" "${EVAL_OUTPUT_PATH}"

export PROJECT_ROOT RUN_OUTPUT_DIR EVAL_OUTPUT_PATH
export LOG
export START_KEEPALIVE="${START_KEEPALIVE:-1}"
export KEEPALIVE_LOG="${KEEPALIVE_LOG:-${RUN_OUTPUT_DIR}/keepalive.log}"

{
    echo "created_at=$(date --iso-8601=seconds)"
    echo "project_root=${PROJECT_ROOT}"
    echo "run_output_dir=${RUN_OUTPUT_DIR}"
    echo "eval_output_path=${EVAL_OUTPUT_PATH}"
    echo "log=${LOG}"
    echo "start_keepalive=${START_KEEPALIVE}"
    echo "wrapper=${WRAPPER:-OmniZip}"
    echo "tasks=${TASKS:-videomme}"
    echo "pretrained=${PRETRAINED:-${PROJECT_ROOT}/models/Qwen3-Omni-30B-A3B-Instruct}"
    echo "omnizip_rho_audio=${OMNIZIP_RHO_AUDIO:-0.3}"
    echo "omnizip_rho_video=${OMNIZIP_RHO_VIDEO:-0.6}"
    echo "omnizip_g=${OMNIZIP_G:-3}"
    echo "omnizip_contextual_ratio=${OMNIZIP_CONTEXTUAL_RATIO:-0.05}"
    echo "max_num_frames=${MAX_NUM_FRAMES:-768}"
} > "${RUN_OUTPUT_DIR}/launch_env.txt"

setsid nohup bash "${PROJECT_ROOT}/scripts/worker_entry_omnizip_qwen3_omni.sh" \
    > "${LOG}" 2>&1 < /dev/null &
PID=$!
echo "${PID}" > "${RUN_OUTPUT_DIR}/launcher.pid"
disown "${PID}" 2>/dev/null || true

echo "Started OmniZip Qwen3-Omni eval"
echo "launcher_pid=${PID}"
echo "run_output_dir=${RUN_OUTPUT_DIR}"
echo "eval_output_path=${EVAL_OUTPUT_PATH}"
echo "log=${LOG}"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/bn/isp-traindata-lf-1/wq/OmniZip}"
WORKER_ID="${WORKER_ID:-3907005}"
TS="${TS:-omnizip_qwen3_omni_videomme_$(date +%Y%m%d_%H%M%S)}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${PROJECT_ROOT}/logs/${TS}}"
EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH:-${RUN_OUTPUT_DIR}/lmms_eval}"
LOG="${LOG:-${RUN_OUTPUT_DIR}/worker_${WORKER_ID}.log}"
BACKGROUND="${BACKGROUND:-1}"
START_KEEPALIVE="${START_KEEPALIVE:-1}"

mkdir -p "${RUN_OUTPUT_DIR}" "${EVAL_OUTPUT_PATH}"

{
    echo "created_at=$(date --iso-8601=seconds)"
    echo "worker_id=${WORKER_ID}"
    echo "project_root=${PROJECT_ROOT}"
    echo "run_output_dir=${RUN_OUTPUT_DIR}"
    echo "eval_output_path=${EVAL_OUTPUT_PATH}"
    echo "log=${LOG}"
    echo "background=${BACKGROUND}"
    echo "start_keepalive=${START_KEEPALIVE}"
    echo "wrapper=${WRAPPER:-OmniZip}"
    echo "tasks=${TASKS:-videomme}"
    echo "pretrained=${PRETRAINED:-${PROJECT_ROOT}/models/Qwen3-Omni-30B-A3B-Instruct}"
    echo "omnizip_rho_audio=${OMNIZIP_RHO_AUDIO:-0.3}"
    echo "omnizip_rho_video=${OMNIZIP_RHO_VIDEO:-0.6}"
    echo "omnizip_g=${OMNIZIP_G:-3}"
    echo "omnizip_contextual_ratio=${OMNIZIP_CONTEXTUAL_RATIO:-0.05}"
    echo "max_num_frames=${MAX_NUM_FRAMES:-768}"
} > "${RUN_OUTPUT_DIR}/launch_command.txt"

if [[ "${BACKGROUND}" == "1" ]]; then
    setsid env NO_COLOR=1 TERM=dumb mlx worker login "${WORKER_ID}" -- env \
        PROJECT_ROOT="${PROJECT_ROOT}" \
        RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR}" \
        EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH}" \
        START_KEEPALIVE="${START_KEEPALIVE}" \
        KEEPALIVE_LOG="${KEEPALIVE_LOG:-${RUN_OUTPUT_DIR}/keepalive.log}" \
        TARGET_UTIL="${TARGET_UTIL:-0.3}" \
        MEM_GB="${MEM_GB:-0.6}" \
        DEVICE="${DEVICE:-0}" \
        WRAPPER="${WRAPPER:-OmniZip}" \
        OMNIZIP_RHO_AUDIO="${OMNIZIP_RHO_AUDIO:-0.3}" \
        OMNIZIP_RHO_VIDEO="${OMNIZIP_RHO_VIDEO:-0.6}" \
        OMNIZIP_G="${OMNIZIP_G:-3}" \
        OMNIZIP_CONTEXTUAL_RATIO="${OMNIZIP_CONTEXTUAL_RATIO:-0.05}" \
        MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-768}" \
        TASKS="${TASKS:-videomme}" \
        PRETRAINED="${PRETRAINED:-${PROJECT_ROOT}/models/Qwen3-Omni-30B-A3B-Instruct}" \
        MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12347}" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        EVAL_LIMIT="${EVAL_LIMIT:-}" \
        bash "${PROJECT_ROOT}/scripts/worker_entry_omnizip_qwen3_omni.sh" \
        > "${LOG}" 2>&1 < /dev/null &
    launcher_pid=$!
    echo "${launcher_pid}" > "${RUN_OUTPUT_DIR}/launcher.pid"
    disown "${launcher_pid}" 2>/dev/null || true
    echo "Started OmniZip Qwen3-Omni eval on worker ${WORKER_ID}"
    echo "launcher_pid=${launcher_pid}"
    echo "run_output_dir=${RUN_OUTPUT_DIR}"
    echo "eval_output_path=${EVAL_OUTPUT_PATH}"
    echo "log=${LOG}"
else
    NO_COLOR=1 TERM=dumb mlx worker login "${WORKER_ID}" -- env \
        PROJECT_ROOT="${PROJECT_ROOT}" \
        RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR}" \
        EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH}" \
        START_KEEPALIVE="${START_KEEPALIVE}" \
        KEEPALIVE_LOG="${KEEPALIVE_LOG:-${RUN_OUTPUT_DIR}/keepalive.log}" \
        TARGET_UTIL="${TARGET_UTIL:-0.3}" \
        MEM_GB="${MEM_GB:-0.6}" \
        DEVICE="${DEVICE:-0}" \
        WRAPPER="${WRAPPER:-OmniZip}" \
        OMNIZIP_RHO_AUDIO="${OMNIZIP_RHO_AUDIO:-0.3}" \
        OMNIZIP_RHO_VIDEO="${OMNIZIP_RHO_VIDEO:-0.6}" \
        OMNIZIP_G="${OMNIZIP_G:-3}" \
        OMNIZIP_CONTEXTUAL_RATIO="${OMNIZIP_CONTEXTUAL_RATIO:-0.05}" \
        MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-768}" \
        TASKS="${TASKS:-videomme}" \
        PRETRAINED="${PRETRAINED:-${PROJECT_ROOT}/models/Qwen3-Omni-30B-A3B-Instruct}" \
        MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12347}" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        EVAL_LIMIT="${EVAL_LIMIT:-}" \
        bash "${PROJECT_ROOT}/scripts/worker_entry_omnizip_qwen3_omni.sh"
fi

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${PROJECT_ROOT}/logs/videomme_qwen3_budget_dynamic/${TS}}"
BUDGET_CONFIG="${BUDGET_CONFIG:-${PROJECT_ROOT}/moe_expert_budget/configs/confidence_dynamic_default.json}"
EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH:-${RUN_OUTPUT_DIR}/lmms_eval}"

mkdir -p "${RUN_OUTPUT_DIR}" "${EVAL_OUTPUT_PATH}"

export PROJECT_ROOT
export TASKS="${TASKS:-videomme}"
export WRAPPER="${WRAPPER:-None}"
export PRETRAINED="${PRETRAINED:-${PROJECT_ROOT}/models/Qwen3-Omni-30B-A3B-Instruct}"
export MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-768}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12357}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EVAL_OUTPUT_PATH
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export MOE_EXPERT_BUDGET_CONFIG="${BUDGET_CONFIG}"
export MOE_EXPERT_BUDGET_RUN_DIR="${RUN_OUTPUT_DIR}"
export MOE_EXPERT_BUDGET_LOG="${RUN_OUTPUT_DIR}/expert_budget_samples.jsonl"

{
    echo "[setup] project_root=${PROJECT_ROOT}"
    echo "[setup] run_output_dir=${RUN_OUTPUT_DIR}"
    echo "[setup] eval_output_path=${EVAL_OUTPUT_PATH}"
    echo "[setup] budget_config=${BUDGET_CONFIG}"
    echo "[setup] wrapper=${WRAPPER}"
    echo "[setup] tasks=${TASKS}"
    echo "[setup] pretrained=${PRETRAINED}"
    echo "[setup] max_num_frames=${MAX_NUM_FRAMES}"
    echo "[setup] eval_limit=${EVAL_LIMIT:-}"
    echo "[setup] cuda_visible_devices=${CUDA_VISIBLE_DEVICES} main_process_port=${MAIN_PROCESS_PORT}"
} > "${RUN_OUTPUT_DIR}/worker_entry_setup.log"

exec bash "${PROJECT_ROOT}/eval.sh"

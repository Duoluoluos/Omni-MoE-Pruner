#!/usr/bin/env bash
# Download Qwen3-Omni-30B-A3B-Instruct into the existing local-dir.
# Resumable; safe to re-run. Uses HF token from ~/.cache/huggingface/token (or HF_TOKEN env).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models/Qwen3-Omni-30B-A3B-Instruct}"
HF_BIN="${HF_BIN:-${PROJECT_ROOT}/.venv/bin/hf}"
REPO_ID="${REPO_ID:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/model_download}"
mkdir -p "${MODEL_DIR}" "${LOG_DIR}"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
# Use a cache dir on the same big mount so blobs are local before symlinking.
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/eval_data/huggingface}"

LOG_FILE="${LOG_DIR}/download_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${LOG_DIR}/download.pid"

cmd=(
  "${HF_BIN}" download "${REPO_ID}"
  --local-dir "${MODEL_DIR}"
  --max-workers 8
)

echo "Starting download:"
echo "  cmd: ${cmd[*]}"
echo "  log: ${LOG_FILE}"
echo "  pid: ${PID_FILE}"

setsid nohup "${cmd[@]}" > "${LOG_FILE}" 2>&1 < /dev/null &
pid=$!
disown
echo "${pid}" > "${PID_FILE}"
echo "started pid=${pid}"
echo "tail -f ${LOG_FILE}"

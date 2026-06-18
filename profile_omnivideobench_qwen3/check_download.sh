#!/usr/bin/env bash
# Tail the Qwen3-Omni download progress running on worker 3911988.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner}"
MODEL_DIR="${PROJECT_ROOT}/models/Qwen3-Omni-30B-A3B-Instruct"
LOG="${PROJECT_ROOT}/logs/model_download/worker_download.log"
WORKER_ADDR="${WORKER_ADDR:-fdbd:dc03:9:654::66}"
WORKER_PORT="${WORKER_PORT:-10678}"

echo "=== model dir size ==="
du -sh "${MODEL_DIR}" "${MODEL_DIR}/.cache" 2>&1 | head -5
echo ""
echo "=== shard files ==="
ls -la "${MODEL_DIR}"/model-*.safetensors 2>/dev/null | awk '{printf "%10s  %s\n", $5, $9}' | tail -20
echo ""
echo "=== incomplete files (in-flight) ==="
ls -la "${MODEL_DIR}/.cache/huggingface/download/"*.incomplete 2>/dev/null | awk '{printf "%10s  %s\n", $5, $9}' | sed "s|.*/\(.\{12\}\)[^/]*\.incomplete|\1...|" | head -16
echo ""
echo "=== remote process ==="
ssh -p "${WORKER_PORT}" -o StrictHostKeyChecking=no -o BatchMode=yes "${WORKER_ADDR}" \
  'pgrep -af "hf download" | head -3; echo; pgrep -af "_remote_download_qwen3_omni" | head -3' 2>/dev/null | tail -10
echo ""
echo "=== last 10 log lines ==="
tail -n 10 "${LOG}" 2>&1

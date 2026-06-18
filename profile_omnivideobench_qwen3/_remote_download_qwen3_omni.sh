#!/usr/bin/env bash
# Run on the GPU worker — download Qwen3-Omni-30B-A3B-Instruct.
# Resumable + auto-retry loop. HF hangs are mitigated via HF_HUB_DOWNLOAD_TIMEOUT
# and an outer retry loop that re-invokes the resumable downloader.
set -uo pipefail
cd /mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner
export HF_TOKEN="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"
export HF_HOME="/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/eval_data/huggingface"
# huggingface.co stalls at 0 B/s from this worker; use the hf-mirror endpoint.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# Disable any inherited network proxies (they cause the 0 B/s stalls).
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset no_proxy NO_PROXY
export http_proxy="" https_proxy="" all_proxy="" HTTP_PROXY="" HTTPS_PROXY="" ALL_PROXY=""
# Fail fast on stalled connections so the retry loop can resume.
export HF_HUB_DOWNLOAD_TIMEOUT="30"
export HF_HUB_ENABLE_HF_TRANSFER="0"

MODEL_DIR=/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/models/Qwen3-Omni-30B-A3B-Instruct
LOG=/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/logs/model_download/worker_download.log
HF=/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/.venv/bin/hf
mkdir -p "$(dirname "$LOG")"

echo "[$(date)] === download session start ===" >> "$LOG"

max_attempts=50
attempt=1
while [ "$attempt" -le "$max_attempts" ]; do
  echo "[$(date)] attempt ${attempt}/${max_attempts}" >> "$LOG"
  "$HF" download \
    Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --local-dir "$MODEL_DIR" \
    --max-workers 8 \
    >> "$LOG" 2>&1
  rc=$?
  echo "[$(date)] hf download exit=${rc}" >> "$LOG"
  if [ "$rc" -eq 0 ]; then
    echo "[$(date)] === download complete ===" >> "$LOG"
    break
  fi
  echo "[$(date)] retrying in 5s..." >> "$LOG"
  sleep 5
  attempt=$((attempt + 1))
done

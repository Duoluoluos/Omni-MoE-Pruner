#!/usr/bin/env bash
# Watchdog wrapper run on the GPU worker. Keeps the hf download alive: if the
# model dir size stalls for > STALL_LIMIT seconds, kill the downloader so the
# inner retry loop resumes from the last byte.
set -uo pipefail

MODEL_DIR=/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/models/Qwen3-Omni-30B-A3B-Instruct
RUNNER=/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/profile_omnivideobench_qwen3/_remote_download_qwen3_omni.sh
LOG=/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/logs/model_download/worker_download.log
WLOG=/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/logs/model_download/watchdog.log
INDEX_FILE="${MODEL_DIR}/model.safetensors.index.json"
STALL_LIMIT="${STALL_LIMIT:-180}"   # seconds of no growth before kill+resume
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"
mkdir -p "$(dirname "$WLOG")"

echo "[$(date)] watchdog start (stall_limit=${STALL_LIMIT}s)" >> "$WLOG"

start_runner() {
  nohup bash "$RUNNER" >/dev/null 2>&1 &
  echo $!
}

runner_pid=$(start_runner)
echo "[$(date)] started runner pid=${runner_pid}" >> "$WLOG"

last_size=0
stall=0
while true; do
  # Done when all 15 shards + index exist and no incomplete files remain.
  shards=$(ls "${MODEL_DIR}"/model-*-of-00015.safetensors 2>/dev/null | wc -l)
  incompletes=$(ls "${MODEL_DIR}/.cache/huggingface/download/"*.incomplete 2>/dev/null | wc -l)
  if [ -f "$INDEX_FILE" ] && [ "$shards" -ge 15 ] && [ "$incompletes" -eq 0 ]; then
    echo "[$(date)] all 15 shards + index present, incompletes=0 -> done" >> "$WLOG"
    break
  fi

  size=$(du -sb "$MODEL_DIR" 2>/dev/null | awk '{print $1}')
  size=${size:-0}
  if [ "$size" -le "$last_size" ]; then
    stall=$((stall + CHECK_INTERVAL))
  else
    stall=0
    last_size=$size
  fi
  echo "[$(date)] size=${size} stall=${stall}s shards=${shards} incompletes=${incompletes} runner_alive=$(kill -0 "$runner_pid" 2>/dev/null && echo yes || echo no)" >> "$WLOG"

  # If the runner process tree died entirely, restart it.
  if ! kill -0 "$runner_pid" 2>/dev/null; then
    if [ -f "$INDEX_FILE" ] && [ "$incompletes" -eq 0 ] && [ "$shards" -ge 15 ]; then
      echo "[$(date)] runner exited and download looks complete" >> "$WLOG"
      break
    fi
    echo "[$(date)] runner exited early, restarting" >> "$WLOG"
    runner_pid=$(start_runner)
    stall=0
    continue
  fi

  # Stall detected -> kill hf workers so inner retry loop resumes.
  if [ "$stall" -ge "$STALL_LIMIT" ]; then
    echo "[$(date)] STALL ${stall}s >= ${STALL_LIMIT}s -> killing hf download to force resume" >> "$WLOG"
    pkill -9 -f "hf download" 2>/dev/null
    stall=0
  fi

  sleep "$CHECK_INTERVAL"
done

echo "[$(date)] watchdog done" >> "$WLOG"

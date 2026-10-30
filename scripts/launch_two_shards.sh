#!/bin/bash
set -euo pipefail
cd /mnt/bn/isp-traindata-lf-1/wq/OmniZip
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BASE="${PROJECT_ROOT:-/mnt/bn/isp-traindata-lf-1/wq/OmniZip}/logs/parallel_${RUN_ID}"
mkdir -p "$BASE"
for shard in 0 1; do
    out="$BASE/shard_${shard}"
    log="$BASE/run_shard_${shard}.log"
    mkdir -p "$out"
    port=$((12347 + shard))
    echo "Starting shard ${shard}/2: log=$log output=$out port=$port"
    nohup bash -c "LMMS_EVAL_SHARD_RANK=${shard} LMMS_EVAL_SHARD_WORLD=2 EVAL_LIMIT='' EVAL_LOG_SAMPLES=1 EVAL_OUTPUT_PATH='${out}' MAIN_PROCESS_PORT=${port} bash eval.sh" > "$log" 2>&1 &
    echo "  pid=$!"
done
echo "$BASE" > logs/latest_parallel_dir.txt
echo "latest_parallel_dir=$BASE"
sleep 3
ps -ef | grep -E "lmms_eval|accelerate|eval.sh" | grep -v grep | head -20

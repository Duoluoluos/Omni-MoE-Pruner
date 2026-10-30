#!/bin/bash
cd /mnt/bn/isp-traindata-lf-1/wq/OmniZip
BASE="${1:-$(cat logs/latest_parallel_dir.txt 2>/dev/null)}"
if [ -z "${BASE:-}" ]; then
    echo "No parallel run dir found"
    exit 1
fi
echo "BASE=$BASE"
echo "=== processes ==="
ps -ef | grep -E "lmms_eval|accelerate|eval.sh" | grep -v grep | head -20
echo "=== gpu ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
for shard in 0 1; do
    log="$BASE/run_shard_${shard}.log"
    echo "=== shard $shard log: $log ==="
    if [ -f "$log" ]; then
        tr '\r' '\n' < "$log" | tail -20
    else
        echo "missing log"
    fi
done

#!/bin/bash
# Check if .incomplete file size matches expected size
HUB=/mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/hub/datasets--lmms-lab--Video-MME
echo "=== process state ==="
pid=$(pgrep -f "lmms_eval --model" | head -1)
echo "pid=$pid"
[ -n "$pid" ] && cat /proc/$pid/status | grep -E "State|Threads|VmRSS"
echo "=== child threads ==="
[ -n "$pid" ] && ls /proc/$pid/task/ 2>/dev/null | wc -l
echo "=== tcp connections (rsync/hf_transfer) ==="
[ -n "$pid" ] && ss -tnp 2>/dev/null | grep "pid=$pid" | head -5
echo "=== last log ==="
tr '\r' '\n' < /mnt/bn/isp-traindata-lf-1/wq/OmniZip/logs/run.log | tail -20
echo "=== incomplete sizes vs hash files (24 expected) ==="
ls -la $HUB/blobs/*.incomplete 2>/dev/null | awk '{print $5, $NF}' | sed "s|.*/||"

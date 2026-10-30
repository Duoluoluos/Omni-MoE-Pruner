#!/bin/bash
cd /mnt/bn/isp-traindata-lf-1/wq/OmniZip
mkdir -p logs
# Start eval in background with nohup, no --limit
nohup bash -c 'EVAL_LIMIT="" bash eval.sh' > logs/run.log 2>&1 &
PID=$!
echo "Started eval PID=$PID"
sleep 3
# Verify
ps -p $PID -o pid,stat,cmd --no-headers 2>/dev/null && echo "Process is alive" || echo "Process died!"

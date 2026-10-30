#!/usr/bin/env bash
set -e
cd /mnt/bn/isp-traindata-lf-1/wq/OmniZip
TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/qwen3_omni_videomme_${TS}.log
echo "Log: $LOG" > "$LOG.banner"
setsid nohup bash eval.sh > "$LOG" 2>&1 < /dev/null &
EVAL_PID=$!
disown
echo "$EVAL_PID" > logs/qwen3_omni_videomme_${TS}.pid
echo "Started PID=$EVAL_PID log=$LOG"
sleep 5
ps -ef > /tmp/ps_$$.txt
grep -E "eval.sh|accelerate" /tmp/ps_$$.txt > "$LOG.ps" 2>&1 || true
grep -E "eval.sh|accelerate" /tmp/ps_$$.txt | head -5

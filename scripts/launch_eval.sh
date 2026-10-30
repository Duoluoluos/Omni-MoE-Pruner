#!/bin/bash
cd /mnt/bn/isp-traindata-lf-1/wq/OmniZip
mkdir -p logs
nohup setsid bash -c "EVAL_LIMIT=${EVAL_LIMIT:-2} bash eval.sh" > logs/run.log 2>&1 < /dev/null &
disown -a 2>/dev/null
echo "started; sleeping 2"
sleep 2
ls -la logs/run.log
ps -ef | grep -E "lmms_eval|accelerate|eval.sh" | grep -v grep | head -10

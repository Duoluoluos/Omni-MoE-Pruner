#!/bin/bash
ps -ef | grep -E "lmms_eval|accelerate|eval.sh" | grep -v grep | head -10
echo "==="
ls -la /mnt/bn/isp-traindata-lf-1/wq/OmniZip/logs/run.log
echo "==="
tr '\r' '\n' < /mnt/bn/isp-traindata-lf-1/wq/OmniZip/logs/run.log | tail -15

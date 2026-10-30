#!/bin/bash
echo "=== process ==="
ps -ef | grep -E "lmms_eval" | grep -v grep | head -3
echo "=== HF cache (eval_data) ==="
du -sh /mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/hub/datasets--lmms-lab--Video-MME 2>/dev/null
ls /mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/hub/datasets--lmms-lab--Video-MME/blobs/ 2>/dev/null | wc -l
echo "=== output logs/videomme ==="
find /mnt/bn/isp-traindata-lf-1/wq/OmniZip/logs -maxdepth 4 -type f 2>/dev/null | head -10
echo "=== last log lines ==="
tr '\r' '\n' < /mnt/bn/isp-traindata-lf-1/wq/OmniZip/logs/run.log | tail -50

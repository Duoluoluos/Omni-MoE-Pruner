#!/bin/bash
# Quick snapshot of HF cache size + active blob.
HUB=/mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/hub/datasets--lmms-lab--Video-MME
date +"[%H:%M:%S]"
du -sb $HUB/blobs 2>/dev/null | awk '{printf "total_bytes=%d\n", $1}'
echo "complete=$(ls $HUB/blobs 2>/dev/null | grep -v incomplete | wc -l) incomplete=$(ls $HUB/blobs 2>/dev/null | grep incomplete | wc -l)"
echo "active_files (mtime within 60s):"
find $HUB/blobs -name '*.incomplete' -mmin -1 -exec ls -la {} \; 2>/dev/null | awk '{print $5, $NF}'
ps -ef | grep lmms_eval | grep -v grep | head -1 | awk '{print "worker_pid="$2}'

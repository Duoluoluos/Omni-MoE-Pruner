#!/bin/bash
echo "=== blobs ==="
ls -laSr /mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/hub/datasets--lmms-lab--Video-MME/blobs/ | tail -10
echo "=== incomplete files ==="
find /mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/hub/datasets--lmms-lab--Video-MME -name "*.incomplete" -exec ls -la {} \;
echo "=== network/process ==="
ls /proc/4591/fd 2>/dev/null | wc -l
cat /proc/4591/status 2>/dev/null | grep -E "State|Threads"
echo "=== open https ==="
ls -la /proc/4591/fd 2>/dev/null | grep -i sock | head

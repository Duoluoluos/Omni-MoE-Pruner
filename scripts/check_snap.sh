#!/bin/bash
SNAP=/mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots
echo "=== snapshots ==="
ls "$SNAP"/ 2>/dev/null
echo "=== zip files ==="
find "$SNAP" -name "*.zip" 2>/dev/null | sort
echo "=== extracted mp4 count ==="
find /mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/videomme -name "*.mp4" 2>/dev/null | wc -l
echo "=== videomme dir size ==="
du -sh /mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/videomme/ 2>/dev/null

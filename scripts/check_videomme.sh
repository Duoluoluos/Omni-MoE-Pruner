#!/bin/bash
echo "=== videomme structure ==="
ls /mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/videomme/ 2>/dev/null
echo "=== data dir ==="
ls /mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/videomme/data/ 2>/dev/null | head -5
echo "=== fFjv93ACGo8 ==="
find /mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/videomme -name "fFjv93ACGo8*" 2>/dev/null
echo "=== mp4 count ==="
find /mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/videomme -name "*.mp4" 2>/dev/null | wc -l

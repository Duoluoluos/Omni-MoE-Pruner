#!/bin/bash
echo "=== before ==="
du -sh /mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/videomme/data/ 2>/dev/null
ls /mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/videomme/data/ 2>/dev/null | wc -l
echo "=== deleting partial extraction ==="
rm -rf /mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/videomme/data/
echo "=== after ==="
ls /mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/videomme/ 2>/dev/null

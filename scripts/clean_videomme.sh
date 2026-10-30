#!/bin/bash
echo "=== deleting entire videomme cache ==="
rm -rf /mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/videomme/
echo "=== verify ==="
ls /mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/videomme/ 2>&1
echo "=== hub blobs still intact ==="
ls /mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/hub/datasets--lmms-lab--Video-MME/blobs/ | wc -l
du -sh /mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/hub/datasets--lmms-lab--Video-MME/blobs/

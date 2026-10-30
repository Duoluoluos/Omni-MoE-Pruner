#!/bin/bash
HUB=/mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/hub/datasets--lmms-lab--Video-MME/blobs
echo "=== before ==="
ls "$HUB"/*.incomplete 2>/dev/null | wc -l
du -sh "$HUB"/..
echo "=== deleting incomplete ==="
rm -f "$HUB"/*.incomplete
echo "=== after ==="
ls "$HUB"/*.incomplete 2>/dev/null | wc -l
ls "$HUB"/ | wc -l
du -sh "$HUB"/..

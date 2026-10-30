unam#!/bin/bash
SNAP=/mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b
DEST=/mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/videomme
mkdir -p "$DEST"
for zip in "$SNAP"/*.zip; do
    name=$(basename "$zip")
    echo "[$(date +%H:%M:%S)] Extracting $name ..."
    unzip -o "$zip" -d "$DEST" > /dev/null 2>&1
    echo "  done"
done
echo "=== done ==="
echo "mp4 count: $(find "$DEST" -name '*.mp4' | wc -l)"
du -sh "$DEST"

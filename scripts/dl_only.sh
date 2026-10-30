#!/bin/bash
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"
source eval.sh > /dev/null 2>&1
# Kill any stale processes first
pkill -9 -f "lmms_eval" 2>/dev/null
pkill -9 -f "accelerate launch" 2>/dev/null
sleep 2

# Pure dataset download (no model loading, no eval)
mkdir -p logs
nohup setsid bash -c '
export PROJECT_ROOT="${PROJECT_ROOT}"
export HF_HOME="${PROJECT_ROOT}/eval_data/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
export HF_HUB_ENABLE_HF_TRANSFER=0
$PROJECT_ROOT/.venv/bin/python -c "
from huggingface_hub import snapshot_download
import os
print(\"start\", flush=True)
path = snapshot_download(
    repo_id=\"lmms-lab/Video-MME\",
    repo_type=\"dataset\",
    token=os.environ.get(\"HF_TOKEN\") or None,
    max_workers=4,
    etag_timeout=60,
)
print(\"done\", path, flush=True)
"
' > logs/dl.log 2>&1 &
disown -a 2>/dev/null
echo "started"
sleep 3
ls -la logs/dl.log
ps -ef | grep snapshot_download | grep -v grep

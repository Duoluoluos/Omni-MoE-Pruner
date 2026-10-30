#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner
python_bin="$repo/.venv/bin/python"
runner="$repo/scripts/run_e16b_auto_config_validation.py"
protocol="$repo/research_paper/iclr2027/evidence/e16b_auto_config_validation_protocol.json"
output_dir="$repo/research_paper/iclr2027/evidence/runs/e16b_auto_config_validation_worker4193170_20260827"
log_file="$repo/logs/e16b_auto_config_validation_worker4193170_20260827.log"

mkdir -p "$output_dir"
cd "$repo"
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
FORCE_QWENVL_VIDEO_READER=decord \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
E16B_WORKER_ID=4193170 \
nohup "$python_bin" "$runner" \
  --protocol "$protocol" \
  --output-dir "$output_dir" \
  >>"$log_file" 2>&1 </dev/null &
run_pid=$!
echo "[launcher] E16-B pid=$run_pid log=$log_file output=$output_dir"

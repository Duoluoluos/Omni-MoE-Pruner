#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner
python_bin="$repo/.venv/bin/python"
runner="$repo/scripts/run_pm2_uniform_frontier.py"
output_dir="$repo/research_paper/iclr2027/evidence/runs/pm2_uniform_frontier_worker4193170_20260828"
log_file="$repo/logs/pm2_uniform_frontier_worker4193170_20260828.log"

mkdir -p "$output_dir" "$repo/logs"
cd "$repo"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export FORCE_QWENVL_VIDEO_READER=decord
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PM2_WORKER_ID=4193170

CUDA_VISIBLE_DEVICES=0 nohup "$python_bin" "$runner" \
  --output-dir "$output_dir" \
  >>"$log_file" 2>&1 </dev/null &
run_pid=$!
echo "[launcher] P-M2 pid=$run_pid log=$log_file output=$output_dir"
echo "$run_pid" > "$output_dir/launcher.pid"

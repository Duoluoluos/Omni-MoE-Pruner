#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner
python_bin="$repo/.venv/bin/python"
runner="$repo/scripts/run_e16_auto_routing_selector.py"
protocol="$repo/research_paper/iclr2027/evidence/e16_auto_selector_protocol.json"
output_dir="$repo/research_paper/iclr2027/evidence/runs/e16_auto_selector_worker4193170_20260826"
log_file="$repo/logs/e16_auto_selector_worker4193170_20260826.log"

mkdir -p "$output_dir"
cd "$repo"
CUDA_VISIBLE_DEVICES=0 nohup "$python_bin" "$runner" \
  --protocol "$protocol" \
  --output-dir "$output_dir" \
  >>"$log_file" 2>&1 </dev/null &
run_pid=$!
echo "[launcher] E16-A pid=$run_pid log=$log_file output=$output_dir"

#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner
python_bin="$repo/.venv/bin/python"
runner="$repo/scripts/run_pm3c_audio_directionality_control.py"
output_dir="$repo/research_paper/iclr2027/evidence/runs/pm3c_audio_directionality_control_worker4207577_20260830"
log_file="$repo/logs/pm3c_audio_directionality_control_worker4207577_20260830.log"

mkdir -p "$output_dir" "$repo/logs"
cd "$repo"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export FORCE_QWENVL_VIDEO_READER=decord
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PM3C_WORKER_ID=4207577

# Both A800-80GB GPUs stay visible so the model auto-shards and the longest
# OmniVideoBench prefills fit without OOM.
nohup "$python_bin" "$runner" \
  --output-dir "$output_dir" \
  >>"$log_file" 2>&1 </dev/null &
run_pid=$!
echo "[launcher] P-M3c pid=$run_pid log=$log_file output=$output_dir"
echo "$run_pid" > "$output_dir/launcher.pid"

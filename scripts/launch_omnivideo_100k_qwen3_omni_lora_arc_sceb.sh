#!/usr/bin/env bash
# Start an independent ARC-SCEB Megatron LoRA training run in tmux.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${LORA_RUNTIME_DIR:-/tmp/omnivideo_100k_qwen3_omni_lora}"
run_name="${RUN_NAME:-omnivideo_100k_qwen3_omni_lora_arc_sceb}"
output_dir="${OUTPUT_DIR:-${project_root}/models/${run_name}}"
policy_path="${ARC_SCEB_POLICY:-${project_root}/moe_expert_budget/configs/arc_video_boundary_audio_floor6_full_v1.json}"
plugin_path="${project_root}/scripts/megatron_arc_sceb_router.py"
mode="full"
run_arg=()

if [[ "${1:-}" == "--smoke" ]]; then
  mode="smoke"
  run_arg=(--smoke)
elif [[ $# -gt 0 ]]; then
  printf 'Usage: %s [--smoke]\n' "$0" >&2
  exit 2
fi

if [[ ! -x "${runtime_root}/venv/bin/python" ]]; then
  printf 'Missing LoRA environment: %s\n' "${runtime_root}/venv" >&2
  exit 1
fi
if [[ ! -f "${plugin_path}" || ! -f "${policy_path}" ]]; then
  printf 'Missing ARC-SCEB plugin or policy: %s / %s\n' "${plugin_path}" "${policy_path}" >&2
  exit 1
fi

log_dir="${LORA_LOG_DIR:-${output_dir}/logs}"
log_file="${log_dir}/${mode}.log"
session_file="${log_dir}/${mode}.tmux-session"
session_name="${TMUX_SESSION_NAME:-omnivideo_lora_arc_sceb_${mode}}"

unset TMUX_TMPDIR || true
mkdir -p "/tmp/tmux-$(id -u)" "${log_dir}"
if ! command -v tmux >/dev/null 2>&1; then
  printf 'tmux is required to keep the ARC-SCEB run alive after logout.\n' >&2
  exit 1
fi
if tmux has-session -t "${session_name}" 2>/dev/null; then
  printf 'ARC-SCEB %s run is already active in tmux session %s. Log: %s\n' \
    "${mode}" "${session_name}" "${log_file}"
  exit 0
fi

environment=(
  "RUN_NAME=${run_name}"
  "OUTPUT_DIR=${output_dir}"
  "EXTERNAL_PLUGINS=${plugin_path}"
  "ARC_SCEB_POLICY=${policy_path}"
  "NPROC_PER_NODE=${NPROC_PER_NODE:-2}"
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}"
  "EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-2}"
  "FINETUNE=true"
  "ADD_VERSION=true"
)
# The 8-record smoke split produces fewer packed sequences than one normal
# global batch (16), which makes Megatron's one-epoch LR schedule degenerate
# to zero steps before any forward pass.  This affects only the smoke gate;
# the full ARC-SCEB run retains the baseline global batch size of 16.
if [[ "${mode}" == "smoke" ]]; then
  environment+=("GLOBAL_BATCH_SIZE=${SMOKE_GLOBAL_BATCH_SIZE:-2}")
fi

printf -v quoted_runner '%q' "${project_root}/scripts/run_omnivideo_100k_qwen3_omni_lora.sh"
run_command='exec env'
for variable in "${environment[@]}"; do
  printf -v quoted_variable ' %q' "${variable}"
  run_command+="${quoted_variable}"
done
run_command+=" bash ${quoted_runner}"
for arg in "${run_arg[@]}"; do
  printf -v quoted_arg ' %q' "${arg}"
  run_command+="${quoted_arg}"
done
printf -v quoted_log ' %q' "${log_file}"
run_command+=" >${quoted_log} 2>&1"

tmux new-session -d -s "${session_name}" "${run_command}"
printf '%s\n' "${session_name}" > "${session_file}"
printf 'Started ARC-SCEB LoRA %s run in tmux session %s. Log: %s\n' \
  "${mode}" "${session_name}" "${log_file}"

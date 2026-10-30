#!/usr/bin/env bash
# Resume the interrupted OmniVideo LoRA run without creating a new version.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="${OUTPUT_DIR:-${project_root}/models/omnivideo_100k_qwen3_omni_lora/v0-20260728-094600}"
checkpoint_dir="${RESUME_CHECKPOINT:-${run_dir}/checkpoint-2100}"
runtime_root="${LORA_RUNTIME_DIR:-/tmp/omnivideo_100k_qwen3_omni_lora}"
log_dir="${LORA_LOG_DIR:-${run_dir}/logs}"
log_file="${log_dir}/resume_from_2100.log"
session_file="${log_dir}/resume.tmux-session"
session_name="${TMUX_SESSION_NAME:-omnivideo_lora_resume}"

if [[ ! -d "${run_dir}" || ! -f "${checkpoint_dir}/adapter_model.safetensors" ]]; then
  printf 'Missing resume checkpoint: %s\n' "${checkpoint_dir}" >&2
  exit 1
fi
if [[ ! -x "${runtime_root}/venv/bin/python" ]]; then
  printf 'Missing LoRA environment: %s\n' "${runtime_root}/venv" >&2
  exit 1
fi

mkdir -p "${log_dir}"
unset TMUX_TMPDIR || true
mkdir -p "/tmp/tmux-$(id -u)"

if ! command -v tmux >/dev/null 2>&1; then
  printf 'tmux is required to keep the LoRA run alive after logout.\n' >&2
  exit 1
fi
if tmux has-session -t "${session_name}" 2>/dev/null; then
  printf 'LoRA resume is already running in tmux session %s. Log: %s\n' \
    "${session_name}" "${log_file}"
  exit 0
fi

"${runtime_root}/venv/bin/python" \
  "${project_root}/scripts/prepare_omnivideo_100k_qwen3_omni_lora_resume.py" \
  --output-dir "${run_dir}" \
  --checkpoint "${checkpoint_dir}"

environment=(
  "OUTPUT_DIR=${run_dir}"
  "ADAPTERS=${checkpoint_dir}"
  "FINETUNE=false"
  "ADD_VERSION=false"
  "NO_LOAD_OPTIM=true"
  "NO_LOAD_RNG=true"
  "EXTERNAL_PLUGINS=${project_root}/scripts/megatron_resume_lr_scheduler.py"
  "NPROC_PER_NODE=2"
  "CUDA_VISIBLE_DEVICES=0,1"
  "EXPERT_MODEL_PARALLEL_SIZE=2"
)

printf -v quoted_runner '%q' "${project_root}/scripts/run_omnivideo_100k_qwen3_omni_lora.sh"
run_command='exec env'
for variable in "${environment[@]}"; do
  printf -v quoted_variable ' %q' "${variable}"
  run_command+="${quoted_variable}"
done
run_command+=" bash ${quoted_runner}"
printf -v quoted_log ' %q' "${log_file}"
run_command+=" >>${quoted_log} 2>&1"

tmux new-session -d -s "${session_name}" "${run_command}"
printf '%s\n' "${session_name}" > "${session_file}"
printf 'Started LoRA resume in tmux session %s. Log: %s\n' "${session_name}" "${log_file}"

#!/usr/bin/env bash
# Launch the LoRA job in tmux so it survives the worker login session.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${LORA_RUNTIME_DIR:-/tmp/omnivideo_100k_qwen3_omni_lora}"
run_name="${RUN_NAME:-omnivideo_100k_qwen3_omni_lora}"
output_dir="${OUTPUT_DIR:-${project_root}/models/${run_name}}"
mode="full"
run_arg=()
if [[ "${1:-}" == "--smoke" ]]; then
  mode="smoke"
  run_arg=(--smoke)
fi

# Logs live next to the checkpoints, not in /tmp, so both survive a worker
# recycle and are available from the shared workspace.
log_dir="${LORA_LOG_DIR:-${output_dir}/logs}"
log_file="${log_dir}/${mode}.log"
session_file="${log_dir}/${mode}.tmux-session"
session_name="${TMUX_SESSION_NAME:-omnivideo_lora_${mode}}"
mkdir -p "${log_dir}"
# This worker image lacks tmux's user socket directory by default.  Use the
# proven default location so the session is visible to subsequent logins.
unset TMUX_TMPDIR || true
mkdir -p "/tmp/tmux-$(id -u)"

if ! command -v tmux >/dev/null 2>&1; then
  printf 'tmux is required to keep the LoRA run alive after logout.\n' >&2
  exit 1
fi

if tmux has-session -t "${session_name}" 2>/dev/null; then
  printf 'LoRA %s run is already running in tmux session %s. Log: %s\n' \
    "${mode}" "${session_name}" "${log_file}"
  exit 0
fi

printf -v quoted_runner '%q' "${project_root}/scripts/run_omnivideo_100k_qwen3_omni_lora.sh"
run_command="exec bash ${quoted_runner}"
for arg in "${run_arg[@]}"; do
  printf -v quoted_arg '%q' "${arg}"
  run_command+=" ${quoted_arg}"
done
printf -v quoted_log '%q' "${log_file}"
run_command+=" >${quoted_log} 2>&1"

tmux new-session -d -s "${session_name}" "${run_command}"
printf '%s\n' "${session_name}" > "${session_file}"
printf 'Started LoRA %s run in tmux session %s. Log: %s\n' \
  "${mode}" "${session_name}" "${log_file}"

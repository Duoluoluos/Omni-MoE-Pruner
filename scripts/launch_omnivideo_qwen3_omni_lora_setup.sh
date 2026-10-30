#!/usr/bin/env bash
# Launch isolated environment setup independently of the worker login session.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${LORA_RUNTIME_DIR:-/tmp/omnivideo_100k_qwen3_omni_lora}"
log_dir="${runtime_root}/launch_logs"
log_file="${log_dir}/environment_setup.log"
session_file="${log_dir}/environment_setup.tmux-session"
session_name="${TMUX_SESSION_NAME:-omnivideo_lora_setup}"
mkdir -p "${log_dir}"
unset TMUX_TMPDIR || true
mkdir -p "/tmp/tmux-$(id -u)"

if ! command -v tmux >/dev/null 2>&1; then
  printf 'tmux is required to keep environment setup alive after logout.\n' >&2
  exit 1
fi
if tmux has-session -t "${session_name}" 2>/dev/null; then
  printf 'Environment setup is already running in tmux session %s. Log: %s\n' \
    "${session_name}" "${log_file}"
  exit 0
fi

printf -v quoted_setup '%q' "${project_root}/scripts/setup_omnivideo_qwen3_omni_lora_env.sh"
printf -v quoted_log '%q' "${log_file}"
tmux new-session -d -s "${session_name}" "exec bash ${quoted_setup} >${quoted_log} 2>&1"
printf '%s\n' "${session_name}" > "${session_file}"
printf 'Started isolated LoRA environment setup in tmux session %s. Log: %s\n' \
  "${session_name}" "${log_file}"

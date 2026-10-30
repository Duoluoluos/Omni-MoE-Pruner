#!/usr/bin/env bash
# Print concise, login-safe status for data preparation and LoRA jobs.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${LORA_RUNTIME_DIR:-/tmp/omnivideo_100k_qwen3_omni_lora}"
run_name="${RUN_NAME:-omnivideo_100k_qwen3_omni_lora}"
output_dir="${OUTPUT_DIR:-${project_root}/models/${run_name}}"
status_dir="${LORA_LOG_DIR:-${output_dir}/logs}"
unset TMUX_TMPDIR || true

show_pid() {
  local label="$1"
  local pid_file="$2"
  if [[ -s "${pid_file}" ]]; then
    local pid
    pid="$(<"${pid_file}")"
    if ps -p "${pid}" -o pid=,stat=,etime=,args= >/dev/null 2>&1; then
      printf '%s: running\n' "${label}"
      ps -p "${pid}" -o pid=,stat=,etime=,args=
    else
      printf '%s: exited (last pid %s)\n' "${label}" "${pid}"
    fi
  else
    printf '%s: not started\n' "${label}"
  fi
}

show_pid 'environment setup' "${runtime_root}/launch_logs/environment_setup.pid"

for mode in smoke full; do
  session_file="${status_dir}/${mode}.tmux-session"
  if [[ -s "${session_file}" ]] && command -v tmux >/dev/null 2>&1; then
    session_name="$(<"${session_file}")"
    if tmux has-session -t "${session_name}" 2>/dev/null; then
      printf 'LoRA %s run: running in tmux session %s\n' "${mode}" "${session_name}"
    else
      printf 'LoRA %s run: tmux session %s has exited\n' "${mode}" "${session_name}"
    fi
  else
    printf 'LoRA %s run: not started\n' "${mode}"
  fi
done

if pgrep -af extract_omnivideo_100k_videos >/dev/null 2>&1; then
  printf 'video extraction: running\n'
  pgrep -af extract_omnivideo_100k_videos
else
  printf 'video extraction: not running\n'
fi
if [[ -d "${project_root}/ft_data/videos" ]]; then
  du -sh "${project_root}/ft_data/videos"
  find "${project_root}/ft_data/videos" -type f -name '*.mp4' -printf . | wc -c | xargs printf 'extracted mp4 files: %s\n'
fi

for log_file in "${runtime_root}/launch_logs/environment_setup.log" "${status_dir}/smoke.log" "${status_dir}/full.log"; do
  if [[ -f "${log_file}" ]]; then
    printf '\n==> %s <==\n' "${log_file}"
    tail -n 20 "${log_file}"
  fi
done

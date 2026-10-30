#!/usr/bin/env bash
# Background launcher for the single-H20, one-seed baseline/ARC-SCEB pair.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pair_root="${PAIR_ROOT:-${project_root}/models/arc_sceb_mask_only_h20_seed42_20260804}"
mode="full"
run_arg=()

if [[ "${1:-}" == "--smoke" ]]; then
  mode="smoke"
  run_arg=(--smoke)
elif [[ $# -gt 0 ]]; then
  printf 'Usage: %s [--smoke]\n' "$0" >&2
  exit 2
fi

runner="${project_root}/scripts/run_omnivideo_100k_qwen3_omni_lora_h20_pair.sh"
log_dir="${pair_root}/launcher"
log_file="${log_dir}/${mode}.log"
session_name="${TMUX_SESSION_NAME:-omnivideo_lora_h20_seed42_${mode}}"

if [[ ! -f "${runner}" ]]; then
  printf 'Missing H20 pair runner: %s\n' "${runner}" >&2
  exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
  printf 'tmux is required to keep the H20 pair alive after logout.\n' >&2
  exit 1
fi

mkdir -p "${log_dir}" "/tmp/tmux-$(id -u)"
unset TMUX_TMPDIR || true
if tmux has-session -t "${session_name}" 2>/dev/null; then
  printf 'H20 paired %s run is already active in tmux session %s. Log: %s\n' \
    "${mode}" "${session_name}" "${log_file}"
  exit 0
fi

printf -v quoted_runner '%q' "${runner}"
printf -v quoted_log '%q' "${log_file}"
run_command="exec bash ${quoted_runner}"
for arg in "${run_arg[@]}"; do
  printf -v quoted_arg ' %q' "${arg}"
  run_command+="${quoted_arg}"
done
run_command+=" >${quoted_log} 2>&1"

tmux new-session -d -s "${session_name}" "${run_command}"
printf 'Started H20 paired %s run in tmux session %s. Log: %s\n' \
  "${mode}" "${session_name}" "${log_file}"

#!/usr/bin/env bash
# Start the single-H20 smoke-gated baseline/ARC-SCEB training pipeline.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pair_root="${PAIR_ROOT:-${project_root}/models/arc_sceb_mask_only_h20_seed42_20260804_native_adamw}"
runner="${project_root}/scripts/run_omnivideo_100k_qwen3_omni_lora_h20_pipeline.sh"
log_dir="${pair_root}/launcher"
log_file="${log_dir}/pipeline.log"
session_name="${TMUX_SESSION_NAME:-omnivideo_lora_h20_seed42_pipeline}"
skip_precheck="${SKIP_PRECHECK:-false}"
pair_master_port="${PAIR_MASTER_PORT:-}"
max_steps="${MAX_STEPS:-}"

if [[ ! -f "${runner}" ]]; then
  printf 'Missing H20 pipeline runner: %s\n' "${runner}" >&2
  exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
  printf 'tmux is required to keep the H20 pipeline alive after logout.\n' >&2
  exit 1
fi

mkdir -p "${log_dir}" "/tmp/tmux-$(id -u)"
unset TMUX_TMPDIR || true
if tmux has-session -t "${session_name}" 2>/dev/null; then
  printf 'H20 pipeline is already active in tmux session %s. Log: %s\n' \
    "${session_name}" "${log_file}"
  exit 0
fi

printf -v quoted_runner '%q' "${runner}"
printf -v quoted_log '%q' "${log_file}"
printf -v quoted_pair_root '%q' "${pair_root}"
printf -v quoted_skip_precheck '%q' "${skip_precheck}"
printf -v quoted_pair_master_port '%q' "${pair_master_port}"
printf -v quoted_max_steps '%q' "${max_steps}"
# A pre-existing tmux server does not reliably inherit variables introduced by
# a later client. Pass the run identity, smoke-gate decision, and optional
# rendezvous-port override explicitly.
tmux new-session -d -s "${session_name}" \
  "exec env PAIR_ROOT=${quoted_pair_root} SKIP_PRECHECK=${quoted_skip_precheck} PAIR_MASTER_PORT=${quoted_pair_master_port} MAX_STEPS=${quoted_max_steps} bash ${quoted_runner} >${quoted_log} 2>&1"
printf 'Started H20 smoke-gated pipeline in tmux session %s. Log: %s\n' \
  "${session_name}" "${log_file}"

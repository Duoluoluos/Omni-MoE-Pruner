#!/usr/bin/env bash
# Keep the short native-sparse versus dense-fallback benchmark alive after the
# worker shell disconnects.  It is intentionally sequential because this is a
# one-GPU H20 recipe.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="${BENCHMARK_ROOT:-${project_root}/models/arc_sceb_router_benchmark_h20_seed42_20260807}"
runner="${project_root}/scripts/run_omnivideo_100k_qwen3_omni_lora_h20_router_benchmark.sh"
session_name="${TMUX_SESSION_NAME:-arc_sceb_router_benchmark_h20_seed42}"
log_dir="${benchmark_root}/launcher"
log_file="${log_dir}/benchmark.log"
max_steps="${MAX_STEPS:-25}"
master_port="${BENCHMARK_MASTER_PORT:-25380}"
benchmark_cases="${BENCHMARK_CASES:-dense sparse}"

if [[ ! -f "${runner}" ]]; then
  printf 'Missing router benchmark runner: %s\n' "${runner}" >&2
  exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
  printf 'tmux is required to keep the router benchmark alive after logout.\n' >&2
  exit 1
fi

mkdir -p "${log_dir}" "/tmp/tmux-$(id -u)"
unset TMUX_TMPDIR || true
if tmux has-session -t "${session_name}" 2>/dev/null; then
  printf 'Router benchmark is already active in tmux session %s. Log: %s\n' \
    "${session_name}" "${log_file}"
  exit 0
fi

printf -v q_runner '%q' "${runner}"
printf -v q_log '%q' "${log_file}"
printf -v q_root '%q' "${benchmark_root}"
printf -v q_cases '%q' "${benchmark_cases}"
tmux new-session -d -s "${session_name}" \
  "exec env BENCHMARK_ROOT=${q_root} MAX_STEPS=${max_steps} BENCHMARK_MASTER_PORT=${master_port} BENCHMARK_CASES=${q_cases} bash ${q_runner} >${q_log} 2>&1"
printf 'Started router benchmark in tmux session %s. Log: %s\n' \
  "${session_name}" "${log_file}"

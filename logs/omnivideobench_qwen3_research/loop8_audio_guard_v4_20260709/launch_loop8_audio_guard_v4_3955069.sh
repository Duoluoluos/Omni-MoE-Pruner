#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner
EXP="${ROOT}/logs/omnivideobench_qwen3_research/loop8_audio_guard_v4_20260709"
LAUNCHER_DIR="${EXP}/launcher"

mkdir -p "${LAUNCHER_DIR}"
cd "${ROOT}"

ts="$(date +%Y%m%d_%H%M%S)"
log_path="${LAUNCHER_DIR}/loop8_audio_guard_v4_worker_3955069_${ts}.log"
pid_path="${LAUNCHER_DIR}/loop8_audio_guard_v4_worker_3955069.pid"

nohup env NO_COLOR=1 TERM=dumb mlx worker login 3955069 -- \
  bash -lc "bash '${EXP}/run_smoke_audio_guard_v4.sh' && bash '${EXP}/run_limit200_audio_guard_v4.sh'" \
  >"${log_path}" 2>&1 &
pid="$!"

printf '%s\n' "${pid}" >"${pid_path}"
printf '%s\n' "${log_path}" >"${LAUNCHER_DIR}/latest.logpath"
echo "launched_pid=${pid}"
echo "log_path=${log_path}"
echo "pid_path=${pid_path}"

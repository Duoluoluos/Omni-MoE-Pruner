#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner
EXP_DIR="${ROOT}/logs/omnivideobench_qwen3_research/loop3_audio_guard_20260709"
LAUNCHER_DIR="${EXP_DIR}/launcher"

mkdir -p "${LAUNCHER_DIR}"
cd "${ROOT}"

ts="$(date +%Y%m%d_%H%M%S)"
log_path="${LAUNCHER_DIR}/limit200_audio_guard_3955069_${ts}.log"
pid_path="${LAUNCHER_DIR}/limit200_audio_guard_3955069.pid"

nohup "${EXP_DIR}/run_limit200_audio_guard.sh" >"${log_path}" 2>&1 &
pid="$!"
printf '%s\n' "${pid}" >"${pid_path}"

echo "launched_pid=${pid}"
echo "log_path=${log_path}"
echo "pid_path=${pid_path}"

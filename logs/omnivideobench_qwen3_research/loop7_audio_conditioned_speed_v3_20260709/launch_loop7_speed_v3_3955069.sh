#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner
EXP="${ROOT}/logs/omnivideobench_qwen3_research/loop7_audio_conditioned_speed_v3_20260709"
LAUNCHER_DIR="${EXP}/launcher"
SESSION="loop7_audio_conditioned_speed_v3"

mkdir -p "${LAUNCHER_DIR}"
mkdir -p "/tmp/tmux-$(id -u)"
chmod 700 "/tmp/tmux-$(id -u)"
cd "${ROOT}"

ts="$(date +%Y%m%d_%H%M%S)"
log_path="${LAUNCHER_DIR}/loop7_speed_v3_3955069_${ts}.log"
session_path="${LAUNCHER_DIR}/loop7_speed_v3_3955069.session"

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" "bash '${EXP}/run_smoke_speed_v3.sh' >'${log_path}' 2>&1 && bash '${EXP}/run_limit200_speed_v3.sh' >>'${log_path}' 2>&1"

printf '%s\n' "${SESSION}" >"${session_path}"
printf '%s\n' "${log_path}" >"${LAUNCHER_DIR}/latest.logpath"
echo "session=${SESSION}"
echo "log_path=${log_path}"
echo "session_path=${session_path}"
tmux ls

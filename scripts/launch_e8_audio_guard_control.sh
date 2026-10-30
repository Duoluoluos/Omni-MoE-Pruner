#!/usr/bin/env bash
set -euo pipefail

root=/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner
run_dir="${root}/research_paper/iclr2027/evidence/runs/e8_audio_guard_worker4177575_20260824"
log_dir="${root}/logs/e8_audio_guard_worker4177575_20260824"
log="${log_dir}/run.log"
pid_file="${run_dir}/launcher.pid"
mkdir -p "${run_dir}" "${log_dir}"

if [[ -s "${pid_file}" ]]; then
  old_pid="$(<"${pid_file}")"
  if kill -0 "${old_pid}" 2>/dev/null; then
    echo "ALREADY_RUNNING pid=${old_pid} log=${log}"
    exit 0
  fi
fi

cd "${root}"
setsid env \
  CUDA_VISIBLE_DEVICES=0 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  FORCE_QWENVL_VIDEO_READER=decord \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  E8_WORKER_ID=4177575 \
  "${root}/.venv/bin/python" "${root}/scripts/run_e8_audio_guard_control.py" \
  --output-dir "${run_dir}" >"${log}" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" >"${pid_file}"
echo "LAUNCHED pid=${pid} log=${log} output=${run_dir}"

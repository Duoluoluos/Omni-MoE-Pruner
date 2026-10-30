#!/usr/bin/env bash
set -euo pipefail
project_root=/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner
export LORA_RUNTIME_DIR="${project_root}/.omni_lora_runtime"
export MAX_STEPS="${MAX_STEPS:-25}"
export BENCHMARK_CASES="${BENCHMARK_CASES:-native guard}"
export ARC_AUDIO_GUARD_POLICY="${project_root}/moe_expert_budget/configs/arc_audio_guard_static_topk_v1.json"
if [[ ! -x "${LORA_RUNTIME_DIR}/venv/bin/python" ]]; then
  echo "ABORT: shared venv missing at ${LORA_RUNTIME_DIR}/venv" >&2
  exit 1
fi
log="${project_root}/logs/arc_speed_opt/arc_audio_guard_bench_$(date +%Y%m%d_%H%M%S).log"
ln -sfn "${log}" "${project_root}/logs/arc_speed_opt/arc_audio_guard_bench_latest.log"
setsid bash "${project_root}/scripts/run_arc_audio_guard_benchmark.sh" >"${log}" 2>&1 < /dev/null &
echo "LAUNCHED pid=$! log=${log}"
sleep 3
head -12 "${log}" 2>/dev/null || true

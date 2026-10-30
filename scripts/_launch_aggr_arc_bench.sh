#!/usr/bin/env bash
set -euo pipefail
proj=/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner
export LORA_RUNTIME_DIR="${proj}/.omni_lora_runtime"
export MAX_STEPS="${MAX_STEPS:-25}"
export BENCHMARK_CASES="${BENCHMARK_CASES:-native aggr}"
export AGGR_ARC_MID_TOPK="${AGGR_ARC_MID_TOPK:-4}"
export AGGR_ARC_LAYER_START="${AGGR_ARC_LAYER_START:-10}"
export AGGR_ARC_LAYER_END="${AGGR_ARC_LAYER_END:-34}"
if [[ ! -x "${LORA_RUNTIME_DIR}/venv/bin/python" ]]; then
  echo "ABORT: shared venv missing at ${LORA_RUNTIME_DIR}/venv" >&2; exit 1
fi
log="${proj}/logs/arc_speed_opt/aggr_arc_bench_$(date +%Y%m%d_%H%M%S).log"
ln -sf "${log}" "${proj}/logs/arc_speed_opt/aggr_arc_bench_latest.log"
setsid bash "${proj}/scripts/run_aggressive_arc_benchmark.sh" >"${log}" 2>&1 < /dev/null &
echo "LAUNCHED pid=$! log=${log}"
sleep 3; head -8 "${log}" 2>/dev/null || true

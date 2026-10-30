#!/usr/bin/env bash
# Fire the global-topk benchmark (baseline top-8 vs topk4) detached, using the
# reusable shared-mount runtime.  Logs to shared mount for polling.
set -euo pipefail
proj=/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner
export LORA_RUNTIME_DIR="${proj}/.omni_lora_runtime"
export BENCHMARK_ROOT="${proj}/models/moe_global_topk_benchmark_h20_seed42"
export MAX_STEPS="${MAX_STEPS:-25}"
export BENCHMARK_CASES="${BENCHMARK_CASES:-baseline 4}"
# Guard: refuse to start if the shared venv is not actually usable yet.
if [[ ! -x "${LORA_RUNTIME_DIR}/venv/bin/python" ]]; then
  echo "ABORT: shared venv missing at ${LORA_RUNTIME_DIR}/venv" >&2; exit 1
fi
log="${proj}/logs/arc_speed_opt/global_topk_bench_$(date +%Y%m%d_%H%M%S).log"
ln -sf "${log}" "${proj}/logs/arc_speed_opt/global_topk_bench_latest.log"
setsid bash "${proj}/scripts/run_global_topk_benchmark.sh" >"${log}" 2>&1 < /dev/null &
echo "LAUNCHED pid=$! log=${log}"
sleep 3; head -8 "${log}" 2>/dev/null || true

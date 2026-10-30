#!/usr/bin/env bash
# Compare ARC-SCEB's legacy dense router mask with the native sparse rewrite
# on one H20.  The two arms intentionally differ only in
# ARC_SCEB_ROUTER_MODE; compact dispatch remains disabled.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${LORA_RUNTIME_DIR:-/tmp/omnivideo_100k_qwen3_omni_lora}"
benchmark_root="${BENCHMARK_ROOT:-${project_root}/models/arc_sceb_router_benchmark_h20_seed42_20260807}"
seed="${SEED:-42}"
max_train_steps="${MAX_STEPS:-25}"
master_port_base="${BENCHMARK_MASTER_PORT:-25380}"
benchmark_cases="${BENCHMARK_CASES:-dense sparse}"

if ! [[ "${max_train_steps}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'MAX_STEPS must be a positive integer, got: %s\n' "${max_train_steps}" >&2
  exit 2
fi
if ! [[ "${master_port_base}" =~ ^[0-9]+$ ]] || (( master_port_base < 1024 || master_port_base > 65534 )); then
  printf 'BENCHMARK_MASTER_PORT must leave room for two unprivileged ports, got: %s\n' \
    "${master_port_base}" >&2
  exit 2
fi

runner="${project_root}/scripts/run_omnivideo_100k_qwen3_omni_lora.sh"
router_plugin="${project_root}/scripts/megatron_arc_sceb_router.py"
optimizer_plugin="${project_root}/scripts/megatron_single_gpu_torch_adamw.py"
policy="${project_root}/moe_expert_budget/configs/arc_video_boundary_audio_floor6_full_v1.json"
if [[ ! -x "${runtime_root}/venv/bin/python" || ! -f "${runner}" || ! -f "${router_plugin}" || \
      ! -f "${optimizer_plugin}" || ! -f "${policy}" ]]; then
  printf 'Missing H20 runtime, runner, router plugin, optimizer plugin, or policy.\n' >&2
  exit 1
fi

# Preserve the proven single-H20 recipe.  In particular, micro batch remains
# one (inside the shared runner), while global batch sixteen gives the original
# accumulation behavior and keeps peak memory under the H20 budget.
common_env=(
  "LORA_RUNTIME_DIR=${runtime_root}"
  "NPROC_PER_NODE=1"
  "CUDA_VISIBLE_DEVICES=0"
  "EXPERT_MODEL_PARALLEL_SIZE=1"
  "GLOBAL_BATCH_SIZE=16"
  "USE_DISTRIBUTED_OPTIMIZER=false"
  "SEED=${seed}"
  "DATA_SEED=${seed}"
  "FINETUNE=true"
  "ADD_VERSION=true"
  "TRAIN_DATASET=${project_root}/ft_data/omnivideo_100k_lora_seed42_train.jsonl"
  "VAL_DATASET=${project_root}/ft_data/omnivideo_100k_lora_seed42_val.jsonl"
  "SPLIT_DATASET_RATIO=0"
  "MAX_STEPS=${max_train_steps}"
  "EVAL_STEPS=${EVAL_STEPS:-${max_train_steps}}"
  "SAVE_STEPS=${SAVE_STEPS:-${max_train_steps}}"
  "LOGGING_STEPS=${LOGGING_STEPS:-5}"
)

run_case() {
  local mode="$1"
  local ordinal="$2"
  local run_name="arc_sceb_${mode}"
  local external_plugins="${optimizer_plugin}"
  local -a case_env=()
  if [[ "${mode}" != "baseline" ]]; then
    external_plugins+=":${router_plugin}"
    case_env+=(
      "ARC_SCEB_POLICY=${policy}"
      "ARC_SCEB_DEBUG_EVERY=${ARC_SCEB_DEBUG_EVERY:-0}"
      "ARC_SCEB_ROUTER_MODE=${mode}"
    )
    # Optional opt-in profiler / experimental plugin, appended after the router
    # so it can wrap the already-installed ARC functions.  No effect unless set.
    if [[ -n "${ARC_SCEB_EXTRA_PLUGIN:-}" && -f "${ARC_SCEB_EXTRA_PLUGIN}" ]]; then
      external_plugins+=":${ARC_SCEB_EXTRA_PLUGIN}"
      case_env+=(
        "ARC_SCEB_PROFILE=${ARC_SCEB_PROFILE:-0}"
        "ARC_SCEB_PROFILE_EVERY=${ARC_SCEB_PROFILE_EVERY:-50}"
        "ARC_SCEB_PROFILE_WARMUP=${ARC_SCEB_PROFILE_WARMUP:-40}"
        "ARC_SCEB_SORT_MODE=${ARC_SCEB_SORT_MODE:-full}"
      )
    fi
  fi
  if [[ "${mode}" == "baseline" ]]; then
    run_name="baseline"
  elif [[ "${mode}" != "dense" && "${mode}" != "sparse" ]]; then
    printf 'BENCHMARK_CASES only accepts baseline, dense, or sparse; got: %s\n' "${mode}" >&2
    exit 2
  fi
  local output_dir="${benchmark_root}/${run_name}"
  local master_port="$((master_port_base + ordinal))"

  printf '\n===== ARC-SCEB %s router benchmark: %s steps, seed=%s =====\n' \
    "${mode}" "${max_train_steps}" "${seed}"
  printf 'MASTER_PORT=%s OUTPUT_DIR=%s\n' "${master_port}" "${output_dir}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits
  env "${common_env[@]}" \
    "RUN_NAME=${run_name}" \
    "OUTPUT_DIR=${output_dir}" \
    "MASTER_PORT=${master_port}" \
    "EXTERNAL_PLUGINS=${external_plugins}" \
    "${case_env[@]}" \
    bash "${runner}"
}

mkdir -p "${benchmark_root}"
case_ordinal=0
for benchmark_case in ${benchmark_cases}; do
  run_case "${benchmark_case}" "${case_ordinal}"
  case_ordinal="$((case_ordinal + 1))"
done
printf '\n===== Requested ARC-SCEB router benchmark cases completed =====\n'

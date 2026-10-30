#!/usr/bin/env bash
# Aggressive ARC-SCEB speed benchmark on one H20: native top-8 vs per-layer
# static top-k reduction in the middle band (megatron_aggressive_arc.py).
# Arms differ only in the AGGR_ARC_* env; the shared runner recipe is otherwise
# identical. Both use the reusable shared-mount runtime.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${LORA_RUNTIME_DIR:-${project_root}/.omni_lora_runtime}"
benchmark_root="${BENCHMARK_ROOT:-${project_root}/models/aggressive_arc_benchmark_h20_seed42}"
seed="${SEED:-42}"
max_train_steps="${MAX_STEPS:-25}"
master_port_base="${BENCHMARK_MASTER_PORT:-25480}"
# Arms: "native" = top-8 baseline; "aggr" = middle-band reduced top-k.
benchmark_cases="${BENCHMARK_CASES:-native aggr}"
# Aggressive-arm knobs (defaults hit eff_avg=6.0 => ~12% predicted).
aggr_mid_topk="${AGGR_ARC_MID_TOPK:-4}"
aggr_layer_start="${AGGR_ARC_LAYER_START:-10}"
aggr_layer_end="${AGGR_ARC_LAYER_END:-34}"

if ! [[ "${max_train_steps}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'MAX_STEPS must be a positive integer, got: %s\n' "${max_train_steps}" >&2
  exit 2
fi

runner="${project_root}/scripts/run_omnivideo_100k_qwen3_omni_lora.sh"
aggr_plugin="${project_root}/scripts/megatron_aggressive_arc.py"
optimizer_plugin="${project_root}/scripts/megatron_single_gpu_torch_adamw.py"
if [[ ! -x "${runtime_root}/venv/bin/python" || ! -f "${runner}" || ! -f "${aggr_plugin}" || \
      ! -f "${optimizer_plugin}" ]]; then
  printf 'Missing runtime venv, runner, aggressive-arc plugin, or optimizer plugin.\n' >&2
  exit 1
fi

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
  local run_name external_plugins="${optimizer_plugin}"
  local -a case_env=()
  if [[ "${mode}" == "native" ]]; then
    run_name="native"
  elif [[ "${mode}" == "aggr" ]]; then
    run_name="aggr_mid${aggr_mid_topk}_L${aggr_layer_start}-${aggr_layer_end}"
    external_plugins+=":${aggr_plugin}"
    case_env+=(
      "AGGR_ARC_MID_TOPK=${aggr_mid_topk}"
      "AGGR_ARC_LAYER_START=${aggr_layer_start}"
      "AGGR_ARC_LAYER_END=${aggr_layer_end}"
    )
  else
    printf 'BENCHMARK_CASES only accepts "native" or "aggr"; got: %s\n' "${mode}" >&2
    exit 2
  fi
  local output_dir="${benchmark_root}/${run_name}"
  local master_port="$((master_port_base + ordinal))"

  printf '\n===== Aggressive-ARC benchmark: %s, %s steps, seed=%s =====\n' \
    "${run_name}" "${max_train_steps}" "${seed}"
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
printf '\n===== Aggressive-ARC benchmark cases completed =====\n'

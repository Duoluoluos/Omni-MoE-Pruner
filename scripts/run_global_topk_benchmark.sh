#!/usr/bin/env bash
# Simplest speed experiment: bluntly lower the MoE router top-k for *every*
# token in *every* layer (native 8 -> 4) and measure throughput against the
# unmodified top-8 baseline on one H20.
#
# Unlike ARC-SCEB (per-token, video-only, middle-layer, mask-only) this is NOT
# modality aware and NOT a deployment policy -- it is a pure "fewer experts =>
# faster?" ablation.  The two arms differ only in MOE_GLOBAL_TOPK:
#   baseline : native top-8, only the single-GPU optimizer shim loaded.
#   topk4    : optimizer shim + megatron_global_topk.py with MOE_GLOBAL_TOPK=4.
#
# The reduction is *static* (both router.topk and config.moe_router_topk are
# lowered together), so the token dispatcher sizes num_out_tokens = tokens * 4
# at compile time with no GPU->CPU sync -- the property the removed compact
# dispatch lacked.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${LORA_RUNTIME_DIR:-/tmp/omnivideo_100k_qwen3_omni_lora}"
benchmark_root="${BENCHMARK_ROOT:-${project_root}/models/moe_global_topk_benchmark_h20_seed42}"
seed="${SEED:-42}"
max_train_steps="${MAX_STEPS:-25}"
master_port_base="${BENCHMARK_MASTER_PORT:-25460}"
benchmark_cases="${BENCHMARK_CASES:-baseline 4}"

if ! [[ "${max_train_steps}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'MAX_STEPS must be a positive integer, got: %s\n' "${max_train_steps}" >&2
  exit 2
fi
if ! [[ "${master_port_base}" =~ ^[0-9]+$ ]] || (( master_port_base < 1024 || master_port_base > 65534 )); then
  printf 'BENCHMARK_MASTER_PORT must leave room for the arm ports, got: %s\n' \
    "${master_port_base}" >&2
  exit 2
fi

runner="${project_root}/scripts/run_omnivideo_100k_qwen3_omni_lora.sh"
topk_plugin="${project_root}/scripts/megatron_global_topk.py"
optimizer_plugin="${project_root}/scripts/megatron_single_gpu_torch_adamw.py"
if [[ ! -x "${runtime_root}/venv/bin/python" || ! -f "${runner}" || ! -f "${topk_plugin}" || \
      ! -f "${optimizer_plugin}" ]]; then
  printf 'Missing H20 runtime, runner, global-topk plugin, or optimizer plugin.\n' >&2
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
  local run_name
  local external_plugins="${optimizer_plugin}"
  local -a case_env=()
  if [[ "${mode}" == "baseline" ]]; then
    run_name="baseline"
  elif [[ "${mode}" =~ ^[1-9][0-9]*$ ]]; then
    run_name="topk${mode}"
    external_plugins+=":${topk_plugin}"
    case_env+=("MOE_GLOBAL_TOPK=${mode}")
  else
    printf 'BENCHMARK_CASES only accepts "baseline" or positive integers; got: %s\n' "${mode}" >&2
    exit 2
  fi
  local output_dir="${benchmark_root}/${run_name}"
  local master_port="$((master_port_base + ordinal))"

  printf '\n===== MoE global top-k benchmark: %s, %s steps, seed=%s =====\n' \
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
printf '\n===== Requested MoE global top-k benchmark cases completed =====\n'

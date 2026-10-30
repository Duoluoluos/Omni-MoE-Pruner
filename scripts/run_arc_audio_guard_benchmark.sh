#!/usr/bin/env bash
# Paired single-H20 benchmark: native top-8 vs ARC audio-guard static schedule.
# Both arms use identical data, seed, optimizer, evaluation, and checkpointing.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${LORA_RUNTIME_DIR:-${project_root}/.omni_lora_runtime}"
benchmark_root="${BENCHMARK_ROOT:-${project_root}/models/arc_audio_guard_benchmark_h20_seed42}"
policy="${ARC_AUDIO_GUARD_POLICY:-${project_root}/moe_expert_budget/configs/arc_audio_guard_static_topk_v1.json}"
seed="${SEED:-42}"
max_train_steps="${MAX_STEPS:-25}"
master_port_base="${BENCHMARK_MASTER_PORT:-25680}"
benchmark_cases="${BENCHMARK_CASES:-native guard}"

if ! [[ "${max_train_steps}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'MAX_STEPS must be a positive integer, got: %s\n' "${max_train_steps}" >&2
  exit 2
fi

runner="${project_root}/scripts/run_omnivideo_100k_qwen3_omni_lora.sh"
guard_plugin="${project_root}/scripts/megatron_arc_audio_guard_static.py"
optimizer_plugin="${project_root}/scripts/megatron_single_gpu_torch_adamw.py"
for required in "${runtime_root}/venv/bin/python" "${runner}" "${guard_plugin}" \
                "${optimizer_plugin}" "${policy}"; do
  if [[ ! -e "${required}" ]]; then
    printf 'Missing required benchmark input: %s\n' "${required}" >&2
    exit 1
  fi
done

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
  "PYTHONUNBUFFERED=1"
)

run_case() {
  local mode="$1"
  local ordinal="$2"
  local run_name external_plugins="${optimizer_plugin}"
  local -a case_env=()
  if [[ "${mode}" == "native" ]]; then
    run_name="native"
  elif [[ "${mode}" == "guard" ]]; then
    run_name="arc_audio_guard_mid12-32_p22-24-25"
    external_plugins+="${external_plugins:+:}${guard_plugin}"
    case_env+=("ARC_AUDIO_GUARD_POLICY=${policy}")
  else
    printf 'BENCHMARK_CASES only accepts "native" or "guard"; got: %s\n' "${mode}" >&2
    exit 2
  fi
  local output_dir="${benchmark_root}/${run_name}"
  local master_port="$((master_port_base + ordinal))"

  printf '\n===== ARC-Audio-Guard benchmark: %s, %s steps, seed=%s =====\n' \
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
printf '\n===== ARC-Audio-Guard benchmark cases completed =====\n'

#!/usr/bin/env bash
# Run one clean baseline/ARC-SCEB LoRA pair on a single NVIDIA H20.
# This deliberately keeps ARC-SCEB mask-only; compact dispatch is unsupported.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${LORA_RUNTIME_DIR:-/tmp/omnivideo_100k_qwen3_omni_lora}"
pair_root="${PAIR_ROOT:-${project_root}/models/arc_sceb_mask_only_h20_seed42_20260804_native_adamw}"
seed="${SEED:-42}"
mode="full"
max_train_steps="${MAX_STEPS:-1000}"

# ms-swift delegates to torchrun whenever NPROC_PER_NODE is set.  Its implicit
# rendezvous port is 29500, which is shared by every job on a worker and can
# collide with an unrelated/stale launcher.  Each sequential arm must also use
# a distinct port: torchrun can retain its previous port briefly after the
# baseline exits.  `PAIR_MASTER_PORT`, when given, is the baseline port and
# ARC-SCEB receives the following port.
if [[ -n "${PAIR_MASTER_PORT:-}" ]]; then
  master_port_base="${PAIR_MASTER_PORT}"
else
  master_port_base=""
fi
if [[ -n "${master_port_base}" ]] && \
    { ! [[ "${master_port_base}" =~ ^[0-9]+$ ]] || (( master_port_base < 1024 || master_port_base > 65534 )); }; then
  printf 'PAIR_MASTER_PORT must leave room for two unprivileged ports, got: %s\n' "${master_port_base}" >&2
  exit 2
fi

if [[ "${1:-}" == "--smoke" ]]; then
  mode="smoke"
elif [[ $# -gt 0 ]]; then
  printf 'Usage: %s [--smoke]\n' "$0" >&2
  exit 2
fi

if [[ ! -x "${runtime_root}/venv/bin/python" ]]; then
  printf 'Missing LoRA environment: %s\n' "${runtime_root}/venv" >&2
  exit 1
fi

runner="${project_root}/scripts/run_omnivideo_100k_qwen3_omni_lora.sh"
plugin="${project_root}/scripts/megatron_arc_sceb_router.py"
optimizer_plugin="${project_root}/scripts/megatron_single_gpu_torch_adamw.py"
policy="${project_root}/moe_expert_budget/configs/arc_video_boundary_audio_floor6_full_v1.json"
if [[ ! -f "${runner}" || ! -f "${plugin}" || ! -f "${optimizer_plugin}" || ! -f "${policy}" ]]; then
  printf 'Missing runner, ARC plugin, or policy.\n' >&2
  exit 1
fi

# A single H20 owns all experts, so EP and process count must both be one.
# Peak activation memory is governed by micro-batch one; global batch sixteen
# only changes gradient accumulation and preserves the original recipe.
global_batch_size=16
if [[ "${mode}" == "smoke" ]]; then
  global_batch_size=2
fi
common_env=(
  "LORA_RUNTIME_DIR=${runtime_root}"
  "NPROC_PER_NODE=1"
  "CUDA_VISIBLE_DEVICES=0"
  "EXPERT_MODEL_PARALLEL_SIZE=1"
  "GLOBAL_BATCH_SIZE=${global_batch_size}"
  "USE_DISTRIBUTED_OPTIMIZER=false"
  "SEED=${seed}"
  "DATA_SEED=${seed}"
  "FINETUNE=true"
  "ADD_VERSION=true"
)
if [[ "${mode}" == "full" ]]; then
  common_env+=(
    "TRAIN_DATASET=${project_root}/ft_data/omnivideo_100k_lora_seed42_train.jsonl"
    "VAL_DATASET=${project_root}/ft_data/omnivideo_100k_lora_seed42_val.jsonl"
    "SPLIT_DATASET_RATIO=0"
    "MAX_STEPS=${max_train_steps}"
    "EVAL_STEPS=${EVAL_STEPS:-250}"
    "SAVE_STEPS=${SAVE_STEPS:-250}"
  )
fi

run_case() {
  local name="$1"
  local output_dir="${pair_root}/${name}"
  local case_master_port
  if [[ -n "${master_port_base}" ]]; then
    if [[ "${name}" == "baseline" ]]; then
      case_master_port="${master_port_base}"
    else
      case_master_port="$((master_port_base + 1))"
    fi
  else
    local case_port_hash
    case_port_hash="$(printf '%s' "${pair_root}:${seed}:${name}" | cksum | awk '{print $1}')"
    case_master_port="$((20000 + case_port_hash % 20000))"
  fi
  local -a case_env=(
    "RUN_NAME=${name}"
    "OUTPUT_DIR=${output_dir}"
    "MASTER_PORT=${case_master_port}"
  )
  local external_plugins="${optimizer_plugin}"
  if [[ "${name}" == "arc_sceb" ]]; then
    external_plugins+=":${plugin}"
    case_env+=(
      "ARC_SCEB_POLICY=${policy}"
    )
  fi
  case_env+=("EXTERNAL_PLUGINS=${external_plugins}")

  printf '\n===== %s (%s, seed=%s) =====\n' "${name}" "${mode}" "${seed}"
  printf 'Using per-arm torchrun MASTER_PORT=%s.\n' "${case_master_port}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits
  if [[ "${mode}" == "smoke" ]]; then
    env "${common_env[@]}" "${case_env[@]}" bash "${runner}" --smoke
  else
    env "${common_env[@]}" "${case_env[@]}" bash "${runner}"
  fi
}

mkdir -p "${pair_root}"
run_case baseline
run_case arc_sceb
printf '\n===== H20 paired %s run completed =====\n' "${mode}"

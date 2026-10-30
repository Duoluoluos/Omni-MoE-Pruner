#!/usr/bin/env bash
# Restart only the ARC-SCEB arm after a launcher-level failure.
# The paired baseline must already have completed; this never resumes it.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${LORA_RUNTIME_DIR:-/tmp/omnivideo_100k_qwen3_omni_lora}"
pair_root="${PAIR_ROOT:-${project_root}/models/arc_sceb_mask_only_h20_seed42_20260804_iter500_persistent_cache}"
runner="${project_root}/scripts/run_omnivideo_100k_qwen3_omni_lora.sh"
optimizer_plugin="${project_root}/scripts/megatron_single_gpu_torch_adamw.py"
arc_plugin="${project_root}/scripts/megatron_arc_sceb_router.py"
policy="${project_root}/moe_expert_budget/configs/arc_video_boundary_audio_floor6_full_v1.json"
session_name="${TMUX_SESSION_NAME:-omnivideo_lora_h20_seed42_arc_retry}"
master_port="${ARC_MASTER_PORT:-24763}"
log_dir="${pair_root}/launcher"
log_file="${log_dir}/arc_sceb_retry.log"

if [[ ! -x "${runtime_root}/venv/bin/python" || ! -f "${runner}" || ! -f "${optimizer_plugin}" || \
      ! -f "${arc_plugin}" || ! -f "${policy}" ]]; then
  printf 'Missing H20 runtime, runner, ARC plugin, optimizer plugin, or policy.\n' >&2
  exit 1
fi
if ! [[ "${master_port}" =~ ^[0-9]+$ ]] || (( master_port < 1024 || master_port > 65535 )); then
  printf 'ARC_MASTER_PORT must be an unprivileged TCP port, got: %s\n' "${master_port}" >&2
  exit 2
fi
if ! command -v tmux >/dev/null 2>&1; then
  printf 'tmux is required to keep the ARC retry alive after logout.\n' >&2
  exit 1
fi

mkdir -p "${log_dir}" "/tmp/tmux-$(id -u)"
unset TMUX_TMPDIR || true
if tmux has-session -t "${session_name}" 2>/dev/null; then
  printf 'ARC retry is already active in tmux session %s. Log: %s\n' "${session_name}" "${log_file}"
  exit 0
fi

printf -v q_runner '%q' "${runner}"
printf -v q_log '%q' "${log_file}"
printf -v q_runtime '%q' "${runtime_root}"
printf -v q_output '%q' "${pair_root}/arc_sceb"
printf -v q_train '%q' "${project_root}/ft_data/omnivideo_100k_lora_seed42_train.jsonl"
printf -v q_val '%q' "${project_root}/ft_data/omnivideo_100k_lora_seed42_val.jsonl"
printf -v q_policy '%q' "${policy}"
printf -v q_plugins '%q' "${optimizer_plugin}:${arc_plugin}"

tmux new-session -d -s "${session_name}" \
  "exec env LORA_RUNTIME_DIR=${q_runtime} NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 EXPERT_MODEL_PARALLEL_SIZE=1 GLOBAL_BATCH_SIZE=16 USE_DISTRIBUTED_OPTIMIZER=false MASTER_PORT=${master_port} SEED=42 DATA_SEED=42 FINETUNE=true ADD_VERSION=true RUN_NAME=arc_sceb OUTPUT_DIR=${q_output} TRAIN_DATASET=${q_train} VAL_DATASET=${q_val} SPLIT_DATASET_RATIO=0 MAX_STEPS=500 EVAL_STEPS=250 SAVE_STEPS=250 ARC_SCEB_POLICY=${q_policy} EXTERNAL_PLUGINS=${q_plugins} bash ${q_runner} >${q_log} 2>&1"
printf 'Started ARC-SCEB retry in tmux session %s. Log: %s\n' "${session_name}" "${log_file}"

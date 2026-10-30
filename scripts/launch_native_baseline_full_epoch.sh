#!/usr/bin/env bash
# Launch the native top-8 baseline paired with the ARC-SCEB full-epoch run.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
worker_id="${WORKER_ID:-4181860}"
runtime_root="${LORA_RUNTIME_DIR:-${project_root}/.omni_lora_runtime}"
run_tag="${RUN_TAG:-20260821_worker${worker_id}}"
validation_root="${VALIDATION_ROOT:-${project_root}/models/native_baseline_full_epoch_h20_seed42_${run_tag}}"
output_dir="${validation_root}/baseline"
launcher_dir="${validation_root}/launcher"
log_file="${launcher_dir}/train.log"
manifest_file="${launcher_dir}/manifest.txt"
session_name="${TMUX_SESSION_NAME:-native_baseline_epoch_${worker_id}}"
master_port="${MASTER_PORT:-25910}"
train_iters="${TRAIN_ITERS:-4106}"
save_steps="${SAVE_STEPS:-250}"
eval_steps="${EVAL_STEPS:-${train_iters}}"
worker_hard_expiry_cst="${WORKER_HARD_EXPIRY_CST:-2026-08-25T15:00:09+08:00}"

runner="${project_root}/scripts/run_omnivideo_100k_qwen3_omni_lora.sh"
optimizer_plugin="${project_root}/scripts/megatron_single_gpu_torch_adamw.py"
train_data="${project_root}/ft_data/omnivideo_100k_lora_seed42_train.jsonl"
val_data="${project_root}/ft_data/omnivideo_100k_lora_seed42_val.jsonl"
paired_arc_root="${project_root}/models/arc_audio_guard_full_epoch_h20_seed42_20260820_worker4177575"

for required in "${runtime_root}/venv/bin/python" "${runner}" "${optimizer_plugin}" \
                "${train_data}" "${val_data}"; do
  if [[ ! -e "${required}" ]]; then
    printf 'Missing required baseline input: %s\n' "${required}" >&2
    exit 1
  fi
done
if ! [[ "${train_iters}" =~ ^[1-9][0-9]*$ && "${save_steps}" =~ ^[1-9][0-9]*$ && \
        "${eval_steps}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'TRAIN_ITERS, SAVE_STEPS, and EVAL_STEPS must be positive integers.\n' >&2
  exit 2
fi
if ! command -v tmux >/dev/null 2>&1; then
  printf 'tmux is required for a durable background run.\n' >&2
  exit 1
fi
if [[ "${ARNOLD_WORKER_ID:-}" != "${worker_id}" ]]; then
  printf 'Refusing to launch on worker id %s; expected %s.\n' \
    "${ARNOLD_WORKER_ID:-unknown}" "${worker_id}" >&2
  exit 2
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  printf 'GPU already has a compute process; refusing to oversubscribe it.\n' >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits >&2
  exit 1
fi

mkdir -p "${launcher_dir}" "${output_dir}" "/tmp/tmux-$(id -u)" "${project_root}/logs/arc_speed_opt"
unset TMUX_TMPDIR || true
if tmux has-session -t "${session_name}" 2>/dev/null; then
  printf 'Baseline full-epoch run is already active in tmux session %s. Log: %s\n' \
    "${session_name}" "${log_file}"
  exit 0
fi
if find "${output_dir}" -mindepth 1 -print -quit | grep -q .; then
  printf 'Output directory is not empty; refusing to overwrite: %s\n' "${output_dir}" >&2
  exit 1
fi

{
  printf 'purpose=native top-8 baseline full-epoch wall-clock validation\n'
  printf 'worker_id=%s\n' "${worker_id}"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'launch_time=%s\n' "$(date --iso-8601=seconds)"
  printf 'worker_hard_expiry_cst=%s\n' "${worker_hard_expiry_cst}"
  printf 'train_iters=%s\n' "${train_iters}"
  printf 'global_batch_size=16\n'
  printf 'save_steps=%s\n' "${save_steps}"
  printf 'eval_steps=%s\n' "${eval_steps}"
  printf 'output_dir=%s\n' "${output_dir}"
  printf 'paired_arc_root=%s\n' "${paired_arc_root}"
  printf 'method=native_top8_no_arc_sceb_plugin\n'
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  sha256sum "${runner}" "${optimizer_plugin}" "${train_data}" "${val_data}"
} >"${manifest_file}"

printf -v q_runner '%q' "${runner}"
printf -v q_log '%q' "${log_file}"
printf -v q_runtime '%q' "${runtime_root}"
printf -v q_output '%q' "${output_dir}"
printf -v q_train '%q' "${train_data}"
printf -v q_val '%q' "${val_data}"
printf -v q_plugin '%q' "${optimizer_plugin}"

tmux new-session -d -s "${session_name}" \
  "exec env LORA_RUNTIME_DIR=${q_runtime} NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 EXPERT_MODEL_PARALLEL_SIZE=1 GLOBAL_BATCH_SIZE=16 USE_DISTRIBUTED_OPTIMIZER=false MASTER_PORT=${master_port} SEED=42 DATA_SEED=42 FINETUNE=true ADD_VERSION=false RUN_NAME=native_baseline_full_epoch OUTPUT_DIR=${q_output} TRAIN_DATASET=${q_train} VAL_DATASET=${q_val} SPLIT_DATASET_RATIO=0 MAX_STEPS=${train_iters} EVAL_STEPS=${eval_steps} SAVE_STEPS=${save_steps} SAVE_TOTAL_LIMIT=2 LOGGING_STEPS=5 NO_SAVE_OPTIM=false NO_SAVE_RNG=false EXTERNAL_PLUGINS=${q_plugin} PYTHONUNBUFFERED=1 bash ${q_runner} >${q_log} 2>&1"

ln -sfn "${log_file}" "${project_root}/logs/arc_speed_opt/native_baseline_full_epoch_latest.log"
printf 'LAUNCHED session=%s log=%s output=%s manifest=%s\n' \
  "${session_name}" "${log_file}" "${output_dir}" "${manifest_file}"

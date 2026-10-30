#!/usr/bin/env bash
# Resume the interrupted native top-8 full-epoch run from its latest durable checkpoint.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
worker_id="${WORKER_ID:-4193170}"
runtime_root="${LORA_RUNTIME_DIR:-${project_root}/.omni_lora_runtime}"
run_root="${RUN_ROOT:-${project_root}/models/native_baseline_full_epoch_h20_seed42_20260821_worker4181860}"
output_dir="${run_root}/baseline"
checkpoint_dir="${RESUME_CHECKPOINT:-${output_dir}/checkpoint-3250}"
launcher_dir="${run_root}/launcher"
log_file="${RESUME_LOG_FILE:-${launcher_dir}/resume_worker${worker_id}.log}"
manifest_file="${launcher_dir}/resume_worker${worker_id}_manifest.txt"
session_name="${TMUX_SESSION_NAME:-native_baseline_resume_${worker_id}}"
master_port="${MASTER_PORT:-25931}"
train_iters="${TRAIN_ITERS:-4106}"

runner="${project_root}/scripts/run_omnivideo_100k_qwen3_omni_lora.sh"
resume_preparer="${project_root}/scripts/prepare_omnivideo_100k_qwen3_omni_lora_resume.py"
optimizer_plugin="${project_root}/scripts/megatron_single_gpu_torch_adamw.py"
train_data="${project_root}/ft_data/omnivideo_100k_lora_seed42_train.jsonl"
val_data="${project_root}/ft_data/omnivideo_100k_lora_seed42_val.jsonl"

for required in "${runtime_root}/venv/bin/python" "${runner}" "${resume_preparer}" \
                "${optimizer_plugin}" "${train_data}" "${val_data}" \
                "${checkpoint_dir}/adapter_model.safetensors" \
                "${checkpoint_dir}/latest_checkpointed_iteration.txt"; do
  if [[ ! -e "${required}" ]]; then
    printf 'Missing required baseline-resume input: %s\n' "${required}" >&2
    exit 1
  fi
done

checkpoint_step="$(<"${checkpoint_dir}/latest_checkpointed_iteration.txt")"
if [[ "${checkpoint_step}" != "3250" ]]; then
  printf 'Refusing unexpected resume iteration %s from %s.\n' \
    "${checkpoint_step}" "${checkpoint_dir}" >&2
  exit 2
fi
if [[ -d "${output_dir}/checkpoint-${train_iters}" ]]; then
  printf 'Final checkpoint already exists: %s\n' "${output_dir}/checkpoint-${train_iters}"
  exit 0
fi
if [[ "${ARNOLD_WORKER_ID:-}" != "${worker_id}" ]]; then
  printf 'Refusing to launch on worker id %s; expected %s.\n' \
    "${ARNOLD_WORKER_ID:-unknown}" "${worker_id}" >&2
  exit 2
fi
if ! command -v tmux >/dev/null 2>&1; then
  printf 'tmux is required for a durable background resume.\n' >&2
  exit 1
fi

mkdir -p "${launcher_dir}" "/tmp/tmux-$(id -u)"
unset TMUX_TMPDIR || true
if tmux has-session -t "${session_name}" 2>/dev/null; then
  printf 'Baseline resume already active in tmux session %s. Log: %s\n' \
    "${session_name}" "${log_file}"
  exit 0
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  printf 'GPU already has a compute process; refusing to oversubscribe it.\n' >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits >&2
  exit 1
fi

"${runtime_root}/venv/bin/python" "${resume_preparer}" \
  --output-dir "${output_dir}" \
  --checkpoint "${checkpoint_dir}"

{
  printf 'purpose=native top-8 full-epoch exact-state resume\n'
  printf 'worker_id=%s\n' "${worker_id}"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'resume_time=%s\n' "$(date --iso-8601=seconds)"
  printf 'checkpoint=%s\n' "${checkpoint_dir}"
  printf 'checkpoint_step=%s\n' "${checkpoint_step}"
  printf 'train_iters=%s\n' "${train_iters}"
  printf 'global_batch_size=16\n'
  printf 'state_restore=iteration,optimizer,scheduler,rng\n'
  printf 'output_dir=%s\n' "${output_dir}"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  sha256sum "${runner}" "${resume_preparer}" "${optimizer_plugin}" \
    "${train_data}" "${val_data}"
} >"${manifest_file}"

printf -v q_runner '%q' "${runner}"
printf -v q_log '%q' "${log_file}"
printf -v q_runtime '%q' "${runtime_root}"
printf -v q_output '%q' "${output_dir}"
printf -v q_checkpoint '%q' "${checkpoint_dir}"
printf -v q_train '%q' "${train_data}"
printf -v q_val '%q' "${val_data}"
printf -v q_plugin '%q' "${optimizer_plugin}"

tmux new-session -d -s "${session_name}" \
  "exec env LORA_RUNTIME_DIR=${q_runtime} NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 EXPERT_MODEL_PARALLEL_SIZE=1 GLOBAL_BATCH_SIZE=16 USE_DISTRIBUTED_OPTIMIZER=false MASTER_PORT=${master_port} SEED=42 DATA_SEED=42 FINETUNE=false ADD_VERSION=false RUN_NAME=native_baseline_full_epoch OUTPUT_DIR=${q_output} TRAIN_DATASET=${q_train} VAL_DATASET=${q_val} SPLIT_DATASET_RATIO=0 MAX_STEPS=${train_iters} EVAL_STEPS=${train_iters} SAVE_STEPS=250 SAVE_TOTAL_LIMIT=2 LOGGING_STEPS=5 MCORE_ADAPTER=${q_checkpoint} NO_LOAD_OPTIM=false NO_LOAD_RNG=false NO_SAVE_OPTIM=false NO_SAVE_RNG=false EXTERNAL_PLUGINS=${q_plugin} PYTHONUNBUFFERED=1 bash ${q_runner} >>${q_log} 2>&1"

ln -sfn "${log_file}" "${project_root}/logs/arc_speed_opt/native_baseline_full_epoch_resume_latest.log"
printf 'RESUMED session=%s log=%s output=%s manifest=%s checkpoint=%s\n' \
  "${session_name}" "${log_file}" "${output_dir}" "${manifest_file}" "${checkpoint_dir}"

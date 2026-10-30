#!/usr/bin/env bash
# Wait for the worker-local environment, then gate a one-seed H20 pair through
# check-env and full-length smoke before launching the two capped training arms.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
setup_session="${SETUP_SESSION_NAME:-omnivideo_lora_h20_setup}"
pair_runner="${project_root}/scripts/run_omnivideo_100k_qwen3_omni_lora_h20_pair.sh"
base_runner="${project_root}/scripts/run_omnivideo_100k_qwen3_omni_lora.sh"
skip_precheck="${SKIP_PRECHECK:-false}"

if [[ ! -f "${pair_runner}" || ! -f "${base_runner}" ]]; then
  printf 'Missing H20 pair or base runner.\n' >&2
  exit 1
fi

if [[ "${skip_precheck}" != "true" ]]; then
  printf 'Waiting for setup tmux session %s to finish.\n' "${setup_session}"
  while tmux has-session -t "${setup_session}" 2>/dev/null; do
    sleep 15
  done

  # A failed setup causes this import/CLI gate to fail before any GPU allocation.
  NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 EXPERT_MODEL_PARALLEL_SIZE=1 GLOBAL_BATCH_SIZE=16 SEED=42 DATA_SEED=42 bash "${base_runner}" --check-env

  # The smoke uses the same max length, video token cap, micro-batch, precision,
  # recomputation, EP, and ARC policy as the capped training run. Do not bypass it
  # on a fresh launch; it may be skipped only after both arms have passed here.
  bash "${pair_runner}" --smoke
else
  printf 'Skipping precheck and smoke because both arms already passed this worker gate.\n'
fi
bash "${pair_runner}"

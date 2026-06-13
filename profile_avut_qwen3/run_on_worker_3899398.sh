#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/bn/isp-traindata-lf-1/wq/OmniZip}"
WORKER_ID="${WORKER_ID:-3899398}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/logs/avut_qwen3_profile/${TS}}"
BACKGROUND="${BACKGROUND:-1}"

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3-Omni-30B-A3B-Instruct}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/eval_data/AVUTBenchmark/AV_Human_data.json}"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/eval_data/AVUTBenchmark}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
ROUTE_DUMP="${ROUTE_DUMP:-head}"
ROUTE_HEAD_TOKENS="${ROUTE_HEAD_TOKENS:-64}"
TORCH_PROFILE_SAMPLES="${TORCH_PROFILE_SAMPLES:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"

mkdir -p "${OUT_DIR}"

args=(
  --model-path "${MODEL_PATH}"
  --data-path "${DATA_PATH}"
  --video-dir "${VIDEO_DIR}"
  --output-dir "${OUT_DIR}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --route-dump "${ROUTE_DUMP}"
  --route-head-tokens "${ROUTE_HEAD_TOKENS}"
  --torch-profile-samples "${TORCH_PROFILE_SAMPLES}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --device-map "${DEVICE_MAP}"
)

if [[ -n "${LIMIT:-}" ]]; then
  args+=(--limit "${LIMIT}")
fi
if [[ -n "${START_INDEX:-}" ]]; then
  args+=(--start-index "${START_INDEX}")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  args+=(--resume)
fi
if [[ "${NO_AUDIO_IN_VIDEO:-0}" == "1" ]]; then
  args+=(--no-audio-in-video)
fi
if [[ -n "${VIDEO_FPS:-}" ]]; then
  args+=(--video-fps "${VIDEO_FPS}")
fi
if [[ -n "${VIDEO_MAX_FRAMES:-}" ]]; then
  args+=(--video-max-frames "${VIDEO_MAX_FRAMES}")
fi

log_file="${OUT_DIR}/worker_${WORKER_ID}.log"
cmd=(NO_COLOR=1 TERM=dumb mlx worker login "${WORKER_ID}" -- bash "${PROJECT_ROOT}/profile_avut_qwen3/worker_entry.sh" "${args[@]}")

printf "%q " "${cmd[@]}" > "${OUT_DIR}/launch_command.txt"
printf "\n" >> "${OUT_DIR}/launch_command.txt"

if [[ "${BACKGROUND}" == "1" ]]; then
  setsid env NO_COLOR=1 TERM=dumb mlx worker login "${WORKER_ID}" -- \
    bash "${PROJECT_ROOT}/profile_avut_qwen3/worker_entry.sh" "${args[@]}" \
    > "${log_file}" 2>&1 < /dev/null &
  launcher_pid=$!
  disown
  echo "${launcher_pid}" > "${OUT_DIR}/launcher.pid"
  echo "Started AVUT Qwen3-Omni profile on worker ${WORKER_ID}"
  echo "launcher_pid=${launcher_pid}"
  echo "output_dir=${OUT_DIR}"
  echo "log=${log_file}"
else
  exec env NO_COLOR=1 TERM=dumb mlx worker login "${WORKER_ID}" -- \
    bash "${PROJECT_ROOT}/profile_avut_qwen3/worker_entry.sh" "${args[@]}"
fi

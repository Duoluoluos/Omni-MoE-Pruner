#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/bn/isp-traindata-lf-1/wq/OmniZip}"
WORKER_ID="${WORKER_ID:-3898754}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/logs/omnivideobench_qwen3_twopass/${TS}}"
BACKGROUND="${BACKGROUND:-1}"

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3-Omni-30B-A3B-Instruct}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/eval_data/OmniVideoBench/data.json}"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/eval_data/OmniVideoBench}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-30}"
SEGMENT_CONTEXT_SECONDS="${SEGMENT_CONTEXT_SECONDS:-8}"
TOP_SEGMENTS="${TOP_SEGMENTS:-2}"
SEGMENT_MAX_FRAMES="${SEGMENT_MAX_FRAMES:-128}"
AUDIO_ENERGY_SR="${AUDIO_ENERGY_SR:-8000}"

mkdir -p "${OUT_DIR}"

args=(
  --model-path "${MODEL_PATH}"
  --data-path "${DATA_PATH}"
  --video-dir "${VIDEO_DIR}"
  --output-dir "${OUT_DIR}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --device-map "${DEVICE_MAP}"
  --segment-seconds "${SEGMENT_SECONDS}"
  --segment-context-seconds "${SEGMENT_CONTEXT_SECONDS}"
  --top-segments "${TOP_SEGMENTS}"
  --segment-max-frames "${SEGMENT_MAX_FRAMES}"
  --audio-energy-sr "${AUDIO_ENERGY_SR}"
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
if [[ "${NO_AUDIO_ENERGY:-0}" == "1" ]]; then
  args+=(--no-audio-energy)
fi
if [[ "${NO_ORACLE_REASONING_TIMES:-0}" == "1" ]]; then
  args+=(--no-oracle-reasoning-times)
fi
if [[ "${RETRIEVAL_ONLY:-0}" == "1" ]]; then
  args+=(--retrieval-only)
fi

log_file="${OUT_DIR}/worker_${WORKER_ID}.log"
cmd=(NO_COLOR=1 TERM=dumb mlx worker login "${WORKER_ID}" -- bash "${PROJECT_ROOT}/profile_omnivideobench_qwen3/worker_entry_twopass.sh" "${args[@]}")

printf "%q " "${cmd[@]}" > "${OUT_DIR}/launch_command.txt"
printf "\n" >> "${OUT_DIR}/launch_command.txt"

if [[ "${BACKGROUND}" == "1" ]]; then
  setsid env NO_COLOR=1 TERM=dumb mlx worker login "${WORKER_ID}" -- \
    bash "${PROJECT_ROOT}/profile_omnivideobench_qwen3/worker_entry_twopass.sh" "${args[@]}" \
    > "${log_file}" 2>&1 < /dev/null &
  launcher_pid=$!
  disown
  echo "${launcher_pid}" > "${OUT_DIR}/launcher.pid"
  echo "Started OmniVideoBench Qwen3-Omni two-pass run on worker ${WORKER_ID}"
  echo "launcher_pid=${launcher_pid}"
  echo "output_dir=${OUT_DIR}"
  echo "log=${log_file}"
else
  exec env NO_COLOR=1 TERM=dumb mlx worker login "${WORKER_ID}" -- \
    bash "${PROJECT_ROOT}/profile_omnivideobench_qwen3/worker_entry_twopass.sh" "${args[@]}"
fi

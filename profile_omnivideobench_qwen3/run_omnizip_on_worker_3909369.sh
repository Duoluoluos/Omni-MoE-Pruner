#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/bn/isp-traindata-lf-1/wq/OmniZip}"
WORKER_ID="${WORKER_ID:-3909369}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/logs/omnivideobench_qwen3_omnizip/${TS}}"
BACKGROUND="${BACKGROUND:-1}"

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3-Omni-30B-A3B-Instruct}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/eval_data/OmniVideoBench/data.json}"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/eval_data/OmniVideoBench}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
ROUTE_DUMP="${ROUTE_DUMP:-none}"
ROUTE_HEAD_TOKENS="${ROUTE_HEAD_TOKENS:-64}"
TRAJECTORY_BINS="${TRAJECTORY_BINS:-16}"
TRAJECTORY_TOP_EXPERTS="${TRAJECTORY_TOP_EXPERTS:-8}"
TORCH_PROFILE_SAMPLES="${TORCH_PROFILE_SAMPLES:-0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
OMNIZIP_RHO_AUDIO="${OMNIZIP_RHO_AUDIO:-0.3}"
OMNIZIP_RHO_VIDEO="${OMNIZIP_RHO_VIDEO:-0.6}"
OMNIZIP_G="${OMNIZIP_G:-3}"
OMNIZIP_CONTEXTUAL_RATIO="${OMNIZIP_CONTEXTUAL_RATIO:-0.05}"

mkdir -p "${OUT_DIR}"

args=(
  --model-path "${MODEL_PATH}"
  --data-path "${DATA_PATH}"
  --video-dir "${VIDEO_DIR}"
  --output-dir "${OUT_DIR}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --route-dump "${ROUTE_DUMP}"
  --route-head-tokens "${ROUTE_HEAD_TOKENS}"
  --trajectory-bins "${TRAJECTORY_BINS}"
  --trajectory-top-experts "${TRAJECTORY_TOP_EXPERTS}"
  --torch-profile-samples "${TORCH_PROFILE_SAMPLES}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --device-map "${DEVICE_MAP}"
  --use-omnizip
  --omnizip-rho-audio "${OMNIZIP_RHO_AUDIO}"
  --omnizip-rho-video "${OMNIZIP_RHO_VIDEO}"
  --omnizip-g "${OMNIZIP_G}"
  --omnizip-contextual-ratio "${OMNIZIP_CONTEXTUAL_RATIO}"
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
cmd=(NO_COLOR=1 TERM=dumb mlx worker login "${WORKER_ID}" -- bash "${PROJECT_ROOT}/profile_omnivideobench_qwen3/worker_entry_omnizip.sh" "${args[@]}")

printf "%q " "${cmd[@]}" > "${OUT_DIR}/launch_command.txt"
printf "\n" >> "${OUT_DIR}/launch_command.txt"
{
  echo "worker_id=${WORKER_ID}"
  echo "output_dir=${OUT_DIR}"
  echo "model_path=${MODEL_PATH}"
  echo "data_path=${DATA_PATH}"
  echo "video_dir=${VIDEO_DIR}"
  echo "omnizip_rho_audio=${OMNIZIP_RHO_AUDIO}"
  echo "omnizip_rho_video=${OMNIZIP_RHO_VIDEO}"
  echo "omnizip_g=${OMNIZIP_G}"
  echo "omnizip_contextual_ratio=${OMNIZIP_CONTEXTUAL_RATIO}"
  echo "route_dump=${ROUTE_DUMP}"
  echo "torch_profile_samples=${TORCH_PROFILE_SAMPLES}"
} > "${OUT_DIR}/launch_env.txt"

if [[ "${BACKGROUND}" == "1" ]]; then
  setsid env NO_COLOR=1 TERM=dumb mlx worker login "${WORKER_ID}" -- \
    env RUN_OUTPUT_DIR="${OUT_DIR}" KEEPALIVE_LOG="${OUT_DIR}/keepalive.log" START_KEEPALIVE="${START_KEEPALIVE:-1}" \
      TARGET_UTIL="${TARGET_UTIL:-0.3}" MEM_GB="${MEM_GB:-0.6}" DEVICE="${DEVICE:-0}" \
      bash "${PROJECT_ROOT}/profile_omnivideobench_qwen3/worker_entry_omnizip.sh" "${args[@]}" \
    > "${log_file}" 2>&1 < /dev/null &
  launcher_pid=$!
  disown
  echo "${launcher_pid}" > "${OUT_DIR}/launcher.pid"
  echo "Started OmniVideoBench OmniZip Qwen3-Omni run on worker ${WORKER_ID}"
  echo "launcher_pid=${launcher_pid}"
  echo "output_dir=${OUT_DIR}"
  echo "log=${log_file}"
else
  exec env NO_COLOR=1 TERM=dumb mlx worker login "${WORKER_ID}" -- \
    env RUN_OUTPUT_DIR="${OUT_DIR}" KEEPALIVE_LOG="${OUT_DIR}/keepalive.log" START_KEEPALIVE="${START_KEEPALIVE:-1}" \
      TARGET_UTIL="${TARGET_UTIL:-0.3}" MEM_GB="${MEM_GB:-0.6}" DEVICE="${DEVICE:-0}" \
      bash "${PROJECT_ROOT}/profile_omnivideobench_qwen3/worker_entry_omnizip.sh" "${args[@]}"
fi

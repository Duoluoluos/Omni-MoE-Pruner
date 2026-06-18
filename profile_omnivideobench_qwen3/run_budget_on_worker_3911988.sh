#!/usr/bin/env bash
# Launch the Fixed Top-M Mask quality-floor A/B on worker 3911988.
#
# Steps (sequential on the same worker):
#   1) baseline run via worker_entry.sh (no budget)
#   2) treatment run via worker_entry_budget.sh (top_m=4, layers [12,32), renorm)
#   3) compare_ab.py paired report
#
# By default this runs the SMOKE size (--limit 50). Set LIMIT="" to run the full
# 411 samples. Both runs share the same --limit so paired indices align.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner}"
WORKER_ID="${WORKER_ID:-3911988}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
BACKGROUND="${BACKGROUND:-1}"

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3-Omni-30B-A3B-Instruct}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/eval_data/OmniVideoBench/data.json}"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/eval_data/OmniVideoBench}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"

LIMIT="${LIMIT-50}"
START_INDEX="${START_INDEX-0}"

# Lightweight profile dump for compare_ab.py.
ROUTE_DUMP="${ROUTE_DUMP:-none}"
ROUTE_HEAD_TOKENS="${ROUTE_HEAD_TOKENS:-0}"
TRAJECTORY_BINS="${TRAJECTORY_BINS:-4}"
TRAJECTORY_TOP_EXPERTS="${TRAJECTORY_TOP_EXPERTS:-4}"
TORCH_PROFILE_SAMPLES="${TORCH_PROFILE_SAMPLES:-0}"

BUDGET_CONFIG="${BUDGET_CONFIG:-${PROJECT_ROOT}/moe_expert_budget/configs/fixed_topm_default.json}"
TAG="${TAG:-topm4_renorm_smoke}"

BASELINE_DIR="${BASELINE_DIR:-${PROJECT_ROOT}/logs/omnivideobench_qwen3_profile/baseline_${TAG}_${TS}}"
TREATMENT_DIR="${TREATMENT_DIR:-${PROJECT_ROOT}/logs/omnivideobench_qwen3_budget_topm/${TAG}_${TS}}"
ROOT_LOG_DIR="${ROOT_LOG_DIR:-${PROJECT_ROOT}/logs/omnivideobench_qwen3_budget_topm/${TAG}_${TS}_launcher}"
mkdir -p "${BASELINE_DIR}" "${TREATMENT_DIR}" "${ROOT_LOG_DIR}"

worker_log="${ROOT_LOG_DIR}/worker_${WORKER_ID}.log"
remote_script="${ROOT_LOG_DIR}/remote_run.sh"
ab_report="${TREATMENT_DIR}/ab_compare.json"

LIMIT_BASELINE=""
LIMIT_TREATMENT=""
if [[ -n "${LIMIT}" ]]; then
  LIMIT_BASELINE="--limit ${LIMIT}"
  LIMIT_TREATMENT="--limit ${LIMIT}"
fi
START_ARG=""
if [[ -n "${START_INDEX}" ]]; then
  START_ARG="--start-index ${START_INDEX}"
fi

# Build the remote script as a real file on the shared FS so mlx login -> ssh
# can just `bash <file>` it without inline-quoting hell.
cat > "${remote_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT}"
cd "\${PROJECT_ROOT}"

echo "[\$(date)] === step 1/3: baseline ==="
bash "\${PROJECT_ROOT}/profile_omnivideobench_qwen3/worker_entry.sh" \\
  --model-path "${MODEL_PATH}" \\
  --data-path "${DATA_PATH}" \\
  --video-dir "${VIDEO_DIR}" \\
  --output-dir "${BASELINE_DIR}" \\
  --max-new-tokens ${MAX_NEW_TOKENS} \\
  --route-dump ${ROUTE_DUMP} \\
  --route-head-tokens ${ROUTE_HEAD_TOKENS} \\
  --trajectory-bins ${TRAJECTORY_BINS} \\
  --trajectory-top-experts ${TRAJECTORY_TOP_EXPERTS} \\
  --torch-profile-samples ${TORCH_PROFILE_SAMPLES} \\
  --attn-implementation ${ATTN_IMPLEMENTATION} \\
  --device-map ${DEVICE_MAP} \\
  ${LIMIT_BASELINE} ${START_ARG}

echo "[\$(date)] === step 2/3: treatment (fixed top-m mask) ==="
bash "\${PROJECT_ROOT}/profile_omnivideobench_qwen3/worker_entry_budget.sh" \\
  --model-path "${MODEL_PATH}" \\
  --data-path "${DATA_PATH}" \\
  --video-dir "${VIDEO_DIR}" \\
  --output-dir "${TREATMENT_DIR}" \\
  --max-new-tokens ${MAX_NEW_TOKENS} \\
  --route-dump ${ROUTE_DUMP} \\
  --route-head-tokens ${ROUTE_HEAD_TOKENS} \\
  --trajectory-bins ${TRAJECTORY_BINS} \\
  --trajectory-top-experts ${TRAJECTORY_TOP_EXPERTS} \\
  --torch-profile-samples ${TORCH_PROFILE_SAMPLES} \\
  --attn-implementation ${ATTN_IMPLEMENTATION} \\
  --device-map ${DEVICE_MAP} \\
  --budget-config "${BUDGET_CONFIG}" \\
  ${LIMIT_TREATMENT} ${START_ARG}

echo "[\$(date)] === step 3/3: paired A/B compare ==="
"\${PROJECT_ROOT}/.venv/bin/python" \\
  "\${PROJECT_ROOT}/profile_omnivideobench_qwen3/compare_ab.py" \\
  --baseline "${BASELINE_DIR}" \\
  --treatment "${TREATMENT_DIR}" \\
  --output "${ab_report}"

echo "[\$(date)] === done ==="
echo "ab_report=${ab_report}"
EOF
chmod +x "${remote_script}"

if [[ "${BACKGROUND}" == "1" ]]; then
  setsid env NO_COLOR=1 TERM=dumb mlx worker login "${WORKER_ID}" -- \
    bash "${remote_script}" > "${worker_log}" 2>&1 < /dev/null &
  launcher_pid=$!
  disown
  echo "${launcher_pid}" > "${ROOT_LOG_DIR}/launcher.pid"
  echo "Started Fixed Top-M Mask A/B on worker ${WORKER_ID}"
  echo "launcher_pid=${launcher_pid}"
  echo "remote_script=${remote_script}"
  echo "baseline_dir=${BASELINE_DIR}"
  echo "treatment_dir=${TREATMENT_DIR}"
  echo "log=${worker_log}"
  echo "ab_report=${ab_report}"
else
  exec env NO_COLOR=1 TERM=dumb mlx worker login "${WORKER_ID}" -- \
    bash "${remote_script}"
fi

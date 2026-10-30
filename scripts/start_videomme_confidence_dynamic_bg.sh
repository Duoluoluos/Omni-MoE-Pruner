#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/bn/isp-traindata-lf-1/wq/OmniZip}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${PROJECT_ROOT}/logs/videomme_qwen3_budget_dynamic/${TS}}"

mkdir -p "${RUN_OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

export PROJECT_ROOT
export TS
export RUN_OUTPUT_DIR
nohup bash "${PROJECT_ROOT}/scripts/run_videomme_confidence_dynamic.sh" \
    > "${RUN_OUTPUT_DIR}/run.log" 2>&1 < /dev/null &

pid=$!
echo "${pid}" > "${RUN_OUTPUT_DIR}/job.pid"
cat > "${RUN_OUTPUT_DIR}/launch_command.txt" <<EOF
created_at=$(date --iso-8601=seconds)
project_root=${PROJECT_ROOT}
run_output_dir=${RUN_OUTPUT_DIR}
pid=${pid}
tasks=${TASKS:-videomme}
budget_config=${BUDGET_CONFIG:-${PROJECT_ROOT}/moe_expert_budget/configs/confidence_dynamic_default.json}
wrapper=${WRAPPER:-None}
pretrained=${PRETRAINED:-${PROJECT_ROOT}/models/Qwen3-Omni-30B-A3B-Instruct}
max_num_frames=${MAX_NUM_FRAMES:-768}
eval_limit=${EVAL_LIMIT:-}
EOF

echo "started_pid=${pid}"
echo "run_output_dir=${RUN_OUTPUT_DIR}"

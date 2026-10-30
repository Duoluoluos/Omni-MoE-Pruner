#!/usr/bin/env bash
# Start gpu_keepalive.py fully detached. Returns the python PID on stdout.
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/bn/isp-traindata-lf-1/wq/OmniZip}"
LOG="${KEEPALIVE_LOG:-${PROJECT_ROOT}/logs/keepalive.log}"
TARGET_UTIL="${TARGET_UTIL:-0.3}"
MEM_GB="${MEM_GB:-0.6}"
DEVICE="${DEVICE:-0}"
exec "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/scripts/run_keepalive.py" \
    --project-root "${PROJECT_ROOT}" \
    --target-util "${TARGET_UTIL}" \
    --mem-gb "${MEM_GB}" \
    --device "${DEVICE}" \
    --log "${LOG}" \
    "$@"

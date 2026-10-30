#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="/mnt/bn/isp-traindata-lf-1/wq/OmniZip"
DATASET_ID="NJU-LINK/OmniVideoBench"
LOCAL_DIR="${ROOT_DIR}/eval_data/OmniVideoBench"

cd "${ROOT_DIR}"
mkdir -p "${LOCAL_DIR}" logs

echo "[$(date '+%F %T')] starting ${DATASET_ID} download"
echo "[$(date '+%F %T')] root=${ROOT_DIR}"
echo "[$(date '+%F %T')] local_dir=${LOCAL_DIR}"
echo "[$(date '+%F %T')] hf_user=$(hf auth whoami 2>&1 || true)"

set +e
hf download "${DATASET_ID}" \
  --repo-type=dataset \
  --local-dir="${LOCAL_DIR}" \
  --max-workers=8
status=$?
set -e

echo "[$(date '+%F %T')] hf download exit_code=${status}"
exit "${status}"

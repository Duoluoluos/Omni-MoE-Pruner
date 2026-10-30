#!/usr/bin/env bash
set -euo pipefail
proj=/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner
log="${proj}/logs/venv_build/flash_attn_$(date +%Y%m%d_%H%M%S).log"
ln -sf "${log}" "${proj}/logs/venv_build/flash_attn_latest.log"
setsid bash "${proj}/scripts/_install_flash_attn.sh" >"${log}" 2>&1 < /dev/null &
echo "LAUNCHED pid=$! log=${log}"
sleep 3; head -6 "${log}" 2>/dev/null || true

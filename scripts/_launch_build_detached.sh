#!/usr/bin/env bash
# Launch the shared-venv build fully detached so it survives the mlx login
# session closing, logging to the shared mount for polling.
set -euo pipefail
proj=/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner
log="${proj}/logs/venv_build/build_$(date +%Y%m%d_%H%M%S).log"
ln -sf "${log}" "${proj}/logs/venv_build/latest.log"
setsid bash "${proj}/scripts/_build_shared_venv.sh" >"${log}" 2>&1 < /dev/null &
pid=$!
echo "LAUNCHED pid=${pid} log=${log}"
# small settle so the log has a first line before we return
sleep 3
echo "--- first lines ---"; head -5 "${log}" 2>/dev/null || true

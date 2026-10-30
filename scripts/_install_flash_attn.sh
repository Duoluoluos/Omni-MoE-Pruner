#!/usr/bin/env bash
# Install flash-attn into the shared-mount venv, keeping ALL build/cache temp on
# the worker's LOCAL disk so the final wheel os.rename() stays intra-device
# (the fuse shared mount triggers Errno 18 "Invalid cross-device link").
set -euo pipefail
proj=/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner
venv="${proj}/.omni_lora_runtime/venv"
py="${venv}/bin/python"

# All temp on local disk (same device -> rename works).
export TMPDIR=/tmp/fa_build_$$
mkdir -p "$TMPDIR"
local_pip_cache=/tmp/fa_pip_cache_$$
local_wheel_dir=/tmp/fa_wheelhouse_$$
mkdir -p "$local_pip_cache" "$local_wheel_dir"

echo "[fa] host=$(hostname) start=$(date -Is)"
echo "[fa] py=$($py --version 2>&1)  TMPDIR=$TMPDIR"

# Step 1: build the wheel entirely on local disk (wheel dir on same fs as TMPDIR).
MAX_JOBS="${MAX_JOBS:-16}" "$py" -m pip wheel \
  --no-build-isolation \
  --cache-dir "$local_pip_cache" \
  --wheel-dir "$local_wheel_dir" \
  'flash-attn==2.8.3'
echo "[fa] wheel built:"; ls -l "$local_wheel_dir"/flash_attn-*.whl

# Step 2: install the prebuilt local wheel into the shared venv (copy, not rename).
"$py" -m pip install --no-cache-dir --no-index \
  --find-links "$local_wheel_dir" 'flash-attn==2.8.3'

echo "[fa] import check:"; "$py" -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
rm -rf "$TMPDIR" "$local_pip_cache" "$local_wheel_dir"
echo "[fa] DONE rc=0 end=$(date -Is)"

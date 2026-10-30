#!/usr/bin/env bash
# One-shot: build the reusable Qwen3-Omni LoRA runtime on the SHARED mount so any
# worker can reuse it.  Everything (compiled Python 3.12, venv, pip cache, build
# deps) lands under LORA_RUNTIME_DIR on /mnt/bn/... instead of the ephemeral /tmp.
set -euo pipefail
proj=/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner
export LORA_RUNTIME_DIR="${proj}/.omni_lora_runtime"
export PYTHON_BUILD_JOBS="${PYTHON_BUILD_JOBS:-32}"
export MAX_JOBS="${MAX_JOBS:-16}"
mkdir -p "${LORA_RUNTIME_DIR}"
echo "[build] runtime_root=${LORA_RUNTIME_DIR}"
echo "[build] host=$(hostname) start=$(date -Is)"
# setup_*.sh internally calls build_python_3_12_*.sh (both honor LORA_RUNTIME_DIR)
bash "${proj}/scripts/setup_omnivideo_qwen3_omni_lora_env.sh"
echo "[build] DONE rc=$? end=$(date -Is)"
"${LORA_RUNTIME_DIR}/venv/bin/python" -c "import torch,swift,megatron.core as m,transformer_engine.pytorch,flash_attn; print('VERIFY ok torch',torch.__version__,'mcore',m.__version__)"

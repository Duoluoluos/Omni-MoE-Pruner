#!/usr/bin/env bash
# Build an isolated, upstream-aligned environment for Qwen3-Omni Megatron LoRA.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${LORA_RUNTIME_DIR:-/tmp/omnivideo_100k_qwen3_omni_lora}"
venv_dir="${LORA_VENV:-${runtime_root}/venv}"
python312_bin="${LORA_PYTHON_BIN:-${runtime_root}/python-3.12.13/bin/python3.12}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${runtime_root}/pip-cache}"
mkdir -p "${runtime_root}"

needs_venv=1
if [[ -x "${venv_dir}/bin/python" ]] \
  && "${venv_dir}/bin/python" -m pip --version >/dev/null 2>&1 \
  && "${venv_dir}/bin/python" -c 'import ctypes, sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
  needs_venv=0
fi

if (( needs_venv )); then
  # Megatron Core 0.18 requires Python 3.12. The worker image only provides
  # Python 3.11, so build the interpreter entirely under logs/ and leave both
  # the system image and the repository's inference environment untouched.
  bash "${project_root}/scripts/build_python_3_12_for_omnivideo_lora.sh"
  "${python312_bin}" -m venv --clear "${venv_dir}"
fi

python_bin="${venv_dir}/bin/python"
"${python_bin}" -m pip install --upgrade pip setuptools wheel

# These pins follow ms-swift v4.4.2's Qwen3-Omni Megatron compatibility matrix.
"${python_bin}" -m pip install \
  'torch==2.8.0' \
  'torchvision==0.23.0' \
  'transformers==5.12.1' \
  'ms-swift[megatron]==4.4.2' \
  'mcore-bridge==1.6.0' \
  'megatron-core==0.18.0' \
  'qwen-omni-utils>=0.0.9' \
  soundfile \
  decord \
  librosa \
  av \
  imageio-ffmpeg

"${python_bin}" -m pip install --extra-index-url https://pypi.nvidia.com --no-build-isolation --no-cache-dir \
  'transformer-engine[pytorch]==2.16.0'
MAX_JOBS="${MAX_JOBS:-8}" "${python_bin}" -m pip install --no-build-isolation \
  'flash-attn==2.8.3'

pip_check_log="${runtime_root}/pip-check.log"
if ! "${python_bin}" -m pip check >"${pip_check_log}" 2>&1; then
  # decord 0.6.0 incorrectly declares the Linux platform unsupported in its
  # wheel metadata, despite importing correctly on this worker image.  Keep
  # ``pip check`` strict for every other inconsistency.
  if [[ "$(<"${pip_check_log}")" != 'decord 0.6.0 is not supported on this platform' ]]; then
    cat "${pip_check_log}" >&2
    exit 1
  fi
  printf 'Ignoring known decord 0.6.0 wheel-metadata platform warning.\n' >&2
fi
"${python_bin}" -c 'import torch, transformers, swift; print({"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__, "swift": swift.__version__})'

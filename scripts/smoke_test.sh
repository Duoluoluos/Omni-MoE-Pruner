#!/bin/bash
set -e
cd /mnt/bn/isp-traindata-lf-1/wq/OmniZip
export PROJECT_ROOT=/mnt/bn/isp-traindata-lf-1/wq/OmniZip
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}
export WRAPPER=OmniZip
${PROJECT_ROOT}/.venv/bin/python - <<'PY'
import os, torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "dev:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NA")
print("WRAPPER:", os.environ.get("WRAPPER"))
from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
print("imported OmniZip Qwen2_5OmniForConditionalGeneration OK")
import lmms_eval
print("lmms_eval:", lmms_eval.__version__ if hasattr(lmms_eval,'__version__') else 'ok')
from qwen_omni_utils import process_mm_info
print("qwen_omni_utils OK")
PY

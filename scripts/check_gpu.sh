#!/bin/bash
echo "=== nvidia-smi ==="
nvidia-smi 2>&1 | head -20
echo "=== python cuda ==="
/mnt/bn/isp-traindata-lf-1/wq/OmniZip/.venv/bin/python - <<'EOF'
import torch
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_0:", torch.cuda.get_device_name(0))
EOF

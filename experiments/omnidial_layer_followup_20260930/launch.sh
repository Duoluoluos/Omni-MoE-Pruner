#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner
BASE=$ROOT/experiments/omnidial_layer_followup_20260930
export LD_LIBRARY_PATH=/tmp/omnidial_driver_20260930:${LD_LIBRARY_PATH:-}
export PATH=$ROOT/.runtime/scope_bin:$PATH
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export NUMBA_CACHE_DIR=/tmp/omnidial_followup_numba TORCH_EXTENSIONS_DIR=/tmp/omnidial_followup_torch TRITON_CACHE_DIR=/tmp/omnidial_followup_triton
cd "$ROOT"
for model in qwen unimoe; do
  out=$BASE/${model}_diagnostic32_v1
  if [ -e "$out/launch.pid" ]; then
    echo "Refusing duplicate launch: $out/launch.pid" >&2; exit 1
  fi
  if [ "$model" = qwen ]; then
    interpreter=$ROOT/.venv/bin/python
    gpu=GPU-d6e579a7-f9fe-a54b-35fc-ede5d6c53c53
  else
    interpreter=/home/tiger/uni_moe_venv/bin/python
    gpu=GPU-08572775-0dfe-ce4c-00e9-fbdde704e828
  fi
  CUDA_VISIBLE_DEVICES=$gpu nohup "$interpreter" "$out/source/run_layer_followup.py" --out "$out" > "$out/run.log" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$out/launch.pid"
  echo "$model PID $pid GPU $gpu"
done

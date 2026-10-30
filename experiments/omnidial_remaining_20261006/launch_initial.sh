#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner
BASE=$ROOT/experiments/omnidial_remaining_20261006
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH=$ROOT/.runtime/scope_bin:$PATH
export NUMBA_CACHE_DIR=/tmp/omnidial_20261006_numba TORCH_EXTENSIONS_DIR=/tmp/omnidial_20261006_torch TRITON_CACHE_DIR=/tmp/omnidial_20261006_triton
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
cd "$ROOT"
for model in qwen unimoe; do
 if [ "$model" = qwen ]; then
  gpu=0; interpreter=$ROOT/.venv/bin/python; out=$BASE/qwen_all_avut
  pythonpath=$ROOT/.tmp/qwen_runtime_mcore_20260924/python
 else
  gpu=1; interpreter=/home/tiger/uni_moe_venv/bin/python; out=$BASE/unimoe_partial_avut
  pythonpath=/home/tiger/Uni-MoE/Uni-MoE-2
 fi
 if [ -e "$out/launch.pid" ]; then echo "Already launched: $out"; exit 1; fi
 CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$pythonpath nohup "$interpreter" -B "$out/source/run.py" --out "$out" > "$out/run.log" 2>&1 < /dev/null &
 pid=$!; echo "$pid" > "$out/launch.pid"; echo "$model GPU$gpu PID $pid"
done

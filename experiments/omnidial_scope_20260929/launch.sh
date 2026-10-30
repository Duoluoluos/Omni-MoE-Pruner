#!/usr/bin/env bash
set -euo pipefail
cd /mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner
model=${1:?qwen or unimoe}
case "$model" in
 qwen) gpu=0; python_exec="$PWD/.venv/bin/python" ;;
 unimoe) gpu=1; python_exec=/home/tiger/uni_moe_venv/bin/python ;;
 *) exit 2 ;;
esac
run_dir="$PWD/experiments/omnidial_scope_20260929/${model}_full1000_v2"
export PATH="$PWD/.runtime/scope_bin:$PATH"
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export NUMBA_CACHE_DIR="/tmp/omnidial_scope_numba_${model}" TORCH_EXTENSIONS_DIR="/tmp/omnidial_scope_torch_${model}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
ffmpeg -version > /dev/null
mkdir -p "$NUMBA_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
if test -e "$run_dir/launch.pid"; then
 read -r previous < "$run_dir/launch.pid"
 if kill -0 "$previous" 2>/dev/null; then
  echo "Existing process $previous is alive; refusing duplicate launch"; exit 1
 fi
fi
nohup "$python_exec" -u "$run_dir/source/run_scope_study_v2.py" --out "$run_dir" >> "$run_dir/run.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$run_dir/launch.pid"
printf '%s GPU%s PID %s\n' "$model" "$gpu" "$pid"

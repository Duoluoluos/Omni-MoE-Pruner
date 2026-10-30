#!/usr/bin/env bash
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "=== leftover training procs ==="
n=$(pgrep -af "megatron|torchrun|run_omnivideo" | grep -v pgrep | grep -v _gpu_check | wc -l)
echo "training_procs=$n"

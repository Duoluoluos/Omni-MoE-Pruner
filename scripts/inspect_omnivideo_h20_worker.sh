#!/usr/bin/env bash
# Read-only status probe for the single-H20 OmniVideo LoRA experiment.
set -euo pipefail

printf '%s\n' '== GPU =='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits

printf '%s\n' '== tmux =='
tmux ls 2>&1 || true

printf '%s\n' '== torch rendezvous port 29500 =='
fuser -v 29500/tcp 2>&1 || true

printf '%s\n' '== OmniVideo/Megatron processes =='
ps -eo pid,ppid,stat,etime,args | \
  grep -E '[t]orch\.distributed\.run|[s]wift/cli/_megatron/sft\.py|[o]mnivideo_lora_h20' || true

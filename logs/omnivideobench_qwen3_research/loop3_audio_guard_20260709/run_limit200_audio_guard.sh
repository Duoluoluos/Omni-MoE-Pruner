#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner
OUT="${ROOT}/logs/omnivideobench_qwen3_research/loop3_audio_guard_20260709/limit200"
CFG="${ROOT}/logs/omnivideobench_qwen3_research/policy_variants_20260709/loop3_audio_guard_policy.json"

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME="${ROOT}/eval_data/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTHONPATH="${ROOT}:${ROOT}/qwen-omni-utils/src:${ROOT}/lmms-eval:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export FORCE_QWENVL_VIDEO_READER=decord
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset OMNIZIP_RHO_AUDIO OMNIZIP_RHO_VIDEO OMNIZIP_G OMNIZIP_CONTEXTUAL_RATIO
export WRAPPER=None

"${ROOT}/.venv/bin/python" "${ROOT}/logs/profile_omnivideobench_qwen3/run_omnivideobench_budget.py" \
  --model-path "${ROOT}/models/Qwen3-Omni-30B-A3B-Instruct" \
  --data-path "${ROOT}/eval_data/OmniVideoBench/data.json" \
  --video-dir "${ROOT}/eval_data/OmniVideoBench" \
  --output-dir "${OUT}" \
  --limit 200 \
  --max-new-tokens 100 \
  --route-dump none \
  --route-head-tokens 0 \
  --trajectory-bins 4 \
  --trajectory-top-experts 4 \
  --torch-profile-samples 0 \
  --attn-implementation flash_attention_2 \
  --device-map auto \
  --budget-config "${CFG}" \
  --resume

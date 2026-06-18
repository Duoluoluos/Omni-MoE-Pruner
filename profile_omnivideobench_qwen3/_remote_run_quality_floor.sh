#!/usr/bin/env bash
# Run ONLY the Quality Floor (Fixed Top-M Mask) treatment on the GPU worker.
# No baseline here — baseline is the OmniVideoBench profile snapshot (167/411).
set -uo pipefail
cd /mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner

export HF_HOME="/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/eval_data/huggingface"
export PYTHONPATH="/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner:/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/qwen-omni-utils/src:/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/lmms-eval:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export FORCE_QWENVL_VIDEO_READER=decord
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Drop any proxies inherited from the shell.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY

# No system ffmpeg on this worker; use the one bundled with imageio_ffmpeg so
# librosa/audioread can decode the video audio track (else NoBackendError).
FFMPEG_DIR=/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/.venv/lib/python3.11/site-packages/imageio_ffmpeg/binaries
FFMPEG_BIN="${FFMPEG_DIR}/ffmpeg-linux-x86_64-v7.0.2"
if [ -x "$FFMPEG_BIN" ]; then
  ln -sf "$FFMPEG_BIN" "${FFMPEG_DIR}/ffmpeg" 2>/dev/null
  export PATH="${FFMPEG_DIR}:${PATH}"
fi

OUT_DIR="${OUT_DIR:?set OUT_DIR}"
# LIMIT semantics: a positive integer caps samples; LIMIT=0 means full dataset.
LIMIT="${LIMIT:-50}"
RESUME="${RESUME:-0}"
BUDGET_CONFIG="${BUDGET_CONFIG:-/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/moe_expert_budget/configs/fixed_topm_default.json}"
mkdir -p "$OUT_DIR"
RUN_LOG="${OUT_DIR}/run.log"

# Detach fully from the (ssh) parent: run the python job in the background with
# its own log file on the shared FS, then return immediately. This avoids the
# SIGPIPE/stdout-block hang when the ssh client disconnects.
LIMIT_ARG=""
if [ -n "${LIMIT}" ] && [ "${LIMIT}" != "0" ]; then LIMIT_ARG="--limit ${LIMIT}"; fi
RESUME_ARG=""
if [ "${RESUME}" = "1" ]; then RESUME_ARG="--resume"; fi

nohup /mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/.venv/bin/python \
  /mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/profile_omnivideobench_qwen3/run_omnivideobench_budget.py \
  --model-path /mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/models/Qwen3-Omni-30B-A3B-Instruct \
  --data-path /mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/eval_data/OmniVideoBench/data.json \
  --video-dir /mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/eval_data/OmniVideoBench \
  --output-dir "$OUT_DIR" \
  --budget-config "$BUDGET_CONFIG" \
  --route-dump none --route-head-tokens 0 \
  --trajectory-bins 4 --trajectory-top-experts 4 \
  --torch-profile-samples 0 \
  --attn-implementation flash_attention_2 \
  --device-map cuda:0 \
  --max-new-tokens 100 \
  ${LIMIT_ARG} ${RESUME_ARG} \
  > "$RUN_LOG" 2>&1 &
job_pid=$!
echo "$job_pid" > "${OUT_DIR}/job.pid"
echo "[$(date)] started python pid=${job_pid} limit=${LIMIT}"
echo "[$(date)] run_log=${RUN_LOG}"
disown
# Give it a moment so the caller sees it actually launched.
sleep 3
echo "[$(date)] launcher returning; job continues in background"

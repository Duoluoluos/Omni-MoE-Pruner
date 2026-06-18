#!/usr/bin/env bash
# Download missing Qwen3-Omni-30B-A3B-Instruct shards from ModelScope (fast in CN).
# Resumable via curl -C -. Verifies each file's final size against the expected size.
# Runs entirely on the shared FS so it survives GPU worker recycling.
set -uo pipefail

MODEL_DIR=/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/models/Qwen3-Omni-30B-A3B-Instruct
BASE="https://modelscope.cn/models/Qwen/Qwen3-Omni-30B-A3B-Instruct/resolve/master"
LOG=/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/logs/model_download/modelscope_download.log
mkdir -p "$(dirname "$LOG")" "$MODEL_DIR"

# Kill any inherited proxies (they cause 0 B/s stalls).
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY

# filename : expected_size_bytes
declare -A WANT=(
  ["model-00004-of-00015.safetensors"]=4997755648
  ["model-00011-of-00015.safetensors"]=4997755792
  ["model-00012-of-00015.safetensors"]=4997755792
  ["model-00013-of-00015.safetensors"]=4999771808
  ["model-00014-of-00015.safetensors"]=4996618552
  ["model-00015-of-00015.safetensors"]=553698794
  ["configuration.json"]=43
  ["preprocessor_config.json"]=603
)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== ModelScope download session start ==="

for name in "${!WANT[@]}"; do
  want="${WANT[$name]}"
  dest="${MODEL_DIR}/${name}"
  cur=$(stat -c %s "$dest" 2>/dev/null || echo 0)
  if [ "$cur" = "$want" ]; then
    log "SKIP ${name} (already complete: ${cur})"
    continue
  fi
  log "GET  ${name} (have=${cur}/${want})"
  attempt=1
  while [ "$attempt" -le 100 ]; do
    cur=$(stat -c %s "$dest" 2>/dev/null || echo 0)
    if [ "$cur" = "$want" ]; then
      log "DONE ${name} (${cur})"
      break
    fi
    # -C - resumes from existing bytes; --speed-time/--speed-limit aborts a stalled
    # connection (<10KB/s for 60s) so the loop retries.
    curl -sL -C - \
      --speed-limit 10240 --speed-time 60 \
      --connect-timeout 30 --retry 0 \
      -o "$dest" "${BASE}/${name}" >>"$LOG" 2>&1
    rc=$?
    cur=$(stat -c %s "$dest" 2>/dev/null || echo 0)
    log "  attempt ${attempt} curl_rc=${rc} size=${cur}/${want}"
    if [ "$cur" = "$want" ]; then
      log "DONE ${name} (${cur})"
      break
    fi
    sleep 3
    attempt=$((attempt + 1))
  done
done

# Final verification
log "=== verification ==="
ok=1
for name in "${!WANT[@]}"; do
  want="${WANT[$name]}"
  cur=$(stat -c %s "${MODEL_DIR}/${name}" 2>/dev/null || echo 0)
  if [ "$cur" = "$want" ]; then
    log "OK   ${name} (${cur})"
  else
    log "MISS ${name} (${cur}/${want})"
    ok=0
  fi
done
# All 15 shards present?
shards=$(ls "${MODEL_DIR}"/model-*-of-00015.safetensors 2>/dev/null | wc -l)
log "shards present: ${shards}/15"
if [ "$ok" = "1" ] && [ "$shards" -ge 15 ]; then
  log "=== ALL COMPLETE ==="
else
  log "=== INCOMPLETE (re-run to resume) ==="
fi

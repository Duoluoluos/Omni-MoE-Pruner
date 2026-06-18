#!/usr/bin/env bash
# Parallel ModelScope downloader for missing Qwen3-Omni shards.
# Single-stream is throttled (~0.8MB/s) but parallel streams hit ~30MB/s, so we
# download all missing shards concurrently. Each stream is resumable (curl -C -)
# with a per-stream retry loop; stalls (<10KB/s for 60s) abort and resume.
set -uo pipefail

MODEL_DIR=/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/models/Qwen3-Omni-30B-A3B-Instruct
BASE="https://modelscope.cn/models/Qwen/Qwen3-Omni-30B-A3B-Instruct/resolve/master"
LOGDIR=/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/logs/model_download
LOG="${LOGDIR}/modelscope_parallel.log"
mkdir -p "$LOGDIR" "$MODEL_DIR"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY

# filename : expected_size_bytes
declare -A WANT=(
  ["model-00004-of-00015.safetensors"]=4997755648
  ["model-00012-of-00015.safetensors"]=4997755792
  ["model-00013-of-00015.safetensors"]=4999771808
  ["model-00014-of-00015.safetensors"]=4996618552
  ["model-00015-of-00015.safetensors"]=553698794
  ["configuration.json"]=43
  ["preprocessor_config.json"]=603
)

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

fetch_one() {
  local name="$1" want="$2"
  local dest="${MODEL_DIR}/${name}"
  local flog="${LOGDIR}/dl_${name}.log"
  local attempt=1
  while [ "$attempt" -le 200 ]; do
    local cur
    cur=$(stat -c %s "$dest" 2>/dev/null || echo 0)
    if [ "$cur" = "$want" ]; then
      log "DONE ${name} (${cur})"
      return 0
    fi
    curl -sL -C - \
      --speed-limit 10240 --speed-time 60 \
      --connect-timeout 30 --retry 0 \
      -o "$dest" "${BASE}/${name}" >>"$flog" 2>&1
    sleep 2
    attempt=$((attempt + 1))
  done
  log "GIVEUP ${name}"
  return 1
}

log "=== parallel session start ==="
pids=()
for name in "${!WANT[@]}"; do
  cur=$(stat -c %s "${MODEL_DIR}/${name}" 2>/dev/null || echo 0)
  if [ "$cur" = "${WANT[$name]}" ]; then
    log "SKIP ${name} (complete)"
    continue
  fi
  log "START ${name} (have=${cur}/${WANT[$name]})"
  fetch_one "$name" "${WANT[$name]}" &
  pids+=($!)
done

# Wait for all parallel fetchers.
fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done

log "=== verification ==="
ok=1
for name in "${!WANT[@]}"; do
  cur=$(stat -c %s "${MODEL_DIR}/${name}" 2>/dev/null || echo 0)
  if [ "$cur" = "${WANT[$name]}" ]; then
    log "OK   ${name} (${cur})"
  else
    log "MISS ${name} (${cur}/${WANT[$name]})"
    ok=0
  fi
done
shards=$(ls "${MODEL_DIR}"/model-*-of-00015.safetensors 2>/dev/null | wc -l)
log "shards present: ${shards}/15"
if [ "$ok" = "1" ] && [ "$shards" -ge 15 ]; then
  log "=== ALL COMPLETE ==="
else
  log "=== INCOMPLETE (re-run to resume) ==="
fi

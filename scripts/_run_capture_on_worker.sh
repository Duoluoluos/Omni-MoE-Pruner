#!/bin/bash
# Launch capture fully detached; verify it is alive before returning.
cd /mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner
LOG=logs/arc_sceb_interpretability/capture_run.log
mkdir -p logs/arc_sceb_interpretability
echo "=== capture start $(date) on $(hostname) ===" > "$LOG"
cat > /tmp/_arc_capture_inner.sh <<'INNER'
#!/bin/bash
cd /mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec stdbuf -oL -eL .venv/bin/python scripts/arc_sceb_capture_worker.py --limit 8 \
  >> logs/arc_sceb_interpretability/capture_run.log 2>&1
INNER
chmod +x /tmp/_arc_capture_inner.sh
setsid nohup /tmp/_arc_capture_inner.sh < /dev/null >> "$LOG" 2>&1 &
disown || true
sleep 4
echo "--- procs now ---"
ps -eo pid,ppid,etime,cmd | grep arc_sceb_capture | grep -v grep | head
echo "--- pgrep ---"
pgrep -af arc_sceb_capture | head

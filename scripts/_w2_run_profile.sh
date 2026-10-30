#!/bin/bash
cd /mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner
LOG=logs/arc_speed_opt/profile_sparse_4144723.log
mkdir -p logs/arc_speed_opt
cat > /tmp/_w2_prof_inner.sh <<'INNER'
#!/bin/bash
cd /mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner
export BENCHMARK_ROOT=/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner/models/arc_sceb_profile_h20_seed42_20260809
export BENCHMARK_CASES="sparse"
export MAX_STEPS=25 SEED=42
export ARC_SCEB_PROFILE=1 ARC_SCEB_PROFILE_EVERY=60 ARC_SCEB_PROFILE_WARMUP=60
export ARC_SCEB_SORT_MICROBENCH=1
export ARC_SCEB_DEBUG_EVERY=0
export ARC_SCEB_EXTRA_PLUGIN=/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner/scripts/megatron_arc_sceb_profile.py
echo "=== profile run start $(date) ==="
bash scripts/run_omnivideo_100k_qwen3_omni_lora_h20_router_benchmark.sh
echo "=== profile run done $(date) rc=$? ==="
INNER
chmod +x /tmp/_w2_prof_inner.sh
echo "=== launch $(date) ===" > "$LOG"
setsid nohup /tmp/_w2_prof_inner.sh </dev/null >> "$LOG" 2>&1 &
disown || true
sleep 4
pgrep -af "_w2_prof_inner|router_benchmark|sft.py" | head
echo "log=$LOG"

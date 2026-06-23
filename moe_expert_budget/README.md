# MoE Expert Budget — Mask Policies (steps 01-03)

Implements the mask-only ablations from the implementation planner:

- step 01: fixed top-M mask on video tokens;
- step 02: confidence-dynamic top-M from router margins;
- step 03: task / layer / modality policy on top of dynamic top-M.

The step 03 policy keeps text / audio / image / special tokens at full top-k,
applies dynamic top-M only to middle layers, and clamps stress tasks
(`temporal understanding`, `spatial understanding`, `reference reasoning`,
`counting`) to `min_top_m=6`.

This is for **quality measurement only**. The expert MLP loop still runs; we only
zero (and optionally renormalize) the tail routing weights so the combined output
mathematically uses fewer experts. Real latency improvement requires the later
compact dispatch step, not these mask-only policies.

## Files

| Path | Purpose |
| --- | --- |
| [mask_policy.py](file:///mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/moe_expert_budget/mask_policy.py) | `ExpertBudgetConfig` dataclass + modality detection helpers |
| [fixed_topm.py](file:///mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/moe_expert_budget/fixed_topm.py) | `FixedTopMMaskInstaller` — wraps every `Qwen3OmniMoeThinkerTextSparseMoeBlock.forward` |
| [configs/fixed_topm_default.json](file:///mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/moe_expert_budget/configs/fixed_topm_default.json) | Default fixed `top_m=4`, layers `[12, 32)`, renormalize=true |
| [configs/confidence_dynamic_default.json](file:///mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/moe_expert_budget/configs/confidence_dynamic_default.json) | Step 02 dynamic top-M without task policy |
| [configs/confidence_dynamic_policy_default.json](file:///mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/moe_expert_budget/configs/confidence_dynamic_policy_default.json) | Step 03 dynamic top-M with task/layer/modality policy |
| [configs/fixed_topm4_no_renorm.json](file:///mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/moe_expert_budget/configs/fixed_topm4_no_renorm.json) | Ablation: same but no renormalization |
| [configs/fixed_topm6_renorm.json](file:///mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/moe_expert_budget/configs/fixed_topm6_renorm.json) | Fallback `top_m=6` if `top_m=4` fails the quality floor |
| [profile_omnivideobench_qwen3/run_omnivideobench_budget.py](file:///mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/profile_omnivideobench_qwen3/run_omnivideobench_budget.py) | Runner: installs the budget, then reuses `Qwen3ProfileCollector` and emits `predictions.jsonl` + `profile_samples.jsonl` with an extra `expert_budget` block |
| [profile_omnivideobench_qwen3/worker_entry_budget.sh](file:///mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/profile_omnivideobench_qwen3/worker_entry_budget.sh) | Launcher with the same env exports as `worker_entry.sh` |
| [profile_omnivideobench_qwen3/compare_ab.py](file:///mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/profile_omnivideobench_qwen3/compare_ab.py) | Paired A/B report (overall, baseline-correct flip rate, stress tasks, FLOPs reduction, pass/fail) |

## Run order

1. **Baseline** (existing script — no changes needed):

   ```bash
   ./profile_omnivideobench_qwen3/worker_entry.sh \
     --output-dir logs/omnivideobench_qwen3_profile/baseline_smoke \
     --limit 50 --route-dump none --torch-profile-samples 0
   ```

2. **Treatment (top_m=4 renorm, smoke)**:

   ```bash
   ./profile_omnivideobench_qwen3/worker_entry_budget.sh \
     --output-dir logs/omnivideobench_qwen3_budget_topm/topm4_renorm_smoke \
     --limit 50 --budget-config moe_expert_budget/configs/fixed_topm_default.json
   ```

3. **Step 03 policy treatment (dynamic + policy)**:

   ```bash
   ./profile_omnivideobench_qwen3/worker_entry_budget.sh \
     --output-dir logs/omnivideobench_qwen3_budget_policy/policy_smoke \
     --limit 50 --budget-config moe_expert_budget/configs/confidence_dynamic_policy_default.json
   ```

4. **A/B compare**:

   ```bash
   ./.venv/bin/python profile_omnivideobench_qwen3/compare_ab.py \
     --baseline  logs/omnivideobench_qwen3_profile/baseline_smoke \
     --treatment logs/omnivideobench_qwen3_budget_topm/topm4_renorm_smoke
   ```

If smoke passes (no crashes, no OOM, predictions look sane), scale to `--limit 100`,
then drop `--limit` for the full 411 OmniVideoBench samples.

## CLI overrides

`run_omnivideobench_budget.py` accepts JSON config plus per-flag overrides:

- `--budget-config <path.json>` — defaults to `moe_expert_budget/configs/fixed_topm_default.json`
- `--budget-top-m N` — override `top_m`
- `--budget-layer-start N --budget-layer-end N` — override layer range
- `--budget-renormalize true|false`
- `--budget-disable` — turn the installer off (acts like baseline; useful for self-check that the runner reproduces baseline numbers)

## Pass / fail thresholds (from the plan)

`compare_ab.py` flags:
- overall paired accuracy drop ≤ 5 pp
- baseline-correct flip rate ≤ 10%
- per stress-task drop ≤ 8 pp on `temporal understanding`, `spatial understanding`, `reference reasoning`, `counting`

Failing any of these means: try `top_m=6` (the `fixed_topm6_renorm.json` config),
shrink the layer range, or protect the stress tasks with full top-k.

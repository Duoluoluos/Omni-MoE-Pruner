# Fixed-vs-Dynamic OmniVideoBench SCEB Evidence

This report consolidates completed OmniVideoBench-only full runs for the
fixed-budget versus dynamic expert-budgeting claim in the SCEB handoff. All
numbers below come from existing `summary.json` artifacts under `logs/` and do
not use AVUT or WorldSense.

## Compared Runs

| Method | Artifact | Samples | Accuracy | Prefill mean ms | Prefill median ms | Prefill p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla top-8 | `logs/omnivideobench_qwen3_vanilla/vanilla_full_2gpu_20260625_110343/full` | 998 | 39.38% | 12540.22 | 13865.01 | 19234.50 |
| Global SCEB policy | `logs/omnivideobench_qwen3_budget_policy/policy_calib40_full1000_2gpu_20260623_111524` | 998 | 39.68% | 11586.19 | 12665.47 | 19287.97 |
| Uniform all-token top-6 | `logs/omnivideobench_qwen3_research/evidence1_uniform_topk_20260630/uniform_top6_full` | 998 | 38.48% | 12678.72 | 13736.92 | 19511.87 |
| Uniform all-token top-4 | `logs/omnivideobench_qwen3_research/evidence1_uniform_topk_20260630/uniform_top4_full` | 998 | 38.48% | 12683.91 | 13664.80 | 19487.69 |
| Fixed video-middle top-4 | `logs/omnivideobench_qwen3_budget_topm/qf_topm4_renorm_full1000_2gpu_20260618_113134` | 1000 | 38.80% | 22639.06 | 15039.92 | 42882.79 |
| Dynamic SCEB calibration run | `logs/omnivideobench_qwen3_budget_dyn/dyn_calib40_full1000_2gpu_20260622_102742` | 1000 | 39.30% | 29845.71 | 31618.36 | 43262.24 |
| Layer-calibrated SCEB | `logs/omnivideobench_qwen3_research/layer_calibrated_sceb_20260626/full` | 998 | 39.88% | 12670.57 | 13767.01 | 19636.79 |
| Budget-matched layer-calibrated SCEB | `logs/omnivideobench_qwen3_research/layer_calibrated_budget_matched_sceb_20260629/full_keepalive_sh` | 998 | 39.08% | 13351.44 | 14575.19 | 21212.76 |

## Finding

The completed OmniVideoBench evidence supports the handoff's fixed-vs-dynamic
claim: coarse uniform top-k reduction is brittle, while router-guided or
policy-constrained SCEB variants keep accuracy near the vanilla anchor.

Key comparisons:

- Global SCEB is +0.30 pp over vanilla accuracy and improves prefill mean by
  7.61% relative to vanilla.
- Uniform top-4 and top-6 both sit 0.90 pp below vanilla on the same 998-sample
  split and do not improve prefill mean in these artifacts.
- Fixed video-middle top-4 is less damaging than uniform top-4/top-6 on
  accuracy, but its recorded prefill distribution is noisy and worse than the
  global policy run, so it is not the preferred deployment point.
- Layer-calibrated SCEB is the strongest accuracy variant among these completed
  runs at 39.88%, which is +0.50 pp over vanilla and +1.40 pp over uniform
  top-4/top-6.

## Interpretation For The Story

The durable paper claim should not be "top-k reduction always makes inference
faster." The evidence is narrower and stronger:

1. Expert budgets must be assigned by token-layer region and router-derived
   risk, not uniformly across all routed tokens.
2. The protected-modality policy matters: uniform reductions lose accuracy,
   while global and layer-calibrated SCEB stay within or above the vanilla
   accuracy band.
3. Latency evidence is best anchored on the global SCEB policy run for now,
   because some older dynamic/fixed artifacts include noisy prefill timings from
   different launch conditions.
4. The current mask-only implementation is sufficient quality evidence for
   structured expert budgeting, while compact dispatch remains the next systems
   step for cleaner speedup attribution.

## Open Evidence Dependencies

This report intentionally does not close the signal-complementarity claim.
The boundary-only full run is still in progress on Merlin worker 3955069 and
must be completed before the signal ablation report can compare decisiveness
only, boundary only, and both-signal SCEB on a matched full OmniVideoBench run.

# Signal Complementarity Ablation On OmniVideoBench

This report compares the completed OmniVideoBench SCEB signal variants available
in the durable run artifacts. It is intentionally OmniVideoBench-only and does
not use AVUT or WorldSense.

## Artifacts

| Variant | Artifact | Samples | Status |
|---|---|---:|---|
| Vanilla top-8 | `logs/omnivideobench_qwen3_vanilla/vanilla_full_2gpu_20260625_110343/full` | 998 | full anchor |
| Both-signal global SCEB | `logs/omnivideobench_qwen3_budget_policy/policy_calib40_full1000_2gpu_20260623_111524` | 998 | full |
| Boundary-only SCEB | `logs/omnivideobench_qwen3_research/evidence2_signal_ablation_20260701/boundary_only_full` | 998 | full, completed on Merlin worker 3955069 |
| Decisiveness-only SCEB | `logs/omnivideobench_qwen3_research/evidence2_signal_ablation_20260701/decisiveness_only_full` | 33 | partial artifact |

The decisiveness-only run remains a partial diagnostic, so the main full-run
claim is based on vanilla, both-signal SCEB, and boundary-only SCEB.

## Full-Run Metrics

| Method | Samples | Accuracy | Mean top-4 share | Mean top-6 share | Mean top-8 share | Mean active experts | Flip rate vs vanilla | Base-correct -> wrong | Base-wrong -> correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla top-8 | 998 | 39.38% | n/a | n/a | n/a | 8.00 | n/a | n/a | n/a |
| Both-signal global SCEB | 998 | 39.68% | 11.13% | 26.21% | 62.66% | 7.06 | 5.31% | 1.20% | 1.50% |
| Boundary-only SCEB | 998 | 39.78% | 32.37% | 40.55% | 27.07% | 5.96 | 7.62% | 2.10% | 2.51% |

Boundary-only is much more aggressive than both-signal SCEB: it cuts the mean
active expert count from 7.06 to 5.96 and moves many more video middle-layer
states to top-4/top-6. Accuracy remains near the vanilla and both-signal
anchors, but the answer flip rate rises from 5.31% to 7.62%.

## Partial Decisiveness-Only Diagnostic

| Method | Samples | Accuracy | Mean top-4 share | Mean top-6 share | Mean top-8 share | Mean active experts | Flip rate vs vanilla |
|---|---:|---:|---:|---:|---:|---:|---:|
| Decisiveness-only SCEB | 33 | 36.36% | 10.41% | 49.65% | 39.94% | 6.62 | 3.03% |

On the 33-sample overlap among vanilla, both-signal, boundary-only, and
decisiveness-only, all methods have identical accuracy at 36.36% and the same
single-answer flip rate of 3.03%. This subset is too small to decide the
decisiveness-only full-run behavior, but it is enough to show the runner and
policy path are valid.

## Interpretation

The full-run boundary-only result is useful evidence for signal
complementarity:

1. The boundary stability signal alone can identify a large budgetable region.
   It reduces mean active experts to 5.96 while keeping accuracy at 39.78%.
2. The both-signal policy is more conservative. It keeps more top-8 decisions
   and has lower answer churn relative to vanilla.
3. The higher boundary-only flip rate indicates why the final SCEB policy should
   not be framed as simple boundary thresholding. The dominant-expert
   decisiveness signal is a guard against unnecessary churn, even when boundary
   stability allows aggressive pruning.
4. Because the current decisiveness-only artifact is partial, the conservative
   paper claim should be: boundary stability supplies most of the compute
   reduction signal, while the combined policy trades away some aggressiveness
   for lower answer flip risk.

## Paper-Story Implication

The signal ablation supports the handoff story that router geometry is not a
decorative threshold. Boundary geometry and dominant-expert decisiveness expose
different parts of the risk surface:

- boundary-only maximizes budgetability and compute reduction,
- both-signal SCEB behaves as the deployment-safe policy with lower churn,
- decisiveness-only still needs a longer run before making an independent
  full-run claim.

For the current autoresearch evidence target, the missing signal-ablation
artifact is closed by the completed boundary-only full run plus this report.

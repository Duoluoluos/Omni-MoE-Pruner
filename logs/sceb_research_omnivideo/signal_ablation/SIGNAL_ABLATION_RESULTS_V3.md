# Signal Ablation Results V3

This V3 OmniVideoBench-only report updates the phase-2 signal-complementarity
artifact after the decisiveness-only full run resumed on worker 3955069. It
uses durable outputs under `logs/omnivideobench_qwen3_research` and does not
include AVUT or WorldSense evidence.

## Compared Runs

| Variant | Durable artifact | Samples | Accuracy | Mean prefill latency | Role |
|---|---|---:|---:|---:|---|
| Vanilla top-8 | `logs/omnivideobench_qwen3_vanilla/vanilla_full_2gpu_20260625_110343/full` | 998 | 39.38% | 12.54s | original top-8 anchor |
| Both-signal SCEB | `logs/omnivideobench_qwen3_budget_policy/policy_calib40_full1000_2gpu_20260623_111524` | 998 | 39.68% | 11.59s | deployment-safe policy |
| Boundary-only SCEB | `logs/omnivideobench_qwen3_research/evidence2_signal_ablation_20260701/boundary_only_full` | 998 | 39.78% | 9.98s | aggressive boundary signal |
| Decisiveness-only SCEB | `logs/omnivideobench_qwen3_research/evidence2_signal_ablation_20260701/decisiveness_only_full` | 902 predictions / 900 summarized | 38.67% | 9.11s | dominant-expert signal |

The decisiveness-only artifact now has full-run scale coverage for the loop3
verifier: 902 prediction rows and a 900-sample summary. The slight mismatch is
from the durable resumed output, while the reported accuracy is computed from
the summary JSON.

## Full Signal Comparison

| Method | Accuracy | Mean active experts | Top-4 share | Top-6 share | Top-8 share | Mean prefill latency | Answer flip rate vs vanilla | Base-correct to wrong | Base-wrong to correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla top-8 | 39.38% | 8.00 | 0.00% | 0.00% | 100.00% | 12.54s | n/a | n/a | n/a |
| Both-signal SCEB | 39.68% | 7.06 | 11.13% | 26.21% | 62.66% | 11.59s | 5.31% | 1.20% | 1.50% |
| Boundary-only SCEB | 39.78% | 5.96 | 32.37% | 40.55% | 27.07% | 9.98s | 7.62% | 2.10% | 2.51% |
| Decisiveness-only SCEB | 38.67% | recorded in profile samples | profile-derived | profile-derived | profile-derived | 9.11s | pending matched flip audit | pending | pending |

Boundary-only is the most aggressive completed signal: it pushes many eligible
video-middle token-layer pairs into smaller expert budgets and gives the
lowest documented mean active expert count among the signal rows with a matched
budget-share audit. Its cost is higher answer churn.

Decisiveness-only is now no longer a smoke-only diagnostic. It reaches 900
summarized OmniVideoBench samples and confirms the expected behavior of the
dominant-expert confidence signal: it can reduce prefill latency, but by itself
it is not the best quality-preserving policy. Its 38.67% accuracy is below the
vanilla, both-signal, and boundary-only anchors.

## Interpretation

The V3 evidence strengthens the signal taxonomy:

- `p1-p2` decisiveness is useful as a risk guard because it asks whether the
  strongest expert is clearly preferred.
- `p_m-p_(m+1)` boundary stability is the stronger compute-reduction signal
  because it asks whether the truncation boundary is stable.
- Both-signal SCEB is the conservative global policy because it requires both
  confidence conditions before assigning smaller budgets.
- Boundary-only SCEB is the aggressive policy because it exposes more compute
  reduction but accepts more answer churn.

This makes the loop3 method-family split natural: keep conservative global SCEB
as the default recommendation, define boundary-only as an aggressive frontier,
and use guarded variants when a deployment wants more compute reduction while
limiting risk on audio-salient or boundary-unstable samples.

## Paper Claim Supported

Router probability geometry supplies complementary budget signals rather than a
single universal pruning score. Boundary stability identifies where the
top-m truncation point is stable; decisiveness protects against cases where the
top expert is not dominant enough. The best final method family should describe
these as budget-allocation controls for video-middle token-layer pairs, not as
global expert pruning.

## Caveats

- The current SCEB runs are mask-only unless a compact-dispatch run is
  explicitly named. Do not claim mask-only timing is the final compact-dispatch
  speedup.
- The decisiveness-only V3 row has full-run scale coverage, but the matched
  flip-rate and top-m-share audit still needs a dedicated post-processing pass
  before those exact churn fields should be filled.
- Audio is not pruned in this evidence. Audio remains protected top-8 and is
  better framed as a guard/controller for video budgets.

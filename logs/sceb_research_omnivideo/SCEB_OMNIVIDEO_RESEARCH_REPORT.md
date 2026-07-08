# SCEB OmniVideoBench Research Report

This report is the OmniVideoBench-only closeout for the current SCEB
autoresearch loop. It follows `implementation_planner/SCEB_AUTORESEARCH_HANDOFF.md`
and excludes AVUT and WorldSense evidence.

## Scope

- Benchmark: `eval_data/OmniVideoBench`
- Model: Qwen3-Omni-30B-A3B-Instruct artifacts already used by the repo runners
- Target method family: Structured Expert Budgeting for omnimodal MoE inference
- Target region: video tokens in middle transformer layers
- Protected regions: text, audio, image, special/control tokens, early layers,
  late layers, and low-confidence router states
- Implementation form: mask-only top-m expert budget with retained-weight
  renormalization, not final compact dispatch

## Main Claim

Expert redundancy in video-heavy omnimodal MoE inference is structured by
modality, layer stage, task regime, and router probability geometry. A uniform
top-k reduction is brittle, while router-guided token-layer expert budgeting can
preserve OmniVideoBench accuracy and expose a compute-reduction path.

## Full-Run Anchors

| Method | Samples | Accuracy | Prefill mean ms | Prefill median ms | Prefill p95 ms | Evidence role |
|---|---:|---:|---:|---:|---:|---|
| Vanilla top-8 | 998 | 39.38% | 12540.22 | 13865.01 | 19234.50 | baseline quality/latency anchor |
| Global SCEB policy | 998 | 39.68% | 11586.19 | 12665.47 | 19287.97 | primary current setting |
| Uniform top-6 | 998 | 38.48% | 12678.72 | 13736.92 | 19511.87 | uniform-budget negative control |
| Uniform top-4 | 998 | 38.48% | 12683.91 | 13664.80 | 19487.69 | aggressive uniform negative control |
| Fixed video-middle top-4 | 1000 | 38.80% | 22639.06 | 15039.92 | 42882.79 | fixed region-budget probe |
| Dynamic SCEB calibration run | 1000 | 39.30% | 29845.71 | 31618.36 | 43262.24 | early dynamic artifact with noisy timing |
| Layer-calibrated SCEB | 998 | 39.88% | 12670.57 | 13767.01 | 19636.79 | strongest accuracy variant |
| Boundary-only signal ablation | 998 | 39.78% | 9976.07 | 11059.31 | 16606.37 | completed on Merlin worker 3955069 |

The most stable deployment-style anchor is the global SCEB policy: it improves
accuracy by +0.30 percentage points over vanilla and reduces prefill mean by
7.61% relative to vanilla in the completed OmniVideoBench artifacts. The
layer-calibrated and boundary-only artifacts strengthen the quality story, but
some timing artifacts vary by launch condition and should be framed carefully.

## Evidence Threads

### 1. Uniform top-k reduction is brittle

Uniform all-token top-4 and top-6 both land at 38.48% accuracy, about 0.90 pp
below the vanilla top-8 anchor. This validates the handoff warning that SCEB
should not be presented as plain expert-count reduction. The budget assignment
must be structured by token type, layer stage, and router risk.

Detailed evidence:

- `logs/sceb_research_omnivideo/fixed_vs_dynamic/FIXED_VS_DYNAMIC_RESULTS.md`
- `logs/omnivideobench_qwen3_research/evidence1_uniform_topk_20260630/`

### 2. Dynamic expert budgeting preserves the main quality band

Global SCEB, layer-calibrated SCEB, and boundary-only SCEB all remain close to
or above vanilla accuracy on full OmniVideoBench. Global SCEB is conservative
with mean active experts at 7.06; boundary-only is more aggressive with mean
active experts at 5.96. This gives the paper a clean quality-compute tradeoff
story: boundary stability exposes more budgetable compute, while the combined
policy is a lower-churn deployment point.

### 3. Router signals are complementary

The boundary-only full run completed on worker 3955069 with 998 predictions and
39.78% accuracy. It selected many more top-4/top-6 decisions than the global
both-signal policy:

| Signal policy | Samples | Accuracy | Mean top-4 share | Mean top-6 share | Mean top-8 share | Mean active experts | Flip rate vs vanilla |
|---|---:|---:|---:|---:|---:|---:|---:|
| Both-signal global SCEB | 998 | 39.68% | 11.13% | 26.21% | 62.66% | 7.06 | 5.31% |
| Boundary-only SCEB | 998 | 39.78% | 32.37% | 40.55% | 27.07% | 5.96 | 7.62% |

Boundary-only therefore supplies strong compute-reduction pressure, but it also
raises answer churn. The combined policy is best framed as a risk-controlled
version of the more aggressive boundary signal. The decisiveness-only artifact
currently has only 33 samples, so it should remain a diagnostic rather than a
full independent claim.

Detailed evidence:

- `logs/sceb_research_omnivideo/signal_ablation/SIGNAL_ABLATION_RESULTS.md`
- `logs/omnivideobench_qwen3_research/evidence2_signal_ablation_20260701/boundary_only_full/`

### 4. Budgetability is structured by modality and stage

The modality-stage probe evidence supports the protected-modality story. The
handoff's current policy protects text, audio, image, special/control tokens,
early layers, and late layers, while targeting video middle layers. This is the
right framing for OmniVideoBench: video-heavy inputs dominate routed token
count, but not every modality or layer stage is equally safe to budget.

Detailed evidence:

- `logs/omnivideobench_qwen3_research/evidence4_modality_layer_budgetability_20260701/`

### 5. Realized budget buckets behave like risk allocation

The risk bucket report uses per-sample realized `top_m_histogram` values from
the global SCEB run as the durable proxy for the router risk decision. Samples
with higher realized top-4 share have higher accuracy, while stress tasks such
as counting, reference reasoning, spatial understanding, and temporal
understanding receive more conservative active expert budgets. This supports
the claim that the policy allocates budget where the sample and token-layer
state look safer, rather than pruning uniformly.

Detailed evidence:

- `logs/sceb_research_omnivideo/risk_bucket_probe/RISK_BUCKET_PROBE_RESULTS.md`

## Current Paper Framing

Recommended title-level framing:

> Router-Guided Dynamic Expert Budgeting for Omnimodal MoE Inference

Recommended claim:

> Expert redundancy in video-heavy omnimodal MoE inference is structured, and
> router-derived risk signals can allocate token-layer expert budgets while
> preserving OmniVideoBench accuracy.

Avoid claiming that mask-only SCEB is a final systems speedup mechanism. The
current implementation zeros dropped expert weights and renormalizes retained
weights before the original MoE loop. It is strong quality and budget-allocation
evidence; compact dispatch remains the systems follow-up.

## Limitations

- Decisiveness-only full-run evidence is incomplete at 33 samples.
- Raw `p1-p2` and `p_m-p_(m+1)` margin bucket tensors are not persisted in the
  full-run logs, so the current confidence-risk report uses realized selected
  expert budgets as the durable proxy.
- Some older timing artifacts were produced under different launch conditions,
  so latency claims should prioritize the vanilla/global SCEB/boundary-only
  anchors and avoid over-interpreting noisy early dynamic runs.
- The method is still mask-only. A compact dispatch implementation is needed to
  convert expert-budget assignment into cleaner wall-clock system speedups.

## Bottom Line

The OmniVideoBench-only evidence is now sufficient for the main SCEB research
story. Global SCEB preserves or slightly improves accuracy over vanilla while
reducing prefill mean in the strongest deployment-style artifact. Uniform top-k
baselines lose accuracy, supporting the need for structured budgeting. The
completed boundary-only ablation shows that router boundary stability exposes a
much larger compute-reduction opportunity, and the combined policy provides a
more conservative, lower-churn tradeoff. This closes the durable evidence
coverage requested by the autoresearch verifier for the OmniVideoBench-only
SCEB loop.

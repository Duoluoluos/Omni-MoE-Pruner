# SCEB Final Research Story

Date: 2026-07-13

This note consolidates the final paper story after the related-work deep review,
the completed OmniVideoBench experiments, and auxiliary AVUT/WorldSense transfer
runs. It intentionally narrows the claim away from broad multimodal dynamic
expert allocation and toward a defensible frozen-model deployment setting.

## One-Sentence Position

SCEB studies post-hoc router-risk expert budgeting for frozen omnimodal MoE
video inference: it uses existing router geometry and conservative
modality/layer/audio-query guards to reduce video-token expert budgets without
retraining, auxiliary importance predictors, virtual experts, or direct audio
expert pruning.

## Why The Broad Story Is Not Safe

The original broad framing, "structured dynamic expert budgeting for
multimodal MoE inference", collides with close prior work:

| Prior line | What it already covers | SCEB must avoid claiming |
|---|---|---|
| AnyExperts | trained multimodal token-level dynamic expert allocation with importance MLPs and virtual experts | first multimodal dynamic expert allocation |
| MoDES | training-free MLLM expert skipping with layer calibration and modality-specific thresholds | first training-free expert skipping for MLLMs |
| MACS / ReaLB / SMIDT | modality-aware MoE serving, capacity, precision, and dynamic-top-k systems | final systems speedup without compact dispatch evidence |
| OmniZip / OmniSelect / OmniSIFT | audio/query-guided AV token compression | audio token/expert pruning from the current evidence |

The safe paper target is therefore a stricter deployment question:

```text
Given a frozen omnimodal audio-video MoE, can we identify where video-heavy
inference has safe expert redundancy using only router geometry and conservative
guards?
```

## Core Observation

Expert redundancy is not uniform. It is localized to a region:

```text
video tokens x middle layers x stable router-boundary states
```

The protected region is equally important:

```text
audio / text / image / special tokens
early and late layers
low-confidence or fragile audio-video query regimes
```

This is the story transition from simple to deep:

1. Uniformly lowering top-k is too blunt.
2. Modality and layer stage define a safer budgetable region.
3. Router-boundary geometry decides whether a token-layer pair is safe inside
   that region.
4. Audio/query semantics do not become a pruning target; they become a guard
   that decides when video expert budgeting is too risky.

## Method Family

### Main SCEB

Main SCEB is a frozen-model, mask-only policy over the original top-8 expert
candidates. It does not retrain the router and does not add an importance
network.

For target video tokens in middle layers:

```text
top-m = 4 if dominant-expert confidence is high and the top-4 boundary is stable
top-m = 6 if confidence is medium and the top-6 boundary is stable
top-m = 8 otherwise
```

All protected tokens keep top-8. Weights of retained experts are renormalized.
The current measured implementation is mask-only; compact dispatch is the
systems step needed to convert effective assignment reduction into robust
wall-clock speedup.

### Audio-Guard Extension

Audio-Guard is not audio expert pruning. It is audio-conditioned video expert
budgeting:

```text
audio/query regime -> AV grounding risk -> allowed video expert budget
```

The final Audio-Guard v4 policy is:

- `background&music understanding`: video remains full top-8;
- temporal, reference, spatial, and ego rows: top-6 floor plus layers 24-31
  full guard;
- lower-risk counting, causal, relationship, sentiment, summarization,
  hypothetical, fine-grained, and attribute rows: speed-oriented router
  thresholds;
- audio/text/image/special tokens always remain top-8.

This gives a clear "audio as protected controller" story that is consistent
with token-compression prior work without claiming direct audio compression.

## Evidence Chain

### Evidence 1: Uniform Top-K Reduction Is Brittle

Uniform top-k reduction lowers expert count without considering modality,
layer stage, or router risk. It is a necessary negative control because it
answers the simplest reviewer objection: "Is SCEB just lowering top-k?"

Existing full-run anchors show that structured policies are needed. Uniform
top-k rows should remain in the main experiment table; SCEB should be compared
against them as the first baseline, not as the only baseline.

### Evidence 2: Main SCEB Preserves Quality While Improving Prefill

Full OmniVideoBench anchors:

| Method | Samples | Correct | Accuracy | Prefill mean | Prefill median | Prefill p95 |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla top-8 | 998 | 393 | 39.38% | 12.54s | 13.87s | 19.23s |
| Global SCEB | 998 | 396 | 39.68% | 11.59s | 12.67s | 19.29s |
| Boundary-only SCEB | 998 | 397 | 39.78% | 9.98s | 11.06s | 16.61s |
| Audio-Guard v4 | 998 | 390 | 39.08% | 9.19s | 10.12s | 14.24s |

Interpretation:

- Global SCEB is the conservative main quality anchor: slightly above vanilla
  with lower prefill.
- Boundary-only is the aggressive compute frontier: better prefill, higher
  churn risk.
- Audio-Guard v4 is the audio-conditioned frontier: fastest full-set prefill
  among current anchors, with a sub-1pp drop against Global SCEB.

### Evidence 2b: Global SCEB Transfers To Other Audio-Video Benchmarks

AVUT and WorldSense are auxiliary transfer checks, not the main benchmark used
to tune the story. Both were rerun fairly on the same `3955069` worker with
2x A100-80GB, full data, same model, and zero OOM/error samples. The transferred
policy is Global SCEB, not Audio-Guard v4, so the clean claim is that the
post-hoc protected-modality/video-middle budgeting rule is not OmniVideoBench
specific.

| Benchmark | Method | Samples | Correct | Accuracy | Prefill mean | E2E/generate mean |
|---|---|---:|---:|---:|---:|---:|
| AVUT | Vanilla top-8 | 1734 | 1234 | 71.16% | 3409.9 ms | 4263.2 ms |
| AVUT | Global SCEB | 1734 | 1226 | 70.70% | 3372.3 ms | 3698.7 ms |
| WorldSense | Vanilla top-8 | 3172 | 1579 | 49.78% | 5277.3 ms | 6591.4 ms |
| WorldSense | Global SCEB | 3172 | 1581 | 49.84% | 5189.8 ms | 5988.1 ms |

Transfer deltas:

| Benchmark | Accuracy delta | Prefill mean delta | E2E/generate mean delta | Mean active experts |
|---|---:|---:|---:|---:|
| AVUT | -0.46 pp | -1.10% | -13.24% | 6.895 |
| WorldSense | +0.06 pp | -1.66% | -9.15% | 6.895 |

Interpretation:

- AVUT is the stricter transfer result: SCEB costs 0.46pp accuracy but improves
  generation-side latency substantially.
- WorldSense is the cleanest transfer result: SCEB slightly improves accuracy
  while reducing both prefill and generation-side latency.
- These results support the deployment story that SCEB is a frozen-model
  budgeting rule with cross-dataset robustness, while OmniVideoBench remains the
  primary benchmark for mechanism analysis and Audio-Guard design.

Loop12 completes the full-set cross-benchmark Audio-Guard transfer check:

| Benchmark | Method | Correct | Accuracy | Prefill mean | Mean active | Churn vs Global SCEB |
|---|---|---:|---:|---:|---:|---:|
| AVUT | Vanilla top-8 | 1234/1734 | 71.16% | 3409.9 ms | - | - |
| AVUT | Global SCEB | 1226/1734 | 70.70% | 3372.3 ms | 6.895 | - |
| AVUT | Audio-Guard v1 | 1228/1734 | 70.82% | 3467.9 ms | 6.957 | 12 flips |
| WorldSense | Vanilla top-8 | 1579/3172 | 49.78% | 5277.3 ms | - | - |
| WorldSense | Global SCEB | 1581/3172 | 49.84% | 5189.8 ms | 6.895 | - |
| WorldSense | Audio-Guard v1 | 1591/3172 | 50.16% | 5401.4 ms | 6.996 | 24 flips |

This strengthens the "audio/query as guard" story, but with an important
qualification: the current cross-benchmark v1 is a quality/churn guard, not a
speed frontier. It recovers fragile-task accuracy on AVUT and improves full-set
WorldSense accuracy, while giving back some prefill improvement. The next
cross-benchmark step should sweep guard strength to recover speed.

### Evidence 3: Router-Boundary Risk Is A Real Signal

The confidence-risk probe shows boundary stability is monotonic with the
observed budget/quality behavior:

| Boundary bucket | Samples | Mean boundary margin | Accuracy | Flip rate | Mean active experts | Top-8 share |
|---|---:|---:|---:|---:|---:|---:|
| Low | 332 | 0.000545 | 37.35% | 4.82% | 7.12 | 66.40% |
| Mid | 333 | 0.000590 | 38.14% | 6.01% | 7.07 | 62.64% |
| High | 333 | 0.000660 | 43.54% | 5.11% | 7.01 | 58.96% |

This supports the core difference from a pure local-probability threshold:
SCEB is not only asking whether a selected expert has high probability; it also
asks whether the truncation boundary is stable enough to drop the tail.

### Evidence 4: Signal Variants Expose A Quality-Speed Frontier

The signal-ablation evidence separates deployment and frontier settings:

| Policy | Samples | Accuracy | Mean active experts | Prefill mean | Role |
|---|---:|---:|---:|---:|---|
| Global both-signal SCEB | 998 | 39.68% | 7.06 | 11.59s | conservative deployment setting |
| Boundary-only SCEB | 998 | 39.78% | 5.96 | 9.98s | aggressive speed frontier |
| Decisiveness-only SCEB | partial | incomplete | 6.62 | 9.11s partial | diagnostic, not final claim |

The paper should not overstate one setting. The method family is better
presented as a controllable budget frontier.

### Evidence 5: Audio Is A Guard, Not A Target

Loop6 measured that audio and video routes are coupled but not
interchangeable:

| Measurement | Value |
|---|---:|
| Audio/video top-8 expert Jaccard overlap | 0.142 |
| Audio trajectory mean `p1-p2` | 0.01009 |
| Video trajectory mean `p1-p2` | 0.00723 |
| Audio/video token ratio on audio-salient subset | 2.53% |

This motivates audio-as-query:

```text
audio is too semantically important to drop blindly, but useful enough to tell
us when video expert reduction is risky
```

Audio-Guard v4 validates this framing. On the audio-salient subset, it recovers
all four samples lost by the aggressive v3 speed policy while preserving almost
the same effective expert-budget distribution.

| Method | Correct | Accuracy | Mean active | Top-4 | Top-6 | Top-8 |
|---|---:|---:|---:|---:|---:|---:|
| Audio-Guard v2 | 83/200 | 41.50% | 7.38 | 5.09% | 25.64% | 69.27% |
| Audio-Conditioned speed v3 | 79/200 | 39.50% | 6.90 | 18.84% | 25.15% | 56.01% |
| Audio-Guard v4 | 83/200 | 41.50% | 6.92 | 18.01% | 25.83% | 56.15% |

Full-set v4 extends the same story:

| Method | Samples | Accuracy | Prefill mean | Mean active | Top-4 / Top-6 / Top-8 |
|---|---:|---:|---:|---:|---|
| Audio-Guard v4 full | 998 | 39.08% | 9.19s | 6.68 | 19.11 / 31.95 / 48.93% |

The correct claim is:

```text
Audio-conditioned guards can recover the quality loss of aggressive video
expert budgeting while preserving most of the effective expert-budget
reduction.
```

## Final Claims

Use these claims:

1. Frozen omnimodal MoE video inference contains structured expert redundancy.
2. The safe region is video middle-layer token routing, not all modalities.
3. Router-boundary risk is a lightweight post-hoc proxy for safe tail expert
   removal.
4. Audio/query semantics should be used as a guard for video expert budgets,
   not as an audio expert pruning target.
5. Global SCEB transfers to AVUT and WorldSense with near-flat accuracy and
   modest prefill improvement, supporting the claim that the protected-region
   rule is not overfit to OmniVideoBench.
6. Current mask-only experiments establish the policy and prefill frontier;
   compact dispatch is needed for final serving-level speedup claims.

Avoid these claims:

- first multimodal dynamic expert allocation;
- first training-free MLLM expert skipping;
- audio expert pruning;
- mask-only assignment reduction equals final compact-dispatch speedup;
- Audio-Guard v4 beats the best quality anchor.

## Paper Structure

1. **Introduction.** Frozen omnimodal MoE inference is expensive; broad dynamic
   allocation exists but often requires training or calibration. We study a
   stricter deployment question.
2. **Motivation.** Uniform expert reduction is brittle; modality/layer/router
   structure matters.
3. **Method.** SCEB as post-hoc router-boundary risk calibration over video
   middle-layer token budgets, with protected modalities.
4. **Audio-Guard.** Audio/query regimes are risk controllers for video budget;
   audio tokens stay protected.
5. **Experiments.** Vanilla, uniform top-k, Global SCEB, Boundary-only,
   Audio-Guard v4, related-work-inspired local-probability baseline, and
   AVUT/WorldSense transfer checks.
6. **Systems note.** Mask-only validates policy behavior; compact dispatch is
   the path to realized serving speedup.

## Loop9 Result: Related-Work-Inspired Local Probability Baseline

Loop9 directly addresses the MoDES-style reviewer comparison:

```text
local router-probability threshold baseline
```

It keeps the same protected modalities and video-middle target region as SCEB,
but replaces router-boundary risk with a local selected-expert probability
threshold. This isolates whether SCEB's boundary rule adds value beyond a
generic local-router skipping rule.

Completed result:

| Method | Samples | Correct | Accuracy | Prefill mean | Mean active experts | Top-4 / Top-6 / Top-8 |
|---|---:|---:|---:|---:|---:|---|
| Local-prob threshold baseline | 998 | 392 | 39.28% | 9.36s | 5.42 | 42.65 / 49.78 / 7.57% |

Interpretation:

- Local probability thresholding is a strong speed baseline, but it is much
  more aggressive than Global SCEB and does not improve quality.
- It chooses top-4/top-6 for more than 92% of measured assignments, confirming
  that local probability alone tends to collapse toward aggressive expert
  reduction.
- The result strengthens the need to present SCEB as router-boundary and
  protected-region budgeting, not as a generic local probability threshold.

1. Audio Query在别的Benchmark上也验证有效
2. 读其他文章，token裁剪/纯文本/VLM，全模态上面有什么特殊的挑战，在全模态上加速的意义和动机说清楚，做出特殊的发现
3. 方法和其他流派（例如token裁剪方法）能否结合使用

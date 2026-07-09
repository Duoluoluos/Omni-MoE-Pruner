# SCEB Method Taxonomy

This taxonomy defines the OmniVideoBench SCEB method family after phase 2 and
loop3 signal finalization. SCEB means Structured Expert Budgeting: assigning a
token-layer expert budget from `{4, 6, 8}` for video-heavy omnimodal MoE
inference under modality, layer-stage, and router-risk constraints.

The taxonomy is intentionally OmniVideoBench-only. It does not use AVUT or
WorldSense evidence, and it does not claim that mask-only timing is the final
compact-dispatch speedup.

## Shared Base Policy

All SCEB variants share these structural constraints:

```text
target modality: video
target layer stage: middle transformer layers
protected modalities: text, audio, image, special/control
protected stages: early layers, late layers, decode path
budget set: top-4, top-6, top-8
execution status: mask-only unless compact dispatch is explicitly named
```

The common risk signals are:

```text
decisiveness = p1 - p2
boundary stability = p_m - p_(m+1)
```

`decisiveness` asks whether the dominant expert is clearly preferred.
`boundary stability` asks whether the truncation point for the selected budget
is stable. The loop3 signal-ablation V3 report shows that these are
complementary controls rather than one interchangeable scalar score.

## Conservative Global SCEB

Conservative Global SCEB is the default deployment-style method.

```text
if token is video and layer is middle and both risk tests pass:
    assign top-4 or top-6 by calibrated thresholds
else:
    keep top-8
```

Evidence anchor:

`logs/omnivideobench_qwen3_budget_policy/policy_calib40_full1000_2gpu_20260623_111524`

Observed OmniVideoBench behavior:

| Metric | Value |
|---|---:|
| Samples | 998 |
| Accuracy | 39.68% |
| Mean prefill latency | 11.59s |
| Mean active experts | 7.06 |
| Top-4 / top-6 / top-8 share | 11.13% / 26.21% / 62.66% |
| Answer flip rate vs vanilla | 5.31% |

Use this variant when the goal is a conservative quality-preserving expert
budget with clear protection for non-video modalities and uncertain router
states. This is the primary paper recommendation because it gives a better
accuracy-latency tradeoff than uniform top-k while keeping answer churn lower
than aggressive boundary-only SCEB.

## Aggressive Boundary-only SCEB

Aggressive Boundary-only SCEB removes the dominant-expert decisiveness guard and
uses truncation-boundary stability as the main budgetability signal.

```text
if token is video and layer is middle and boundary stability passes:
    assign top-4 or top-6
else:
    keep top-8
```

Evidence anchor:

`logs/omnivideobench_qwen3_research/evidence2_signal_ablation_20260701/boundary_only_full`

Observed OmniVideoBench behavior:

| Metric | Value |
|---|---:|
| Samples | 998 |
| Accuracy | 39.78% |
| Mean prefill latency | 9.98s |
| Mean active experts | 5.96 |
| Top-4 / top-6 / top-8 share | 32.37% / 40.55% / 27.07% |
| Answer flip rate vs vanilla | 7.62% |

Use this variant as the aggressive frontier. It exposes the largest measured
mask-only compute reduction among the signal-ablation methods, but it should be
presented with its higher churn risk. It is a method-family member, not the
default final recommendation.

## Decisiveness-only Diagnostic

Decisiveness-only SCEB keeps the dominant-expert confidence signal and removes
the explicit boundary-stability condition.

Evidence anchor:

`logs/omnivideobench_qwen3_research/evidence2_signal_ablation_20260701/decisiveness_only_full`

Observed OmniVideoBench behavior:

| Metric | Value |
|---|---:|
| Predictions | 902 |
| Summarized samples | 900 |
| Accuracy | 38.67% |
| Mean prefill latency | 9.11s |

This variant is a diagnostic, not the recommended deployment policy. It shows
that `p1-p2` can support compute reduction, but by itself it does not preserve
quality as well as Conservative Global SCEB or Boundary-only SCEB in the
current durable artifacts. Its best role is as a guard component.

## Boundary-Guarded SCEB

Boundary-Guarded SCEB starts from Aggressive Boundary-only SCEB and adds a risk
guard to reduce answer churn while preserving most of the prefill gain.

Recommended loop3 guard shape:

```text
base policy: boundary-only SCEB
guard conditions:
    low decisiveness on video-middle routing
    stress task type such as temporal, spatial, reference, or counting
    unstable realized budget region with high top-8 protection in prior runs
action:
    raise guarded video top-4 decisions to at least top-6
    keep already top-6 or top-8 decisions unchanged
    keep text/audio/image/special top-8
```

Expected role:

- preserve the boundary-only compute opportunity,
- reduce churn on unstable or stress-task samples,
- remain video-only in the sense that only video budgets are changed.

This is the main loop3 probe target for method finalization.

## Audio-as-Guard SCEB

Audio-as-Guard SCEB keeps audio tokens protected at top-8 and uses audio-side
salience as a controller for video expert budgets.

Recommended loop3 guard shape:

```text
audio tokens: full top-8
if audio salience is high or speech-plus-stress-task:
    raise guarded video top-4 decisions to at least top-6
else:
    use normal video SCEB decisions
```

Evidence anchor:

`logs/sceb_research_omnivideo/audio_guard/AUDIO_GUARD_OMNIVIDEO_RESULTS.md`

The phase-2 counterfactual selected 546 of 998 samples and raised mean active
experts by only 0.069 if guarded top-4 video decisions were lifted to top-6.
This makes audio-as-guard a conservative deployment mode: it spends a small
amount of expert compute on audio-sensitive samples without directly pruning
audio.

## Recommended Naming

| Family name | Short name | Role |
|---|---|---|
| Conservative Global SCEB | Global SCEB | default paper recommendation |
| Aggressive Boundary-only SCEB | Boundary-only SCEB | compute frontier |
| Boundary-Guarded SCEB | Boundary-Guarded | aggressive with churn guard |
| Audio-as-Guard SCEB | Audio-Guarded | audio-aware conservative mode |

## Final Recommendation

Use Conservative Global SCEB as the primary method and report Boundary-only SCEB
as an aggressive frontier. Treat guarded variants as deployment controls:
Boundary-Guarded SCEB for lower churn from the aggressive frontier, and
Audio-as-Guard SCEB for cross-modal samples where audio should control video
budget aggressiveness.

The final method story should say:

```text
SCEB is a family of router-guided expert budget allocation policies. The
default policy is conservative and global; the boundary-only variant exposes the
compute frontier; guarded variants selectively raise video budgets when risk
signals indicate higher churn.
```

# SCEB Audio Loop8 Report

Loop8 tested a stronger Audio-Guard v4 policy family for AV-SCEB on the
OmniVideoBench audio-salient subset. The final method decision for the long
paper is to keep Audio-Guard v4 as the recommended audio-aware policy-family
member, while keeping Global SCEB as the main matched quality anchor.

## Method Decision

Audio-Guard v4 should be presented as:

```text
Audio-conditioned video expert budgeting with restored fragile-row guards.
```

It should not be presented as audio token pruning. Audio, text, image, and
special/control tokens remain protected at full top-8. The expert-budget target
is video tokens in middle layers, and audio/query metadata controls when video
budgeting is allowed to be aggressive.

The selected v4 policy is the guard-restored v3.5 candidate:

- `background&music understanding`: video remains full top-8;
- temporal, reference, spatial, and ego rows: top-6 floor plus late-middle
  full guards on layers 24-31;
- lower-risk counting, causal, relationship, sentiment, summarization,
  hypothetical, fine-grained, and attribute rows: Loop7 v3 speed thresholds;
- early and late layers: full top-8;
- implementation: `confidence_dynamic_topk_mask`, mask-only, no compact
  dispatch claim.

## Measured Result

The worker-3955069 smoke2 and limit-200 runs completed successfully:

| Method | Accuracy | Correct | Mean active experts | Top-4 | Top-6 | Top-8 |
|---|---:|---:|---:|---:|---:|---:|
| Audio-Guard v4 | 41.50% | 83/200 | 6.92 | 18.01% | 25.83% | 56.15% |
| Audio-Guard v2 | 41.50% | 83/200 | 7.38 | 5.09% | 25.64% | 69.27% |
| Loop7 speed v3 | 39.50% | 79/200 | 6.90 | 18.84% | 25.15% | 56.01% |
| Global SCEB | 42.50% | 85/200 | 7.04 | 13.27% | 22.08% | 64.65% |
| Boundary-only SCEB | 41.50% | 83/200 | 5.84 | 40.38% | 31.15% | 28.47% |
| Vanilla top-8 | 42.00% | 84/200 | n/a | n/a | n/a | n/a |

V4 exactly matches the Audio-Guard v2 score while preserving most of the Loop7
speed-policy budget reduction. Against v3, it recovers four samples with no
correct-to-wrong losses in the direct 200-row churn comparison. Against v2, it
ties total accuracy with two correct-to-wrong and two wrong-to-correct moves.

## Paper Claim

The long paper should use Loop8 to support this claim:

```text
Audio evidence is useful as a controller for video expert budgeting. Restoring
guards for fragile audio-video regimes can recover the quality loss of an
aggressive speed policy while keeping most of its effective expert-budget
reduction.
```

This is stronger than the Loop7 speed-only story because v4 no longer pays the
four-sample loss. It is also more informative than Loop4 v2 because v4 reaches
the same 83/200 score with a much lower mean active expert count and a much
higher top-4 share.

## What Not To Claim

Do not claim that Audio-Guard v4 beats the best matched quality anchor. Global
SCEB remains higher on this subset at 85/200, and Vanilla top-8 remains higher
than v4 by one sample at 84/200. V4 is an AV-SCEB family result, not a global
replacement for the main SCEB quality setting.

Do not claim realized wall-clock speedup. V4 has a lower effective expert
budget than v2, but the measured mask-only wall-clock numbers are slightly
slower:

| Method | Prefill mean | Generation / E2E mean |
|---|---:|---:|
| Audio-Guard v4 | 9,330 ms | 12,200 ms |
| Audio-Guard v2 | 9,078 ms | 11,720 ms |
| Loop7 speed v3 | 9,162 ms | 11,788 ms |

The systems conclusion is that compact dispatch is still required to turn the
effective budget reduction into wall-clock speed.

## Final Recommendation

Keep Audio-Guard v4 as the final Loop8 method. In the paper structure:

1. Use Global SCEB as the main quality anchor for structured expert budgeting.
2. Use Audio-Guard v2 as the conservative audio-aware baseline.
3. Use Loop7 speed v3 as the aggressive audio-conditioned speed frontier.
4. Use Audio-Guard v4 as the preferred AV-SCEB policy because it recovers v3
   quality while preserving most of the effective budget reduction.

The final wording should emphasize protected audio and controlled video budget:

```text
We do not drop audio experts. Instead, audio/query regimes identify when video
expert budgeting is risky. Audio-Guard v4 restores full or top-6 video budgets
for fragile audio-video rows and applies speed-oriented router thresholds only
where prior routing evidence marked the row as lower risk.
```

That is the clearest Loop8 contribution for the long paper.

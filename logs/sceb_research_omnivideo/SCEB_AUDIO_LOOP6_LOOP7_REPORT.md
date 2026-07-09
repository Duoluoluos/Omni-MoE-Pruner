# SCEB Audio Loop6 Loop7 Report

This report closes Loop6 and Loop7 for the audio-aware SCEB long paper track.
Loop6 built routing-risk and audio-video expert coupling evidence from
measured OmniVideoBench artifacts. Loop7 converted that evidence into a
speed-oriented Audio-Conditioned policy v3 and measured it on worker 3955069.

## Completed Artifacts

| Loop | Artifact | Role |
|---|---|---|
| Loop6 | `logs/sceb_research_omnivideo/loop6/AUDIO_ROUTING_RISK_MAP.md` | Audio routing risk map and policy zones |
| Loop6 | `logs/sceb_research_omnivideo/loop6/AUDIO_VIDEO_EXPERT_COUPLING.md` | Audio-video expert coupling evidence |
| Loop7 | `logs/sceb_research_omnivideo/loop7/AUDIO_CONDITIONED_SPEED_POLICY_V3.md` | Policy definition and measurement plan |
| Loop7 | `logs/omnivideobench_qwen3_research/loop7_audio_conditioned_speed_v3_20260709/smoke2` | Smoke2 measured worker run |
| Loop7 | `logs/omnivideobench_qwen3_research/loop7_audio_conditioned_speed_v3_20260709/limit200` | Full limit-200 measured worker run |
| Loop7 | `logs/sceb_research_omnivideo/loop7/AUDIO_CONDITIONED_SPEED_V3_RESULTS.md` | Matched speed-quality tradeoff report |

All measured Loop7 generation used only OmniVideoBench data. No WorldSense or
AVUT runs are used for this Loop6/Loop7 closeout.

## Loop6 Finding

Loop6 shows that audio and video expert routing are coupled but not
interchangeable. On the 200-row audio-salient subset, mean audio/video top-8
expert Jaccard overlap is 0.142 over 9,600 sample-layer comparisons. Audio
trajectory margins are higher than video trajectory margins on average
(0.01009 vs 0.00723), while audio token volume is only 2.53% of video token
volume.

This supports an audio-as-controller framing:

- audio should remain protected unless direct audio-pruning evidence exists;
- audio/query regime is still useful for deciding video expert-budget risk;
- video tokens dominate the expert-compute frontier and are the proper speed
  target for this stage of SCEB.

The Loop6 risk map divides the subset into three policy zones. Audio-core rows
(`background&music understanding`) should protect video at full top-8. Fragile
AV grounding rows such as temporal, reference, and spatial should keep a
top-6/late-middle guard. Lower-risk audio-salient rows can test speed-oriented
video budgets.

## Loop7 Policy And Measurement

Audio-Conditioned speed policy v3 implements that map with the existing
`confidence_dynamic_topk_mask` mechanism:

- target modality: video;
- protected modalities: text, audio, image, special/control;
- audio-core rows: full video top-8;
- temporal/reference/spatial rows: top-6 floor with late-middle full guard;
- lower-risk rows: top-4 allowed under router confidence and boundary margins.

The measured limit-200 result:

| Method | Matched | Accuracy | Correct | Mean active experts | Top-4 | Top-6 | Top-8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Audio-Conditioned speed v3 | 200 | 39.50% | 79/200 | 6.90 | 18.84% | 25.15% | 56.01% |
| Audio-Guard v2 | 200 | 41.50% | 83/200 | 7.38 | 5.09% | 25.64% | 69.27% |
| Global SCEB | 200 | 42.50% | 85/200 | 7.04 | 13.27% | 22.08% | 64.65% |
| Boundary-only SCEB | 200 | 41.50% | 83/200 | 5.84 | 40.38% | 31.15% | 28.47% |
| Vanilla top-8 | 200 | 42.00% | 84/200 | - | - | - | - |

v3 loses 4 samples against Audio-Guard v2 and 6 samples against Global SCEB.
Its answer flip rate against Global is 5.50%, with 6 Global-correct rows
becoming wrong and no Global-wrong rows becoming correct. That is too much
quality loss for default promotion, but it is useful speed-frontier evidence.

## Speed Tradeoff

The speed-oriented result is strongest in effective expert budget rather than
wall-clock latency. Against Audio-Guard v2, v3 reduces mean active experts from
7.38 to 6.90, raises top-4 assignment share from 5.09% to 18.84%, and lowers
top-8 share from 69.27% to 56.01%.

Wall-clock prefill and generation/E2E latency are roughly flat on this subset:
v3 prefill mean is 9,162 ms versus 9,078 ms for Audio-Guard v2, and generation
mean is 11,788 ms versus 11,720 ms. This is not contradictory. The current
implementation is mask-only: dropped experts are zeroed and weights are
renormalized, but the original expert loop still executes. Therefore the paper
should distinguish effective expert-budget reduction from realized compact
dispatch speedup.

## Long-Paper Position

Loop6+Loop7 strengthen the long-paper audio-aware SCEB section in three ways:

1. They provide routing-risk evidence explaining when audio should protect
   video expert budgets.
2. They provide expert-coupling evidence showing audio and video routes share
   some expert neighborhoods but are not interchangeable.
3. They provide a measured speed policy that trades accuracy for lower
   effective video expert budget while keeping audio protected.

The recommended wording is:

> Audio-aware SCEB treats audio as a protected controller for video expert
> budgeting. Audio-core queries protect video expert compute; lower-risk
> audio-salient queries can move toward smaller video expert budgets. In the
> current mask-only implementation this primarily reduces effective expert
> assignments; compact dispatch is needed to convert the budget reduction into
> robust wall-clock speedup.

Do not claim audio token pruning. Do not claim v3 dominates Global SCEB or
Audio-Guard v2 in quality. The correct claim is that Loop7 v3 exposes a
measured speed-quality frontier for audio-conditioned video expert budgeting.

## Final Decision

Keep Audio-Guard v2 as the conservative audio-aware extension. Keep
Audio-Conditioned speed v3 as the speed-oriented ablation. Do not promote v3 as
the mainline quality policy. The best main paper structure is:

- mainline SCEB: Global / router-guided dynamic video expert budgeting;
- audio-aware extension: Audio-Guard v2 for conservative AV protection;
- speed-frontier ablation: Audio-Conditioned v3 for lower effective expert
  budgets and explicit quality tradeoff;
- future systems work: compact dispatch to realize the effective expert-budget
  reduction as wall-clock latency improvement.

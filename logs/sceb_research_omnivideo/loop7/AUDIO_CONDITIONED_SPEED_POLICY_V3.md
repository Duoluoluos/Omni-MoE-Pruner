# Loop7 Audio-Conditioned Speed Policy v3

Audio-Conditioned speed policy v3 is the Loop7 policy candidate for
speed-oriented AV-SCEB evidence on the OmniVideoBench audio-salient subset. It
uses the Loop6 routing-risk map to decide when audio should protect video
expert budgets and when video-token budgeting can be more aggressive.

## Policy File

`logs/omnivideobench_qwen3_research/policy_variants_20260709/loop7_audio_conditioned_speed_v3_policy.json`

## Supported Mechanism

The policy uses the existing `confidence_dynamic_topk_mask` implementation in
`moe_expert_budget`. It does not introduce a new runtime mechanism. The policy
remains mask-only rather than compact dispatch:

- target modality: video;
- protected modalities: text, audio, image, special/control;
- dynamic layers: middle layers by the established 0.25-0.67 layer ratio;
- early and late layers: full top-8;
- audio tokens: always protected at full top-8;
- video-token budgets: dynamic top-4 / top-6 / top-8 by router geometry.

This means any measured speed effect is from changing the video-token expert
budget distribution in the existing code path. It is not an audio token pruning
claim.

## Difference From Audio-Guard v2

Loop4 Audio-Guard v2 was conservative. It set `background&music understanding`
to full top-8, raised counting/temporal/reference/spatial and several reasoning
tasks to a minimum top-6, and protected late-middle layers for audio-core and
reference/spatial rows.

Policy v3 keeps the audio-core protection but shifts the non-core rows toward a
speed search:

| Regime | v3 rule | Reason |
|---|---|---|
| `background&music understanding` | min top-8, layers 24-31 full | Audio-core rows have the clearest sound dependence and lowest measured accuracy. |
| temporal / spatial / reference | min top-6, layers 28-31 full | Fragile AV grounding rows keep a late-middle guard. |
| counting / causal / relationship / sentiment / summarization | min top-4 | Loop6 marked these as better speed candidates when audio remains protected. |
| hypothetical / ego / fine-grained / attribute | min top-4 | High-latency or lower audio-core rows are useful speed probes. |

The explicit router thresholds are also slightly more speed-oriented than v2:

| Threshold | v2 | v3 |
|---|---:|---:|
| top1-top2 high | 0.008606 | 0.006500 |
| top1-top2 medium | 0.002971 | 0.002300 |
| selected boundary | 0.000559 | 0.000450 |

Lowering these thresholds should increase top-4/top-6 video-token assignments
on rows that are already not audio-core. The expected risk is one or two extra
wrong answers on the 200-row subset; the expected benefit is lower prefill and
generation latency or at least lower effective expert assignments.

## Measurement Plan

Run the smoke first, then the real limit-200 generation on the same
audio-salient subset:

1. `logs/omnivideobench_qwen3_research/loop7_audio_conditioned_speed_v3_20260709/run_smoke_speed_v3.sh`
2. `logs/omnivideobench_qwen3_research/loop7_audio_conditioned_speed_v3_20260709/run_limit200_speed_v3.sh`

The launcher for worker 3955069 is:

`logs/omnivideobench_qwen3_research/loop7_audio_conditioned_speed_v3_20260709/launch_loop7_speed_v3_3955069.sh`

## Keep Criteria For Long-Paper Evidence

Policy v3 should be kept as useful speed evidence if the limit-200 run:

- completes at least 180 rows on the audio-salient subset;
- keeps audio tokens protected and reports `policy_version =
  sceb_loop7_audio_conditioned_speed_v3`;
- loses no more than a small number of extra samples relative to Loop4
  Audio-Guard v2 or Global SCEB on the matched subset;
- improves prefill latency, generation latency, or effective expert assignment
  share enough to explain the speed/quality tradeoff.

If v3 is faster but slightly less accurate, the paper framing should present it
as the speed-oriented member of the AV-SCEB family rather than as the default
quality policy. The result is still useful because it demonstrates that audio
can condition the video expert budget frontier without pruning audio tokens.

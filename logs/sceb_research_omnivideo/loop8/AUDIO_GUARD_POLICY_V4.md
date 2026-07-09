# Loop8 Audio-Guard Policy v4

Audio-Guard v4 is the first runnable Loop8 policy for AV-SCEB on the
OmniVideoBench audio-salient subset. It implements the guard-restored v3.5
candidate from `AUDIO_GUARD_V4_CANDIDATES.md`.

## Policy File

`logs/omnivideobench_qwen3_research/policy_variants_20260709/loop8_audio_guard_v4_policy.json`

The policy version is:

```text
sceb_loop8_audio_guard_v4
```

## Mechanism

V4 uses the existing `confidence_dynamic_topk_mask` path in
`moe_expert_budget`. It remains a mask-only expert-budget policy rather than a
compact-dispatch implementation. The target is video-token expert budget, not
audio pruning:

- target modality: video;
- protected modalities: text, audio, image, and special/control tokens;
- original top-k: 8;
- dynamic top-m options: 4, 6, and 8;
- early layers: full top-8;
- middle layers: dynamic by router geometry;
- late layers: full top-8;
- audio tokens: always full top-8.

The thresholds intentionally reuse the Loop7 speed v3 frontier:

| Threshold | Value |
|---|---:|
| top1-top2 high | 0.006500 |
| top1-top2 medium | 0.002300 |
| selected boundary | 0.000450 |

Those thresholds are only allowed to act freely in rows that Loop6 marked as
lower audio-risk. Fragile audio-video rows receive restored guards.

## Task Policy

| Task cluster | Tasks | Minimum video top-m | Reason |
|---|---|---:|---|
| audio core | `background&music understanding` | 8 | Explicit sound/music dependence and low measured v2 accuracy. |
| fragile AV restored | `temporal understanding`, `spatial understanding`, `reference reasoning`, `ego reasoning` | 6 | Loop7 showed quality loss in fragile grounding and ego regimes. |
| speed-safe audio | `counting`, `causal reasoning`, `relationship reasoning`, `sentiment analysis`, `summarization` | 4 | Loop6 marked these as better speed candidates when audio remains protected. |
| speed-probe reasoning | `hypothetical reasoning`, `fine-grained perception`, `attribute comparison` | 4 | Useful for preserving the v3 expert-budget frontier. |
| default | all other rows | 4 | Fall back to speed-oriented dynamic video budgeting. |

## Task-Layer Full Guards

V4 adds full top-8 video protection for late-middle integration layers:

| Guard | Tasks | Layers |
|---|---|---|
| audio-core late-middle full | `background&music understanding` | 24-31 |
| fragile AV late-middle full | `temporal understanding`, `spatial understanding`, `reference reasoning`, `ego reasoning` | 24-31 |

The v3 policy only forced layers 28-31 full for temporal, reference, and
spatial rows, and left ego at the speed-probe floor. V4 widens that guard to
24-31 and adds ego because Loop7 dropped ego accuracy from 30% in v2 to 20%.

## Difference From v2 And v3

Compared with Audio-Guard v2, v4 keeps the quality-first audio-core protection
but uses the v3 router thresholds for lower-risk rows. This should increase
top-4 assignments relative to v2.

Compared with Loop7 speed v3, v4 restores top-6 floors and late-middle full
guards for temporal, reference, spatial, and ego rows. This should recover a
portion of the four-sample loss against v2 while preserving some effective
expert-budget reduction.

## Expected Measurement

The intended result band is:

- at least 82/200 correct on the audio-salient limit-200 run;
- mean active experts below the v2 value of 7.38, or top-4 share above v2's
  5.09%;
- no claim of audio token pruning;
- wall-clock speed interpreted cautiously because the implementation is
  mask-only.

If v4 reaches 82/200 or better, it becomes the recommended quality-aware
Audio-Guard member of the AV-SCEB family. If it stays near v3 at 79/200, the
negative result indicates that the lost answers are not fixed by restoring
late-middle fragile-row guards alone.

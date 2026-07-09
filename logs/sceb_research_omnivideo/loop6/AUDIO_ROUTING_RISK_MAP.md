# Loop6 Audio Routing Risk Map

This note maps audio-aware routing risk for the OmniVideoBench audio-salient
subset used by Loop4 and Loop6. The evidence is measured from the durable
worker-3955069 Audio-Guard v2 limit-200 artifacts:

- Subset:
  `logs/omnivideobench_qwen3_research/loop4_audio_guard_v2_20260709/audio_salient_subset_data.json`
- Predictions:
  `logs/omnivideobench_qwen3_research/loop4_audio_guard_v2_20260709/limit200/predictions.jsonl`
- Routing profiles:
  `logs/omnivideobench_qwen3_research/loop4_audio_guard_v2_20260709/limit200/profile_samples.jsonl`
- Summary:
  `logs/omnivideobench_qwen3_research/loop4_audio_guard_v2_20260709/limit200/summary.json`

The map is intentionally audio-aware without claiming audio token pruning.
Audio, text, image, and control tokens remain protected. The risk question is:
when should audio/query evidence protect video expert budgeting, and when is a
speed-oriented video budget safe enough to test?

## Measured Subset Shape

| Signal | Value |
|---|---:|
| Samples | 200 |
| Accuracy | 41.50% |
| Correct | 83/200 |
| Mean video tokens per row | 35,030 |
| Mean audio tokens per row | 375 |
| Mean text tokens per row | 139 |
| Mean audio/video token ratio | 2.53% |
| Speech / Music / Sound rows | 130 / 50 / 20 |
| Mean active experts | 7.38 |
| Top-4 / Top-6 / Top-8 assignment share | 5.09% / 25.64% / 69.27% |
| Mean prefill latency | 9,078 ms |
| Mean generation latency | 11,720 ms |

The key routing asymmetry is that audio is small in token volume but high in
semantic control value. The measured subset averages only about 375 audio
tokens against about 35k video tokens, so protecting audio tokens has limited
direct expert-compute cost. The compute frontier is therefore mostly controlled
by video-token budgets, while audio/query metadata decides whether those video
budgets should be conservative.

## Risk Buckets

| Bucket | Rows | Accuracy | Routing / policy behavior | Risk interpretation |
|---|---:|---:|---|---|
| audio-core: background and music | 47 | 34.04% | 100.00% top-8 share, 8.00 mean active experts | Highest AV risk. Keep video full because the question is directly about sound or music. |
| audio-stress: counting, temporal, reference, spatial | 66 | 39.39% | About 62-64% top-8 share, about 7.28-7.31 active experts | Medium-high risk. Use a top-6 floor and late-middle protection for grounding-sensitive regimes. |
| audio-reasoning: causal, hypothetical, relation, ego | 47 | 53.19% | About 62-65% top-8 share, about 7.26-7.31 active experts | Medium risk. Audio can guide the event semantics, but full top-8 everywhere is not obviously required. |
| perception / default | 40 | 47.50% | About 63-67% top-8 share, 6.89-6.98 active experts | Lower audio-specific risk. These are the safest rows for speed-oriented video budgeting. |

The risk map preserves the Loop4 finding: `background&music understanding` is
the clean audio-core regime. It is also the least accurate measured bucket
(16/47), so it should not be used as a speed-policy target until a stronger
quality guard exists. By contrast, many non-core audio-salient rows tolerate
dynamic video expert budgeting while keeping audio tokens protected.

## Router Geometry Evidence

The profile samples expose per-modality top expert lists and token trajectory
signatures. Across 9,600 sample-layer comparisons, the mean Jaccard overlap
between audio and video top-8 expert sets is 0.142, with median 0.143 and p90
0.333. This shows coupling is real but sparse: audio and video do share some
expert neighborhoods, yet their dominant experts are not identical enough to
justify blindly pruning or protecting both modalities with one rule.

Trajectory margins also separate the modalities:

| Modality trajectory | Mean top1-top2 margin | Median | p90 | Segment count |
|---|---:|---:|---:|---:|
| Audio | 0.01009 | 0.00980 | 0.01470 | 38,400 |
| Video | 0.00723 | 0.00711 | 0.01094 | 38,400 |

Audio trajectory margins are higher on average than video trajectory margins in
this measured subset. That should not be read as "audio is cheap to prune":
Loop6 treats it as evidence that audio can be a stable controller signal while
remaining protected. The speed opportunity remains in video tokens, whose much
larger token count dominates expert assignment volume.

## Layer Risk

The Audio-Guard v2 policy targets layers 12-31. Mean active experts per video
token by layer ranged from 7.08 to 7.49. The lowest-budget measured layers were
30 (7.08), 31 (7.13), 24 (7.15), 29 (7.17), and 18 (7.18). The highest were
13 (7.49), 15 (7.48), 14 (7.43), 12 (7.39), and 25 (7.36).

This supports a layer-aware risk rule:

- early-middle layers 12-15 often remain conservative despite being inside the
  dynamic region;
- layers 18, 24, 29-31 are more budgetable under the current routing geometry;
- late-middle overrides for audio-core and reference/spatial rows are justified
  because the same layers are where speed pressure would otherwise concentrate.

## Policy Implication

The Loop6 risk map recommends three policy zones:

1. **Protect audio-core rows.** If the task is `background&music understanding`,
   keep video full top-8. The measured accuracy is already low, and the task is
   explicitly sound-grounded.
2. **Budget video cautiously for audio-stress rows.** Counting, temporal,
   reference, and spatial rows should keep at least top-6 and use late-middle
   full protection where the answer likely needs AV alignment.
3. **Spend speed budget on non-core rows.** Perception/default and some
   audio-reasoning rows are the safest candidates for Loop7 speed v3 because
   they keep audio protected while allowing more aggressive video top-m choices.

The long-paper framing should be "audio-conditioned video expert budgeting",
not audio expert pruning. Audio evidence decides when video expert compute is a
risk, and speed-oriented policies should trade compute only where that risk is
measured to be lower.

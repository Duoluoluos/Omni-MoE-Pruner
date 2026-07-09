# Loop6 Audio Video Expert Coupling

This report summarizes measured audio-video expert coupling for the
OmniVideoBench audio-salient subset. It uses the same durable worker-3955069
limit-200 profile artifacts as the Loop6 routing risk map and keeps the claim
strictly scoped: audio stays protected, while audio/query signals condition the
video-token expert budget.

## Evidence Source

| Artifact | Path |
|---|---|
| Audio-salient subset | `logs/omnivideobench_qwen3_research/loop4_audio_guard_v2_20260709/audio_salient_subset_data.json` |
| Audio-Guard v2 predictions | `logs/omnivideobench_qwen3_research/loop4_audio_guard_v2_20260709/limit200/predictions.jsonl` |
| Per-layer routing profiles | `logs/omnivideobench_qwen3_research/loop4_audio_guard_v2_20260709/limit200/profile_samples.jsonl` |
| Limit-200 summary | `logs/omnivideobench_qwen3_research/loop4_audio_guard_v2_20260709/limit200/summary.json` |

The 200-row subset contains 130 Speech rows, 50 Music rows, and 20 Sound rows.
Mean token counts are 35,030 video tokens, 375 audio tokens, and 139 text
tokens per sample. The audio/video token ratio is only 2.53%, so audio-video
coupling should be interpreted as a controller and grounding relation rather
than a symmetric compute relation.

## Coupling Measurements

Across the 200 profiles, every prefill layer records top experts separately for
video and audio token groups. Comparing the top-8 expert sets for audio and
video over 9,600 sample-layer pairs gives:

| Coupling statistic | Value |
|---|---:|
| Mean audio/video top-expert Jaccard overlap | 0.142 |
| Median overlap | 0.143 |
| p90 overlap | 0.333 |
| Segment comparisons | 9,600 |

This is the central Loop6 coupling result. Audio and video are not independent:
many layers share at least one dominant expert. But they are also not identical:
the overlap is far below a common-expert regime. A single global pruning rule
would therefore be too blunt for long-paper evidence. The better story is that
audio conditions where video budgeting is allowed.

## Modality Routing Margins

The token-trajectory statistics give another view of coupling. Audio trajectory
segments have higher top1-top2 router margins than video segments:

| Modality | Mean margin | Median | p90 | Segments |
|---|---:|---:|---:|---:|
| Audio | 0.01009 | 0.00980 | 0.01470 | 38,400 |
| Video | 0.00723 | 0.00711 | 0.01094 | 38,400 |

The interpretation is speed-oriented but conservative. Higher audio margins
make audio a useful routing signal, yet the policy should not drop audio
experts without direct measurement. Since video has far more tokens and lower
average margins, video is the natural compute-control target.

## Task-Coupled Budget Behavior

| Task regime | Rows | Mean top-8 share | Mean active experts | Accuracy |
|---|---:|---:|---:|---:|
| background&music understanding | 47 | 100.00% | 8.00 | 34.04% |
| counting | 28 | 62.21% | 7.28 | 42.86% |
| causal reasoning | 20 | 63.26% | 7.29 | 50.00% |
| temporal understanding | 18 | 62.07% | 7.27 | 27.78% |
| reference reasoning | 15 | 63.57% | 7.29 | 40.00% |
| sentiment analysis | 15 | 63.28% | 6.89 | 60.00% |
| summarization | 12 | 63.00% | 6.90 | 58.33% |
| fine-grained perception | 10 | 63.04% | 6.90 | 20.00% |
| ego reasoning | 10 | 65.24% | 7.31 | 30.00% |
| relationship reasoning | 9 | 61.67% | 7.26 | 55.56% |
| hypothetical reasoning | 8 | 65.18% | 7.31 | 75.00% |
| spatial understanding | 5 | 64.19% | 7.31 | 40.00% |
| attribute comparison | 3 | 66.56% | 6.98 | 0.00% |

The audio-core task is deliberately isolated: it runs full top-8. Other
audio-salient tasks retain dynamic video budgets, with top-6 and top-8
dominating. This task-coupled behavior is useful for a speed policy because it
creates a clean tradeoff surface: keep the audio-core protected, then seek
latency savings in non-core rows where video routing is less risky.

## Speed Coupling

The measured subset has mean prefill latency of 9,078 ms and mean generation
latency of 11,720 ms. Prefill and generation vary strongly by task:

| Regime | Mean prefill | Mean generation | Coupling implication |
|---|---:|---:|---|
| background/music | 8,191 ms | 10,590 ms | quality-risk dominated, not the first speed target |
| counting | 7,011 ms | 8,855 ms | good Loop7 speed candidate with top-6 floor |
| temporal | 10,161 ms | 13,154 ms | expensive but accuracy-fragile, needs guard |
| sentiment/summarization | 10,754 ms | 14,014 ms | lower audio-core risk, useful for speed evidence |
| hypothetical/ego | 12,986 ms | 16,672 ms | high latency; budget only if churn remains small |

The coupling evidence says the speed policy should not simply lower budgets for
all audio-salient rows. It should use audio/query regime and router stability to
decide whether the video budget can move down.

## Long-Paper Claim

Loop6 supports the following long-paper claim:

> Audio and video expert routes are coupled but not interchangeable. Audio
> evidence is a reliable controller for video expert budgeting, while audio
> tokens themselves remain protected unless direct pruning measurements justify
> otherwise.

This claim bridges Loop4 and Loop7. Loop4 proved a conservative audio-aware
extension can run on worker 3955069 and change behavior on a matched
audio-salient subset. Loop6 explains why that result should be converted into a
speed-oriented Audio-Conditioned policy v3: audio-core rows protect video, but
lower-risk rows can spend less video expert compute to test prefill and
generation latency gains.

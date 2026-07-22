# Audio-Router-Conditioned Video SCEB Loop 2026-07-19

## Goal

Move beyond text-query guards by using audio token router geometry to control
video-token expert budgets.

This loop keeps the deployable constraints:

- frozen model;
- no task labels;
- no query regexes;
- no audio token pruning;
- audio/text/image/special tokens remain protected top-8.

## Mechanism

For each dynamic video layer, compute same-layer audio-token router statistics:

```text
audio_margin_mean = mean(p_audio_top1 - p_audio_top2)
audio_boundary_mean = mean(p_audio_top4 - p_audio_top5)
```

If the sample-layer has enough audio tokens and `audio_margin_mean < 0.0082`,
the layer is treated as audio-router fragile and video-token `chosen_m` is
clamped to at least 6.

The threshold `0.0082` is the approximate p25 audio trajectory margin observed
on the first completed OmniVideoBench `audio_top68` 200-sample probe.

## Variants

### `boundary_audio_floor6`

Base policy: aggressive boundary-only video SCEB.

Audio-router guard:

- insufficient audio: video floor 8;
- fragile audio router: video floor 6;
- otherwise no additional floor.

This is the speed candidate.

### `global_audio_floor6`

Base policy: Global SCEB.

Audio-router guard:

- insufficient audio: video floor 8;
- fragile audio router: video floor 6;
- otherwise no additional floor.

This is the pure guard/control ablation.

## Sample Limits

- OmniVideoBench: `limit=200`;
- AVUT: `limit=346`;
- WorldSense: `limit=634`.

## Files

Configs:

- `logs/audio_router_conditioned_video_sceb_20260719/arc_video_boundary_audio_floor6_policy.json`
- `logs/audio_router_conditioned_video_sceb_20260719/arc_video_global_audio_floor6_policy.json`

Run scripts:

- `logs/audio_router_conditioned_video_sceb_20260719/run_arc_video_sceb_loop.sh`
- `logs/audio_router_conditioned_video_sceb_20260719/launch_arc_video_sceb_loop_worker_4065939.sh`
- `logs/audio_router_conditioned_video_sceb_20260719/queue_arc_video_sceb_after_audio_only_worker_4065939.sh`
- `logs/audio_router_conditioned_video_sceb_20260719/wait_and_launch_arc_video_sceb_worker_4065939.sh`

Queue status:

- queued at 2026-07-19 16:41 CST on worker `4065939`;
- queue session: `arc_video_sceb_wait_audio_only_20260719`;
- queue log:
  `logs/audio_router_conditioned_video_sceb_20260719/launcher/arc_video_sceb_wait_audio_only_worker_4065939_20260719_164136.log`;
- the queue waits for `audio_router_sceb_loop_20260719` to finish, then starts
  `arc_video_sceb_loop_20260719`.

Queue repair:

- the first queue attempt failed due to shell quoting in the tmux command;
- repaired at 2026-07-19 17:12 CST by moving the wait/launch logic into
  `wait_and_launch_arc_video_sceb_worker_4065939.sh`;
- relaunched successfully at 2026-07-19 17:13 CST;
- repaired queue log:
  `logs/audio_router_conditioned_video_sceb_20260719/launcher/arc_video_sceb_wait_audio_only_worker_4065939_20260719_171300.log`;
- confirmed queue log contains a live wait line for `audio_router_sceb_loop_20260719`.

Code support:

- `ExpertBudgetConfig.audio_router_conditioning`;
- `ConfidenceDynamicMaskInstaller._audio_condition_floor`;
- per-layer `audio_condition_*` stats in `expert_budget.layer_summaries`.

## Launch Verification

The direct-audio probe was stopped after completing `audio_top68` on the three
1/5 benchmark prefixes because direct audio-token budgeting only reduced total
MoE assignments by 0.05%-0.27%.

`arc_video_sceb_loop_20260719` was launched by the queue at 2026-07-19 18:28
CST on worker `4065939`.

Initial verification on `omnivideobench_boundary_audio_floor6_limit200`:

- first 8 prediction/profile rows were written;
- `policy_version = arc_video_boundary_audio_floor6_v1`;
- `target_modality = video`;
- `protected_modalities = text, audio, image, special`;
- `query_policy.enabled = false`;
- `audio_router_conditioning.enabled = true`;
- per-sample dynamic layers report 20 audio-condition calls;
- the first three samples each show 16 neutral layers and 4 fragile layers;
- average active top-m for the first three samples is about 5.59, compared with
  about 7.98 for direct audio-token budgeting.

## Interim Results

`boundary_audio_floor6` on OmniVideoBench `limit=200` completed at 2026-07-19
20:14 CST:

| Variant | Accuracy | E2E Mean (ms) | Prefill Mean (ms) | Avg Active Top-m | Assignment Reduction |
|---|---:|---:|---:|---:|---:|
| Vanilla | 38.00 | 20458.7 | 16410.3 | - | - |
| Global SCEB | 39.00 | 15890.9 | 12366.4 | 7.0958 | - |
| Video AQG v2 | 39.50 | 15838.8 | 12410.7 | 7.0130 | - |
| Direct audio top68 | 38.00 | 14873.5 | 11425.8 | 7.9950 | 0.0535% |
| ARC video boundary audio floor6 | 42.00 | 14813.4 | 11446.4 | 5.4860 | 31.37% |

Audio-router condition summary for this Omni run:

- neutral layer decisions: 3735;
- fragile layer decisions: 265;
- top-m histogram: top4 104,989,714; top6 34,144,765; top8 53,350,321.

Interpretation:

- Direct audio budgeting reached similar latency mostly without reducing MoE
  assignments; ARC video SCEB reaches similar latency while actually reducing
  video-token expert work by about 31%.
- Accuracy on the 200-prefix improves over all currently available anchors,
  so the audio-router signal is useful as a controller rather than as a direct
  pruning target.

`boundary_audio_floor6` on AVUT `limit=346` completed at 2026-07-19 20:50 CST:

| Variant | Accuracy | E2E Mean (ms) | Prefill Mean (ms) | Avg Active Top-m | Assignment Reduction |
|---|---:|---:|---:|---:|---:|
| Vanilla | 73.12 | 4237.6 | 3143.3 | - | - |
| Global SCEB | 74.28 | 3448.2 | 3155.6 | 6.9460 | - |
| Video AQG v2 | 72.54 | 3537.0 | 3250.4 | 7.1742 | - |
| Direct audio top68 | 73.99 | 3236.9 | 2948.2 | 7.9758 | 0.2685% |
| ARC video boundary audio floor6 | 74.28 | 3331.6 | 3036.3 | 5.6655 | 29.45% |

Audio-router condition summary for this AVUT run:

- neutral layer decisions: 6102;
- fragile layer decisions: 818;
- top-m histogram: top4 30,900,046; top6 11,428,222; top8 16,282,612.

`boundary_audio_floor6` on WorldSense `limit=634` completed at 2026-07-19
22:08 CST:

| Variant | Accuracy | E2E Mean (ms) | Prefill Mean (ms) | Avg Active Top-m | Assignment Reduction |
|---|---:|---:|---:|---:|---:|
| Vanilla | 53.00 | 5625.1 | 4661.6 | - | - |
| Global SCEB | 52.84 | 5443.8 | 4692.9 | 6.9224 | - |
| Direct audio top68 | 52.52 | 5200.8 | 4403.8 | 7.9842 | 0.1314% |
| ARC video boundary audio floor6 | 51.89 | 5266.3 | 4463.3 | 5.6956 | 29.75% |

Audio-router condition summary for this WorldSense run:

- neutral layer decisions: 10296;
- fragile layer decisions: 2384;
- top-m histogram: top4 103,820,587; top6 46,247,452; top8 56,660,701.

Decision after the three completed `boundary_audio_floor6` runs:

- the speed-candidate ARC method is strong on OmniVideoBench and AVUT;
- WorldSense drops 1.10pp against Vanilla, still within the 2pp tolerance;
- the next most important evidence is not `global_audio_floor6`, but a matched
  boundary-only ablation with audio-router conditioning disabled.

The partial `global_audio_floor6` control branch was stopped after 28
OmniVideoBench samples. A matched ablation was launched instead:

- session: `arc_video_boundary_ablation_20260719`;
- config:
  `logs/audio_router_conditioned_video_sceb_ablation_20260719/arc_video_boundary_no_audio_guard_policy.json`;
- launch time: 2026-07-19 22:19 CST;
- first 15 OmniVideoBench samples verified `audio_router_conditioning.enabled =
  false`, zero audio-condition calls, and average active top-m around 5.36.

Matched ablation result on OmniVideoBench `limit=200`:

| Variant | Accuracy | E2E Mean (ms) | Prefill Mean (ms) | Avg Active Top-m | Assignment Reduction |
|---|---:|---:|---:|---:|---:|
| ARC video boundary audio floor6 | 42.00 | 14813.4 | 11446.4 | 5.4860 | 31.37% |
| Boundary no audio guard | 40.00 | 14733.9 | 11399.1 | 5.4083 | 32.37% |

Interpretation:

- Removing the audio-router guard makes the policy slightly faster and more
  aggressive, but loses 4/200 OmniVideoBench samples.
- This supports the method claim that audio-router conditioning is a risk
  control signal rather than a direct compute target.

Matched ablation result on AVUT `limit=346`:

| Variant | Accuracy | E2E Mean (ms) | Prefill Mean (ms) | Avg Active Top-m | Assignment Reduction |
|---|---:|---:|---:|---:|---:|
| ARC video boundary audio floor6 | 74.28 | 3331.6 | 3036.3 | 5.6655 | 29.45% |
| Boundary no audio guard | 74.28 | 3242.9 | 2969.7 | 5.5341 | 30.85% |

Interpretation:

- On AVUT, the audio-router guard does not improve accuracy; no-audio boundary
  matches it while being slightly faster.
- The emerging picture is benchmark-dependent: audio-router conditioning helps
  OmniVideoBench robustness, while AVUT appears budgetable with a more
  aggressive boundary-only rule.

Matched ablation result on WorldSense `limit=634`:

| Variant | Accuracy | E2E Mean (ms) | Prefill Mean (ms) | Avg Active Top-m | Assignment Reduction |
|---|---:|---:|---:|---:|---:|
| ARC video boundary audio floor6 | 51.89 | 5266.3 | 4463.3 | 5.6956 | 29.75% |
| Boundary no audio guard | 51.58 | 5223.0 | 4445.6 | 5.4878 | 31.87% |

Final matched-ablation summary:

| Benchmark | ARC Audio Guard | No Audio Guard | Accuracy Delta | E2E Delta | Interpretation |
|---|---:|---:|---:|---:|---|
| OmniVideoBench 200 | 42.00 | 40.00 | +2.00pp | +0.54% slower | audio guard protects 4 samples |
| AVUT 346 | 74.28 | 74.28 | 0.00pp | +2.73% slower | no guard is enough |
| WorldSense 634 | 51.89 | 51.58 | +0.32pp | +0.83% slower | small protection effect |

Current method framing:

- Base policy: aggressive video-token boundary SCEB.
- Audio-router conditioning: same-layer audio router fragility raises video
  top-m floor from 4 to 6.
- The guard is not a direct speed source. It is a risk-control layer over an
  aggressive video budget.
- For a robust default across benchmarks, use ARC audio guard.
- For a max-speed variant, expose boundary no-audio guard as an ablation or
  speed mode.

Worker cleanup:

- `arc_video_sceb_loop_20260719` was stopped after partial `global_audio_floor6`;
- `arc_video_boundary_ablation_20260719` completed all three benchmark prefixes;
- idle tmux sessions were cleaned at 2026-07-20 02:06 CST.

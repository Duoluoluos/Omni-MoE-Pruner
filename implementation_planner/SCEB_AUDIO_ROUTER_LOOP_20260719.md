# Audio-Router SCEB Loop 2026-07-19

## Goal

Test whether audio tokens themselves can be dynamically assigned fewer experts
in a frozen Qwen3-Omni MoE model.

This loop deliberately does not use task labels, query regexes, or audio token
compression. It only changes the SCEB target modality:

```text
target_modality = audio
protected_modalities = text, video, image, special
```

The purpose is to decide whether "audio as budget target" has empirical
support before implementing the heavier audio-router-conditioned video policy.

## Variants

### `audio_top68`

Conservative probe:

- audio tokens in middle layers use dynamic `top_m_options=[6,8]`;
- video/text/image/special remain full top-8;
- early/late layers remain full top-8.

Config:

- `logs/audio_router_sceb_loop_20260719/audio_only_top68_policy.json`

### `audio_top468`

Aggressive probe:

- audio tokens in middle layers use dynamic `top_m_options=[4,6,8]`;
- video/text/image/special remain full top-8;
- early/late layers remain full top-8.

Config:

- `logs/audio_router_sceb_loop_20260719/audio_only_top468_policy.json`

## Sample Limits

Each benchmark uses approximately one fifth of the data:

- OmniVideoBench: `limit=200` out of 1000 raw rows;
- AVUT: `limit=346` out of 1734 rows;
- WorldSense: `limit=634` out of 3172 flattened QA rows.

## Outputs

For each variant:

- `logs/audio_router_sceb_loop_20260719/omnivideobench_<variant>_limit200`
- `logs/audio_router_sceb_loop_20260719/avut_<variant>_limit346`
- `logs/audio_router_sceb_loop_20260719/worldsense_<variant>_limit634`

Launcher:

- `logs/audio_router_sceb_loop_20260719/launch_audio_router_sceb_loop_worker_4065939.sh`

Launched:

- time: 2026-07-19 14:38 CST;
- worker: `4065939`;
- session: `audio_router_sceb_loop_20260719`;
- launcher log:
  `logs/audio_router_sceb_loop_20260719/launcher/audio_router_sceb_loop_worker_4065939_20260719_143855.log`.

Initial verification:

- first run entered OmniVideoBench `audio_top68` with `limit=200`;
- wrapper log shows `target=audio`, `protected=['text', 'video', 'image', 'special']`;
- first 8 prediction/profile rows were written;
- first profile rows show audio target tokens only:
  sample 0 has `video=29952`, `audio=390`, and `protected_tokens=30102`;
- `top_m_histogram` is recorded for audio-target tokens, while the overall
  average active experts stays near 8 because video/text tokens dominate the
  sequence and remain full top-8.

Protocol:

- run on worker `4065939`;
- use two GPUs via `CUDA_VISIBLE_DEVICES=0,1`;
- no `run_keepalive.sh`;
- primary metric is model-side E2E `generate_latency_ms`;
- compare against matched-prefix Vanilla / Global SCEB / video-target SCEB anchors.

## Interim Results

Status after stopping the direct-audio branch at 2026-07-19 18:25 CST:

- `audio_top68` OmniVideoBench completed: 76/200 = 38.00%;
- `audio_top68` AVUT completed: 256/346 = 73.99%;
- `audio_top68` WorldSense completed: 333/634 = 52.52%;
- `audio_top468` was intentionally stopped after 6 OmniVideoBench samples because
  direct audio-token budgeting had already shown negligible total assignment
  reduction and would delay the conditioned-video mainline.

Matched-prefix comparisons:

| Benchmark Prefix | Variant | Accuracy | E2E Mean (ms) | Prefill Mean (ms) | Avg Active Top-m | Assignment Reduction |
|---|---:|---:|---:|---:|---:|---:|
| OmniVideoBench 200 | Vanilla | 38.00 | 20458.7 | 16410.3 | - | - |
| OmniVideoBench 200 | Global SCEB | 39.00 | 15890.9 | 12366.4 | 7.0958 | - |
| OmniVideoBench 200 | Video AQG v2 | 39.50 | 15838.8 | 12410.7 | 7.0130 | - |
| OmniVideoBench 200 | Audio top68 | 38.00 | 14873.5 | 11425.8 | 7.9950 | 0.0535% |
| AVUT 346 | Vanilla | 73.12 | 4237.6 | 3143.3 | - | - |
| AVUT 346 | Global SCEB | 74.28 | 3448.2 | 3155.6 | 6.9460 | - |
| AVUT 346 | Video AQG v2 | 72.54 | 3537.0 | 3250.4 | 7.1742 | - |
| AVUT 346 | Audio top68 | 73.99 | 3236.9 | 2948.2 | 7.9758 | 0.2685% |
| WorldSense 634 | Audio top68 | 52.52 | 5200.8 | 4403.8 | 7.9842 | 0.1314% |

Paired flips:

- OmniVideoBench 200, audio top68 vs Vanilla: both correct 75, vanilla only 1, audio only 1, both wrong 123;
- OmniVideoBench 200, audio top68 vs Global SCEB: both correct 74, global only 4, audio only 2, both wrong 120;
- OmniVideoBench 200, audio top68 vs Video AQG v2: both correct 74, video AQG only 5, audio only 2, both wrong 119;
- AVUT 346, audio top68 vs Vanilla: both correct 250, vanilla only 3, audio only 6, both wrong 87;
- AVUT 346, audio top68 vs Global SCEB: both correct 252, global only 5, audio only 4, both wrong 85;
- AVUT 346, audio top68 vs Video AQG v2: both correct 249, video AQG only 2, audio only 7, both wrong 88.

Interpretation:

- Directly budgeting audio tokens barely reduces total MoE assignments because
  audio tokens are a small fraction of the full multimodal sequence.
- The useful signal is not "audio as a compute target"; it is audio-router
  confidence as a controller for video-token budgeting.
- The next mainline experiment should prioritize Audio-Router-Conditioned Video
  SCEB and avoid spending excessive worker time on direct-audio `audio_top468`.

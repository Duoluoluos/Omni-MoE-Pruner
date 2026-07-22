# Label-Free Audio+Query Guard Loop

## Problem

Task-label Audio-Guard is not a fair deployable method because benchmark task
labels are not generally available at inference time. It should be treated as a
diagnostic upper bound, not as the final method.

## Method Direction

Use only input-visible signals:

- query text;
- answer options;
- whether audio tokens are present in the processed input;
- router confidence and SCEB modality/layer policy.

Benchmark task labels remain in logs for analysis, but they are not used to
choose the expert budget.

## Implementation

`ExpertBudgetConfig` now supports `query_policy`.

When `query_policy.enabled=true`, `ConfidenceDynamicMaskInstaller.begin_sample`
selects `task_min_top_m` from `query_text` and audio token count. It does not
use `task_policy` or `task_layer_policy` for budget decisions.

The three runners pass:

```text
question + answer options
```

as `query_text`. The recorded `task_type` field is analysis metadata only.

## Current Variant

`audio_query_guard_v1_label_free`

Config:

- `logs/audio_query_guard_label_free_20260716/audio_query_guard_v1_label_free_policy.json`

Rules:

- `audio_reference`: requires audio tokens and matches audio/speech/mention
  words; sets `min_top_m=6`.
- `screen_text_diagram_reference`: matches text/screen/diagram/OCR words; sets
  `min_top_m=6`.
- `fine_visual_relation_reference`: matches spatial/relation/count/attribute
  words; sets `min_top_m=6`.
- otherwise use default `min_top_m=4`, equivalent to Global SCEB's dynamic
  budget floor.

## Running Experiment

Started 2026-07-16 11:40 CST on worker `4008126`.

Session:

- `audio_query_guard_v1_three_bench`

Launcher log:

- `logs/audio_query_guard_label_free_20260716/launcher/audio_query_guard_v1_three_bench_worker_4008126_20260716_114023.log`

Outputs:

- `logs/audio_query_guard_label_free_20260716/omnivideobench_audio_query_guard_v1_full`
- `logs/audio_query_guard_label_free_20260716/avut_audio_query_guard_v1_full`
- `logs/audio_query_guard_label_free_20260716/worldsense_audio_query_guard_v1_full`

Protocol:

- run OmniVideoBench full;
- then AVUT full;
- then WorldSense full;
- no `run_keepalive.sh`.

Primary metric:

- model-side E2E latency: `generate_latency_ms`.

Secondary metrics:

- accuracy;
- paired flips vs Vanilla, Global SCEB, task-label Audio-Guard;
- query guard cluster distribution;
- avg active experts and top-m histogram.

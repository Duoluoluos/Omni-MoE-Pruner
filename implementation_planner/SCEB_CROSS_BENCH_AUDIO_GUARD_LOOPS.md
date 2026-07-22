# Cross-Benchmark Audio-Guard Loops

Date: 2026-07-13

This plan extends the Audio-Guard story from OmniVideoBench to AVUT and
WorldSense. The goal is not to claim a new per-benchmark tuned method, but to
show that the same principle transfers:

```text
audio/query-sensitive regimes guard video expert budgets;
audio/text/image/special tokens remain protected.
```

## Current Status

Loop9 local-probability threshold baseline on OmniVideoBench is complete:

| Method | Samples | Correct | Accuracy | Prefill mean | Mean active experts | Top-4 / Top-6 / Top-8 |
|---|---:|---:|---:|---:|---:|---|
| Local-prob threshold baseline | 998 | 392 | 39.28% | 9.36s | 5.42 | 42.65 / 49.78 / 7.57% |

This is useful for the related-work story: a generic local router-probability
threshold is very aggressive, but it does not beat Global or Boundary SCEB on
quality. It supports keeping router-boundary risk and protected-region policy
as the main method.

The `4008126` worker is currently free: both A100 GPUs report 0 MiB used.

Loop10 and Loop11 are now complete:

| Loop | Deliverable | Result |
|---|---|---|
| Loop10 | paired AVUT/WorldSense risk map | `logs/cross_bench_audio_guard_20260713/cross_bench_audio_guard_risk_map.md` |
| Loop11 | AVUT Audio-Guard v1, 200 samples | 143/200 = 71.50%; same accuracy as Global SCEB subset, lower churn, slower prefill |
| Loop11 | WorldSense Audio-Guard v1, 200 samples | 107/200 = 53.50%; +1 sample over Global SCEB subset, lower churn, slower prefill |

Detailed results are saved in
`logs/cross_bench_audio_guard_20260713/LOOP11_RESULTS.md`.

Loop12 full validation is now complete:

| Loop | Deliverable | Result |
|---|---|---|
| Loop12 | AVUT Audio-Guard v1 full | 1228/1734 = 70.82%; +2 samples over Global SCEB, -6 samples vs vanilla |
| Loop12 | WorldSense Audio-Guard v1 full | 1591/3172 = 50.16%; +10 samples over Global SCEB, +12 samples vs vanilla |

Detailed full results are saved in
`logs/cross_bench_audio_guard_20260713/LOOP12_FULL_RESULTS.md`.

## Existing Transfer Anchors

Global SCEB already transfers reasonably to AVUT and WorldSense:

| Benchmark | Vanilla | Global SCEB | Accuracy delta | Prefill delta | E2E/generate delta |
|---|---:|---:|---:|---:|---:|
| AVUT | 1234/1734 = 71.16% | 1226/1734 = 70.70% | -0.46 pp | -1.10% | -13.24% |
| WorldSense | 1579/3172 = 49.78% | 1581/3172 = 49.84% | +0.06 pp | -1.66% | -9.15% |

Audio-Guard should improve the interpretation of these transfer results by
showing that the guard can be derived from task/query risk rather than from an
OmniVideoBench-only task list.

## Loop 10: Paired Risk Map From Existing Full Runs

No GPU is needed.

Inputs:

- `logs/avut_qwen3_fair_rerun_20260706/vanilla_full/predictions.jsonl`
- `logs/avut_qwen3_fair_rerun_20260706/sceb_full/predictions.jsonl`
- `logs/worldsense_qwen3_fair_rerun_20260707/vanilla_full/predictions.jsonl`
- `logs/worldsense_qwen3_fair_rerun_20260707/sceb_full/predictions.jsonl`

Deliverables:

- task-level vanilla-only losses, SCEB-only gains, same-prediction rate;
- task-level mean active experts and top-m distribution;
- an initial guard map for each benchmark.

Initial AVUT risk map from existing full summaries:

| Group | Tasks | Policy intent |
|---|---|---|
| Fragile audio-localization/counting | Audio Event Location, Audio Content Counting | raise floor to top-6; restore late-middle full layers |
| Mild-risk matching | Audio Object Matching, Audio Information Extraction | keep dynamic, but avoid top-4 if paired losses concentrate |
| Safe or positive | Audio Character Matching, Audio OCR Matching | keep speed-oriented dynamic budget |

Initial WorldSense risk map from existing full summaries:

| Group | Tasks | Policy intent |
|---|---|---|
| Fragile AV/query tasks | Anomaly Recognition, Attribute Reasoning, Human-object Interaction, Text and Diagram Understanding, Audio Counting, Video Emotions, Emotion Change, Spatial Relation | raise floor to top-6; restore late-middle full layers for the first pass |
| Neutral tasks | Attribute Recognition, Object Counting, Object State Change, Relation Reasoning | keep Global SCEB default |
| Safe or positive tasks | Audio Change, Audio Recognition, Audio Source Localization, Event Sorting, Temporal Localization, Temporal Prediction, Object Existence Recognition, Human Emotions, Hallucination, Scene Recognition, Event Recognition, Action Counting | allow speed-oriented dynamic budget |

Exit criterion:

- a reproducible `cross_bench_audio_guard_risk_map.md`;
- one AVUT config and one WorldSense config for Loop 11.

## Loop 11: 200-Sample Audio-Guard Transfer Probe

Run sequentially on one 2xA100 worker. Do not run AVUT and WorldSense
concurrently, and do not use `run_keepalive.sh`.

Variants:

1. `avut_audio_guard_v1`: AVUT risk-map guard.
2. `worldsense_audio_guard_v1`: WorldSense risk-map guard.

Protocol:

- smoke2 first;
- then `--limit 200`;
- if full is launched later, stop active polling after 2 samples and leave it
  running for manual validation.

Decision rule:

- If accuracy is within 1pp of Global SCEB and prefill is not worse than
  Global SCEB, keep the guard map.
- If AVUT loses more than Global SCEB, restore `Audio Event Location` to top-8.
- If WorldSense loses more than Global SCEB, split the fragile group into
  "audio-counting full" and "visual-query top-6" instead of guarding all of it
  equally.

## Loop 12: Guard Strength Sweep

Run only the dataset that fails Loop 11, or both if time allows.

Variants:

- `quality_guard`: fragile group min top-8 or late-middle full;
- `balanced_guard`: fragile group min top-6 plus late-middle full;
- `speed_guard`: fragile group min top-6 only, no full-layer restoration.

Use 200 samples first. If two variants tie in accuracy, prefer the lower mean
active expert count. This loop turns Audio-Guard from a hand-picked task list
into a tunable guard strength axis.

## Loop 13: Full Fair Cross-Benchmark Validation

Run final AVUT and WorldSense Audio-Guard policies sequentially.

Current run, started 2026-07-15 12:50 CST on worker `4008126`:

- variant: `audio_guard_v2_balanced`;
- launcher log: `logs/cross_bench_audio_guard_20260713/launcher/loop13_balanced_full_worker_4008126_20260715_125041.log`;
- configs:
  - `logs/cross_bench_audio_guard_20260713/avut_audio_guard_v2_balanced_policy.json`;
  - `logs/cross_bench_audio_guard_20260713/worldsense_audio_guard_v2_balanced_policy.json`;
- outputs:
  - `logs/cross_bench_audio_guard_20260713/avut_audio_guard_v2_balanced_full`;
  - `logs/cross_bench_audio_guard_20260713/worldsense_audio_guard_v2_balanced_full`;
- protocol: AVUT full first, then WorldSense full in the same tmux chain; no
  `run_keepalive.sh`;
- design change from v1: keep fragile-task `min_top_m=6`, remove the v1
  late-middle `task_layer_policy` full restoration, so this is a speed-recovery
  guard rather than a quality-heavy guard.

Report against existing fair anchors:

| Benchmark | Compare against | Required claim |
|---|---|---|
| AVUT | Vanilla and Global SCEB | recover part of the -0.46pp Global SCEB drop, or preserve it with lower latency |
| WorldSense | Vanilla and Global SCEB | stay near or above vanilla accuracy while improving latency |

The final paper table should keep AVUT/WorldSense as transfer evidence. The
main mechanism claims should still be grounded in OmniVideoBench, because it is
where we have uniform top-k, signal ablation, confidence-risk probes, and
Audio-Guard development evidence.

## Loop 14: Cross-Benchmark Story Consolidation

No GPU is needed.

Deliverables:

- one table with OmniVideoBench main methods;
- one transfer table with AVUT/WorldSense Global SCEB and Audio-Guard;
- one figure or appendix table showing per-benchmark guard maps;
- final claim wording:

```text
SCEB is a post-hoc protected-region expert budgeting rule. Audio-Guard extends
it by using audio/query-sensitive task regimes as risk guards for video expert
budgets. The rule transfers to AVUT and WorldSense without changing model
weights, while benchmark-specific guard maps expose where audio-video tasks are
too fragile for aggressive video expert reduction.
```

## Loop 14b: Selective E2E Audio-Guard

Started 2026-07-16 11:24 CST on worker `4008126`.

Motivation: `generate_latency_ms` is the correct model-side E2E metric because
it wraps the full `model.generate(...)` call and already includes prefill. Loop
13 improved cross-benchmark accuracy but was slower than Global SCEB in E2E, so
this loop narrows Audio-Guard to the task groups with clear full-run positive
flips.

Variant: `audio_guard_v3_selective_e2e`.

- AVUT: keep top-6 floor only for `Audio Event Location`; move
  `Audio Content Counting` back to top-4 dynamic budget.
- WorldSense: keep top-6 floor only for `Anomaly Recognition`,
  `Attribute Reasoning`, `Human-object Interaction`, and
  `Text and Diagram Understanding`; move `Audio Counting`, `Emotion Change`,
  `Spatial Relation`, and `Video Emotions` back to top-4 dynamic budget.
- No task-layer full restoration.
- No `run_keepalive.sh`.

Paths:

- configs:
  - `logs/cross_bench_audio_guard_20260713/avut_audio_guard_v3_selective_policy.json`
  - `logs/cross_bench_audio_guard_20260713/worldsense_audio_guard_v3_selective_policy.json`
- launcher log:
  - `logs/cross_bench_audio_guard_20260713/launcher/loop14_selective_full_worker_4008126_20260716_112407.log`
- outputs:
  - `logs/cross_bench_audio_guard_20260713/avut_audio_guard_v3_selective_full`
  - `logs/cross_bench_audio_guard_20260713/worldsense_audio_guard_v3_selective_full`

Expected readout:

- primary: E2E `generate_latency_ms` vs Global SCEB, v1, v2, and vanilla;
- secondary: accuracy and paired flips;
- target: recover E2E relative to v1/v2 while retaining part of the
  cross-benchmark quality gain over Global SCEB.

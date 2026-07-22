# ARC Full-Set Loop 2026-07-20

## Goal

Make audio-router-conditioned video SCEB work on full OmniVideoBench, AVUT,
and WorldSense.

Acceptance target:

- primary metric: E2E / prefill speedup versus same-device vanilla;
- quality tolerance: up to about 2pp accuracy drop;
- no task labels, no query regexes, no audio token pruning;
- use resume inference so interrupted runs skip completed sample indices.

## Method

Default method is `arc_video_boundary_audio_floor6_full_v1`:

- target modality: video;
- protected modalities: text, audio, image, special;
- dynamic layers: middle 25%-67%;
- base video top-m: boundary-only dynamic 4/6/8;
- audio-router guard: if same-layer audio top1-top2 margin is fragile
  (`< 0.0082`), clamp video top-m floor to 6;
- insufficient audio tokens: clamp video top-m floor to 8;
- `query_policy.enabled = false`;
- `compact_dispatch = false`.

## Run Protocol

Each benchmark writes smoke2 and full outputs into the same full output
directory:

1. run `--limit 2 --resume`;
2. validate at least two predictions and config invariants;
3. run full with no `--limit` and `--resume`.

This lets the full run skip the two smoke samples and lets later resumes skip
any completed `sample_index`.

## Files

- config:
  `logs/audio_router_conditioned_video_sceb_full_20260720/arc_video_boundary_audio_floor6_policy.json`
- full chain:
  `logs/audio_router_conditioned_video_sceb_full_20260720/run_arc_fullset_chain.sh`
- worker tmux launcher:
  `logs/audio_router_conditioned_video_sceb_full_20260720/launch_arc_fullset_tmux_on_worker.sh`
- master launcher for worker 4065939:
  `logs/audio_router_conditioned_video_sceb_full_20260720/launch_arc_fullset_worker_4065939.sh`

## Output Directories

- OmniVideoBench:
  `logs/audio_router_conditioned_video_sceb_full_20260720/omnivideobench_boundary_audio_floor6_full`
- AVUT:
  `logs/audio_router_conditioned_video_sceb_full_20260720/avut_boundary_audio_floor6_full`
- WorldSense:
  `logs/audio_router_conditioned_video_sceb_full_20260720/worldsense_boundary_audio_floor6_full`

## Launch Status

Launched on worker `4065939` at 2026-07-20 10:07 CST through:

```bash
logs/audio_router_conditioned_video_sceb_full_20260720/launch_arc_fullset_worker_4065939.sh
```

Worker-side tmux session:

```text
arc_fullset_loop_20260720
```

Worker log:

```text
logs/audio_router_conditioned_video_sceb_full_20260720/launcher/arc_fullset_worker_n122-192-156_20260720_100718.log
```

Initial verification:

- current worker list reports worker `4065939` as 2 x A100-SXM-80GB;
- OmniVideoBench smoke2 completed in the full output directory;
- config invariants passed:
  `target_modality=video`, audio/text/image/special protected,
  `query_policy.enabled=false`, `audio_router_conditioning.enabled=true`;
- OmniVideoBench full resume started from the same output directory and skipped
  the two smoke samples;
- at 2026-07-20 10:10 CST, OmniVideoBench had at least 7 predictions written
  with zero logged sample errors.
- at 2026-07-20 10:15 CST, worker-side tmux was still alive and running
  `run_omnivideobench_budget.py --resume`; OmniVideoBench had 19/1000
  predictions, 19/1000 profiles, and zero logged sample errors. AVUT and
  WorldSense had not started yet.
- at 2026-07-20 10:17 CST, worker-side tmux was still alive on worker
  `4065939`; OmniVideoBench had 25/1000 predictions, 25/1000 profiles, and
  zero logged sample errors. The active process was still
  `run_omnivideobench_budget.py --resume`; AVUT and WorldSense had not started.
- at 2026-07-20 10:19 CST, worker-side tmux was still alive and the active
  process remained `run_omnivideobench_budget.py --resume`. OmniVideoBench had
  30/1000 predictions, 30/1000 profiles, and zero logged sample errors; AVUT
  and WorldSense had not started. The partial analysis showed 31.37%
  assignment reduction and 5.490 active top-m. Prefix accuracy was 33.33%, but
  this is only a 30-sample sanity check and should not be used as a full-set
  quality conclusion.
- at 2026-07-20 10:21 CST, worker-side tmux was still alive on worker
  `4065939`; the active process remained `run_omnivideobench_budget.py
  --resume`. OmniVideoBench had 37/1000 predictions, 37/1000 profiles, and
  zero logged sample errors. AVUT and WorldSense had not started.
- at 2026-07-20 10:23 CST, worker-side tmux was still alive on worker
  `4065939`; the active process remained `run_omnivideobench_budget.py
  --resume`. OmniVideoBench had 45/1000 predictions, 45/1000 profiles, and
  zero logged sample errors. AVUT and WorldSense had not started.
- at 2026-07-20 10:24 CST, worker-side tmux was still alive on worker
  `4065939`; the active process remained `run_omnivideobench_budget.py
  --resume`. OmniVideoBench had 51/1000 predictions, 51/1000 profiles, and
  zero logged sample errors. AVUT and WorldSense had not started.
- at 2026-07-20 10:25 CST, worker-side tmux was still alive on worker
  `4065939`; the active process remained `run_omnivideobench_budget.py
  --resume`. OmniVideoBench had 54/1000 predictions, 54/1000 profiles, and
  zero logged sample errors. AVUT and WorldSense had not started. The
  instantaneous tqdm ETA for the remaining OmniVideoBench segment was noisy,
  roughly in the 4-7 hour range.
- at 2026-07-20 10:27 CST, worker-side tmux was still alive on worker
  `4065939`; the active process remained `run_omnivideobench_budget.py
  --resume`. OmniVideoBench had 57/1000 predictions, 57/1000 profiles, and
  zero logged sample errors. AVUT and WorldSense had not started. The
  instantaneous tqdm ETA for the remaining OmniVideoBench segment was around
  seven hours, but still noisy because sample lengths vary.
- at 2026-07-20 10:28 CST, worker-side tmux was still alive on worker
  `4065939`; the active process remained `run_omnivideobench_budget.py
  --resume`. OmniVideoBench had 60/1000 predictions, 60/1000 profiles, and
  zero logged sample errors. AVUT and WorldSense had not started. The
  instantaneous tqdm ETA for the remaining OmniVideoBench segment was about
  7.5 hours.
- at 2026-07-20 10:31 CST, worker-side tmux was still alive on worker
  `4065939`; the active process remained `run_omnivideobench_budget.py
  --resume`. OmniVideoBench had 65/1000 predictions, 65/1000 profiles, and
  zero logged sample errors. AVUT and WorldSense had not started. No keepalive
  process was running.
- at 2026-07-20 10:33 CST, worker-side tmux was still alive on worker
  `4065939`; the active process remained `run_omnivideobench_budget.py
  --resume`. OmniVideoBench had 68/1000 predictions, 68/1000 profiles, and
  zero logged sample errors. Partial summary showed 31.88% assignment
  reduction and 5.450 active top-m. The prefix latency direction was favorable
  versus existing vanilla references, but this is not yet final speed evidence
  because the run is mask-only and the comparison is prefix-vs-full rather than
  full paired.
- at 2026-07-20 10:36 CST, OmniVideoBench had 78/1000 predictions,
  78/1000 profiles, and zero logged sample errors. AVUT and WorldSense had not
  started.
- at 2026-07-20 10:39 CST, OmniVideoBench had 80/1000 predictions,
  80/1000 profiles, and zero logged sample errors. Worker `4065939` was the
  only active worker listed by `mlx worker list`; it was occupied by this
  mask-only run. No compact full-set process had been launched.
- at 2026-07-20 10:41 CST, OmniVideoBench had 82/1000 predictions,
  82/1000 profiles, and zero logged sample errors. Prediction sample indices
  had no duplicates. Worker `4065939` still had only the mask-only
  `run_omnivideobench_budget.py --resume` process inside tmux session
  `arc_fullset_loop_20260720`; compact full-set remained unlaunched.
- at 2026-07-20 10:43 CST, partial summary at OmniVideoBench 84/1000 showed
  46.43% prefix accuracy, 5.469 active top-m, 31.64% assignment reduction, and
  zero errors. The prefix latency direction was still favorable versus the
  existing vanilla reference, but remains non-final because the active run is
  mask-only and incomplete.
- at 2026-07-20 10:46 CST, OmniVideoBench had 91/1000 predictions,
  91/1000 profiles, and zero logged sample errors. Prediction sample indices
  still had no duplicates. Worker `4065939` still had only the mask-only
  `run_omnivideobench_budget.py --resume` process; no compact or keepalive
  process was running. The compact production policy was updated to disable
  debug compare (`compact_debug_compare=false`, `compact_debug_max_calls=0`) so
  future true-dispatch full-set latency is not contaminated by reference
  forwards.
- at 2026-07-20 10:57 CST, OmniVideoBench had 107/1000 predictions,
  107/1000 profiles, and zero logged sample errors. Prediction sample indices
  still had no duplicates. A second 2xA100 workspace worker launch was attempted
  for the compact true-dispatch full-set loop so the mask-only quality-oracle
  run would not need to be interrupted. The new worker id was `4069954`, but it
  was still not ready: `mlx worker list` showed no `podIP`, `mlx worker login
  4069954` returned `worker has not been ready yet`, and the platform API also
  emitted a context deadline timeout. The compact launch process was still alive
  and waiting; compact output directories still had 0 predictions.
- at 2026-07-20 11:01 CST, OmniVideoBench had 114/1000 predictions,
  114/1000 profiles, and zero logged sample errors. Prediction sample indices
  still had no duplicates. Two additional low-CPU compact launch attempts were
  tried against the remaining 2xA100 quota; one failed because the requested
  memory exceeded the per-GPU policy, and the corrected memory request failed
  during worker command execution before writing any compact samples. The
  original compact worker `4069954` remained pending, and one `mlx worker list`
  call returned platform HTTP 500. Compact outputs still had 0 predictions.
- at 2026-07-20 11:03 CST, OmniVideoBench had 117/1000 predictions,
  117/1000 profiles, and zero logged sample errors. Prediction sample indices
  still had no duplicates. Partial summary showed 41.88% prefix accuracy,
  5.476 active top-m, and 31.55% assignment reduction. The original compact
  worker `4069954` still had no `podIP`; `mlx worker login 4069954` returned
  `worker has not been ready yet`, and the worker control plane emitted HTTP
  500/502 errors. Compact outputs still had 0 predictions.
- at 2026-07-20 11:06 CST, OmniVideoBench had 120/1000 predictions,
  120/1000 profiles, and zero logged sample errors. Prediction sample indices
  still had no duplicates. Worker `4069954` had disappeared from
  `mlx worker list`, and direct login returned a worker-not-found error. The
  original compact launch process exited after `exec command: 0`; compact
  outputs still had 0 predictions. Added a safer generic compact launcher that
  assumes an already-ready worker and starts the compact chain via login +
  worker-side tmux instead of embedding the full chain directly in
  `mlx worker launch`.
- at 2026-07-20 11:24 CST, compact was explicitly treated as an invalid /
  non-primary path for the current full-benchmark goal. A later compact worker
  (`4070149`) had started loading but was stopped before writing any
  predictions; compact output had 0 predictions and 2 error rows and should not
  be used for analysis. The worker was killed and the local sleep launch process
  was cleaned up. `mlx worker list` then showed only worker `4065939`.
  OmniVideoBench mask-only ARC had 153/1000 predictions, 153/1000 profiles, and
  zero errors at that check, with AVUT and WorldSense not started yet.
- at 2026-07-20 11:55 CST, worker `4065939` was still alive and running tmux
  session `arc_fullset_loop_20260720`. The only active benchmark process was
  mask-only `run_omnivideobench_budget.py --resume`; no compact or keepalive
  process was running. OmniVideoBench had 199/1000 predictions, 199/1000
  profiles, zero logged sample errors, and no duplicate sample indices. AVUT
  and WorldSense had not started yet. The instantaneous tqdm ETA for the
  remaining OmniVideoBench segment was about 8 hours.
- at 2026-07-20 12:57 CST, worker `4065939` still reported 2 x A100-SXM-80GB
  and tmux session `arc_fullset_loop_20260720` was alive. The only active
  benchmark process remained mask-only `run_omnivideobench_budget.py
  --resume`; no compact or keepalive process was running. OmniVideoBench had
  359/1000 predictions, 359/1000 profiles, zero logged sample errors, and no
  duplicate sample indices. AVUT and WorldSense had not started yet. Progress
  since the previous poll was +160 OmniVideoBench samples in about one hour.
- at 2026-07-20 13:58 CST, worker `4065939` and tmux session
  `arc_fullset_loop_20260720` were still alive. The only active benchmark
  process remained mask-only `run_omnivideobench_budget.py --resume`; no
  compact or keepalive process was running. OmniVideoBench had 450/1000
  predictions, 450/1000 profiles, zero logged sample errors, and no duplicate
  sample indices. AVUT and WorldSense had not started yet. Progress since the
  previous poll was +91 OmniVideoBench samples in about one hour; no adjustment
  was needed.
- at 2026-07-20 14:58 CST, worker `4065939` and tmux session
  `arc_fullset_loop_20260720` were still alive. The only active benchmark
  process remained mask-only `run_omnivideobench_budget.py --resume`; no
  compact or keepalive process was running. OmniVideoBench had 619/1000
  predictions, 619/1000 profiles, zero logged sample errors, and no duplicate
  sample indices. AVUT and WorldSense had not started yet. Progress since the
  previous poll was +169 OmniVideoBench samples in about one hour; no
  adjustment was needed.
- at 2026-07-20 15:59 CST, worker `4065939` and tmux session
  `arc_fullset_loop_20260720` were still alive. The only active benchmark
  process remained mask-only `run_omnivideobench_budget.py --resume`; no
  compact or keepalive process was running. OmniVideoBench had 853/1000
  predictions, 853/1000 profiles, zero logged sample errors, and no duplicate
  sample indices. AVUT and WorldSense had not started yet. Progress since the
  previous poll was +234 OmniVideoBench samples in about one hour; no
  adjustment was needed.
- at 2026-07-20 17:00 CST, worker `4065939` and tmux session
  `arc_fullset_loop_20260720` were still alive. The active benchmark process
  had moved to mask-only `run_avut_budget.py --resume`; no compact or
  keepalive process was running. OmniVideoBench was effectively complete with
  998/1000 predictions, 998/1000 profiles, two logged sample errors, and no
  duplicate sample indices. The two OmniVideoBench errors were sample 942 and
  943 on `videos/video_605.mp4`, both
  `RuntimeError('Expected 2D or 3D ... got: [1, 1, 0]')`; the same two samples
  fail in the same way in the same-device vanilla full run and multiple older
  SCEB/policy runs, so this is not treated as an ARC-specific issue. AVUT had
  80/1734 predictions, 80/1734 profiles, zero logged sample errors, and no
  duplicate sample indices. WorldSense had not started yet. No adjustment was
  needed.
- at 2026-07-20 18:02 CST, worker `4065939` and tmux session
  `arc_fullset_loop_20260720` were still alive. The active benchmark process
  remained mask-only `run_avut_budget.py --resume`; no compact or keepalive
  process was running. AVUT had 620/1734 predictions, 620/1734 profiles, zero
  logged sample errors, and no duplicate sample indices. WorldSense had not
  started yet. Progress since the previous poll was +540 AVUT samples in about
  one hour. Merlin action API emitted one transient timeout during polling, but
  `mlx worker list` and the worker-side probe both succeeded afterward; no
  adjustment was needed.
- at 2026-07-20 19:03 CST, worker `4065939` and tmux session
  `arc_fullset_loop_20260720` were still alive. The active benchmark process
  remained mask-only `run_avut_budget.py --resume`; no compact or keepalive
  process was running. AVUT had 951/1734 predictions, 951/1734 profiles, zero
  logged sample errors, and no duplicate sample indices. WorldSense had not
  started yet. Progress since the previous poll was +331 AVUT samples in about
  one hour; no adjustment was needed.
- at 2026-07-20 20:04 CST, worker `4065939` and tmux session
  `arc_fullset_loop_20260720` were still alive. The active benchmark process
  remained mask-only `run_avut_budget.py --resume`; no compact or keepalive
  process was running. AVUT had 1282/1734 predictions, 1282/1734 profiles, zero
  logged sample errors, and no duplicate sample indices. WorldSense had not
  started yet. Progress since the previous poll was +331 AVUT samples in about
  one hour. Merlin action API again emitted a transient timeout during polling,
  but `mlx worker list` and the worker-side probe both succeeded afterward; no
  adjustment was needed.
- at 2026-07-20 21:05 CST, worker `4065939` and tmux session
  `arc_fullset_loop_20260720` were still alive. The active benchmark process
  remained mask-only `run_avut_budget.py --resume`; no compact or keepalive
  process was running. AVUT had 1493/1734 predictions, 1493/1734 profiles,
  zero logged sample errors, and no duplicate sample indices. WorldSense had
  not started yet. Progress since the previous poll was +211 AVUT samples in
  about one hour; no adjustment was needed.
- at 2026-07-20 22:06 CST, worker `4065939` and tmux session
  `arc_fullset_loop_20260720` were still alive. The active benchmark process
  remained mask-only `run_avut_budget.py --resume`; no compact or keepalive
  process was running. AVUT had 1663/1734 predictions, 1663/1734 profiles,
  zero logged sample errors, and no duplicate sample indices. WorldSense had
  not started yet. Progress since the previous poll was +170 AVUT samples in
  about one hour; no adjustment was needed.
- at 2026-07-20 23:06 CST, worker `4065939` and tmux session
  `arc_fullset_loop_20260720` were still alive. AVUT had completed with
  1734/1734 predictions, 1734/1734 profiles, zero logged sample errors, and no
  duplicate sample indices. The active benchmark process had moved to
  mask-only `run_worldsense_budget.py --resume`; no compact or keepalive
  process was running. WorldSense had 352/3172 predictions, 352/3172 profiles,
  zero logged sample errors, and no duplicate sample indices. No adjustment was
  needed.
- at 2026-07-21 00:07 CST, worker `4065939` and tmux session
  `arc_fullset_loop_20260720` were still alive. The active benchmark process
  remained mask-only `run_worldsense_budget.py --resume`; no compact or
  keepalive process was running. WorldSense had 833/3172 predictions,
  833/3172 profiles, zero logged sample errors, and no duplicate sample
  indices. Progress since the previous poll was +481 WorldSense samples in
  about one hour; no adjustment was needed.
- at 2026-07-21 01:08 CST, worker `4065939` and tmux session
  `arc_fullset_loop_20260720` were still alive. The active benchmark process
  remained mask-only `run_worldsense_budget.py --resume`; no compact or
  keepalive process was running. WorldSense had 1304/3172 predictions,
  1304/3172 profiles, zero logged sample errors, and no duplicate sample
  indices. Progress since the previous poll was +471 WorldSense samples in
  about one hour; no adjustment was needed.
- at 2026-07-21 02:09 CST, worker `4065939` and tmux session
  `arc_fullset_loop_20260720` were still alive. The active benchmark process
  remained mask-only `run_worldsense_budget.py --resume`; no compact or
  keepalive process was running. WorldSense had 1780/3172 predictions,
  1780/3172 profiles, zero logged sample errors, and no duplicate sample
  indices. Progress since the previous poll was +476 WorldSense samples in
  about one hour; no adjustment was needed.
- at 2026-07-21 03:10 CST, worker `4065939` and tmux session
  `arc_fullset_loop_20260720` were still alive. The active benchmark process
  remained mask-only `run_worldsense_budget.py --resume`; no compact or
  keepalive process was running. WorldSense had 2226/3172 predictions,
  2226/3172 profiles, zero logged sample errors, and no duplicate sample
  indices. Progress since the previous poll was +446 WorldSense samples in
  about one hour; no adjustment was needed.
- at 2026-07-21 04:11 CST, worker `4065939` and tmux session
  `arc_fullset_loop_20260720` were still alive. The active benchmark process
  remained mask-only `run_worldsense_budget.py --resume`; no compact or
  keepalive process was running. WorldSense had 2638/3172 predictions,
  2638/3172 profiles, zero logged sample errors, and no duplicate sample
  indices. Progress since the previous poll was +412 WorldSense samples in
  about one hour; no adjustment was needed.
- at 2026-07-21 05:12 CST, worker `4065939` and tmux session
  `arc_fullset_loop_20260720` were still alive. The active benchmark process
  remained mask-only `run_worldsense_budget.py --resume`; no compact or
  keepalive process was running. WorldSense had 3048/3172 predictions,
  3048/3172 profiles, zero logged sample errors, and no duplicate sample
  indices. Progress since the previous poll was +410 WorldSense samples in
  about one hour; no adjustment was needed.
- at 2026-07-21 06:12 CST, the full-set run was complete. Worker `4065939`
  and tmux session `arc_fullset_loop_20260720` were still alive, but no active
  benchmark runner remained. Final sample status: OmniVideoBench 998/1000
  predictions plus the two known vanilla-matching sample errors, AVUT
  1734/1734 predictions with zero errors, and WorldSense 3172/3172 predictions
  with zero errors. No compact or keepalive process was running.

Final analysis was generated by:

```bash
python3 logs/audio_router_conditioned_video_sceb_full_20260720/summarize_arc_fullset.py
```

Outputs:

- `logs/audio_router_conditioned_video_sceb_full_20260720/ARC_FULLSET_ANALYSIS.md`
- `logs/audio_router_conditioned_video_sceb_full_20260720/arc_fullset_analysis.json`

Final summary:

| Benchmark | Done | Acc | Vanilla Acc | Δ Acc | Prefill ms | Δ Prefill | E2E ms | Δ E2E | Active top-m | Assign Red. | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniVideoBench | 998/1000 | 40.08% | 39.38% | +0.70pp | 8710.5 | -30.54% | 11276.3 | -27.86% | 5.541 | 30.74% | 2 |
| AVUT | 1734/1734 | 70.18% | 71.16% | -0.98pp | 3255.0 | -4.54% | 3578.3 | -16.07% | 5.590 | 30.12% | 0 |
| WorldSense | 3172/3172 | 49.15% | 49.78% | -0.63pp | 4902.8 | -7.10% | 5699.1 | -13.54% | 5.602 | 29.97% | 0 |

This satisfies the full-set gate for this mask-only ARC policy: all benchmark
outputs are complete under the established comparable error policy, quality is
within the ±2pp tolerance, and E2E latency improves on all three benchmarks.

## Current Polling Rule

Poll at roughly one-hour cadence while the full-set loop is running. Each poll
should check:

- `mlx worker list` includes worker `4065939`;
- worker tmux session `arc_fullset_loop_20260720` exists;
- exactly one active benchmark runner is progressing through the chain;
- output `predictions.jsonl` and `profile_samples.jsonl` counts increase;
- `errors.jsonl` remains small / explainable and sample indices have no
  duplicates;
- no compact or keepalive process is running.

If worker `4065939` is interrupted, resume only the mask-only full-set chain:

```bash
NO_COLOR=1 TERM=dumb mlx worker login 4065939 -- \
  bash /mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/logs/audio_router_conditioned_video_sceb_full_20260720/launch_arc_fullset_tmux_on_worker.sh
```

Do not launch compact dispatch for this goal.

## Compact Route Archived

The current full-benchmark loop is mask-only: `compact_dispatch=false`.
Previous compact / physical expert deletion experiments were treated as an
invalid route for this research loop and should not be used for analysis or
future resumes unless explicitly reopened.

Historical compact notes are kept only as provenance:

```text
implementation_planner/SCEB_ARC_TRUE_ACCEL_COMPACT_LOOP_20260720.md
```

They used the same label-free audio-router-conditioned policy, but set
`compact_dispatch=true` and wrote outputs under:

```text
logs/audio_router_conditioned_video_sceb_compact_full_20260720/
```

Static validation had passed:

- compact config loads as
  `confidence_dynamic_compact_dispatch` /
  `arc_video_boundary_audio_floor6_compact_full_v1`;
- shell launch scripts pass `bash -n`;
- compact summary script dry-runs and writes empty analysis tables;
- `python3 moe_expert_budget/_test_compact_dispatch.py` passes;
- `python3 moe_expert_budget/_test_confidence_dynamic.py` passes;
- `python3 -m compileall moe_expert_budget
  logs/audio_router_conditioned_video_sceb_compact_full_20260720/summarize_arc_compact_fullset.py`
  passes.

The chain is idempotent at sample granularity because each runner uses
`--resume` and skips completed `sample_index` entries in `predictions.jsonl`.

## Analysis

Use the loop-local summarizer to aggregate all three benchmarks with the same
schema:

```bash
python3 logs/audio_router_conditioned_video_sceb_full_20260720/summarize_arc_fullset.py
```

It writes:

- `logs/audio_router_conditioned_video_sceb_full_20260720/ARC_FULLSET_ANALYSIS.md`
- `logs/audio_router_conditioned_video_sceb_full_20260720/arc_fullset_analysis.json`

The summarizer reads `profile_samples.jsonl` directly, so it includes fields
that are missing from the OmniVideoBench runner's built-in `summary.json`:

- `generate_latency_ms` / E2E latency;
- `prefill_summary.prefill_latency_ms`;
- effective active top-m;
- effective assignment ratio and assignment reduction;
- top-m histogram;
- audio-router condition state counts.

Current partial observations are sanity checks only, not conclusions. The
full-set gate remains all three benchmarks complete with speed/accuracy checked
against the vanilla references.

# ARC True-Acceleration Compact Loop 2026-07-20

## Goal

Run the same audio-router-conditioned video SCEB policy with real compact
dispatch so speed claims are backed by skipped expert MLP assignments, not only
mask-only budget accounting.

This is prepared as the next full-set loop after the current mask-only
quality-oracle run, or earlier if the current run is explicitly stopped.

## Policy

`arc_video_boundary_audio_floor6_compact_full_v1`:

- target modality: video;
- protected modalities: text, audio, image, special;
- dynamic layers: middle 25%-67%;
- base video top-m: boundary-only dynamic 4/6/8;
- same-layer audio router guard:
  - audio token count < 64 -> video top-m floor 8;
  - audio top1-top2 mean margin < 0.0082 -> video top-m floor 6;
- no task labels and no query regex;
- `compact_dispatch = true`, so dropped video expert assignments are removed
  from the expert loop.
- `compact_debug_compare = false` for production latency measurement; compact
  equivalence is covered by local synthetic tests instead of extra reference
  forwards during full-set inference.

## Files

- config:
  `logs/audio_router_conditioned_video_sceb_compact_full_20260720/arc_video_boundary_audio_floor6_compact_policy.json`
- full chain:
  `logs/audio_router_conditioned_video_sceb_compact_full_20260720/run_arc_compact_fullset_chain.sh`
- worker tmux launcher:
  `logs/audio_router_conditioned_video_sceb_compact_full_20260720/launch_arc_compact_fullset_tmux_on_worker.sh`
- master launcher for worker 4065939:
  `logs/audio_router_conditioned_video_sceb_compact_full_20260720/launch_arc_compact_fullset_worker_4065939.sh`
- analysis:
  `logs/audio_router_conditioned_video_sceb_compact_full_20260720/summarize_arc_compact_fullset.py`

## Run Protocol

Each benchmark uses the same smoke2-then-full resume protocol:

1. run `--limit 2 --resume`;
2. validate at least two predictions and config invariants;
3. run full with no `--limit` and `--resume`.

Output directories are separate from the mask-only quality-oracle run:

- OmniVideoBench:
  `logs/audio_router_conditioned_video_sceb_compact_full_20260720/omnivideobench_boundary_audio_floor6_compact_full`
- AVUT:
  `logs/audio_router_conditioned_video_sceb_compact_full_20260720/avut_boundary_audio_floor6_compact_full`
- WorldSense:
  `logs/audio_router_conditioned_video_sceb_compact_full_20260720/worldsense_boundary_audio_floor6_compact_full`

## Launch Command

Only launch when the current worker is free or the active run has been
explicitly stopped:

```bash
NO_COLOR=1 TERM=dumb mlx worker login 4065939 -- \
  bash /mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/logs/audio_router_conditioned_video_sceb_compact_full_20260720/launch_arc_compact_fullset_tmux_on_worker.sh
```

or from the master:

```bash
logs/audio_router_conditioned_video_sceb_compact_full_20260720/launch_arc_compact_fullset_worker_4065939.sh
```

If switching before the mask-only quality-oracle run finishes, first get
explicit approval to stop the active tmux session, then run:

```bash
NO_COLOR=1 TERM=dumb mlx worker login 4065939 -- \
  bash -lc 'tmux kill-session -t arc_fullset_loop_20260720'
```

After that, launch the compact loop with the command above. The compact runners
are resume-safe at sample granularity and write to separate output directories,
so they do not reuse or corrupt the mask-only output.

## Launch Status

At 2026-07-20 10:48 CST, a separate workspace 2xA100 worker was requested to
run the compact full-set loop without interrupting the active mask-only
quality-oracle run:

```bash
NO_COLOR=1 TERM=dumb mlx worker launch \
  --resourcetype workspace \
  --gpu 2 \
  --type A100-SXM-80GB \
  --workdir /mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner \
  --alias arc-compact-fullset-20260720 \
  -- bash /mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/logs/audio_router_conditioned_video_sceb_compact_full_20260720/run_arc_compact_fullset_chain.sh
```

Master-side launch process:

```text
pid: 801760
log: logs/audio_router_conditioned_video_sceb_compact_full_20260720/launcher/arc_compact_fullset_worker_launch_20260720_104826.log
```

Worker id:

```text
4069954
```

At 2026-07-20 10:57 CST, worker `4069954` had been created but was not ready:
`mlx worker list` showed no `podIP`, `mlx worker login 4069954` returned
`worker has not been ready yet`, and the platform API emitted a context
deadline timeout. The launch process remained alive and waiting. Compact output
directories still had zero predictions, so no partial compact run needed
cleanup or resume handling yet.

At 2026-07-20 11:00 CST, a second low-CPU launch was attempted after quota
showed only 3 CPU cores remaining with 2xA100. The first low-CPU attempt used
`--cpu 3 --memory 800 --gpu 2`, but failed immediately because platform policy
limits memory to about 247G per GPU. A corrected attempt used
`--cpu 3 --memory 494 --gpu 2`; it reached "Login Worker..." but then failed
with `exec command: 0` and wrote no compact samples. The original full-size
launch process for worker `4069954` was still alive, but the worker remained
without a `podIP`. A subsequent `mlx worker list` call hit a platform 500. No
additional worker launches should be attempted until `mlx worker list` becomes
stable again or the user provides a fresh worker.

At 2026-07-20 11:03 CST, `4069954` was still pending with no `podIP`.
`mlx worker login 4069954` again returned `worker has not been ready yet`, and
the original launch log had also recorded a worker control-plane HTTP 502 while
querying resources. Compact outputs still had zero predictions. The active
compact next step is therefore to wait for `4069954` to become ready, reuse a
fresh user-provided worker, or launch after worker control-plane stability
returns.

At 2026-07-20 11:06 CST, `4069954` disappeared from `mlx worker list`; direct
login returned a worker-not-found error. The original direct `mlx worker
launch -- bash run_arc_compact_fullset_chain.sh` process had also exited after
`exec command: 0`. Compact outputs still had zero predictions. A safer generic
launcher was added:

```text
logs/audio_router_conditioned_video_sceb_compact_full_20260720/launch_arc_compact_fullset_worker_generic.sh
```

Use it only after a worker is actually ready. The related launch notes are in:

```text
implementation_planner/SCEB_ARC_COMPACT_WORKER_LAUNCH_NOTES_20260720.md
```

At 2026-07-20 11:24 CST, compact was deprioritized for the current goal because
physical expert deletion had already been considered ineffective. A later
worker (`4070149`) successfully started the compact tmux and model loading, but
was stopped before writing predictions. Compact outputs had 0 predictions and 2
error rows. Do not use compact outputs for the resumed full-benchmark analysis
unless a future loop explicitly reopens compact.

Failed low-CPU logs:

```text
logs/audio_router_conditioned_video_sceb_compact_full_20260720/launcher/arc_compact_fullset_worker_lowcpu_launch_20260720_105929.log
logs/audio_router_conditioned_video_sceb_compact_full_20260720/launcher/arc_compact_fullset_worker_lowcpu494_launch_20260720_110035.log
```

Status check:

```bash
NO_COLOR=1 TERM=dumb mlx worker list
pid=$(cat $(cat logs/audio_router_conditioned_video_sceb_compact_full_20260720/launcher/latest_arc_compact_fullset_launch.pidpath))
ps -p "$pid" -o pid,ppid,stat,etime,cmd
tail -n 120 $(cat logs/audio_router_conditioned_video_sceb_compact_full_20260720/launcher/latest_arc_compact_fullset_launch.logpath)
```

## Analysis

```bash
python3 logs/audio_router_conditioned_video_sceb_compact_full_20260720/summarize_arc_compact_fullset.py
```

It writes:

- `logs/audio_router_conditioned_video_sceb_compact_full_20260720/ARC_COMPACT_FULLSET_ANALYSIS.md`
- `logs/audio_router_conditioned_video_sceb_compact_full_20260720/arc_compact_fullset_analysis.json`

The compact summary includes normal quality/latency columns plus compact
assignment reduction. The debug-diff column remains in the table schema, but
production full-set runs leave it empty because debug compare is disabled.

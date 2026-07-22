# ARC Compact Worker Launch Notes 2026-07-20

## What Failed

Directly launching the full compact chain through `mlx worker launch -- bash
run_arc_compact_fullset_chain.sh` was brittle:

- worker `4069954` was created but never became usable before the command hit
  worker-control-plane failures;
- low-CPU retries failed or exited during worker command execution;
- compact output directories stayed at zero predictions, so no cleanup or
  resume correction was needed.

Observed failure snippets:

- `worker has not been ready yet`;
- worker control-plane HTTP 500/502;
- `exec command: 0`;
- no `podIP` for the pending worker while it existed.

## Safer Pattern

For the next available worker, prefer two phases:

1. Launch or receive an idle 2xA100 worker that stays alive.
2. Once `mlx worker list` shows a `podIP`, start the compact chain through
   login + worker-side tmux:

```bash
logs/audio_router_conditioned_video_sceb_compact_full_20260720/launch_arc_compact_fullset_worker_generic.sh <worker_id>
```

This calls:

```bash
NO_COLOR=1 TERM=dumb mlx worker login <worker_id> -- \
  bash logs/audio_router_conditioned_video_sceb_compact_full_20260720/launch_arc_compact_fullset_tmux_on_worker.sh
```

The worker-side tmux session is:

```text
arc_compact_fullset_loop_20260720
```

The compact run writes to separate output directories and uses `--resume`, so
restarting is safe at sample-index granularity.

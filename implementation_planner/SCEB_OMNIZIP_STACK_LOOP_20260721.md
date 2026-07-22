# SCEB OmniZip Stack Loop 2026-07-21

## Goal

Test whether audio-router-conditioned video SCEB can stack with OmniZip token
compression on OmniVideoBench.

Primary comparison:

- `OmniZip-only`: token compression only, budget disabled.
- `OmniZip+ARC`: OmniZip token compression plus mask-only ARC/SCEB expert
  budgeting.

Reference context:

- `Vanilla`: existing OmniVideoBench vanilla full run.
- `ARC-only`: existing full run at
  `logs/audio_router_conditioned_video_sceb_full_20260720/omnivideobench_boundary_audio_floor6_full`.

## Constraints

- OmniVideoBench only for this loop.
- No compact dispatch.
- No keepalive / `run_keepalive.sh`.
- Use `--resume` so smoke2 and interrupted full runs are sample-idempotent.
- Poll at roughly one-hour cadence. Do not query worker more frequently unless
  a concrete failure signal appears.

## Compatibility Fix

OmniZip compresses `inputs_embeds`, `attention_mask`, and `position_ids` with a
runtime `global_mask` before Qwen3 text MoE layers. SCEB previously built its
modality mask from the original `input_ids`; when OmniZip shortened the
sequence, SCEB saw a length mismatch and skipped masking.

Fix:

- `omnizip/qwen3_omni/modeling_qwen3_omni_moe.py` exposes the current
  `_sceb_omnizip_global_mask` on the thinker.
- `moe_expert_budget/confidence_dynamic.py` aligns modality and target masks
  with that `global_mask` when the current MoE hidden length is compressed.
- `moe_expert_budget/_test_confidence_dynamic.py` now checks compressed-mask
  alignment.

Validation:

```bash
python3 moe_expert_budget/_test_confidence_dynamic.py
python3 -m compileall moe_expert_budget omnizip/qwen3_omni/modeling_qwen3_omni_moe.py logs/profile_omnivideobench_qwen3/run_omnivideobench_budget.py
```

Both passed before launch.

## Files

- stack policy:
  `logs/omnizip_sceb_stack_omnivideobench_20260721/arc_omnizip_stack_policy.json`
- chain:
  `logs/omnizip_sceb_stack_omnivideobench_20260721/run_omnizip_stack_omnivideobench_chain.sh`
- worker tmux launcher:
  `logs/omnizip_sceb_stack_omnivideobench_20260721/launch_omnizip_stack_tmux_on_worker.sh`
- master launcher for worker 4065939:
  `logs/omnizip_sceb_stack_omnivideobench_20260721/launch_omnizip_stack_worker_4065939.sh`
- summarizer:
  `logs/omnizip_sceb_stack_omnivideobench_20260721/summarize_omnizip_stack.py`

## Output Directories

- OmniZip-only:
  `logs/omnizip_sceb_stack_omnivideobench_20260721/omnivideobench_omnizip_only_full`
- OmniZip+ARC:
  `logs/omnizip_sceb_stack_omnivideobench_20260721/omnivideobench_omnizip_arc_stack_full`

## Run Protocol

For each case:

1. run `--limit 2 --resume`;
2. validate at least two predictions/profiles and key invariants;
3. run full with no `--limit` and `--resume`.

The stack smoke check additionally requires:

- `expert_budget.enabled = true`;
- `compact_dispatch = false`;
- `target_modality = video`;
- compressed target token count is positive;
- effective assignment count is positive.

## Polling Template

Each hourly poll should check:

- `mlx worker list` includes worker `4065939`;
- tmux session `omnizip_stack_ovb_20260721` exists while the chain is running;
- exactly one active OmniVideoBench runner is present;
- no compact or keepalive process exists;
- output counts increase and sample indices have no duplicates;
- errors are either zero or match known OmniVideoBench `video_605.mp4` behavior.

If interrupted, resume only this chain:

```bash
NO_COLOR=1 TERM=dumb mlx worker login 4065939 -- \
  bash /mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/logs/omnizip_sceb_stack_omnivideobench_20260721/launch_omnizip_stack_tmux_on_worker.sh
```

## Launch Status

Launched on worker `4065939` at 2026-07-21 10:12 CST.

Worker-side tmux session:

```text
omnizip_stack_ovb_20260721
```

Worker log:

```text
logs/omnizip_sceb_stack_omnivideobench_20260721/launcher/omnizip_stack_worker_n122-192-156_20260721_101249.log
```

Initial post-launch probe at 2026-07-21 10:13 CST:

- worker `4065939` was alive;
- tmux session `omnizip_stack_ovb_20260721` existed;
- active process was `run_omnivideobench_budget.py --use-omnizip --budget-disable --limit 2`;
- no compact or keepalive process was running;
- the old `arc_fullset_loop_20260720` tmux session was still present but idle.

Checkpoint at 2026-07-21 11:14 CST:

- worker `4065939` and tmux session `omnizip_stack_ovb_20260721` were alive;
- OmniZip-only smoke2 had passed and the active process was OmniZip-only full
  `run_omnivideobench_budget.py --use-omnizip --budget-disable --resume`;
- OmniZip-only had 157/1000 predictions, 157/1000 profiles, zero errors, and
  no duplicate sample indices;
- OmniZip+ARC had not started yet;
- no compact or keepalive process was running;
- no adjustment was needed.

Checkpoint at 2026-07-21 12:16 CST:

- worker `4065939` and tmux session `omnizip_stack_ovb_20260721` were alive;
- active process was still OmniZip-only full
  `run_omnivideobench_budget.py --use-omnizip --budget-disable --resume`;
- OmniZip-only had 341/1000 predictions, 341/1000 profiles, zero errors, and
  no duplicate sample indices;
- launcher log tail was around sample 340/341, consistent with local outputs;
- OmniZip+ARC had not started yet;
- no compact or keepalive process was running;
- no adjustment was needed.

Checkpoint at 2026-07-21 13:17 CST:

- worker `4065939` and tmux session `omnizip_stack_ovb_20260721` were alive;
- active process was still OmniZip-only full
  `run_omnivideobench_budget.py --use-omnizip --budget-disable --resume`;
- OmniZip-only had 465/1000 predictions, 465/1000 profiles, zero errors, and
  no duplicate sample indices;
- first missing sample index was 465; launcher log tail was around sample
  464/465, consistent with local outputs;
- OmniZip+ARC had not started yet;
- no compact or keepalive process was running;
- progress from 12:16 to 13:17 was 124 samples/hour; OmniZip-only ETA from
  this checkpoint was roughly 4.3 hours, before the stacked run begins;
- no adjustment was needed.

Checkpoint at 2026-07-21 14:18 CST:

- worker `4065939` and tmux session `omnizip_stack_ovb_20260721` were alive;
- active process was still OmniZip-only full
  `run_omnivideobench_budget.py --use-omnizip --budget-disable --resume`;
- OmniZip-only had 747/1000 predictions, 747/1000 profiles, zero errors, and
  no duplicate sample indices;
- first missing sample index was 747; launcher log tail was around sample
  746/747, consistent with local outputs;
- OmniZip+ARC had not started yet;
- no compact or keepalive process was running;
- progress from 13:17 to 14:18 was 282 samples/hour; at this rate
  OmniZip-only should finish before the next hourly poll and then enter the
  OmniZip+ARC smoke/full stage;
- no adjustment was needed.

Checkpoint at 2026-07-21 15:19 CST:

- worker `4065939` and tmux session `omnizip_stack_ovb_20260721` were alive;
- active process was still OmniZip-only full
  `run_omnivideobench_budget.py --use-omnizip --budget-disable --resume`;
- OmniZip-only had 988/1000 predictions, 988/1000 profiles, and 2 errors;
- the 2 errors were sample indices 942 and 943, both matching the known
  `[1, 1, 0]` runtime error seen in vanilla/ARC OmniVideoBench runs;
- no duplicate sample indices were present; launcher log tail was around sample
  989/990, consistent with local outputs;
- OmniZip+ARC had not started yet;
- no compact or keepalive process was running;
- no adjustment was needed.

Checkpoint at 2026-07-21 16:20 CST:

- worker `4065939` and tmux session `omnizip_stack_ovb_20260721` were alive;
- OmniZip-only full was complete: 998/1000 predictions, 998/1000 profiles,
  2 errors, accounted 1000/1000, no duplicate sample indices;
- OmniZip-only error indices were 942 and 943, matching the known
  `[1, 1, 0]` OmniVideoBench runtime error;
- chain had advanced through OmniZip+ARC smoke into OmniZip+ARC full;
- active process was OmniZip+ARC full
  `run_omnivideobench_budget.py --use-omnizip --budget-config .../arc_omnizip_stack_policy.json --resume`;
- OmniZip+ARC full had 156/1000 predictions, 156/1000 profiles, zero errors,
  and no duplicate sample indices;
- launcher log tail showed `avg_active` around 5.3, confirming SCEB masking was
  active under OmniZip-compressed sequences;
- no compact or keepalive process was running;
- no adjustment was needed.

Checkpoint at 2026-07-21 17:20 CST, after worker queue stopped:

- `mlx worker list` no longer listed worker `4065939`;
- no resume/relaunch was performed in this checkpoint;
- local outputs remained usable:
  - OmniZip-only was complete: 998/1000 predictions, 998/1000 profiles,
    2 known errors at sample indices 942 and 943, accounted 1000/1000,
    accuracy 39.68%;
  - OmniZip+ARC stack was interrupted at 336/1000 predictions,
    336/1000 profiles, zero errors, no duplicate sample indices,
    partial accuracy 39.29%;
- generated partial comparison:
  `logs/omnizip_sceb_stack_omnivideobench_20260721/OMNIZIP_STACK_ANALYSIS.md`;
- current partial stack metrics were: prefill 5044.4 ms, E2E 6392.7 ms,
  active top-m 5.437, assignment reduction 32.04%;
- relative to complete OmniZip-only, the interrupted stack partial was
  +7.78% prefill and +7.82% E2E slower, so stackability is not supported yet
  by the current partial run.

Paired analysis at 2026-07-21 17:20 CST:

- the previous speed comparison was not fair because it compared interrupted
  OmniZip+ARC `0..335` against full-set OmniZip-only;
- fair paired scope: exact sample indices `0..335`;
- paired result:

| Run | Samples | Acc | Prefill mean / p50 ms | E2E mean / p50 ms | Active top-m | Assign Red. |
|---|---:|---:|---:|---:|---:|---:|
| OmniZip-only@336 | 336 | 37.80% (127/336) | 5096.6 / 6110.8 | 6475.7 / 7501.7 | - | - |
| OmniZip+ARC@336 | 336 | 39.29% (132/336) | 5044.4 / 6056.1 | 6392.7 / 7444.6 | 5.437 | 32.04% |

- OmniZip+ARC vs OmniZip-only on the paired subset:
  - accuracy: +1.49pp;
  - prefill: -1.03%;
  - E2E: -1.28%;
  - MoE prefill: -1.58%;
- prediction stability: 310/336 identical predictions (92.26%);
- stack changed 26 predictions: 14 wrong-to-correct and 9 correct-to-wrong,
  net +5 correct samples;
- current paired evidence suggests OmniZip and ARC-SCEB are stackable, but the
  speed gain over OmniZip-only is small at this policy point.

Migration note:

- `logs/` is ignored by git. The code/configuration changes and this handoff can
  be synced through git, but raw JSONL outputs and launcher scripts under
  `logs/omnizip_sceb_stack_omnivideobench_20260721/` will not be pushed.
- To resume the interrupted OmniZip+ARC run from exactly 336/1000 on a new
  machine, copy that logs subdirectory separately or rerun with a fresh output
  directory.

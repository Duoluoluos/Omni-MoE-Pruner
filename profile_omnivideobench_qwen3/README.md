# Qwen3-Omni OmniVideoBench profile runner

This directory mirrors `profile_avut_qwen3` but targets `eval_data/OmniVideoBench`.

Default background run:

```bash
profile_omnivideobench_qwen3/run_on_worker_3899398.sh
```

Useful smoke test:

```bash
LIMIT=1 BACKGROUND=0 profile_omnivideobench_qwen3/run_on_worker_3899398.sh
```

Resume an interrupted run:

```bash
OUT_DIR=/path/to/existing/output RESUME=1 profile_omnivideobench_qwen3/run_on_worker_3899398.sh
```

Additional routing outputs:

- `topk_margin`: per-layer top1/top2 and top-k boundary routing margins.
- `modality_expert_distribution`: text/audio/video expert activation distributions per layer.
- `task_expert_overlap_per_layer_top32_jaccard`: per-layer task-wise expert overlap in `summary.json`.
- `token_trajectory`: per-layer video/audio route signatures grouped by video grid timestep or equal token bins.

Defaults use `eval_data/OmniVideoBench/data.json`. The runner can also read the original parquet if `pyarrow` is installed in the worker Python environment.

Create a visualization snapshot from a running or completed profile:

```bash
python profile_omnivideobench_qwen3/OmniVideoBench_profile_visualize.py \
  --run-dir logs/omnivideobench_qwen3_profile/20260612_144602 \
  --output-dir profile_omnivideobench_qwen3/OmniVideoBench_profile_snapshot
```

The visualizer streams `profile_samples.jsonl` and writes compact PNG/CSV/JSON/Markdown outputs without copying the raw profile.

## Two-pass evidence retrieval prototype

`run_omnivideobench_twopass.py` implements a first-pass skim followed by a high-token pass over selected temporal windows.

The default path is GT-free MoE route-signature retrieval:

- fixed short windows, default `SEGMENT_SECONDS=6`;
- low-cost skim pass per window, default `SKIM_VIDEO_FPS=0.5` and `SKIM_SEGMENT_MAX_FRAMES=12`;
- middle-layer MoE route signatures for text/video/audio tokens;
- route similarity between question-text signature and segment video/audio signatures;
- route novelty, neighboring route change, route margin/confidence, audio RMS saliency, and sparse visual scene-change score;
- diverse evidence-chain selection rather than plain top-K.

`reasoning_steps` timestamps are no longer used by default. The old time-cue path remains available only as an oracle upper-bound diagnostic via `--retrieval-mode oracle_time_audio --oracle-reasoning-times`; do not report it as real inference acceleration.

Run retrieval-only sanity check:

```bash
python profile_omnivideobench_qwen3/run_omnivideobench_twopass.py \
  --retrieval-only \
  --limit 3 \
  --output-dir /tmp/omnivideobench_twopass_retrieval_check
```

Run Qwen3-Omni two-pass on worker `3898754`:

```bash
profile_omnivideobench_qwen3/run_twopass_on_worker_3898754.sh
```

Useful knobs:

- `TOP_SEGMENTS=4` selects a route-diverse evidence chain by default.
- `SEGMENT_SECONDS=6` and `SEGMENT_CONTEXT_SECONDS=2` control each candidate window and padding.
- `SEGMENT_MAX_FRAMES=128` caps frames per selected segment.
- `SKIM_VIDEO_FPS=0.5` and `SKIM_SEGMENT_MAX_FRAMES=12` control the low-cost route-signature skim.
- `RETRIEVAL_MODE=oracle_time_audio ORACLE_REASONING_TIMES=1` runs the old annotation-time upper bound.
- `NO_AUDIO_ENERGY=1` disables audio RMS scoring.

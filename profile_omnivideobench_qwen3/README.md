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

`run_omnivideobench_twopass.py` implements a first-pass skim followed by a high-token pass over selected temporal windows. The current v0 scorer uses:

- oracle time cues parsed from `reasoning_steps` evidence/inference text, enabled by default for an upper-bound diagnostic;
- explicit time cues in the question text;
- low-cost audio RMS energy from PyAV, with librosa fallback;
- non-overlap selection so adjacent windows around the same cue do not consume all slots.

The unimplemented skim signals from the analysis are left explicit in the retrieval metadata as `not_enabled_v0`: low-res frame features, MoE route signature, and cheap partial Qwen3-Omni forward.

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

- `TOP_SEGMENTS=2` selects two skim windows by default.
- `SEGMENT_SECONDS=30` and `SEGMENT_CONTEXT_SECONDS=8` control each candidate window and padding.
- `SEGMENT_MAX_FRAMES=128` caps frames per selected segment.
- `NO_ORACLE_REASONING_TIMES=1` disables annotation-derived time cues for a non-oracle retrieval run.
- `NO_AUDIO_ENERGY=1` disables audio RMS scoring.

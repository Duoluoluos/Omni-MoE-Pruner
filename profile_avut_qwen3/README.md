# Qwen3-Omni AVUT profile runner

This directory contains the isolated scripts for running the unpruned Qwen3-Omni model on `eval_data/AVUTBenchmark`.

Default run:

```bash
profile_avut_qwen3/run_on_worker_3899398.sh
```

Useful smoke test:

```bash
LIMIT=1 BACKGROUND=0 profile_avut_qwen3/run_on_worker_3899398.sh
```

Resume an interrupted run:

```bash
OUT_DIR=/path/to/existing/output RESUME=1 profile_avut_qwen3/run_on_worker_3899398.sh
```

Outputs:

- `predictions.jsonl`: one AVUT prediction per sample.
- `profile_samples.jsonl`: per-sample prefill, routing, MoE, KV-cache, and system-overhead profile.
- `summary.json`: aggregate accuracy and profile summary.
- `torch_profiler/`: key averages and Chrome trace for the first `TORCH_PROFILE_SAMPLES` samples.

The runner does not enable OmniZip or token pruning. It loads `Qwen3OmniMoeForConditionalGeneration` from transformers, disables the talker, and records profile data through runtime hooks.

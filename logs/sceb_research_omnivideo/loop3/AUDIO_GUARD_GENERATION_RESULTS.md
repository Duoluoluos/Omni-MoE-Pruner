# Audio-as-Guard SCEB Generation Results

This loop3 artifact records the Audio-as-Guard SCEB probe for OmniVideoBench.
It uses only `eval_data/OmniVideoBench` and runs through Merlin worker 3955069
with `NO_COLOR=1 TERM=dumb`.

## Policy

Policy config:

`logs/omnivideobench_qwen3_research/policy_variants_20260709/loop3_audio_guard_policy.json`

Audio-as-Guard SCEB keeps audio protected at full top-8 and uses audio/task
metadata to make video expert budgets more conservative on audio-sensitive
cases:

```text
audio/text/image/special tokens: full top-8
video middle-layer tokens: global both-signal SCEB thresholds
stress tasks: video min_top_m = 6
background&music understanding: video min_top_m = 6
other tasks: normal top-4/6/8 decisions from both-signal thresholds
```

This is an audio-aware guard over video budgets. It does not prune audio
experts.

## Smoke Run

Smoke runner:

`logs/omnivideobench_qwen3_research/loop3_audio_guard_20260709/run_smoke_audio_guard.sh`

Smoke output:

`logs/omnivideobench_qwen3_research/loop3_audio_guard_20260709/smoke2`

Smoke completed on worker 3955069 on July 9, 2026.

| Metric | Value |
|---|---:|
| Samples | 2 |
| Correct | 0 |
| Accuracy | 0.00% |
| Sample 0 prefill | 10.79s |
| Sample 1 prefill | 6.78s |
| Sample 0 avg active experts | 7.222 |
| Sample 1 avg active experts | 7.221 |

The smoke confirms the audio-guard policy loads, wraps 48 SparseMoeBlock
layers, keeps audio in the protected modality set, and writes normal
`summary.json`, `predictions.jsonl`, and `profile_samples.jsonl` artifacts.

## Limit-200 Probe Launch

Limit-200 runner:

`logs/omnivideobench_qwen3_research/loop3_audio_guard_20260709/run_limit200_audio_guard.sh`

Detached launcher:

`logs/omnivideobench_qwen3_research/loop3_audio_guard_20260709/launch_limit200_audio_guard_3955069.sh`

Worker launch state:

| Field | Value |
|---|---|
| Worker | 3955069 |
| Remote PID | 2507121 |
| PID file | `logs/omnivideobench_qwen3_research/loop3_audio_guard_20260709/launcher/limit200_audio_guard_3955069.pid` |
| Log file | `logs/omnivideobench_qwen3_research/loop3_audio_guard_20260709/launcher/limit200_audio_guard_3955069_20260709_104918.log` |
| Output dir | `logs/omnivideobench_qwen3_research/loop3_audio_guard_20260709/limit200` |

The limit-200 runner uses `--limit 200 --resume` so it can resume from partial
predictions if the worker session is interrupted.

## Interpretation

The smoke average active expert count is higher than the Boundary-Guarded smoke
because Audio-as-Guard starts from global both-signal thresholds and raises
audio-sensitive/stress-task video decisions to at least top-6. That is the
intended tradeoff: it gives up some aggressive top-4 assignment to protect
cross-modal examples while leaving audio itself at top-8.

Expected success criteria for the completed limit-200 summary:

```text
accuracy close to Conservative Global SCEB on the matched subset
lower churn than aggressive boundary-only on audio-sensitive/stress samples
mean active experts near the phase-2 audio guard counterfactual budget
```

This artifact supports the method taxonomy by making Audio-as-Guard a concrete
configured SCEB variant with a real worker smoke and durable limit200 launch
state.

# Boundary-Guarded SCEB Probe Results

This loop3 artifact records the Boundary-Guarded SCEB probe for
OmniVideoBench. It uses only `eval_data/OmniVideoBench` and runs through Merlin
worker 3955069 with `NO_COLOR=1 TERM=dumb`.

## Policy

Policy config:

`logs/omnivideobench_qwen3_research/policy_variants_20260709/loop3_boundary_guarded_policy.json`

Boundary-Guarded SCEB starts from the aggressive boundary-only signal and adds a
decisiveness guard for the most aggressive top-4 decisions:

```text
top-4: boundary stable and p1-p2 >= 0.008606448397040365
top-6: boundary stable and top-4 guard does not pass
top-8: boundary unstable or protected token/layer/modality
stress/audio tasks: min_top_m = 6
```

This keeps boundary stability as the compute-reduction signal while using
dominant-expert decisiveness and task stress as churn guards.

## Smoke Run

Smoke runner:

`logs/omnivideobench_qwen3_research/loop3_boundary_guard_20260709/run_smoke_boundary_guard.sh`

Smoke output:

`logs/omnivideobench_qwen3_research/loop3_boundary_guard_20260709/smoke2`

Smoke completed on worker 3955069 on July 9, 2026.

| Metric | Value |
|---|---:|
| Samples | 2 |
| Correct | 0 |
| Accuracy | 0.00% |
| Sample 0 prefill | 10.74s |
| Sample 1 prefill | 6.95s |
| Sample 0 avg active experts | 6.956 |
| Sample 1 avg active experts | 6.955 |

The smoke confirms the configured policy loads, wraps 48 SparseMoeBlock layers,
uses middle layers 12-31 as dynamic layers, and writes normal
`summary.json`, `predictions.jsonl`, and `profile_samples.jsonl` artifacts.

## Limit-200 Probe Launch

Limit-200 runner:

`logs/omnivideobench_qwen3_research/loop3_boundary_guard_20260709/run_limit200_boundary_guard.sh`

Detached launcher:

`logs/omnivideobench_qwen3_research/loop3_boundary_guard_20260709/launch_limit200_boundary_guard_3955069.sh`

Worker launch state:

| Field | Value |
|---|---|
| Worker | 3955069 |
| Remote PID | 2506229 |
| PID file | `logs/omnivideobench_qwen3_research/loop3_boundary_guard_20260709/launcher/limit200_boundary_guard_3955069.pid` |
| Log file | `logs/omnivideobench_qwen3_research/loop3_boundary_guard_20260709/launcher/limit200_boundary_guard_3955069_20260709_104511.log` |
| Output dir | `logs/omnivideobench_qwen3_research/loop3_boundary_guard_20260709/limit200` |

The limit-200 run is a real background generation probe, not a replay. It uses
`--limit 200 --resume` so interrupted output can be resumed from existing
predictions.

## Interpretation

Boundary-only full-run evidence showed strong compute reduction but higher
answer churn. Boundary-Guarded SCEB is the loop3 method-family bridge: it keeps
the boundary-only compute frontier but prevents top-4 assignment unless the
dominant expert is also decisive. Stress and audio-oriented tasks are clamped
to at least top-6.

Expected success criteria for the completed limit-200 summary:

```text
mean active experts < Conservative Global SCEB
answer churn lower than Boundary-only on the matched subset
accuracy not meaningfully below the vanilla matched subset
```

Until the limit-200 job finishes, the completed evidence is the real worker
smoke plus the durable background launch state.

# 07. Evaluation Report Template

每一次实验都用这个模板写报告，避免只看平均 latency 或只看 overall accuracy。

## Experiment Metadata

```text
experiment_name:
date:
model_path:
benchmark:
sample_indices:
num_samples:
baseline_run:
policy_file:
code_commit:
device:
```

## Policy Summary

```json
{
  "mode": "",
  "target_layers": [],
  "target_modalities": [],
  "protected_modalities": [],
  "top_m_options": [],
  "task_policy": {},
  "capacity_policy": {}
}
```

## Accuracy

| metric | baseline | experiment | delta |
|---|---:|---:|---:|
| overall accuracy | | | |
| baseline-correct flip rate | | | |
| baseline-wrong fixed rate | | | |

## Stress Task Breakdown

| task | n | baseline acc | experiment acc | delta | flip rate |
|---|---:|---:|---:|---:|---:|
| temporal understanding | | | | | |
| spatial understanding | | | | | |
| reference reasoning | | | | | |
| counting | | | | | |

## Cost / Runtime

| metric | baseline | experiment | delta |
|---|---:|---:|---:|
| prefill mean ms | | | |
| prefill p50 ms | | | |
| prefill p95 ms | | | |
| MoE block mean ms | | | |
| MoE block p95 ms | | | |
| assignment count | | | |
| avg active experts/token | | | |
| MoE FFN FLOPs estimate | | | |

## Load Balance

| metric | baseline | experiment | delta |
|---|---:|---:|---:|
| hot/mean expert batch mean | | | |
| hot/mean expert batch p95 | | | |
| empty experts mean | | | |
| experts over capacity | | | |

## Per-Layer Notes

List layers where:

- accuracy-sensitive changes seem likely.
- MoE time improved.
- hot expert got worse.
- assignment count dropped but latency did not.

## Failure Cases

For baseline-correct -> experiment-wrong samples:

| sample_index | task | video_type | duration | baseline | experiment | suspected reason |
|---:|---|---|---:|---|---|---|

## Decision

Choose one:

```text
PASS: proceed to next stage
ADJUST: tune policy and rerun
FAIL: stop this direction
```

Decision rationale:

```text
...
```

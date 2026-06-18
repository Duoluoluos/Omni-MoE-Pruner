# 03. Task / Layer / Modality Policy

第三步把 OmniVideoBench 的 task、layer、modality 现象显式写进 policy。

## 为什么需要 Policy

Profile 里有三个重要现象：

- text/audio token 少但 expert preference 更尖锐，应保护。
- task expert overlap 有分化，`temporal understanding` 等任务更独特。
- per-layer task overlap 显示共享层和任务分化层并存。

所以不能一套 top-k 规则打所有层、所有任务、所有模态。

## Task Cluster

建议第一版：

```json
{
  "stress_tasks": [
    "temporal understanding",
    "spatial understanding",
    "reference reasoning",
    "counting"
  ],
  "medium_tasks": [
    "causal reasoning",
    "hypothetical reasoning",
    "relationship reasoning",
    "ego reasoning"
  ],
  "perception_tasks": [
    "fine-grained perception",
    "attribute comparison",
    "sentiment analysis"
  ],
  "audio_tasks": [
    "background&music understanding"
  ]
}
```

Policy：

```text
stress_tasks: min_top_m = 6
medium_tasks: min_top_m = 4
perception_tasks: min_top_m = 4
audio_tasks: video min_top_m = 4, audio full top-k
```

## Layer Policy

不要永久写死层号。先由比例生成：

```text
early:  [0, 0.25L)
middle: [0.25L, 0.67L)
late:   [0.67L, L)
```

默认：

```text
early: full top-k
middle: dynamic top-k
late: full top-k
```

后续可以用 profile 自动选择 layer：

```text
layer_score =
  routing_margin_score
  + task_separation_score
  + modality_separation_score
  - late_layer_penalty
```

共享层保守：

- 如果某层在 task expert overlap per layer 里对多数 task pair overlap 很高，保持 full top-k 或更高 min_top_m。

任务分化层可动：

- 如果某层 task separation 高，允许 task-specific min_top_m。

## Modality Policy

第一版：

```text
text: full top-k
audio: full top-k
image: full top-k
video: dynamic top-k
```

如果后续发现 audio route signature 能预测 segment confidence，只用它做 controller，不裁它。

## Policy JSON

建议输出：

```json
{
  "version": "omnivideobench_expert_budget_v1",
  "default_top_k": 8,
  "protected_modalities": ["text", "audio", "image", "special"],
  "layer_policy": {
    "early": {"range_ratio": [0.0, 0.25], "mode": "full"},
    "middle": {"range_ratio": [0.25, 0.67], "mode": "dynamic"},
    "late": {"range_ratio": [0.67, 1.0], "mode": "full"}
  },
  "task_policy": {
    "stress": {"tasks": ["temporal understanding", "spatial understanding", "reference reasoning", "counting"], "min_top_m": 6},
    "default": {"min_top_m": 4}
  }
}
```

## Evaluation

对比：

```text
dynamic top-k without policy
dynamic top-k with task/layer/modality policy
```

目标：

- stress task drop 降低。
- baseline-correct flip rate 降低。
- assignment reduction 仍然保留。

如果 policy 后 assignment reduction 太小：

- 只放宽 perception tasks。
- 或只在 task分化高的层做 top_m=4。

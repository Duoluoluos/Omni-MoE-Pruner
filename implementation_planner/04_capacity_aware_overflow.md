# 04. Capacity-Aware Overflow

第四步验证 load balance。目标不是减少平均 assignment，而是减少 hot expert 的过载。

## Profile 依据

OmniVideoBench routing overhead 显示：

- 前中层 hot/mean expert batch 更高。
- 最高超过 10。
- 说明某些层存在明显 expert skew。

这会导致：

- 单卡：某些 expert GEMM 特别大，其他 expert 小，grouped execution 不均。
- 多卡：hot expert 所在 GPU 成为 straggler。

## 核心思想

对每个 target layer 设置 expert capacity：

```text
capacity = ceil(cap_factor * total_assignments / num_experts)
```

如果某 expert 超过 capacity：

```text
保留高优先级 assignment
处理低优先级 overflow assignment
```

第一版只做 mask-only，不做真实 reroute。

## Assignment Priority

推荐：

```text
priority =
  routing_weight
  + protected_modality_bonus
  + low_margin_guard_bonus
  + task_stress_bonus
```

解释：

- routing_weight 高，保留。
- text/audio/special，保留。
- low-margin token，保留更多 experts。
- stress task，保守。

低优先级 overflow：

- 如果 token 已经保留 >= min_top_m experts，则 drop。
- 如果不足 min_top_m，则保留。

## 配置

```json
{
  "mode": "capacity_aware_overflow_mask",
  "capacity_factor": 1.25,
  "target_modality": "video",
  "protected_modalities": ["text", "audio", "image", "special"],
  "min_top_m_by_task": {
    "stress": 6,
    "default": 4
  },
  "reroute": false
}
```

## 可选 Reroute

第二版再做 reroute：

```text
overflow assignment expert=e_hot
look at next-best expert candidates from router logits
choose candidate with:
  not over capacity
  routing_weight close enough
  same layer
```

约束：

```text
only reroute boundary assignment
do not reroute top1
do not reroute protected modality
do not reroute low-margin token
```

## 输出指标

每层记录：

```json
{
  "layer_idx": 16,
  "capacity": 4300,
  "hot_to_mean_before": 10.4,
  "hot_to_mean_after": 4.2,
  "overflow_assignments": 12000,
  "dropped_assignments": 9000,
  "rerouted_assignments": 0,
  "experts_over_capacity_before": 9,
  "experts_over_capacity_after": 0
}
```

## Evaluation

对比：

```text
dynamic top-k policy
dynamic top-k policy + capacity overflow
```

目标：

- hot/mean expert batch 明显下降。
- accuracy 不显著下降。
- assignment reduction 不一定更大，但 expert batch 更规整。

通过标准：

```text
hot_to_mean p95下降 >= 30%
overall accuracy drop <= 额外 1 percentage point
stress task drop <= 额外 2 percentage points
```

如果质量掉：

- capacity_factor 从 1.25 提到 1.5。
- 禁止 reroute/drop top2。
- stress tasks full top-k。

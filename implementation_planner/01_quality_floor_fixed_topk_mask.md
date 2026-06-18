# 01. Quality Floor: Fixed Top-M Mask

这是第一步，也是最粗的操作。目标不是加速，而是测质量底线：

> 如果只在中层把 video token 的 active experts 从原始 top-k 降到固定 top-m，答案会掉多少？

这一步做 **mask-only ablation**。不要改 dispatch，不要写 kernel，不要追求真实 latency。只改 MoE combine 的数学结果，用于质量评估。

## 实验假设

OmniVideoBench profile 显示：

- 中层 routing margin 更高，说明中层 top experts 更可信。
- 后层 margin 下降，可能在做全局整合，应保守。
- text/audio token 少但重要，应保护。

所以第一版只做：

```text
video token only
middle layers only
fixed top-m = 4
text/audio/special full top-k
early/late layers full top-k
```

## 推荐默认配置

如果模型默认 `top_k=8`：

```json
{
  "enabled": true,
  "mode": "fixed_topm_mask",
  "target_modality": "video",
  "protected_modalities": ["text", "audio", "image", "special"],
  "original_top_k": 8,
  "top_m": 4,
  "layer_start": 12,
  "layer_end": 32,
  "renormalize_weights": true
}
```

如果无法确认 48 层边界，coding agent 应从 `model.thinker.config.text_config.num_hidden_layers` 读取，再用比例：

```text
layer_start = floor(0.25 * num_layers)
layer_end   = floor(0.67 * num_layers)
```

## 实现位置

优先复用已有 profiler hook 里的 `SparseMoeBlock.forward` 包装思路。

现有参考：

```text
profile_avut_qwen3/run_avut_profile.py
Qwen3ProfileCollector._wrap_moe_forward()
Qwen3ProfileCollector._summarize_router_logits()
```

需要新增一个轻量 wrapper，例如：

```text
moe_expert_budget/
  __init__.py
  mask_policy.py
  fixed_topm.py
```

## Mask-Only 语义

原始 MoE 大致是：

```text
probs = softmax(router_logits)
top_weights, selected = topk(probs, k)
expert_outputs = expert_mlp(hidden_states, selected)
output = weighted_sum(expert_outputs, top_weights)
```

mask-only 版本：

```text
if layer in target_layers and token_modality == video:
    keep only first top_m experts from selected/top_weights
    zero out remaining experts in combine
    optionally renormalize kept weights
else:
    keep original top_k
```

注意：

- 这一步如果底层仍然计算了 8 个 expert，latency 不会明显变。
- 只用于质量底线，不用于速度结论。
- 必须记录 effective active experts/token。

## Renormalization

第一版建议 `renormalize_weights=true`：

```text
kept_weights = kept_weights / sum(kept_weights)
```

原因：

- 如果直接 zero tail experts，不归一化会改变 residual scale。
- 归一化更像“用前 m 个 expert 近似原 top-k mixture”。

也可以在 ablation 里加一组：

```text
top_m=4 no_renorm
top_m=4 renorm
```

默认先跑 renorm。

## 必须保护的 Token

保护规则：

```text
if modality in text/audio/image/special:
    top_m = original_top_k
```

special token 识别方式：

- 不属于 video/audio/image token id 的普通 text token 都先按 text 保护。
- 如果有 padding token，也保护或直接不参与 MoE。

## 输出 Profile 字段

每个样本新增：

```json
{
  "expert_budget": {
    "enabled": true,
    "mode": "fixed_topm_mask",
    "target_layers": [12, 13, 14],
    "top_m": 4,
    "protected_modalities": ["text", "audio", "image", "special"],
    "renormalize_weights": true,
    "assignment_count_original_est": 123456,
    "assignment_count_effective": 78910,
    "avg_active_experts_per_token_original": 8.0,
    "avg_active_experts_per_token_effective": 5.6,
    "layer_summaries": [
      {
        "layer_idx": 16,
        "video_tokens": 55296,
        "video_active_experts_original": 8,
        "video_active_experts_effective": 4,
        "protected_tokens": 521
      }
    ]
  }
}
```

## Evaluation

先跑小规模：

```text
limit = 50
```

如果无崩溃，再跑：

```text
limit = 100
```

最后再跑完整 411。

必须 paired A/B：

```text
同一批 sample_index
baseline prediction
fixed_topm prediction
```

报告：

- overall accuracy。
- baseline-correct flip rate。
- stress task accuracy：
  - temporal understanding
  - spatial understanding
  - reference reasoning
  - counting
- per-task answer flip。
- average active experts/token effective。
- MoE FFN FLOPs estimate reduction。

不要声称 latency speedup，除非后续做了 real compact dispatch。

## 通过 / 失败标准

建议第一步通过标准：

```text
overall accuracy drop <= 5 percentage points
baseline-correct flip rate <= 10%
stress task drop <= 8 percentage points
```

如果失败：

- top_m 从 4 改成 6。
- layer range 缩小到更中间的层。
- temporal/spatial/reference/counting 全部改成 full top-k。

如果通过：

- 进入 `02_confidence_dynamic_topk_mask.md`。

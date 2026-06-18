# 02. Confidence Dynamic Top-K Mask

第二步在固定 top-m 质量底线基础上加入 routing confidence。目标是验证：

> routing margin 能否预测“这个 video token 可以少算几个 expert”？

仍然是 mask-only ablation，不追求真实 latency。

## 核心规则

对 target middle layers 的 video token：

```text
high confidence  -> top_m = 4
medium confidence -> top_m = 6
low confidence   -> top_m = 8
```

text/audio/image/special token 保持 full top-k。

## Confidence 指标

从 router softmax 后的 top weights 计算：

```text
top1_top2_margin = p1 - p2
boundary_margin = p_top_m - p_(top_m + 1)
entropy = -sum(p log p)
```

推荐第一版只用简单规则：

```text
if top1_top2_margin >= high_margin_threshold
   and boundary_margin_for_4 >= boundary_threshold:
    top_m = 4
elif top1_top2_margin >= medium_margin_threshold
   and boundary_margin_for_6 >= boundary_threshold:
    top_m = 6
else:
    top_m = 8
```

阈值不要手写死。先从 baseline profile 的中层 video token margin 分布取分位数：

```text
high_margin_threshold = p70(top1_top2_margin)
medium_margin_threshold = p40(top1_top2_margin)
boundary_threshold = p50(boundary_margin)
```

如果没有 token-level margin dump，则第一版可以用全局层均值先跑，但要在 report 里标注风险。

## 推荐配置

```json
{
  "enabled": true,
  "mode": "confidence_dynamic_topk_mask",
  "target_modality": "video",
  "protected_modalities": ["text", "audio", "image", "special"],
  "layer_start_ratio": 0.25,
  "layer_end_ratio": 0.67,
  "top_m_options": [4, 6, 8],
  "threshold_source": "profile_percentile",
  "high_margin_percentile": 70,
  "medium_margin_percentile": 40,
  "boundary_margin_percentile": 50,
  "renormalize_weights": true
}
```

## 为什么 Low-Margin 保持 Full Top-K

OmniVideoBench profile 显示后层 margin 下降、route diversity 上升。低 margin 不是“没用”，而是不确定。

所以保护 low-margin token：

```text
low margin = router uncertain = keep more experts
```

不要反过来把 low-margin 当作低重要性。

## 输出 Profile 字段

新增：

```json
{
  "expert_budget": {
    "mode": "confidence_dynamic_topk_mask",
    "top_m_histogram": {"4": 120000, "6": 80000, "8": 30000},
    "margin_thresholds": {
      "top1_top2_high": 0.18,
      "top1_top2_medium": 0.07,
      "boundary": 0.03
    },
    "avg_active_experts_per_video_token_by_layer": {
      "16": 5.1,
      "24": 5.4
    }
  }
}
```

## Evaluation

和 `01` 对比：

```text
baseline
fixed top_m=4
dynamic {4,6,8}
```

关键问题：

- dynamic 是否比 fixed top_m=4 准确率更稳？
- dynamic 是否仍然有明显 assignment reduction？
- stress tasks 是否被保护住？

通过标准：

```text
dynamic accuracy >= fixed_topm accuracy
dynamic assignment reduction >= 20%
baseline-correct flip rate <= fixed_topm flip rate
```

如果 dynamic 不优于 fixed，说明 margin threshold 或 protected policy 有问题，不要继续做真实 dispatch。

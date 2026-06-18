# Qwen3-Omni MoE Expert Compute Optimization Implementation Plan

这个目录给后续 coding agent 使用。目标是围绕 OmniVideoBench profile 里的 MoE 现象，逐步验证“Expert 计算优化，而不是删专家”的可行性。

核心原则：

- 先做质量底线，再做真实加速。
- 先做 mask-only ablation，不急着改 dispatch/kernel。
- 只优化 video token；text/audio/special token 默认保护。
- 只动中层；早层 warmup，后层保守。
- 不做 global expert deletion。
- 每一步都要 paired A/B，对比 baseline 正确样本是否被改错。

推荐执行顺序：

1. [`00_profile_signals_to_design.md`](00_profile_signals_to_design.md)  
   说明为什么这些操作来自 OmniVideoBench profile，而不是照搬纯文本 MoE。

2. [`01_quality_floor_fixed_topk_mask.md`](01_quality_floor_fixed_topk_mask.md)  
   最粗质量底线：中层 video token 固定从 top-k=8 降到 top-m=4，只改 combine 数学输出，不追求速度。

3. [`02_confidence_dynamic_topk_mask.md`](02_confidence_dynamic_topk_mask.md)  
   用 routing margin 做动态 top-k：high-margin 降，low-margin 保持 full top-k。

4. [`03_task_layer_modality_policy.md`](03_task_layer_modality_policy.md)  
   加 task/layer/modality policy：stress tasks 更保守，共享/后层保守。

5. [`04_capacity_aware_overflow.md`](04_capacity_aware_overflow.md)  
   做 expert capacity cap 和 overflow priority，验证 hot expert/load balance 是否改善。

6. [`05_real_compact_dispatch.md`](05_real_compact_dispatch.md)  
   从 mask-only 进入真实加速：减少 token-expert assignment，重建 compact dispatch list。

7. [`06_multigpu_placement_load_balance.md`](06_multigpu_placement_load_balance.md)  
   如果跑多卡，再做 all-to-all、hot expert placement、replication 和 load-aware reroute。

8. [`07_eval_report_template.md`](07_eval_report_template.md)  
   每一轮实验都按同一个模板产出报告，避免只看平均值。

最小闭环：

```text
baseline
-> fixed top-m mask quality floor
-> confidence dynamic top-m
-> task/layer/modality policy
-> real compact dispatch
```

第一轮只需要完成 `01_quality_floor_fixed_topk_mask.md`。如果这一步质量已经大幅崩掉，就不要继续做真实 dispatch；先回到 policy 和保护规则。

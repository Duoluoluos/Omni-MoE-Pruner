# 00. Profile Signals To Design

本文档说明每个 MoE expert compute 优化设计对应 OmniVideoBench profile 的哪个现象。

## 关键 Profile 现象

### 1. 不能做 Global Expert Deletion

现象：

- 全局 top expert 最高频率约 1.07%。
- 128 个 expert 的均匀基准是 1 / 128 = 0.78%。
- 没有少数 expert 全局统治。
- 某些 expert 只是在特定 layer / modality / task 上偏热。

设计：

- 不删全局 expert。
- 不做静态 whitelist / blacklist。
- 优化对象改成 `(layer, token, expert)` assignment。

### 2. MoE 是二阶段优化，不是主矛盾

现象：

- 平均 input length 约 43,788 tokens。
- video tokens 约 43,273，占 98%。
- prefill latency 和 video tokens 相关性约 0.991。
- attention FLOPs 和 prefill latency 相关性约 0.997。
- attention 估算占主要成本，MoE FFN 是次要但仍显著成本。

设计：

- MoE expert compute optimization 作为 token/attention 压缩后的第二阶段。
- 实验里必须单独报告 MoE block latency，不能只报 overall prefill。

### 3. 中层适合动，后层保守

现象：

- 中层 top1-top2 margin 更高。
- 中层 top-k boundary margin 更高。
- 后层 entropy 回升，margin 下降。
- 后层 route diversity 更高，但不一定代表冗余。

设计：

- early layers: no change / warmup。
- middle layers: dynamic expert budget。
- late layers: full top-k。
- low-margin token 保守。

### 4. Text/Audio 是 Controller，不是主要节省对象

现象：

- text/audio token 数量很少。
- video expert distribution 很平滑。
- text/audio distribution 更尖锐，有更明显 expert preference。
- text/audio 更可能携带 question intent 和跨模态控制信号。

设计：

- text token full top-k。
- audio token full top-k。
- special token full top-k。
- 只对 video token 做 expert budget reduction。

### 5. Task Expert Overlap 有分化

现象：

- `temporal understanding` 和其他 task overlap 偏低。
- `ego reasoning` 也有部分独特路径。
- per-layer overlap 显示某些层共享，某些层任务分化。

设计：

- stress tasks 更保守。
- task cluster 而不是每个 task 单独写死。
- 共享层保守，任务分化层才做 task-specific budget。

### 6. Expert Skew 支持 Capacity / Load Smoothing

现象：

- 前中层 hot/mean expert batch 更高，最高超过 10。
- empty experts 主要在前层；中后层多数 expert 都会被用到。

设计：

- 不用“删空专家”作为主策略。
- 可以对 hot expert 做 capacity cap。
- overflow 时只处理低优先级 boundary assignment。
- high-margin / high-weight assignment 不 reroute。

## 最终设计命名

建议把这个方向命名为：

```text
Capacity-Aware Confidence Routing
```

含义：

- confidence：用 routing margin 决定 token 是否能少算 expert。
- capacity-aware：用 expert bucket capacity 规整负载。
- routing：优化 token-expert assignment，而不是删 expert 权重。

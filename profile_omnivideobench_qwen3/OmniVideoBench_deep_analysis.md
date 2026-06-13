# OmniVideoBench Qwen3-Omni Profile 深度分析

分析日期：2026-06-12
输入目录：`OmniVideoBench_profile_snapshot`
对象：Qwen3-Omni MoE 在 OmniVideoBench 上的 profile snapshot。

## 1. 总体口径

- 样本数：411
- 准确率：167 / 411 = 40.63%
- 平均视频时长：515.3s，p50 420s，p95 1380s，最长 1956s
- 平均输入长度：43,788 tokens，p50 55,799，p95 55,840，max 55,990
- 平均 prefill：21.64s，p50 28.14s，p95 28.66s，max 29.48s
- 平均 generate：24.98s，p50 31.68s，p95 33.81s
- FlashAttention requested：全部样本
- Chunked prefill observed：0

Token 构成：

| modality | mean tokens | share |
|---|---:|---:|
| text | 131.3 | 0.6% |
| video | 43,273.2 | 98.0% |
| audio | 383.4 | 1.4% |

这和 AVUT 完全不是一个量级。AVUT 平均约 9.6k tokens，OmniVideoBench 平均约 43.8k tokens，并且超过一半样本被 768 frames / 55k video tokens cap 卡住。这里的核心瓶颈已经不是音频 token，而是超长视频 prefill，尤其是 attention。

## 2. 总体结论

1. OmniVideoBench 是更适合研究 MoE expert insight 的多跳 benchmark。和 AVUT 相比，它的 task expert overlap 更有分化，per-layer overlap 也能看到共享层和差异层。
2. 性能瓶颈主要是视频 token 和 attention。`video_tokens` 与 `prefill_latency_ms` 的相关性约 0.991，attention FLOPs 与 prefill 相关性约 0.997。attention 估算占总 FLOPs 约 80.6%，MoE FFN 约 19.4%。
3. 很多样本被 max frame cap 限制。221 / 411 条样本达到 768 frames；219 条达到 55,296 video tokens。超过约 6 分钟后输入长度基本饱和，prefill 也饱和到约 28s。
4. 准确率与长度/时长几乎不相关。更长、更贵的样本并没有更高准确率，说明当前均匀或固定 cap 采样没有充分解决多跳关键片段选择问题。
5. 全局删专家不合适。全局 top expert 频率最高约 1.07%，只比 1/128 的均匀基准略高。专家没有明显全局坍缩。
6. 中层更适合做 expert-aware 压缩。MoE routing 图显示中层 top1-top2 margin 和 top-k boundary margin 更高，routing 更有信心；后层 entropy 回升、margin 下降，更像全局整合层，应更保守。
7. 多跳任务的 task-specific 信号主要在 per-layer expert overlap 和 token trajectory route signatures，而不是全局 top experts。

## 3. 关键图像解读

### 3.1 `OmniVideoBench_progress_accuracy_prefill.png`

![progress](OmniVideoBench_progress_accuracy_prefill.png)

这张图上半部分是累计准确率，下半部分是样本序列上的 prefill latency。

关键观察：

- 累计准确率最终稳定在 40% 左右，波动后没有明显上升。
- prefill latency 呈现明显的分段结构，中间大段接近 28s，后半段又出现短视频低延迟样本。
- 这个红色 rolling mean 主要反映数据排序中的视频长度分布，而不是模型运行状态自然变快/变慢。

解释：

中间大段样本基本打到 768 frames / 55k video tokens 的上限，所以 prefill 卡在约 28s。准确率没有随高成本样本提升，说明多跳 benchmark 的核心不是“看更多 token 就更好”，而是是否选到关键片段、是否能跨片段推理。

研究启发：

- 后续做 pruning / expert budget 对比时必须随机化样本顺序，或者 paired A/B 交替运行。
- 只报告整体平均 latency 很容易被样本排序和长视频 cap 支配。

### 3.2 `OmniVideoBench_task_latency_accuracy.png`

![task latency accuracy](OmniVideoBench_task_latency_accuracy.png)

左图是各 task 的平均 prefill latency，右图是 accuracy。

关键观察：

- `temporal understanding` 样本最多，准确率只有 27.9%，平均 prefill 23.3s。
- `spatial understanding` 准确率最低之一，25.7%。
- `reference reasoning` 32.4%，`counting` 35.0%，也比较弱。
- `relationship reasoning` 77.8% 最高，但只有 9 条样本，统计不稳定。
- `hypothetical reasoning` 61.5%，也只有 13 条样本。

解释：

真正需要跨时间、空间定位、引用消解的任务准确率偏低。这很符合多跳 benchmark 的定位：瓶颈不是单帧识别，而是跨片段证据收集和推理链保持。

研究启发：

- `temporal / spatial / reference / counting` 应作为核心 stress tasks。
- 对这些任务做 aggressive video pruning 风险最高，但也最有研究价值，因为它们最需要“关键片段选择 + 推理 expert 子图”。

### 3.3 `OmniVideoBench_metadata_field_breakdown.png`

![metadata breakdown](OmniVideoBench_metadata_field_breakdown.png)

这张图把 question type、audio type、video type 的样本数、延迟、准确率放在一起。

关键观察：

- question type 分布比较长尾，`temporal understanding` 和 `counting` 样本最多。
- audio type 以 Speech 为主，Sound 和 Music 少很多。
- video type 以 Arts&TV Show、Ego、Vlog、Cartoon 为主。
- Sports 准确率很低，约 20.8%。
- Documentary 平均 prefill 很高，但只有 8 条样本。

解释：

Sports 这类快速动作场景可能对采样策略特别敏感，关键事件短、运动快、视觉变化大，均匀采样或低保真 token 压缩容易错过证据。Speech 类样本多且更长，所以 latency 更高。

研究启发：

- 不同 video type 应该单独看 route signatures。Sports / Ego / Vlog 的 expert pattern 可能不同。
- 多跳加速策略不能只按 question type 分，还应看 video type 和 audio type。

### 3.4 `OmniVideoBench_prefill_scaling_scatter.png`

![prefill scaling](OmniVideoBench_prefill_scaling_scatter.png)

左图是 input length vs prefill latency，右图是 audio tokens vs prefill latency。

关键观察：

- input length 与 prefill 几乎线性相关，相关性约 0.991。
- video tokens 与 prefill 相关性约 0.991。
- audio tokens 大多数是 390，几乎形成一条竖线，与 prefill 的关系很弱。

解释：

OmniVideoBench 的 latency 几乎完全被视频长度决定。音频 token 自身不是成本瓶颈，但音频可能仍然是选择关键视频片段的重要控制信号。

研究启发：

- 单独剪 audio token 收益很小。
- audio/text 更适合作为 router controller，用来决定哪些 video segment 进入更高预算处理。

### 3.5 `OmniVideoBench_duration_scaling.png`

![duration scaling](OmniVideoBench_duration_scaling.png)

左图是 video duration vs prefill latency，右图是 video duration vs input length。

关键观察：

- 0-6 分钟内，duration 增加会带来 input length 和 prefill 增加。
- 超过约 6 分钟后，input length 基本饱和在 55k 左右。
- prefill 也饱和在约 28s。
- 很多 10-30 分钟视频和 6 分钟视频的输入长度差不多。

解释：

当前 pipeline 对长视频不是按完整时长线性增加 token，而是被 768 frame cap 限制。长视频的信息被压缩到固定帧预算里，多跳问题的关键片段是否被保留，高度依赖采样策略。

研究启发：

- 多跳场景不能只做 token pruning，还要做 segment retrieval / evidence selection。
- 对 6 分钟以上视频，应优先研究 temporal segment selection，而不是进一步压缩已经 cap 后的全部帧。

### 3.6 `OmniVideoBench_prefill_cost_distributions.png`

![prefill costs](OmniVideoBench_prefill_cost_distributions.png)

这张图展示 attention FLOPs、MoE FFN FLOPs、KV cache write 的分布。

关键观察：

- 有一个很尖的高成本峰。
- 高成本峰对应 max-frame 样本，attention 约 1.3 PFLOPs，MoE FFN 约 202 TFLOPs，KV write 约 5GB。
- attention 是主导项，平均占总估算 FLOPs 约 80.6%；MoE FFN 约 19.4%。

解释：

OmniVideoBench 上 attention 成本压倒 AVUT。MoE 不是最大 FLOPs 项，但 202T 级别仍然足够大，尤其在多卡 expert parallel 时还会引入通信和 load imbalance。

研究启发：

- 只做 expert pruning 不够，必须和 token/attention 压缩联动。
- 但 MoE expert insight 仍有价值，特别是用于控制哪些 token 值得进入昂贵的中后层计算。

### 3.7 `OmniVideoBench_moe_layer_routing_overhead.png`

![moe layer routing](OmniVideoBench_moe_layer_routing_overhead.png)

这张图是最关键的 MoE 图之一，包含 normalized entropy、hot/mean expert batch、empty experts、MoE block time、top1-top2 margin、top-k boundary margin。

关键观察：

- 路由 entropy 整体很高，但中层略低，最后几层又升高。
- hot/mean expert batch 在前中层更高，最高超过 10，说明这些层有更明显的 expert skew。
- empty experts 主要出现在前层，中后层几乎所有专家都会被用到。
- MoE block time 多在 93-99ms，层间差异不算极端。
- top1-top2 margin 在中层最高，后层明显下降。
- top-k boundary margin 也在中层较高，后层下降。

解释：

中层的 expert routing 更“有信心”，也更不均匀，说明中层更可能承载可利用的 specialist routing。后层虽然 entropy 更高、专家覆盖更广，但 margin 低，可能是在做全局融合和推理整合。

研究启发：

- 中层适合做 expert budget reduction、route-signature token merging、routing-aware token pruning。
- 后层适合保守处理，不建议简单降低 top-k。
- 前层可以看 empty experts 和 hot experts，做 expert cache/load balance。

### 3.8 `OmniVideoBench_expert_activation_heatmap.png`

![expert heatmap](OmniVideoBench_expert_activation_heatmap.png)

这张图是 layer x expert 的 aggregate activation frequency。

关键观察：

- 没有少数 expert 全局统治。
- top expert 最高频率约 1.07%，理论均匀 1/128 为 0.78%。
- 某些 expert 在某些层偏热，但不是全局可删的强模式。

解释：

Qwen3-Omni 的 MoE routing 在这个 benchmark 上相对分散。全局 expert pruning 很可能损伤质量，因为很多 expert 都有使用。

研究启发：

- 不要做 global static expert deletion。
- 应该做 layer-wise、task-wise、modality-wise、token-wise 的动态 expert budget。

### 3.9 `OmniVideoBench_modality_expert_distribution.png`

![modality expert distribution](OmniVideoBench_modality_expert_distribution.png)

这张图比较 text/video/audio 在 top 32 aggregate experts 上的分布。

关键观察：

- video 分布很平滑。
- text 和 audio 更尖锐，有更明显的 expert 偏好。
- text 对 expert 105、109 等有明显偏好，audio 也有自己的偏好。

解释：

video token 数量巨大，会把分布平均化；text/audio token 少，但更携带任务意图和语义控制信号。多跳任务的 question type 很可能通过 text token 调制后续 video token 的 expert routing。

研究启发：

- text/audio 应该作为 expert controller，而不是主要 pruning 对象。
- 可以研究 text/audio routing signature 是否能预测 task-specific expert budget。

### 3.10 `OmniVideoBench_task_expert_overlap.png`

![task expert overlap](OmniVideoBench_task_expert_overlap.png)

这张图是任务之间 top-32 expert 的 Jaccard overlap。

关键观察：

- 比 AVUT 更有任务分化。
- `temporal understanding` 与其他 task 的 overlap 偏低。
- `ego reasoning` 与部分任务 overlap 也偏低。
- `fine-grained perception / spatial / attribute / sentiment / reference` 等任务之间 overlap 更高。

解释：

OmniVideoBench 的多跳任务确实诱导出不同 expert 子图，但不是每个任务都完全独立。某些任务共享 perception/reasoning expert，某些任务需要独特路径。

研究启发：

- `temporal understanding` 是最适合挖 task-specific expert 的任务。
- 可以按 task cluster 做 expert budget，而不是每个 task 单独建白名单。

### 3.11 `OmniVideoBench_task_expert_overlap_per_layer.png`

![task expert overlap per layer](OmniVideoBench_task_expert_overlap_per_layer.png)

这张图是最有研究价值的图之一：每个 task pair 在每一层的 top-32 expert overlap。

关键观察：

- 某些层出现亮竖条，例如 23、39、41、44 附近，说明大量 task pair 在这些层共享 expert。
- 其他层颜色差异更大，说明 task-specific 分化更强。
- fine-grained/spatial 等任务 pair 常年 overlap 较高。

解释：

模型可能存在两类层：

1. 共享整合层：不同任务都走相似 expert，用于通用融合/推理。
2. 任务分化层：不同任务 expert overlap 低，适合做 task-specific budget 或 routing analysis。

研究启发：

- 不应该对所有层使用同一种 MoE 压缩策略。
- 共享层保守，差异层做 task-specific expert budget。
- 这张图可以直接指导“哪些层做 MoE 加速，哪些层保留原始路由”。

### 3.12 `OmniVideoBench_token_trajectory_route_signatures.png`

![token trajectory](OmniVideoBench_token_trajectory_route_signatures.png)

这张图展示 video/audio token 在不同 time segment / token bin 上的 route signatures 和 route margin。

关键观察：

- unique route signatures 随层数增加而增加，后层 route diversity 更高。
- video route margin 在中层较高，后层下降。
- audio route signatures 比 video 更丰富，margin 也更高一些。
- 时间段之间有差异，但不如层间差异强。

解释：

中层路由既有较高置信度，又保留较强结构性，适合用 route signature 做 token merge/pruning。后层 route diversity 高但 margin 低，可能是多跳整合阶段，不能简单把多样性看成冗余。

研究启发：

- 可以定义 `route_signature_similarity`，合并同一时间段或相邻时间段中 route signature 高度相似的视频 token。
- audio route margin 更高，可以作为 segment-level confidence 或 controller。
- 对多跳任务，route signature 不应只看相邻帧，还要看远距离片段是否激活相同 reasoning expert。

## 4. 与 AVUT 的关键差异

| 维度 | AVUT | OmniVideoBench |
|---|---:|---:|
| 平均输入长度 | 9.6k | 43.8k |
| video token share | 93.8% | 98.0% |
| audio token share | 4.6% | 1.4% |
| prefill mean | 5.47s | 21.64s |
| attention share | 57.7% | 80.6% |
| MoE share | 42.3% | 19.4% |
| task expert overlap | 普遍很高 | 更有分化 |
| 最适合研究点 | audio-guided video pruning baseline | task/layer/trajectory expert insight |

结论：AVUT 更适合验证 audio-video local alignment；OmniVideoBench 更适合研究多跳长视频里的 MoE expert specialization。

## 5. 加速收益粗估

基于平均 token 构成和 FLOPs 拟合，仅做方向判断。

平均样本：

```text
text 131.3 + audio 383.4 + video 43,273.2 = seq 43,788.0
```

如果只剪 video token：

| video keep | seq approx | seq reduction | attention reduction | MoE/KV reduction | fitted latency reduction |
|---:|---:|---:|---:|---:|---:|
| 75% | 32,970 | 24.7% | 41.9% | 24.7% | 26.1% |
| 50% | 22,151 | 49.4% | 72.5% | 49.4% | 52.3% |
| 25% | 11,333 | 74.1% | 91.9% | 74.1% | 78.4% |
| 10% | 4,842 | 88.9% | 98.0% | 88.9% | 94.1% |

如果把 frame cap 从 768 降到固定帧数，假设每帧约 72 video token：

| cap frames | seq approx | attention reduction | MoE/KV reduction | fitted latency |
|---:|---:|---:|---:|---:|
| 512 | 37,379 | 26.2% | 14.6% | 18.3s |
| 384 | 28,163 | 56.9% | 35.7% | 13.5s |
| 256 | 18,947 | 79.4% | 56.7% | 8.7s |
| 128 | 9,731 | 93.8% | 77.8% | 3.8s |
| 64 | 5,123 | 97.9% | 88.3% | 1.4s |

风险：多跳任务不适合盲目降 frame cap。必须配合 segment retrieval / expert-aware token selection，否则 temporal/spatial/reference 这类任务会继续掉。

## 6. 面向 MoE expert insight 的研究方向

### 6.1 Layer-aware expert budget

根据图像信号：

- 中层：routing margin 高，hot/mean 高，适合减少 top-k 或专家预算。
- 后层：entropy 高、margin 低，适合保守。
- 共享 overlap 高的层：保守或共享压缩。
- task overlap 低的层：做 task-specific expert budget。

### 6.2 Task cluster，而不是 task whitelist

可以先把 task 分成几类：

- temporal/reference/spatial：长程定位和证据链。
- fine-grained/attribute/sentiment：细粒度感知和属性判断。
- counting/causal/hypothetical：推理和关系判断。
- background/music：音频相关，但样本少。

不同 cluster 使用不同 token budget 和 expert budget。

### 6.3 Route-signature token merging

对 video token / time segment 计算 route signature：

```text
signature(layer, segment) = top-k expert ids + routing weights summary
```

然后：

- 相邻 segment signature 相似：可 merge 或降低 token budget。
- 远距离 segment signature 相似：可能代表同类事件或重复场景，可做 sparse retention。
- margin 高的中层：更可信地做 merge。
- margin 低的后层：不直接 merge，作为质量保护层。

### 6.4 Audio/text as expert controller

audio token 成本很小，不应优先剪。更适合：

- 预测哪些 temporal segments 需要保留。
- 预测 task cluster。
- 预测中层 expert budget。
- 调整 video token 的 top-k expert 数。

### 6.5 多模型比较

下一步不应只看 Qwen3-Omni。建议横向比较：

- Qwen3-Omni：Thinker-Talker MoE，已有强 baseline。
- Uni-MoE-2.0-Omni：shared/routed/null experts，适合研究 null expert 与 discardable token。
- Ming-Flash-Omni / Ming-Omni：大规模 sparse MoE，适合看 scaling 后 expert specialization 是否更强。
- DeepSeek-VL2 / InternVL3.5-MoE：非全模态但适合隔离视觉 MoE 专家规律。

## 7. 下一轮 profile 建议

1. 保存完整 `profile_samples.jsonl` 和一批 full route dump。
2. 对每个样本记录是否 capped、采样帧的时间戳、关键帧命中情况。
3. 记录 per-layer/per-task/per-modality top experts，而不是只存全局 top experts。
4. 记录 top1/top2 margin、top-k boundary margin 的 token-level 分布。
5. 对 OmniZip 前后分别画同样的 expert overlap、per-layer overlap、route trajectory 图。
6. 增加 task cluster 级别的 quality breakdown。
7. 对 768 / 512 / 384 / 256 frame cap 做 paired A/B，并配合 evidence-aware sampling。

一句话总结：OmniVideoBench 不是 AVUT 那种局部音视频对齐 benchmark，它把 Qwen3-Omni 推到 55k token / 28s prefill 的长视频多跳 regime。这里最有价值的不是复现 audio-guided temporal pruning，而是利用 per-layer task expert overlap 和 route trajectory signatures，研究哪些层、哪些任务、哪些时间段真的需要完整 MoE 计算。

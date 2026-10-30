# P2：紧凑 Contribution Buffer 32 样例复测

## 结论

P2 已完成实现、严格数值身份验证和原 32 样例四臂复测。

P2 对**未经过 OmniZip 压缩的长序列**有稳定收益：

- Native 相对 P0 平均快 **72.34 ms**，配对 95% CI `[-93.17, -51.90] ms`。
- OmniDial 相对 P0 平均快 **54.85 ms**，配对 95% CI `[-72.43, -37.45] ms`。

但对已经经过 OmniZip 压缩的短序列没有稳定收益：

- OmniZip 相对 P0 平均慢 4.25 ms，95% CI `[-16.67, +37.09] ms`。
- Stack 相对 P0 平均慢 5.55 ms，95% CI `[-7.81, +19.69] ms`。

因此：

- P2 适合用作**未压缩长序列的优化路径**。
- 对当前核心 Stack 路径没有可确认收益，不应仅为 Stack 替换 P0。
- 若需要单一统一实现，P2 数值正确且对 Native/OmniDial 有收益，但它会使 OmniDial 相对 Native 的表观增量收益减少，因为 Native 从 P2 获益更大。

## 实现内容

P2 从性能更好的 P0 路径出发，保持以下部分不变：

- Qwen router 和 top-k 选择；
- stable expert-major 排序；
- Megatron-Core GroupedMLP；
- expert probability 在 down projection 后应用；
- ascending expert-ID BF16 累加顺序；
- telemetry-off。

只替换结果恢复路径：

### P0

分配完整的：

`[tokens, topk, hidden]` BF16 contribution buffer

再按 rank 取出并逐步相加。

### P2

改为：

- `[tokens, topk]` int64 location table；
- 每次聚合只 gather 一个 `[tokens, hidden]` 临时 contribution；
- 保持原 expert-ID 加法顺序。

未引入 P1 的 `[tokens, 128 experts]` dense routing map，也未使用 unfused MCore dispatcher。

## 数值身份

P2 与 P0 的 32 样例 × 4 臂逐项比较：

- 输出文本 mismatch：0。
- `max_abs_option_logp_delta=0`。
- 准确率、gold log-prob、JS 和答案翻转完全一致。

因此 P2 没有改变模型语义。

## 四臂时延结果

| Arm | P0 | P2 | P2 − P0 | 配对 95% CI |
|---|---:|---:|---:|---:|
| Native | 11.9694 s | 11.8970 s | **-72.34 ms** | `[-93.17, -51.90] ms` |
| OmniZip | 4.2217 s | 4.2260 s | +4.25 ms | `[-16.67, +37.09] ms` |
| OmniDial | 11.7697 s | 11.7149 s | **-54.85 ms** | `[-72.43, -37.45] ms` |
| Stack | 4.1598 s | 4.1653 s | +5.55 ms | `[-7.81, +19.69] ms` |

负值表示 P2 更快。

与 P1 相比，P2 四个臂全部更快：

- Native：-150.71 ms。
- OmniZip：-47.53 ms。
- OmniDial：-132.04 ms。
- Stack：-51.67 ms。

这进一步说明 P1 的 unfused dispatcher 基础开销不适合当前执行环境。

## 长度依赖

P2 的时延收益与未压缩输入长度高度相关：

- Native：输入长度与节省时延的相关系数约 **0.90**。
- OmniDial：相关系数约 **0.89**。

按 Native 输入长度四分位：

| 平均输入 token | Native 平均节省 |
|---:|---:|
| 6,342 | -0.17 ms |
| 31,186 | 44.04 ms |
| 48,504 | 101.81 ms |
| 55,769 | 143.67 ms |

对于 OmniDial，最长两个四分位平均分别节省约 94 ms 和 105 ms。

OmniZip/Stack 已将保留 token 降到约 40%，P2 的收益被显著压缩，且测量波动占主导。

## 显存结果

Qwen thinker hidden size为 2048，BF16、top-k=8。因此 P0 的全宽 contribution buffer 理论大小是：

`tokens × 8 × 2048 × 2 bytes`

对应：

- Native 平均 35,450 token：单层临时 buffer 理论约 **1.08 GiB**。
- OmniZip 平均保留 14,216 token：单层理论约 **0.43 GiB**。

P2 删除了该全宽 buffer，但进程级 `torch.cuda.max_memory_allocated()` 相对 P0 基本不变：

- Native/OmniDial：差约 `-0.00007 GiB`。
- OmniZip/Stack：差约 0。

原因是全局峰值由模型权重、attention 或其他更高峰值阶段主导；P2 降低了局部 MoE 临时内存，但没有改变进程全局峰值。若要精确验证局部显存收益，需要在单层 MoE 前后采样 memory snapshot，而不是仅看整个 generate 的最大值。

## 对 OmniDial 增量收益的影响

### OmniDial 相对 Native

- P0 平均节省：199.65 ms。
- P2 平均节省：182.16 ms。
- P2 比 P0 少节省：17.49 ms，95% CI `[-27.85, -8.49] ms`。

这是因为 Native 的全 Top-8 contribution buffer 更大，P2 对 Native 的收益高于对 OmniDial 的收益；并不意味着 OmniDial 本身变慢。

### Stack 相对 OmniZip

- P0 平均节省：61.96 ms。
- P2 平均节省：60.67 ms。
- 差异：-1.30 ms，95% CI `[-26.62, +33.58] ms`。

没有显著变化。

## 工程决策

推荐：

1. **保留 P0 作为 Stack 默认基线。** P2 对 Stack 没有稳定收益。
2. **对未启用 OmniZip 的长序列启用 P2。** 特别是输入超过约 30K token 后，收益开始明显。
3. 如果希望维护单一路径，可合入 P2，因为它严格数值等价，且未压缩长序列更快；但论文或报告中的 OmniDial 相对加速比例应重新按 P2 基线计算。
4. 若要进一步优化 Stack，应聚焦压缩后的小规模 dispatch/kernel，而不是 contribution buffer。
5. 如需验证显存，新增 layer-local allocator snapshot 或 CUDA memory history，避免全局峰值遮蔽局部变化。

## 远端产物

实验目录：

`/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner/experiments/omnizip_omnidial_stack_20261015/full32_p2_v1`

关键文件：

- `raw_results.jsonl`
- `summary.json`
- `p2_comparison.json`
- `protocol.json`
- `environment.json`
- `inference_complete.json`
- `../source/run_stack_p2.py`
- `../source/qwen_mcore_controller_p2.py`
- `../source/qwen_mcore_experts_p2.py`
- `../source/analyze_p2.py`

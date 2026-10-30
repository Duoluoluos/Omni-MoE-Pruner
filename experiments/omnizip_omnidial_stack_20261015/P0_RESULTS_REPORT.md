# P0：关闭 Expert-Row 遥测后的 32 样例复测

## 结论

P0 已按要求在**原 32 个样例、相同四臂、相同门限、相同臂顺序**上完成。

实施内容：

- 删除每个 MCore expert forward 内的 `valid.sum().item()`。
- 删除 `executed_rows` / `calls` 的逐层诊断累积和每臂 reset/readback。
- 计时结果中的 `expert_rows` 引用上一轮 `full32_v1` 同样例同臂的机制计数，并标记来源，不在本轮计时前向中统计。
- 保留 `torch.bincount(experts, minlength=num_experts).to('cpu')`：这是 MCore GroupedMLP grouped-GEMM 所需的 `tokens_per_expert` 元数据，不是可选遥测；移除它属于后续 dispatcher 改造，不属于 P0。
- 保留每臂计时边界的 CUDA synchronize，用于获得正确端到端时延。

结果表明：**额外 `.item()` 遥测同步不是当前 OmniDial 时延收益偏小的主要原因。**

## 输出一致性

32 样例 × 4 臂全部完成：

- 输出文本 mismatch：**0**。
- 四个 option log-prob 最大绝对差：**0**（`max_abs_option_logp_delta=0`）。
- 准确率、JS、答案翻转与上一轮完全一致。

因此 P0 只改变了 telemetry 行为，没有改变模型语义。

## 四臂平均时延

| Arm | 原版 | P0 | 均值差 |
|---|---:|---:|---:|
| Native | 11.9805 s | 11.9694 s | -11.1 ms |
| OmniZip | 4.2377 s | 4.2217 s | -16.0 ms |
| OmniDial | 11.7841 s | 11.7697 s | -14.4 ms |
| Stack | 4.1737 s | 4.1598 s | -13.9 ms |

这些臂都略有加速，但幅度接近，未选择性地释放出 OmniDial 的大量额外收益。

注意：由于视频长度差异大，“平均绝对时延差”和“逐样本百分比差”的权重不同。各臂逐样本百分比变化的 95% CI 均跨 0；不能把 11–16 ms 解释成稳定的业务级提升。

## OmniDial 增量收益

### 无 OmniZip 条件：OmniDial vs Native

- 原版平均节省：**196.35 ms**。
- P0 平均节省：**199.65 ms**。
- P0 额外释放：**+3.30 ms**。
- 额外释放的 95% CI：`[-9.61, +16.48] ms`，跨 0。
- 逐样本平均节省比例：**1.87% → 1.91%**。

### 有 OmniZip 条件：Stack vs OmniZip

- 原版平均节省：**64.00 ms**。
- P0 平均节省：**61.96 ms**。
- P0 变化：**-2.04 ms**。
- 变化的 95% CI：`[-13.58, +10.11] ms`，跨 0。
- 逐样本平均节省比例：**1.55% → 1.58%**（比率与绝对均值因样本加权方式不同而略有差异）。

## 工程判断

P0 的假设是：逐层 `valid.sum().item()` GPU→CPU 同步可能把 expert-row 减少的收益抵消。

实验结果不支持它是主瓶颈：

1. P0 对四个臂产生了相近的十几毫秒变化，而不是主要改善 OmniDial/Stack。
2. OmniDial 的额外收益只增加约 3 ms，置信区间跨 0。
3. Stack 相对 OmniZip 的绝对增量收益没有增加。
4. OmniDial 的 12.1% expert-row 减少依旧只转化为约 1.9% 的端到端节省。

这意味着下一主要瓶颈仍在实际 dispatch 路径：

- `bincount → CPU tokens_per_expert` 的 grouped-GEMM 元数据路径；
- stable argsort、gather、全宽 contribution buffer、unpermute/reduction；
- Top-4 条件下仍按宽度 8 构造部分中间结构；
- 音视频编码、attention、router/top-k 等不随 expert rows 下降的固定成本。

下一步若继续工程优化，应进入 P1：原生 MCore token dispatcher / fused permutation-unpermutation，而不是继续清理 telemetry。

## 远端产物

实验目录：

`/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner/experiments/omnizip_omnidial_stack_20261015/full32_p0_v1`

关键文件：

- `raw_results.jsonl`：32 样例四臂输出。
- `summary.json`：P0 四臂 bootstrap 汇总。
- `p0_comparison.json`：与原版 `full32_v1` 的配对比较。
- `protocol.json`：冻结协议。
- `environment.json`：运行环境和 telemetry 说明。
- `inference_complete.json`：完成标记。
- `../source/run_stack_p0.py`
- `../source/qwen_mcore_controller_p0.py`
- `../source/qwen_mcore_experts_p0.py`
- `../source/analyze_p0.py`

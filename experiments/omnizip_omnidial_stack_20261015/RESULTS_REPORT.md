# OmniZip × OmniDial on Qwen3-Omni：叠加效应验证报告

## 结论先行

在**同一 custom-Qwen + Megatron-Core GroupedMLP backend**内，OmniZip 与 OmniDial 的效率收益可以叠加，而且在实际 expert rows 上呈现接近乘法、略优于乘法的结果：

- OmniZip 单独将平均保留 token 降至 **40.76%**，将实际 expert rows 降至 **40.77%**。
- OmniDial `[12,32)` 单独将实际 expert rows 降至 **87.92%**。
- 两者叠加后，实际 expert rows 降至 **35.26%**；按两个单项逐样本比率相乘的期望为 **35.83%**，叠加比预期再低 **0.57 个百分点**（95% CI `[-0.64, -0.52]` 个百分点）。
- 平均端到端时延由 **11.98 s** 降至 **4.17 s**，即保留 **34.84%**、约 **2.87× 加速**。逐样本乘法残差均值为 **+0.12 个百分点**，95% CI `[-0.38, +0.62]`，与理想乘法叠加一致。
- 平均峰值 allocated GPU memory 从 **73.62 GiB** 降至 **65.62 GiB**，下降约 **10.9%**；几乎全部来自 OmniZip，OmniDial 不降低驻留权重。

质量侧没有足够证据支持“无损叠加”，但也不能据此宣称显著退化：

- 32 个分层视频子集上，Native / OmniZip / OmniDial / Stack 分别答对 **9 / 11 / 9 / 8** 题，即 **28.1% / 34.4% / 28.1% / 25.0%**。
- 准确率差分中的差分为 **-9.38 个百分点**，95% CI `[-21.88, +3.13]`，区间跨 0。
- 金标 log-prob 的交互项为 **-0.219**，95% CI `[-0.654, +0.188]`，同样跨 0。
- Stack 相对 Native 的答案翻转率为 **40.6%**，JS divergence 均值为 **0.082**，说明组合臂确实会显著改变输出分布；后续需要扩大到完整 1,000 题后才能对质量下最终结论。

**当前判断：效率叠加已得到机制和系统层验证；质量叠加尚未通过充分统计验证。**

## 理论解释

OmniZip 和 OmniDial 作用于两个近似正交的计算轴：

1. **OmniZip：序列轴**
   - 在 thinker transformer 之前压缩音频/视频 token。
   - 降低后续 attention、KV cache 和所有 MoE 层处理的 token 数。
   - 对 routed MLP 工作量的主效应近似为保留 token 比率 `r_L`。

2. **OmniDial：专家轴**
   - 保留 token 序列，只在 Qwen thinker 的 MoE 层 12–31 中按冻结置信度门限选择 top-4 或 top-8。
   - 对 routed MLP 工作量的主效应近似为条件 expert-row 比率 `r_E`。

因此重叠区的 routed expert work 预计为：

`W_stack / W_native ≈ r_L × r_E`

但 wall-clock latency 不会严格等于该乘积，因为音视频编码器、router、输出头、kernel launch、decode 和部分 attention 开销是固定或不受 OmniDial 影响的。Megatron-Core GroupedMLP 的作用，是将四个臂的 expert dispatch/GEMM 统一到同一高效后端，使真实 expert-row 减少更有机会转化为时延收益。

## 实验协议

### 2×2 因子设计

| 实验臂 | OmniZip | OmniDial |
|---|---:|---:|
| Native | 关 | 关 |
| OmniZip | 开 | 关 |
| OmniDial | 关 | 开 |
| Stack | 开 | 开 |

固定设置：

- 模型：Qwen3-Omni-30B-A3B-Instruct。
- 精度与生成：BF16、batch size 1、greedy、thinker 最多生成 4 token。
- OmniZip：`rho_audio=0.3`、`rho_video=0.6`、`g=3`、`contextual_ratio=0.05`。
- OmniDial：CGCC top-4/top-8，层段 `[12,32)`，冻结门限 `tau_reduce=0.08694529702904027`；不在 OmniZip 输出上重新调参。
- 样本：OmniVideoBench 的 32 个预选视频，按历史输入长度四分位分层，每层 8 个。
- 每个样本内确定性随机化四臂顺序，四臂共享一次模型加载和一次 MCore expert 转换。
- Backend：Qwen 多模态编码器、attention、KV cache 保持原路径；48 层 thinker MoE experts 全部转换为 Megatron-Core 0.12.3 `GroupedMLP`。

### 指标与统计

- 质量：准确率、金标 option log-prob、相对 Native 的 full-vocabulary JS、答案翻转率。
- 效率：CUDA 同步的端到端 `generate` 时延、保留 token、实际 MCore expert rows、峰值 allocated memory。
- 统计：以问题为配对单位，20,000 次 bootstrap。
- 加性交互：`Y_stack - Y_omnizip - Y_omnidial + Y_native`。
- 乘法残差：`Y_stack/Y_native - (Y_omnizip/Y_native)(Y_omnidial/Y_native)`。

## 正式结果（n=32）

| 指标 | Native | OmniZip | OmniDial | Stack |
|---|---:|---:|---:|---:|
| Accuracy | 28.1% | 34.4% | 28.1% | 25.0% |
| Gold log-prob | -2.348 | -2.257 | -2.309 | -2.437 |
| Latency | 11.980 s | 4.238 s | 11.784 s | 4.174 s |
| Token retention | 100.0% | 40.76% | 100.0% | 40.76% |
| Expert-row ratio | 100.0% | 40.77% | 87.92% | 35.26% |
| Peak allocated memory | 73.62 GiB | 65.62 GiB | 73.62 GiB | 65.62 GiB |
| JS to Native | 0 | 0.020 | 0.046 | 0.082 |
| Flip vs Native | 0 | 15.6% | 25.0% | 40.6% |

### 效率交互

- **Expert rows**：逐样本乘法期望 35.83%，实际 Stack 35.26%；乘法残差 `-0.57pp`，95% CI `[-0.64, -0.52]pp`。组合后 OmniDial 的门控在压缩序列上略微更积极，形成小幅超乘法算量节省。
- **Latency**：逐样本乘法期望 42.72%，实际 Stack 42.84%；乘法残差 `+0.12pp`，95% CI `[-0.38, +0.62]pp`。系统时延与乘法模型一致，没有观察到明显 infra 干扰。
- **Memory**：Stack 与 OmniZip 基本相同，说明显存收益来自 token/activation 缩减，而非 expert 权重或路由预算。

注意：比率“均值”采用逐样本先求比再平均，因此与“全体均值之比”略有不同；独立重算得到均值之比为 Native→Stack latency **34.84%**、expert rows **34.81%**，方向与结论一致。

### 质量交互

准确率交互项 `-9.38pp` 和 gold log-prob 交互项 `-0.219` 都是负向，但置信区间跨 0。结合 Stack 的 40.6% 答案翻转率，可以判断组合会改变模型决策；但 32 题无法区分这是偶然波动、门限迁移，还是稳定的质量冲突。

建议把效率结论视为已验证，把质量结论标记为“待 full-1000”：

- 不修改 OmniZip 配置和 OmniDial 门限，原样扩到 1,000 题，保持配对四臂。
- 主要质量门：Stack 相对 Native 的准确率非劣界，以及交互项的 cluster bootstrap CI。
- 若 full-1000 确认负交互，再做输入无标签的轻量 gate correction；在此之前不要用答案调门限。

## 验证与限制

已验证：

- 32 个唯一样本全部完成，四臂齐全；无 failure marker。
- 每个样本 OmniZip 都减少 token；每个样本 OmniDial 都减少 expert rows；每个样本 Stack 的 expert rows 都低于 OmniZip。
- 分析器结果已用独立脚本重算 latency、token、expert-row 均值与交互项。
- 远端 GPU 在结束后已释放。

重要限制：

1. 这是 **custom-Qwen + MCore backend 内部配对实验**，不是标准 HF backend 的绝对质量复现。
2. 关闭 OmniZip 的 custom 模型与历史标准 HF 模型在两个 identity 样本上生成答案一致，但 option log-prob 最大差约 `0.24–0.26`，未通过 `1e-5` identity 门。因此不得把本次数值与历史 HF 主表直接拼接。
3. 历史 MCore conversion 本身也可能产生数值漂移；本实验通过四臂共享 backend 控制这一偏差，但不能据此证明 backend 等价。
4. 质量样本量只有 32，当前置信区间较宽；效率/机制结论远强于质量结论。
5. 当前 Megatron 利用范围是 thinker MoE expert backend，不是完整多模态生成栈的 Megatron 迁移。

## 远端位置

- 实验根目录：`/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner/experiments/omnizip_omnidial_stack_20261015`
- 正式结果：`full32_v1/`
- 协议：`full32_v1/protocol.json`
- 环境：`full32_v1/environment.json`
- 原始逐题结果：`full32_v1/raw_results.jsonl`
- Bootstrap 汇总：`full32_v1/summary.json`
- identity 门：`identity_gate.json`
- 可复现代码：`source/`

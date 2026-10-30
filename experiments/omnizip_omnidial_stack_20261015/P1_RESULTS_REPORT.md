# P1：MCore 原生 Token Dispatcher 32 样例复测

## 结论

P1 已完成实现、数值修复和原 32 样例四臂复测，但**性能验收不通过，不建议合入默认执行路径**。

在当前远端环境中：

- Megatron-Core 版本：0.12.3。
- Transformer Engine 不可用，`HAVE_TE=False`。
- 因而只能运行 MCore 原生 **unfused** permutation；无法启用 `moe_permute_fusion`。

P1 最终实现：

1. 使用 `MoEAllGatherTokenDispatcher`，单卡 `TP=1, EP=1`。
2. 使用 MCore 原生 `token_permutation`，取代旧版手写 stable argsort 和 token gather。
3. GroupedMLP 只处理紧凑 token–expert rows。
4. 删除旧版 `[tokens, topk, hidden]` 全宽 contribution buffer。
5. Transformer Engine 缺失，未使用 fused permute/unpermute。
6. MCore 原生 `scatter_add_` 会改变 BF16 多专家累加顺序，曾导致 Native 答案变化；最终 P1-exact 改为紧凑 `[tokens, topk]` 整数反向表，并按 expert ID 升序做确定性聚合，以恢复历史数值身份。

最终结果：输出身份完全保持，但 P1 相比 P0 的四个臂均稳定变慢约 1.7%–2.6%。

## 数值身份

P1 与 P0 的 32 样例 × 4 臂逐项比较：

- 输出文本 mismatch：0。
- `max_abs_option_logp_delta=0`。
- 准确率、gold log-prob、答案翻转和 JS 指标一致。

因此最终 P1-exact 没有改变模型语义。

## 平均时延

| Arm | P0 | P1 | P1 − P0 | 逐样本百分比变化 95% CI |
|---|---:|---:|---:|---:|
| Native | 11.9694 s | 12.0477 s | +78.4 ms | `[+0.83%, +3.37%]` |
| OmniZip | 4.2217 s | 4.2735 s | +51.8 ms | `[+1.11%, +2.39%]` |
| OmniDial | 11.7697 s | 11.8469 s | +77.2 ms | `[+0.91%, +5.32%]` |
| Stack | 4.1598 s | 4.2170 s | +57.2 ms | `[+1.28%, +2.56%]` |

四个臂的 paired CI 均完全大于 0，说明该性能回退不是随机噪声。

与最初版本相比，P1 也全部变慢：

- Native：+67.3 ms。
- OmniZip：+35.8 ms。
- OmniDial：+62.8 ms。
- Stack：+43.3 ms。

## OmniDial 的增量收益

### OmniDial 相对 Native

| 版本 | 平均节省 |
|---|---:|
| 原始版本 | 196.35 ms |
| P0 | 199.65 ms |
| P1 | 200.82 ms |

P1 相对 P0 只多释放 **1.18 ms**，95% CI `[-33.81, +40.31] ms`，没有显著改善。

### Stack 相对 OmniZip

| 版本 | 平均节省 |
|---|---:|
| 原始版本 | 64.00 ms |
| P0 | 61.96 ms |
| P1 | 56.54 ms |

P1 相对 P0 少节省 **5.43 ms**，95% CI `[-23.02, +12.68] ms`，没有显著改善。

## 为什么 P1 变慢

当前环境缺少 Transformer Engine，因此 MCore dispatcher 走 unfused 路径：

- 每层构造 `[tokens, 128 experts]` routing map 和 dense probability map。
- MCore `permute` 需要转置、masked-select 和 index-select。
- 为通过严格数值身份门，不能使用原生 BF16 `scatter_add_`；需要构造紧凑 `[tokens, topk]` 反向索引，并按历史 expert ID 顺序聚合。
- 这些额外开销超过了删除全宽 contribution buffer 和旧手写 gather 带来的收益。

此外，P1 的 Native/OmniZip 也变慢，说明成本主要来自 dispatcher 基础开销，而不是 OmniDial 策略本身。

## 工程决策

### 当前环境

- **保留 P0 为默认路径。**
- **不合入 P1-exact。** 它数值正确，但性能稳定回退。
- 初版纯原生 `scatter_add_` P1 虽在单样本上快约 1.6%–3.9%，但数值身份失败，Native 输出文本发生变化，不能用于严格比较或默认推理。

### 后续若继续 P1

真正值得继续的版本是 **P1-fused**，前置条件是安装与当前 CUDA/PyTorch/MCore 兼容的 Transformer Engine，使以下接口可用：

- fused permute with probabilities；
- fused unpermute；
- `moe_permute_fusion=True`。

安装后仍须重新通过两道门：

1. 数值门：32 样例四臂文本一致；option log-prob 漂移在预设容差内。
2. 性能门：至少不拖慢 Native，并显著扩大 OmniDial vs Native 或 Stack vs OmniZip 的配对时延收益。

在没有 TE 的当前机器上，继续优化 unfused dispatcher 的优先级低于：

- 使用更紧凑的 top-k→routing-map kernel；
- 避免 `[tokens,128]` dense routing/prob map；
- 专门实现 fused compact permutation + deterministic reduction；
- 或直接部署 TE fused dispatcher。

## 远端产物

实验目录：

`/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner/experiments/omnizip_omnidial_stack_20261015/full32_p1_v1`

关键文件：

- `raw_results.jsonl`
- `summary.json`
- `p1_comparison.json`
- `protocol.json`
- `environment.json`
- `inference_complete.json`
- `../source/run_stack_p1.py`
- `../source/qwen_mcore_controller_p1.py`
- `../source/qwen_mcore_experts_p1.py`
- `../source/analyze_p1.py`

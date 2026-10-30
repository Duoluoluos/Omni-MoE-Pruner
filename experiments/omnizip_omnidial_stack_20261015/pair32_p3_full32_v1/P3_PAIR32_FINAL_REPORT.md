# P3 固定32 配对验证最终报告（已更正 bootstrap CI）

> **更正声明（bootstrap CI 修订）**：旧版 `latency_summary.json` 的 CI 错误地把 B×n 个重采样**个体**比率/差值展平取分位，那是样本散布而非均值的 bootstrap CI。已修正为「每个 bootstrap replicate 先对 n 个配对样本求统计量均值，再对 B 个 replicate-mean 取 2.5/97.5 分位」。**原始推理 raw_results.jsonl / predictions 未改**（raw_sha256 仍 `f0c432754d929a18a856b7e5376c0bc5feb0faa53f2c98097e43aa9d11e9234a`），仅重算派生 summary。新 runner sha256 `9674bb8e6372bc272ec9b161f2de31a87c9d6c402e5589b29c6011d8755f32e5`。

## 1. 执行结论
- 后端：custom-Qwen + Megatron-Core GroupedMLP（vendored mcore **0.12.3**），HAVE_TE=False，无 fused/TE dispatcher。
- 冻结配置：`p3_calibration_prep/run_v2/gates.json`（sha256 `05ccd693…ad49c`），selected=`p3_balanced`，tau2=0.048523，tau4=0.070612，in-scope mean budget≈3.1，scope=[12,32]。
- Runner：`source/run_pair32_p3.py` sha256 **`9674bb8e…5f32e5`**（含更正后 bootstrap；本地/远端一致）。
- 两阶段干净退出：smoke2_v1 `--limit 2` 通过后，full32_v1 跑满 **n=32**，无 failure.json。

## 2. 配对测量（stack=冻结P3+OmniZip；omnizip=native top8+OmniZip）
- 每样例两臂，臂序固定 RNG seed=20261015+idx 随机、可审计（`arm_order` 记录）。
- **bootstrap（per-replicate sample mean；配对索引重采样；B=2000, seed=20261015；分位取在 replicate-mean 上，非 B×n 个体）**：

| 统计量 | 点估计 | 95% CI |
|---|---|---|
| ratio-of-means（mean stack / mean zip） | **0.9907** | **[0.9807, 1.0020]** |
| mean-of-ratios（mean_i s_i/z_i） | **1.0225** | **[0.9924, 1.0579]** |
| paired mean diff (stack−zip) | **−0.0402 s** | **[−0.0894, +0.0076]** |

- mean latency：stack 4.275 s vs omnizip 4.315 s。
- **结论**：ratio-of-means CI 上界 1.0020 几乎贴 1，paired diff CI 含 0 → 墙钟时延在统计上**无显著差异**（轻微倾向 stack 略快但不显著）。
- expert rows（runtime 计数器 `(weights>0).sum()`，source=`runtime_real_row_counter`，非复制 v1）：
  - mean-of-ratios **0.7410**，95% CI **[0.7371, 0.7447]**；ratio-of-means 0.7426 [0.7381,0.7466]。
  - 理论吻合：in-scope 20 层均值 3.1 + 越界 28 层 top8 → (28×8+20×3.1)/(48×8)=0.745。

## 3. 质量变化（冻结后仅用于质量评估）
- 准确率：omnizip **0.344** → stack **0.125**（下降）。
- mean gold logp：omnizip −2.257 → stack −2.054（均值略升，但 argmax 翻转）。
- gold_delta(stack−zip)：恶化 17、改善 15（worst −5.24 / best +6.99）。
- answer 一致：11/32。
- 解读：P3 在 in-scope 层把 top8 压到均值 3.1，省 ~26% 专家行但**未转化为墙钟收益**，argmax 准确率明显下降——真实测得的 tradeoff。

## 4. 可核验清单（全过）
- 32 样例索引精确等于固定集，未扩充/更换；每样例两臂各一条、arm_order 记录。
- omnizip=top8、stack=P3；两臂 retained_tokens 相同（OmniZip 均开）。
- latency / answer / A–D option_logp 全部有限，expert_rows 均为正；无失败样例。
- gates/raw 哈希正确；raw_sha256 经重算前后均为 `f0c43275…234a`。
- verifier-hub：JSON 格式 PASS、raw 必含字段 PASS、`"sample_index"` 计数=32、报告格式与关键字 PASS。

## 5. 产物路径
- 远端：`.../omnizip_omnidial_stack_20261015/pair32_p3_full32_v1/`
  - `latency_summary.json`（已更正 bootstrap）、`bootstrap_correction.json`、`raw_results.jsonl`（未改）、`environment.json`、`protocol.json`、`inference_complete.json`、`predictions/`。
- 本地：`omnizip_omnidial_stack/full32_v1_artifacts/`（同名派生文件，raw sha 已核对一致）。
- smoke：`.../pair32_p3_smoke2_v1/`（独立目录，未污染正式）。

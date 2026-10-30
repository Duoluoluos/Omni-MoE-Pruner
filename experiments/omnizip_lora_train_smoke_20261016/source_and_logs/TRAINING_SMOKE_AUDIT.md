# OmniDial 训练态接入 — 最小机制 Smoke 审计（TRAINING_SMOKE_AUDIT）

日期：2026-10-16（UTC+8）　节点：`ff31d166e82f5d6c.worker_0.job.wangqi-2248@merlin-ssh-proxy-cn.byted.org`（hostname `g340-cd60-...-ff5a`，2× NVIDIA H20）

## 0. 硬边界（务必先读）
- 本轮**只跑了合成张量机制 smoke**：vendored **MCore 0.12.3** 的 `GroupedMLP` + `grouped_gemm`，1-rank 两臂各 **1 个** forward+backward+optimizer.step，外加 EP=2 语义检查。
- **没有运行 Qwen3-Omni-30B 任何 step，没有运行 LoRA 任何 step**。没有短程 benchmark、没有稳态多 step 对照、没有长训。
- 本证据**不得解读为机时/吞吐收益**。它只证明：训练态 autograd 通路存在、稀疏 dispatch 真实发生、EP=2 专家分片语义成立。
- 未复用 P3 的 `model.eval/no_grad` 推理路径；未沿用 ARC-SCEB 命名/结论；custom-MCore P0 后端，**HAVE_TE=False**，未声称 fused。

## 1. 审计依据（源码只读）
- GroupedMLP：`$R/.tmp/qwen_runtime_mcore_20260924/python/megatron/core/transformer/moe/experts.py:90`（`class GroupedMLP`，`expert_parallel`、EP-aware `sharded_state_dict`、selective activation recompute）。
- 可微 gmm：`.../grouped_gemm/ops.py` `GroupedGemm`（`backward` 同时算 `agrad` 与 `bgrad`）。
- P3 推理封装（不可训练，仅对照）：`experiments/omnizip_omnidial_stack_20261015/source/qwen_mcore_experts_p0.py`（`.eval()` + `no_grad` 拷权重 + world_size=1）。

## 2. 实际运行内容与成功门
### 2.1 1-rank 路由 smoke（`moe_train_smoke.py`）
合成 `E=8,H=16,I=32,T=4`；每臂 1 个 optimizer.step；**共 2 step（native_top8、omnidial_budget2），之后立即停止**。

| 臂 | dispatched rows（真实送入 gmm） | loss finite | grad gate | grad expert w1 | grad input x |
|---|---|---|---|---|---|
| native top8 | **32**（4×8） | True | nonzero | nonzero | nonzero |
| omnidial budget2 | **8**（4×2） | True | nonzero | nonzero | nonzero |

- **dispatch rows 真实减少 32→8（4×），非 mask-only**：tail 专家被剪后实际送入 grouped_gemm 的 (token,expert) 行成比例下降。
- optimizer.step 正常；输出 `SMOKE_OK`。

### 2.2 EP=2 语义（`ep2_test.py`，torchrun --nproc_per_node=2）
合成 E=8、ep=2，调用方传 `num_local_experts=E/ep=4`：

| rank | ep_rank | num_local | global expert range | loss finite | expert w1 grad |
|---|---|---|---|---|---|
| rank0 | 0 | 4 | **[0,4)** | True (0.062) | nonzero |
| rank1 | 1 | 4 | **[4,8)** | True (0.002) | nonzero |

- 两 rank 各自的成功门在 barrier 前均已打印通过。
- 收尾 barrier 因 NCCL `Failed to open libnvidia-ml.so.1`（监控库缺失，环境告警）崩溃，**非正确性问题**。

## 3. 版本/环境不匹配阻塞（重要）
- 当前节点唯一可用 mcore 是**推理 drop** `qwen_runtime_mcore_20260924`（0.12.3），无 RouterPlugin API。用户目标栈为 **MCore 0.18 / Megatron-SWIFT + transformer-engine 2.16.0**，该训练 env 正由队友在新 venv（`/tmp/omnivideo_100k_qwen3_omni_lora`）安装中，**尚未就绪**。
- 因此本 smoke 证明的是「mcore GroupedMLP 通路可训练态真实稀疏」，**不是**最终 0.18/SWIFT 栈上的结论。真 30B 1-step 待 env 就绪 + 协议冻结后另做。
- grouped_gemm 约束：仅 bf16；`tokens_per_expert` 必须 CPU int64；权重须显式初始化（`perform_initialization=False` 垃圾内存会让反向读 0）。

## 4. 产物清单（绝对路径 + sha256）
远端共享目录：`/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner/experiments/omnizip_lora_train_smoke_20261016/source_and_logs/`

| 文件 | sha256 |
|---|---|
| `moe_train_smoke.py` | `c610ef939403c0aad1658b81584deeb11f5ebefb7a52decb5893c3b17e2fdefe` |
| `ep2_test.py` | `9c4e4d42e2315d00d14c611dbd2a134d139a80c10f85ff68e7ea7b2fd640b8c3` |
| `log_1rank_smoke.txt` | `c0a68349186fdf67d95cea721fe106cd4357a77ea94422d8e377cf44fe06f60a` |
| `log_ep2.txt` | `1677b44194b005e89198b34dfe847d5cb4d288561e01478727bd52ede1cb5797` |
| `TRAINING_SMOKE_AUDIT.md` | `077e18bf…`（见归档后 sha256sum） |

## 5. 结论
- 训练态**真实稀疏 dispatch 可行**（非 mask-only）：autograd、有限 loss、optimizer step、EP=2 每 rank 专家语义、dispatch rows 32→8 均以合成张量证明。
- **未运行 30B LoRA 任何 step**；不得据此宣称机时收益。真 30B 最小 1-2 step 待 0.18/SWIFT env 就绪与协议冻结。

# 05. Real Compact Dispatch

前面 `01-04` 都是 mask-only ablation，用来确认质量。第五步才开始追求真实速度。

目标：

> 真的减少 token-expert assignment，让未保留的 expert 不执行。

## 前置条件

不要跳过这些条件：

- `01` fixed top-m 质量没有崩。
- `02` dynamic top-k 比 fixed top-m 更稳。
- `03` task/layer/modality policy 能保护 stress tasks。
- mask-only 已经记录了 effective assignment reduction。

如果这些没过，不要做 real dispatch。

## 为什么 Mask-Only 不够

如果只是：

```text
routing_weights[:, m:] = 0
```

但底层仍然把 token 发给原始 top-k expert，那么：

- expert MLP 仍然执行。
- dispatch/combine 仍然执行。
- latency 几乎不会下降。

真实加速必须减少：

```text
flat_token_ids
flat_expert_ids
flat_routing_weights
```

## Compact Dispatch 数据结构

每层构造：

```text
selected_experts_compact: [num_assignments]
token_indices_compact: [num_assignments]
routing_weights_compact: [num_assignments]
expert_offsets: [num_experts + 1]
expert_batch_sizes: [num_experts]
```

流程：

```text
1. router logits -> selected experts / weights
2. apply expert budget policy -> keep mask over assignments
3. compact assignment list
4. sort by expert id
5. gather hidden_states[token_indices]
6. run expert MLP by expert groups
7. weighted scatter-add back to original token positions
```

## Correctness Tests

必须先写小测试：

1. `policy=full_topk` 时，compact dispatch 输出与原始 MoE 输出接近。
2. batch size 1、短 seq、随机 hidden states 可跑通。
3. 所有 expert 空 / 部分 expert 空不崩。
4. top_m=top_k 时输出一致。
5. top_m=1/4/6 时 shape 正确。
6. fp16/bf16 下误差在可接受范围。

## Integration Test

对 1-2 个真实 OmniVideoBench 样本：

```text
max_new_tokens=1
token_pruning disabled
expert_budget enabled
```

确认：

- prefill 不崩。
- decode 第一个 token 不崩。
- KV cache shape 合理。
- output logits 没有 NaN。

## Performance Metrics

必须记录：

```text
moe_forward_time_ms_by_layer
dispatch_time_ms_by_layer
expert_mlp_time_ms_by_layer
combine_time_ms_by_layer
assignment_count_before/after
expert_batch_size_before/after
hot_to_mean_before/after
```

如果只能拿到整体 `moe_forward_time_ms`，至少记录 before/after。

## Evaluation

对比：

```text
baseline
mask-only dynamic top-k
real-dispatch dynamic top-k
```

期望：

- mask-only 和 real-dispatch 的 accuracy 接近。
- real-dispatch 的 MoE block time 下降。
- overall prefill 有小幅下降，但不要期待巨大下降，因为 attention 仍是大头。

失败排查：

- MoE block time 没降：可能仍然走原 top-k，或 Python loop/grouped GEMM overhead 抵消收益。
- accuracy 和 mask-only 不一致：compact dispatch combine 权重、排序、scatter 有 bug。
- 某些样本 NaN：检查 renormalize、空 expert、dtype。

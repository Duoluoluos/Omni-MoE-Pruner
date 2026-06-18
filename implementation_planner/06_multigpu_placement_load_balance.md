# 06. Multi-GPU Placement And Load Balance

只有在多卡 expert parallel 环境下才需要做这一步。单卡先跳过。

## Profile 依据

MoE 多卡推理的额外瓶颈是：

```text
dispatch all-to-all
combine all-to-all
hot expert straggler
cross-GPU token shuffle
```

纯文本 MoE 系统论文的启发：

- DeepSpeed-MoE：hierarchical all-to-all、token-to-expert mapping、kernel fusion。
- Fiddler：expert popularity 可以指导 placement/offload。
- Expert Choice：固定 bucket size 能减少 straggler。

## 要记录的 Profile

多卡 run 必须记录：

```text
all_to_all_time_ms_by_layer
dispatch_bytes_by_layer
combine_bytes_by_layer
expert_batch_size_by_gpu
hot_expert_gpu
gpu_idle_time 或 step straggler time
```

如果 torch profiler 可用，抓：

```text
nccl all_to_all
all_gather
reduce_scatter
```

## Hot Expert Placement

根据 calibration profile：

```text
for each layer:
  sort experts by activation count
  distribute hot experts across GPUs
```

目标：

- 不要把同层 hot experts 放在同一 GPU。
- 如果某 expert 极热，考虑 replica。

## Expert Replication

只对极热 expert：

```text
if expert_load > replica_threshold:
    replicate expert on 2 GPUs
    route token to less loaded replica
```

约束：

- replica 增加显存。
- combine 逻辑必须知道 replica 是同一个 expert。
- 先只在最热的少数层/专家试。

## Locality-Aware Boundary Reroute

只对低优先级 boundary assignment：

```text
if selected expert is remote and next-best local expert score close enough:
    use local expert
```

保护：

```text
do not reroute top1
do not reroute protected modality
do not reroute low-margin token
do not reroute stress task unless explicitly allowed
```

## Evaluation

对比：

```text
baseline expert placement
hot expert distributed placement
hot expert replication
locality-aware boundary reroute
```

指标：

- all-to-all time。
- p95 layer MoE time。
- hot/mean per GPU。
- overall prefill。
- accuracy / flip rate。

如果 all-to-all 不是瓶颈，不要在这一步继续投入。

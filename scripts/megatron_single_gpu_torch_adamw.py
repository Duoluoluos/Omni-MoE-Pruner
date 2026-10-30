"""Use native PyTorch AdamW for the single-GPU Qwen3-Omni LoRA run.

Megatron-Core prefers Transformer Engine's FusedAdam whenever Transformer
Engine is installed. With EP=1, this model places all expert LoRA tensors on
one rank, exceeding FusedAdam's fixed tensor-entry limit during its first
optimizer step. MCore's optimizer factory deliberately supports a native
PyTorch AdamW fallback behind ``USING_PYTORCH_OPTIMIZER``; this plugin selects
that supported path before the trainer or model is constructed.

The paired baseline and ARC-SCEB arms both load this plugin. It changes only
the optimizer implementation required by the one-GPU topology, not the model,
data, token budget, batch semantics, or routing policy.
"""

from __future__ import annotations

import megatron.core.optimizer as _mcore_optimizer


if not _mcore_optimizer.USING_PYTORCH_OPTIMIZER:
    _mcore_optimizer.USING_PYTORCH_OPTIMIZER = True
    print(
        "[single-GPU optimizer] forcing native PyTorch AdamW; "
        "TE FusedAdam cannot represent this EP=1 LoRA tensor count.",
        flush=True,
    )
else:
    print("[single-GPU optimizer] native PyTorch optimizer already selected.", flush=True)

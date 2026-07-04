"""Compact expert dispatch for Qwen3-Omni MoE blocks.

The stock SparseMoeBlock groups by expert over the full top-k assignment matrix.
Mask-only budget methods zero some routing weights but still execute those
assignments. This module builds a compact assignment list from a boolean keep
mask, sorts by expert id, runs each non-empty expert group, and scatter-adds
weighted outputs back to token positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch


@dataclass
class CompactDispatchMetrics:
    assignment_count_before: int
    assignment_count_after: int
    active_experts_before: int
    active_experts_after: int
    hot_to_mean_before: float
    hot_to_mean_after: float
    dispatch_time_ms: float = 0.0
    expert_mlp_time_ms: float = 0.0
    combine_time_ms: float = 0.0


def _hot_to_mean(counts: torch.Tensor) -> float:
    nonzero = counts[counts > 0]
    if nonzero.numel() == 0:
        return 0.0
    return float(nonzero.max().item() / nonzero.float().mean().item())


def compact_moe_forward(
    *,
    experts: Any,
    flat_hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
    keep_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, CompactDispatchMetrics]:
    """Run an MoE expert loop over a compact assignment list.

    Args:
        experts: module list indexed by expert id.
        flat_hidden_states: ``[num_tokens, hidden_dim]`` input rows.
        selected_experts: ``[num_tokens, top_k]`` expert ids from router top-k.
        routing_weights: ``[num_tokens, top_k]`` normalized weights.
        num_experts: total expert count.
        keep_mask: optional bool tensor with the same shape as selected_experts.
            False assignments are removed and their expert MLP is not run.
    """
    t0 = perf_counter()
    num_tokens, hidden_dim = flat_hidden_states.shape
    top_k = int(selected_experts.shape[1])
    if keep_mask is None:
        keep_mask = torch.ones_like(selected_experts, dtype=torch.bool)
    else:
        keep_mask = keep_mask.to(device=selected_experts.device, dtype=torch.bool)

    before_experts = selected_experts.reshape(-1)
    before_counts = torch.bincount(before_experts, minlength=num_experts)
    kept_pos, kept_rank = keep_mask.nonzero(as_tuple=True)
    assignment_count_before = int(selected_experts.numel())
    assignment_count_after = int(kept_pos.numel())
    after_counts = torch.bincount(
        selected_experts[kept_pos, kept_rank],
        minlength=num_experts,
    ) if assignment_count_after else torch.zeros(num_experts, device=selected_experts.device, dtype=torch.long)

    final_hidden_states = torch.zeros(
        (num_tokens, hidden_dim),
        dtype=flat_hidden_states.dtype,
        device=flat_hidden_states.device,
    )
    t1 = perf_counter()

    if assignment_count_after == 0:
        metrics = CompactDispatchMetrics(
            assignment_count_before=assignment_count_before,
            assignment_count_after=0,
            active_experts_before=int((before_counts > 0).sum().item()),
            active_experts_after=0,
            hot_to_mean_before=_hot_to_mean(before_counts),
            hot_to_mean_after=0.0,
            dispatch_time_ms=(t1 - t0) * 1000.0,
        )
        return final_hidden_states, metrics

    compact_experts = selected_experts[kept_pos, kept_rank]
    compact_weights = routing_weights[kept_pos, kept_rank]
    # Match the original Qwen MoE loop order as closely as possible:
    # expert id ascending, and within each expert the order produced by
    # torch.where(expert_mask[expert]) over a [top_k, num_tokens] mask, i.e.
    # rank-major then token id. This is mathematically equivalent to sorting by
    # expert only, but keeps GEMM row order and scatter-add order closer to the
    # mask-only reference in fp16/bf16.
    order_key = compact_experts * (top_k * num_tokens) + kept_rank * num_tokens + kept_pos
    order = torch.argsort(order_key, stable=True)
    expert_ids_sorted = compact_experts[order]
    token_ids_sorted = kept_pos[order]
    rank_sorted = kept_rank[order]
    weights_sorted = compact_weights[order]
    t2 = perf_counter()

    unique_experts, counts = torch.unique_consecutive(expert_ids_sorted, return_counts=True)
    offsets = torch.cat([
        torch.zeros(1, device=counts.device, dtype=torch.long),
        torch.cumsum(counts, dim=0),
    ])

    expert_ms = 0.0
    combine_start = perf_counter()
    for i, expert_id in enumerate(unique_experts.tolist()):
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())
        top_x = token_ids_sorted[start:end]
        idx = rank_sorted[start:end]
        current_state = flat_hidden_states[None, top_x].reshape(-1, hidden_dim)
        e0 = perf_counter()
        current_hidden_states = experts[int(expert_id)](current_state)
        e1 = perf_counter()
        expert_ms += (e1 - e0) * 1000.0
        current_hidden_states = current_hidden_states * weights_sorted[start:end, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(flat_hidden_states.dtype))
        # Keep idx live for parity with the original loop's assignment rank.
        del idx
    combine_end = perf_counter()

    metrics = CompactDispatchMetrics(
        assignment_count_before=assignment_count_before,
        assignment_count_after=assignment_count_after,
        active_experts_before=int((before_counts > 0).sum().item()),
        active_experts_after=int((after_counts > 0).sum().item()),
        hot_to_mean_before=_hot_to_mean(before_counts),
        hot_to_mean_after=_hot_to_mean(after_counts),
        dispatch_time_ms=(t2 - t0) * 1000.0,
        expert_mlp_time_ms=expert_ms,
        combine_time_ms=max(0.0, (combine_end - combine_start) * 1000.0 - expert_ms),
    )
    return final_hidden_states, metrics

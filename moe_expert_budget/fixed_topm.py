"""Fixed top-M mask wrapper for Qwen3-Omni SparseMoeBlock.

Mask-only ablation: still computes the full set of expert MLPs as the original block
would, but zeros out the tail (top_m+1 ... top_k) routing weights for target-modality
tokens in target layers. Optionally renormalizes the kept weights so the residual
scale stays comparable to the original mixture.

This file deliberately mirrors the math in
``Qwen3OmniMoeThinkerTextSparseMoeBlock.forward`` from transformers, so that the
returned ``(hidden_states, router_logits)`` tuple stays compatible with the existing
profiler hook in ``profile_omnivideobench_qwen3.run_omnivideobench_profile``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from .mask_policy import (
    ExpertBudgetConfig,
    MODALITY_CODES,
    MODALITY_NAMES,
    build_modality_vector,
    resolve_target_token_mask,
)


@dataclass
class _BudgetCallStats:
    layer_idx: int
    tokens_total: int = 0
    tokens_target: int = 0
    tokens_protected: int = 0
    assignments_original: int = 0
    assignments_effective: int = 0


@dataclass
class _BudgetSampleStats:
    sample_index: int | None = None
    layers: dict[int, _BudgetCallStats] = field(default_factory=dict)
    target_layers: list[int] = field(default_factory=list)

    def to_dict(self, config: ExpertBudgetConfig) -> dict[str, Any]:
        layer_summaries: list[dict[str, Any]] = []
        total_orig = 0
        total_eff = 0
        total_target_tokens = 0
        total_protected_tokens = 0
        total_tokens = 0
        for idx in sorted(self.layers):
            ls = self.layers[idx]
            total_orig += ls.assignments_original
            total_eff += ls.assignments_effective
            total_target_tokens += ls.tokens_target
            total_protected_tokens += ls.tokens_protected
            total_tokens += ls.tokens_total
            layer_summaries.append({
                "layer_idx": ls.layer_idx,
                "tokens_total": ls.tokens_total,
                "video_tokens": ls.tokens_target,
                "protected_tokens": ls.tokens_protected,
                "video_active_experts_original": config.original_top_k,
                "video_active_experts_effective": min(config.top_m, config.original_top_k),
                "assignments_original": ls.assignments_original,
                "assignments_effective": ls.assignments_effective,
            })
        return {
            "enabled": bool(config.enabled),
            "mode": config.mode,
            "target_layers": sorted(self.target_layers),
            "top_m": config.top_m,
            "original_top_k": config.original_top_k,
            "target_modality": config.target_modality,
            "protected_modalities": list(config.protected_modalities),
            "renormalize_weights": bool(config.renormalize_weights),
            "assignment_count_original_est": total_orig,
            "assignment_count_effective": total_eff,
            "avg_active_experts_per_token_original": (
                total_orig / total_tokens if total_tokens else 0.0
            ),
            "avg_active_experts_per_token_effective": (
                total_eff / total_tokens if total_tokens else 0.0
            ),
            "target_token_count": total_target_tokens,
            "protected_token_count": total_protected_tokens,
            "layer_summaries": layer_summaries,
        }


class _ActiveSampleContext:
    """Shared per-sample state read by the wrapped MoE forward."""

    def __init__(self) -> None:
        self.modalities: np.ndarray | None = None
        self.target_token_mask: torch.Tensor | None = None  # bool, on model device, len = seq
        self.seq_len: int = 0
        self.sample_index: int | None = None
        self.stats: _BudgetSampleStats = _BudgetSampleStats()


class FixedTopMMaskInstaller:
    """Install fixed top-M mask wrappers on every SparseMoeBlock in ``model.thinker``.

    Usage:
        installer = FixedTopMMaskInstaller(model, config)
        installer.install()
        ...
        installer.begin_sample(sample_index, input_ids)  # before model.generate
        cont, _ = model.generate(...)
        budget_profile = installer.end_sample()
    """

    def __init__(self, model: Any, config: ExpertBudgetConfig) -> None:
        self.model = model
        self.config = config
        self.context = _ActiveSampleContext()
        self._wrapped: list[tuple[Any, Callable]] = []
        text_config = model.thinker.config.text_config
        self.num_layers = int(text_config.num_hidden_layers)
        self.num_experts = int(text_config.num_experts)
        self.text_top_k = int(text_config.num_experts_per_tok)
        self.norm_topk_prob = bool(getattr(text_config, "norm_topk_prob", True))
        if self.config.original_top_k != self.text_top_k:
            print(
                f"[expert_budget] config.original_top_k={self.config.original_top_k} "
                f"differs from model top_k={self.text_top_k}; using model top_k.",
                flush=True,
            )
            self.config.original_top_k = self.text_top_k
        self.target_layers = self.config.target_layer_set(self.num_layers)
        self.token_ids = {
            "image": int(model.thinker.config.image_token_id),
            "video": int(model.thinker.config.video_token_id),
            "audio": int(model.thinker.config.audio_token_id),
        }
        self.protected_codes = sorted({MODALITY_CODES[m] for m in self.config.protected_modalities})
        self.target_code = self.config.target_modality_code()
        self._enabled = bool(self.config.enabled) and self.config.top_m < self.text_top_k

    @property
    def enabled(self) -> bool:
        return self._enabled

    def install(self) -> int:
        if not self._enabled:
            return 0
        installed = 0
        for name, module in self.model.thinker.named_modules():
            if not self._is_sparse_moe(module):
                continue
            layer_idx = self._parse_layer_idx(name)
            if layer_idx is None:
                continue
            self._wrap_forward(module, layer_idx)
            installed += 1
        print(
            f"[expert_budget] wrapped {installed} SparseMoeBlock layers; "
            f"target_layers={sorted(self.target_layers)} top_m={self.config.top_m} "
            f"renorm={self.config.renormalize_weights} target={self.config.target_modality} "
            f"protected={self.config.protected_modalities}",
            flush=True,
        )
        return installed

    def restore(self) -> None:
        for module, original in self._wrapped:
            module.forward = original
        self._wrapped.clear()

    @staticmethod
    def _is_sparse_moe(module: Any) -> bool:
        cls_name = module.__class__.__name__.lower()
        return "sparsemoeblock" in cls_name and hasattr(module, "gate") and hasattr(module, "experts")

    @staticmethod
    def _parse_layer_idx(name: str) -> int | None:
        parts = name.split(".")
        for i, part in enumerate(parts[:-1]):
            if part == "layers" and parts[i + 1].isdigit():
                return int(parts[i + 1])
        return None

    def begin_sample(self, sample_index: int | None, input_ids: torch.Tensor, task_type: str | None = None) -> None:
        del task_type
        ids = input_ids[0].detach().cpu().numpy().astype(np.int64)
        modalities = build_modality_vector(ids, self.token_ids)
        target_mask_np = resolve_target_token_mask(modalities, self.target_code, self.protected_codes)
        device = input_ids.device
        target_mask_t = torch.from_numpy(target_mask_np).to(device=device)
        self.context.modalities = modalities
        self.context.target_token_mask = target_mask_t
        self.context.seq_len = int(ids.shape[0])
        self.context.sample_index = sample_index
        self.context.stats = _BudgetSampleStats(
            sample_index=sample_index,
            layers={},
            target_layers=sorted(self.target_layers),
        )

    def end_sample(self) -> dict[str, Any]:
        out = self.context.stats.to_dict(self.config)
        self.context.modalities = None
        self.context.target_token_mask = None
        self.context.seq_len = 0
        self.context.sample_index = None
        self.context.stats = _BudgetSampleStats()
        return out

    def _wrap_forward(self, module: Any, layer_idx: int) -> None:
        original_forward = module.forward
        self._wrapped.append((module, original_forward))
        installer = self
        # Capture block-level constants.
        num_experts = int(getattr(module, "num_experts", installer.num_experts))
        top_k = int(getattr(module, "top_k", installer.text_top_k))
        norm_topk_prob = bool(getattr(module, "norm_topk_prob", installer.norm_topk_prob))

        def wrapped_forward(hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            flat = hidden_states.view(-1, hidden_dim)
            router_logits = module.gate(flat)
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

            apply_mask = (
                installer._enabled
                and (layer_idx in installer.target_layers)
                and installer.context.target_token_mask is not None
                and (batch_size * sequence_length == installer.context.seq_len)
            )

            top_m = min(int(installer.config.top_m), top_k)
            tgt_mask = None
            if apply_mask:
                tgt_mask = installer.context.target_token_mask
                if tgt_mask.shape[0] != routing_weights.shape[0]:
                    apply_mask = False
                    tgt_mask = None
                elif tgt_mask.device != routing_weights.device:
                    tgt_mask = tgt_mask.to(routing_weights.device)

            if apply_mask and not tgt_mask.any():
                apply_mask = False

            if apply_mask:
                rows = tgt_mask.nonzero(as_tuple=True)[0]
                if top_m < top_k:
                    routing_weights[rows, top_m:] = 0.0
                # Renormalize target rows when renormalize_weights is set; otherwise
                # leave their kept weights unscaled (intentional ablation variant).
                if installer.config.renormalize_weights:
                    tgt_sums = routing_weights[rows].sum(dim=-1, keepdim=True).clamp_min(1e-20)
                    routing_weights[rows] = routing_weights[rows] / tgt_sums
                # Non-target rows follow the original norm_topk_prob behaviour.
                if norm_topk_prob:
                    nt_mask = ~tgt_mask
                    if nt_mask.any():
                        nt_rows = nt_mask.nonzero(as_tuple=True)[0]
                        nt_sums = routing_weights[nt_rows].sum(dim=-1, keepdim=True).clamp_min(1e-20)
                        routing_weights[nt_rows] = routing_weights[nt_rows] / nt_sums
                # stats
                stats = installer.context.stats
                layer_stats = stats.layers.get(layer_idx)
                if layer_stats is None:
                    layer_stats = _BudgetCallStats(layer_idx=layer_idx)
                    stats.layers[layer_idx] = layer_stats
                total_rows = int(routing_weights.shape[0])
                n_target = int(rows.numel())
                layer_stats.tokens_total += total_rows
                layer_stats.tokens_target += n_target
                layer_stats.tokens_protected += total_rows - n_target
                layer_stats.assignments_original += total_rows * top_k
                layer_stats.assignments_effective += n_target * top_m + (total_rows - n_target) * top_k
            else:
                if norm_topk_prob:
                    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)

            routing_weights = routing_weights.to(flat.dtype)

            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim),
                dtype=flat.dtype,
                device=flat.device,
            )
            expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
            # Skip experts whose routing weight is effectively zero everywhere they'd be picked.
            # The original code uses `expert_mask.sum > 0` which is binary (any picks at all);
            # to actually save the wasted MM you'd need to also gate by routing_weights > 0.
            # Per the planning doc we explicitly keep mask-only semantics here.
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit:
                expert_layer = module.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
                current_state = flat[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(flat.dtype))
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states, router_logits

        module.forward = wrapped_forward

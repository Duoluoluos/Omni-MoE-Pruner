"""Confidence-aware dynamic top-M mask wrapper for Qwen3-Omni SparseMoeBlock.

Chooses, per target-modality token, how many experts to keep from the router
top-k based on confidence margins. Step 03 adds task/layer/modality policy:
only policy-selected layers are dynamic, protected modalities keep full top-k,
and task clusters can clamp the minimum top-M.
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
    build_modality_vector,
    resolve_target_token_mask,
)

_CALIB_MAX_ROWS_PER_CALL = 4096


@dataclass
class _DynCallStats:
    layer_idx: int
    tokens_total: int = 0
    tokens_target: int = 0
    tokens_protected: int = 0
    assignments_original: int = 0
    assignments_effective: int = 0
    target_active_sum: int = 0


@dataclass
class _DynSampleStats:
    sample_index: int | None = None
    task_type: str | None = None
    task_policy_cluster: str = "default"
    task_min_top_m: int | None = None
    layers: dict[int, _DynCallStats] = field(default_factory=dict)
    target_layers: list[int] = field(default_factory=list)
    top_m_histogram: dict[int, int] = field(default_factory=dict)

    def to_dict(self, installer: "ConfidenceDynamicMaskInstaller") -> dict[str, Any]:
        config = installer.config
        layer_summaries: list[dict[str, Any]] = []
        total_orig = 0
        total_eff = 0
        total_target_tokens = 0
        total_protected_tokens = 0
        total_tokens = 0
        avg_active_by_layer: dict[str, float] = {}
        for idx in sorted(self.layers):
            ls = self.layers[idx]
            total_orig += ls.assignments_original
            total_eff += ls.assignments_effective
            total_target_tokens += ls.tokens_target
            total_protected_tokens += ls.tokens_protected
            total_tokens += ls.tokens_total
            if ls.tokens_target:
                avg_active_by_layer[str(idx)] = ls.target_active_sum / ls.tokens_target
            layer_summaries.append({
                "layer_idx": ls.layer_idx,
                "tokens_total": ls.tokens_total,
                "video_tokens": ls.tokens_target,
                "protected_tokens": ls.tokens_protected,
                "video_active_experts_original": config.original_top_k,
                "video_active_experts_effective": (
                    ls.target_active_sum / ls.tokens_target if ls.tokens_target else config.original_top_k
                ),
                "assignments_original": ls.assignments_original,
                "assignments_effective": ls.assignments_effective,
            })
        hist = {str(k): int(self.top_m_histogram.get(k, 0)) for k in sorted(installer.top_m_options)}
        return {
            "enabled": bool(config.enabled),
            "mode": config.mode,
            "policy_version": config.policy_version,
            "target_layers": sorted(self.target_layers),
            "layer_policy": config.layer_policy,
            "task_type": self.task_type,
            "task_policy_cluster": self.task_policy_cluster,
            "task_min_top_m": self.task_min_top_m,
            "task_policy": config.task_policy,
            "top_m_options": list(installer.top_m_options),
            "original_top_k": config.original_top_k,
            "target_modality": config.target_modality,
            "protected_modalities": list(config.protected_modalities),
            "renormalize_weights": bool(config.renormalize_weights),
            "margin_thresholds": {
                "top1_top2_high": installer.high_thr,
                "top1_top2_medium": installer.medium_thr,
                "boundary": installer.boundary_thr,
            },
            "top_m_histogram": hist,
            "assignment_count_original_est": total_orig,
            "assignment_count_effective": total_eff,
            "avg_active_experts_per_token_original": total_orig / total_tokens if total_tokens else 0.0,
            "avg_active_experts_per_token_effective": total_eff / total_tokens if total_tokens else 0.0,
            "avg_active_experts_per_video_token_by_layer": avg_active_by_layer,
            "target_token_count": total_target_tokens,
            "protected_token_count": total_protected_tokens,
            "layer_summaries": layer_summaries,
        }


class _ActiveSampleContext:
    def __init__(self) -> None:
        self.modalities: np.ndarray | None = None
        self.target_token_mask: torch.Tensor | None = None
        self.seq_len: int = 0
        self.sample_index: int | None = None
        self.task_type: str | None = None
        self.task_policy_cluster: str = "default"
        self.task_min_top_m: int | None = None
        self.stats: _DynSampleStats = _DynSampleStats()


class ConfidenceDynamicMaskInstaller:
    """Install confidence-dynamic top-M mask wrappers on SparseMoeBlock modules."""

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

        opts = sorted({min(int(m), self.text_top_k) for m in (config.top_m_options or [4, 6, 8])})
        while len(opts) < 3:
            opts.append(self.text_top_k)
        self.top_m_options = opts[:3]
        self.m_small, self.m_mid, self.m_large = self.top_m_options

        self.layer_modes = self.config.layer_modes(self.num_layers)
        self.target_layers = self.config.target_layer_set_from_policy(self.num_layers, "dynamic")
        self.token_ids = {
            "image": int(model.thinker.config.image_token_id),
            "video": int(model.thinker.config.video_token_id),
            "audio": int(model.thinker.config.audio_token_id),
        }
        self.protected_codes = sorted({MODALITY_CODES[m] for m in self.config.protected_modalities})
        self.target_code = self.config.target_modality_code()

        self.high_thr: float | None = config.high_margin_threshold
        self.medium_thr: float | None = config.medium_margin_threshold
        self.boundary_thr: float | None = config.boundary_threshold
        self._calibrating = False
        self._calib_margin12: list[np.ndarray] = []
        self._calib_boundary: list[np.ndarray] = []
        self._enabled = bool(self.config.enabled) and self.m_small < self.text_top_k

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def thresholds_ready(self) -> bool:
        return None not in (self.high_thr, self.medium_thr, self.boundary_thr)

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
            f"[expert_budget] (dynamic) wrapped {installed} SparseMoeBlock layers; "
            f"target_layers={sorted(self.target_layers)} options={self.top_m_options} "
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

    def start_calibration(self) -> None:
        self._calibrating = True
        self._calib_margin12 = []
        self._calib_boundary = []
        print("[expert_budget] (dynamic) calibration started", flush=True)

    def finish_calibration(self) -> dict[str, Any]:
        self._calibrating = False
        if not self._calib_margin12:
            self.high_thr = self.high_thr if self.high_thr is not None else 0.0
            self.medium_thr = self.medium_thr if self.medium_thr is not None else 0.0
            self.boundary_thr = self.boundary_thr if self.boundary_thr is not None else 0.0
            print("[expert_budget] (dynamic) calibration saw no target tokens; thresholds=0", flush=True)
            return self._threshold_report(n_margin=0, n_boundary=0, fallback=True)
        margin12 = np.concatenate(self._calib_margin12)
        boundary = np.concatenate(self._calib_boundary)
        self.high_thr = float(np.percentile(margin12, self.config.high_margin_percentile))
        self.medium_thr = float(np.percentile(margin12, self.config.medium_margin_percentile))
        self.boundary_thr = float(np.percentile(boundary, self.config.boundary_margin_percentile))
        self.config.high_margin_threshold = self.high_thr
        self.config.medium_margin_threshold = self.medium_thr
        self.config.boundary_threshold = self.boundary_thr
        report = self._threshold_report(n_margin=int(margin12.size), n_boundary=int(boundary.size))
        self._calib_margin12 = []
        self._calib_boundary = []
        print(f"[expert_budget] (dynamic) calibrated thresholds={report['thresholds']}", flush=True)
        return report

    def _threshold_report(self, n_margin: int, n_boundary: int, fallback: bool = False) -> dict[str, Any]:
        return {
            "thresholds": {
                "top1_top2_high": self.high_thr,
                "top1_top2_medium": self.medium_thr,
                "boundary": self.boundary_thr,
            },
            "percentiles": {
                "high_margin_percentile": self.config.high_margin_percentile,
                "medium_margin_percentile": self.config.medium_margin_percentile,
                "boundary_margin_percentile": self.config.boundary_margin_percentile,
            },
            "calibration_samples_used": self.config.calibrate_samples,
            "margin_values_collected": n_margin,
            "boundary_values_collected": n_boundary,
            "fallback_no_target_tokens": fallback,
        }

    def begin_sample(self, sample_index: int | None, input_ids: torch.Tensor, task_type: str | None = None) -> None:
        ids = input_ids[0].detach().cpu().numpy().astype(np.int64)
        modalities = build_modality_vector(ids, self.token_ids)
        target_mask_np = resolve_target_token_mask(modalities, self.target_code, self.protected_codes)
        device = input_ids.device
        task_min_top_m, task_cluster = self.config.task_min_top_m(task_type)
        task_min_top_m = min(max(int(task_min_top_m), self.m_small), self.text_top_k)
        self.context.modalities = modalities
        self.context.target_token_mask = torch.from_numpy(target_mask_np).to(device=device)
        self.context.seq_len = int(ids.shape[0])
        self.context.sample_index = sample_index
        self.context.task_type = task_type
        self.context.task_policy_cluster = task_cluster
        self.context.task_min_top_m = task_min_top_m
        self.context.stats = _DynSampleStats(
            sample_index=sample_index,
            task_type=task_type,
            task_policy_cluster=task_cluster,
            task_min_top_m=task_min_top_m,
            layers={},
            target_layers=sorted(self.target_layers),
        )

    def end_sample(self) -> dict[str, Any]:
        out = self.context.stats.to_dict(self)
        self.context.modalities = None
        self.context.target_token_mask = None
        self.context.seq_len = 0
        self.context.sample_index = None
        self.context.task_type = None
        self.context.task_policy_cluster = "default"
        self.context.task_min_top_m = None
        self.context.stats = _DynSampleStats()
        return out

    def _wrap_forward(self, module: Any, layer_idx: int) -> None:
        original_forward = module.forward
        self._wrapped.append((module, original_forward))
        installer = self
        num_experts = int(getattr(module, "num_experts", installer.num_experts))
        top_k = int(getattr(module, "top_k", installer.text_top_k))
        norm_topk_prob = bool(getattr(module, "norm_topk_prob", installer.norm_topk_prob))

        def wrapped_forward(hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            flat = hidden_states.view(-1, hidden_dim)
            router_logits = module.gate(flat)
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

            apply_ctx = (
                installer._enabled
                and installer.layer_modes.get(layer_idx, "full") == "dynamic"
                and installer.context.target_token_mask is not None
                and (batch_size * sequence_length == installer.context.seq_len)
            )
            tgt_mask = None
            if apply_ctx:
                tgt_mask = installer.context.target_token_mask
                if tgt_mask.shape[0] != routing_weights.shape[0]:
                    apply_ctx = False
                    tgt_mask = None
                elif tgt_mask.device != routing_weights.device:
                    tgt_mask = tgt_mask.to(routing_weights.device)
            if apply_ctx and not tgt_mask.any():
                apply_ctx = False

            did_mask = False
            if apply_ctx:
                rows = tgt_mask.nonzero(as_tuple=True)[0]
                if installer._calibrating:
                    installer._collect_calibration(routing_weights, rows, top_k)
                elif installer.thresholds_ready:
                    chosen_m = installer._choose_top_m(routing_weights, rows, top_k)
                    if installer.context.task_min_top_m is not None:
                        chosen_m = torch.clamp(chosen_m, min=int(installer.context.task_min_top_m), max=top_k)
                    installer._apply_dynamic_mask(routing_weights, rows, chosen_m, top_k)
                    installer._record_stats(layer_idx, routing_weights.shape[0], rows, chosen_m, top_k)
                    did_mask = True

            if did_mask:
                if installer.config.renormalize_weights:
                    rows = tgt_mask.nonzero(as_tuple=True)[0]
                    tgt_sums = routing_weights[rows].sum(dim=-1, keepdim=True).clamp_min(1e-20)
                    routing_weights[rows] = routing_weights[rows] / tgt_sums
                if norm_topk_prob:
                    nt_mask = ~tgt_mask
                    if nt_mask.any():
                        nt_rows = nt_mask.nonzero(as_tuple=True)[0]
                        nt_sums = routing_weights[nt_rows].sum(dim=-1, keepdim=True).clamp_min(1e-20)
                        routing_weights[nt_rows] = routing_weights[nt_rows] / nt_sums
            elif norm_topk_prob:
                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)

            routing_weights = routing_weights.to(flat.dtype)
            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim),
                dtype=flat.dtype,
                device=flat.device,
            )
            expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
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

    def _collect_calibration(self, routing_weights: torch.Tensor, rows: torch.Tensor, top_k: int) -> None:
        n = int(rows.numel())
        if n == 0:
            return
        if n > _CALIB_MAX_ROWS_PER_CALL:
            perm = torch.randperm(n, device=rows.device)[:_CALIB_MAX_ROWS_PER_CALL]
            rows = rows[perm]
        w = routing_weights[rows]
        self._calib_margin12.append((w[:, 0] - w[:, 1]).detach().to("cpu", dtype=torch.float32).numpy())
        boundary_parts = []
        for m in (self.m_small, self.m_mid):
            if m < top_k:
                boundary_parts.append((w[:, m - 1] - w[:, m]).detach().to("cpu", dtype=torch.float32).numpy())
        if boundary_parts:
            self._calib_boundary.append(np.concatenate(boundary_parts))

    def _choose_top_m(self, routing_weights: torch.Tensor, rows: torch.Tensor, top_k: int) -> torch.Tensor:
        w = routing_weights[rows]
        margin12 = w[:, 0] - w[:, 1]
        b_small = (
            w[:, self.m_small - 1] - w[:, self.m_small]
            if self.m_small < top_k else torch.full_like(margin12, float("inf"))
        )
        b_mid = (
            w[:, self.m_mid - 1] - w[:, self.m_mid]
            if self.m_mid < top_k else torch.full_like(margin12, float("inf"))
        )
        choose_small = (margin12 >= self.high_thr) & (b_small >= self.boundary_thr)
        choose_mid = (~choose_small) & (margin12 >= self.medium_thr) & (b_mid >= self.boundary_thr)
        chosen = torch.full((w.shape[0],), self.m_large, device=w.device, dtype=torch.long)
        chosen[choose_mid] = self.m_mid
        chosen[choose_small] = self.m_small
        return chosen

    @staticmethod
    def _apply_dynamic_mask(
        routing_weights: torch.Tensor,
        rows: torch.Tensor,
        chosen_m: torch.Tensor,
        top_k: int,
    ) -> None:
        pos = torch.arange(top_k, device=routing_weights.device).unsqueeze(0)
        kill = pos >= chosen_m.unsqueeze(1)
        sub = routing_weights[rows]
        sub[kill] = 0.0
        routing_weights[rows] = sub

    def _record_stats(
        self,
        layer_idx: int,
        total_rows: int,
        rows: torch.Tensor,
        chosen_m: torch.Tensor,
        top_k: int,
    ) -> None:
        stats = self.context.stats
        ls = stats.layers.get(layer_idx)
        if ls is None:
            ls = _DynCallStats(layer_idx=layer_idx)
            stats.layers[layer_idx] = ls
        n_target = int(rows.numel())
        active_sum = int(chosen_m.sum().item())
        ls.tokens_total += total_rows
        ls.tokens_target += n_target
        ls.tokens_protected += total_rows - n_target
        ls.assignments_original += total_rows * top_k
        ls.assignments_effective += active_sum + (total_rows - n_target) * top_k
        ls.target_active_sum += active_sum
        for m in self.top_m_options:
            c = int((chosen_m == m).sum().item())
            if c:
                stats.top_m_histogram[m] = stats.top_m_histogram.get(m, 0) + c

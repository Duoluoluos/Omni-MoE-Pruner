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

from .compact_dispatch import compact_moe_forward
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
    compact_assignment_count_before: int = 0
    compact_assignment_count_after: int = 0
    compact_active_experts_before_sum: int = 0
    compact_active_experts_after_sum: int = 0
    compact_hot_to_mean_before_sum: float = 0.0
    compact_hot_to_mean_after_sum: float = 0.0
    compact_calls: int = 0
    compact_dispatch_time_ms: float = 0.0
    compact_expert_mlp_time_ms: float = 0.0
    compact_combine_time_ms: float = 0.0
    compact_debug_max_abs_diff: float = 0.0
    compact_debug_mean_abs_diff_sum: float = 0.0
    compact_debug_calls: int = 0
    audio_condition_calls: int = 0
    audio_condition_audio_tokens_sum: int = 0
    audio_condition_margin12_sum: float = 0.0
    audio_condition_boundary_sum: float = 0.0
    audio_condition_floor_sum: int = 0
    audio_condition_state_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class _DynSampleStats:
    sample_index: int | None = None
    task_type: str | None = None
    task_policy_cluster: str = "default"
    task_min_top_m: int | None = None
    task_layer_full_layers: list[int] = field(default_factory=list)
    query_guard_enabled: bool = False
    query_guard_cluster: str = "disabled"
    query_guard_matched_rules: list[str] = field(default_factory=list)
    query_guard_audio_tokens: int = 0
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
                "compact_assignment_count_before": ls.compact_assignment_count_before,
                "compact_assignment_count_after": ls.compact_assignment_count_after,
                "compact_active_experts_before_avg": (
                    ls.compact_active_experts_before_sum / ls.compact_calls if ls.compact_calls else 0.0
                ),
                "compact_active_experts_after_avg": (
                    ls.compact_active_experts_after_sum / ls.compact_calls if ls.compact_calls else 0.0
                ),
                "compact_hot_to_mean_before_avg": (
                    ls.compact_hot_to_mean_before_sum / ls.compact_calls if ls.compact_calls else 0.0
                ),
                "compact_hot_to_mean_after_avg": (
                    ls.compact_hot_to_mean_after_sum / ls.compact_calls if ls.compact_calls else 0.0
                ),
                "compact_dispatch_time_ms": ls.compact_dispatch_time_ms,
                "compact_expert_mlp_time_ms": ls.compact_expert_mlp_time_ms,
                "compact_combine_time_ms": ls.compact_combine_time_ms,
                "compact_debug_max_abs_diff": ls.compact_debug_max_abs_diff,
                "compact_debug_mean_abs_diff_avg": (
                    ls.compact_debug_mean_abs_diff_sum / ls.compact_debug_calls if ls.compact_debug_calls else 0.0
                ),
                "compact_debug_calls": ls.compact_debug_calls,
                "audio_condition_calls": ls.audio_condition_calls,
                "audio_condition_audio_tokens_avg": (
                    ls.audio_condition_audio_tokens_sum / ls.audio_condition_calls
                    if ls.audio_condition_calls else 0.0
                ),
                "audio_condition_margin12_mean": (
                    ls.audio_condition_margin12_sum / ls.audio_condition_calls if ls.audio_condition_calls else 0.0
                ),
                "audio_condition_boundary_mean": (
                    ls.audio_condition_boundary_sum / ls.audio_condition_calls if ls.audio_condition_calls else 0.0
                ),
                "audio_condition_floor_mean": (
                    ls.audio_condition_floor_sum / ls.audio_condition_calls if ls.audio_condition_calls else 0.0
                ),
                "audio_condition_state_counts": dict(sorted(ls.audio_condition_state_counts.items())),
            })
        hist = {str(k): int(self.top_m_histogram.get(k, 0)) for k in sorted(installer.top_m_options)}
        return {
            "enabled": bool(config.enabled),
            "mode": config.mode,
            "policy_version": config.policy_version,
            "compact_dispatch": bool(installer.compact_dispatch),
            "target_layers": sorted(self.target_layers),
            "layer_policy": config.layer_policy,
            "task_type": self.task_type,
            "task_policy_cluster": self.task_policy_cluster,
            "task_min_top_m": self.task_min_top_m,
            "task_layer_full_layers": list(self.task_layer_full_layers),
            "task_policy": config.task_policy,
            "task_layer_policy": config.task_layer_policy,
            "query_policy": config.query_policy,
            "audio_router_conditioning": config.audio_router_conditioning,
            "query_guard_enabled": bool(self.query_guard_enabled),
            "query_guard_cluster": self.query_guard_cluster,
            "query_guard_matched_rules": list(self.query_guard_matched_rules),
            "query_guard_audio_tokens": int(self.query_guard_audio_tokens),
            "top_m_options": list(installer.top_m_options),
            "original_top_k": config.original_top_k,
            "target_modality": config.target_modality,
            "protected_modalities": list(config.protected_modalities),
            "renormalize_weights": bool(config.renormalize_weights),
            "margin_thresholds": {
                "top1_top2_high": installer.high_thr,
                "top1_top2_medium": installer.medium_thr,
                "boundary": installer.boundary_thr,
                "local_prob": installer.local_prob_threshold if installer.local_prob_mode else None,
            },
            "threshold_granularity": installer.threshold_granularity,
            "per_layer_thresholds": {
                str(k): v for k, v in sorted(installer.per_layer_thresholds.items())
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
        self.task_layer_full_layers: set[int] = set()
        self.query_guard_enabled: bool = False
        self.query_guard_cluster: str = "disabled"
        self.query_guard_matched_rules: list[str] = []
        self.query_guard_audio_tokens: int = 0
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
        self.local_prob_mode = "local_prob" in str(config.mode or "").lower()
        self.local_prob_threshold = float(
            config.local_prob_threshold
            if config.local_prob_threshold is not None
            else (config.high_margin_threshold if config.high_margin_threshold is not None else 0.02)
        )
        self.threshold_granularity = str(getattr(config, "threshold_granularity", "global") or "global").lower()
        if self.threshold_granularity not in {"global", "per_layer"}:
            raise ValueError(f"Unknown threshold_granularity {self.threshold_granularity!r}")
        self.per_layer_thresholds: dict[int, dict[str, float]] = {
            int(k): {
                "top1_top2_high": float(v["top1_top2_high"]),
                "top1_top2_medium": float(v["top1_top2_medium"]),
                "boundary": float(v["boundary"]),
            }
            for k, v in (getattr(config, "per_layer_thresholds", {}) or {}).items()
            if all(name in v for name in ("top1_top2_high", "top1_top2_medium", "boundary"))
        }
        self._calibrating = False
        self._calib_margin12: list[np.ndarray] = []
        self._calib_boundary: list[np.ndarray] = []
        self._calib_margin12_by_layer: dict[int, list[np.ndarray]] = {}
        self._calib_boundary_by_layer: dict[int, list[np.ndarray]] = {}
        self._enabled = bool(self.config.enabled) and self.m_small < self.text_top_k
        self.compact_dispatch = bool(config.compact_dispatch) or "compact" in (config.mode or "").lower()
        self.audio_router_conditioning = getattr(config, "audio_router_conditioning", {}) or {}
        self.audio_conditioning_enabled = bool(self.audio_router_conditioning.get("enabled", False))
        self._compact_debug_calls = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def thresholds_ready(self) -> bool:
        if self.local_prob_mode:
            return True
        if self.threshold_granularity == "per_layer":
            if not self.target_layers:
                return True
            return all(idx in self.per_layer_thresholds for idx in self.target_layers)
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
        self._calib_margin12_by_layer = {}
        self._calib_boundary_by_layer = {}
        print("[expert_budget] (dynamic) calibration started", flush=True)

    def finish_calibration(self) -> dict[str, Any]:
        self._calibrating = False
        if self.threshold_granularity == "per_layer":
            return self._finish_per_layer_calibration()
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

    def _finish_per_layer_calibration(self) -> dict[str, Any]:
        per_layer: dict[str, dict[str, float]] = {}
        counts: dict[str, dict[str, int]] = {}
        fallback_layers: list[int] = []
        for layer_idx in sorted(self.target_layers):
            margin_parts = self._calib_margin12_by_layer.get(layer_idx, [])
            boundary_parts = self._calib_boundary_by_layer.get(layer_idx, [])
            if not margin_parts:
                fallback_layers.append(layer_idx)
                per_layer[str(layer_idx)] = {
                    "top1_top2_high": float(self.high_thr if self.high_thr is not None else 0.0),
                    "top1_top2_medium": float(self.medium_thr if self.medium_thr is not None else 0.0),
                    "boundary": float(self.boundary_thr if self.boundary_thr is not None else 0.0),
                }
                counts[str(layer_idx)] = {"margin": 0, "boundary": 0}
                continue
            margin12 = np.concatenate(margin_parts)
            boundary = np.concatenate(boundary_parts) if boundary_parts else np.array([], dtype=np.float32)
            boundary_source = boundary if boundary.size else margin12
            per_layer[str(layer_idx)] = {
                "top1_top2_high": float(np.percentile(margin12, self.config.high_margin_percentile)),
                "top1_top2_medium": float(np.percentile(margin12, self.config.medium_margin_percentile)),
                "boundary": float(np.percentile(boundary_source, self.config.boundary_margin_percentile)),
            }
            counts[str(layer_idx)] = {"margin": int(margin12.size), "boundary": int(boundary.size)}
        self.per_layer_thresholds = {int(k): v for k, v in per_layer.items()}
        self.config.per_layer_thresholds = per_layer
        report = self._threshold_report(
            n_margin=sum(v["margin"] for v in counts.values()),
            n_boundary=sum(v["boundary"] for v in counts.values()),
            fallback=bool(fallback_layers),
        )
        report["per_layer_thresholds"] = per_layer
        report["per_layer_counts"] = counts
        report["fallback_layers"] = fallback_layers
        self._calib_margin12 = []
        self._calib_boundary = []
        self._calib_margin12_by_layer = {}
        self._calib_boundary_by_layer = {}
        print(
            f"[expert_budget] (dynamic) calibrated per-layer thresholds for {len(per_layer)} layers",
            flush=True,
        )
        return report

    def _threshold_report(self, n_margin: int, n_boundary: int, fallback: bool = False) -> dict[str, Any]:
        return {
            "threshold_granularity": self.threshold_granularity,
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

    def begin_sample(
        self,
        sample_index: int | None,
        input_ids: torch.Tensor,
        task_type: str | None = None,
        query_text: str | None = None,
        sample_meta: dict[str, Any] | None = None,
    ) -> None:
        ids = input_ids[0].detach().cpu().numpy().astype(np.int64)
        modalities = build_modality_vector(ids, self.token_ids)
        target_mask_np = resolve_target_token_mask(modalities, self.target_code, self.protected_codes)
        device = input_ids.device
        audio_token_count = int((modalities == MODALITY_CODES["audio"]).sum())
        query_guard_enabled = self.config.query_policy_enabled()
        query_guard_cluster = "disabled"
        query_guard_matched_rules: list[str] = []
        if query_guard_enabled:
            task_min_top_m, query_guard_cluster, query_guard_matched_rules = self.config.query_min_top_m(
                query_text=query_text,
                audio_token_count=audio_token_count,
            )
            task_cluster = f"query:{query_guard_cluster}"
        else:
            task_min_top_m, task_cluster = self.config.task_min_top_m(task_type)
        task_min_top_m = min(max(int(task_min_top_m), self.m_small), self.text_top_k)
        task_layer_full_layers = (
            set()
            if query_guard_enabled
            else self.config.task_layer_full_layers(task_type, self.num_layers)
        )
        self.context.modalities = modalities
        self.context.target_token_mask = torch.from_numpy(target_mask_np).to(device=device)
        self.context.seq_len = int(ids.shape[0])
        self.context.sample_index = sample_index
        self.context.task_type = task_type
        self.context.task_policy_cluster = task_cluster
        self.context.task_min_top_m = task_min_top_m
        self.context.task_layer_full_layers = task_layer_full_layers
        self.context.query_guard_enabled = query_guard_enabled
        self.context.query_guard_cluster = query_guard_cluster
        self.context.query_guard_matched_rules = list(query_guard_matched_rules)
        self.context.query_guard_audio_tokens = audio_token_count
        self.context.stats = _DynSampleStats(
            sample_index=sample_index,
            task_type=task_type,
            task_policy_cluster=task_cluster,
            task_min_top_m=task_min_top_m,
            task_layer_full_layers=sorted(task_layer_full_layers),
            query_guard_enabled=query_guard_enabled,
            query_guard_cluster=query_guard_cluster,
            query_guard_matched_rules=list(query_guard_matched_rules),
            query_guard_audio_tokens=audio_token_count,
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
        self.context.task_layer_full_layers = set()
        self.context.query_guard_enabled = False
        self.context.query_guard_cluster = "disabled"
        self.context.query_guard_matched_rules = []
        self.context.query_guard_audio_tokens = 0
        self.context.stats = _DynSampleStats()
        return out

    def _target_mask_for_current_forward(self, token_count: int, device: torch.device) -> torch.Tensor | None:
        """Return the target-token mask matching the current MoE hidden length.

        OmniZip shortens the prefill hidden sequence before the text MoE layers.
        `begin_sample()` sees the original `input_ids`, so its modality mask must
        be compressed with OmniZip's `global_mask` before applying SCEB.
        """
        if self.context.modalities is None or self.context.target_token_mask is None:
            return None
        token_count = int(token_count)
        if token_count <= 0:
            return None

        target_mask = self.context.target_token_mask
        if token_count == int(self.context.seq_len) and target_mask.shape[0] == token_count:
            return target_mask.to(device=device) if target_mask.device != device else target_mask

        if token_count == 1 and not self.config.apply_during_decode:
            return None

        thinker = getattr(self.model, "thinker", None)
        global_mask = getattr(thinker, "_sceb_omnizip_global_mask", None)
        if global_mask is None:
            return None

        if isinstance(global_mask, torch.Tensor):
            mask_cpu = global_mask.detach().reshape(-1).to("cpu")
            if mask_cpu.dtype == torch.bool:
                keep_np = mask_cpu.numpy().astype(bool, copy=False)
            else:
                idx_np = mask_cpu.to(dtype=torch.long).numpy()
                keep_np = np.zeros(self.context.modalities.shape[0], dtype=bool)
                if idx_np.size == 0 or idx_np.max(initial=-1) >= keep_np.shape[0] or idx_np.min(initial=0) < 0:
                    return None
                keep_np[idx_np] = True
        else:
            keep_np = np.asarray(global_mask).reshape(-1)
            if keep_np.dtype != np.bool_:
                idx_np = keep_np.astype(np.int64, copy=False)
                keep_bool = np.zeros(self.context.modalities.shape[0], dtype=bool)
                if idx_np.size == 0 or idx_np.max(initial=-1) >= keep_bool.shape[0] or idx_np.min(initial=0) < 0:
                    return None
                keep_bool[idx_np] = True
                keep_np = keep_bool
            else:
                keep_np = keep_np.astype(bool, copy=False)

        if keep_np.shape[0] != self.context.modalities.shape[0]:
            return None
        if int(keep_np.sum()) != token_count:
            return None

        modalities = self.context.modalities[keep_np]
        target_mask_np = resolve_target_token_mask(modalities, self.target_code, self.protected_codes)
        self.context.modalities = modalities
        self.context.target_token_mask = torch.from_numpy(target_mask_np).to(device=device)
        self.context.seq_len = token_count
        return self.context.target_token_mask

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
                and layer_idx not in installer.context.task_layer_full_layers
                and installer.context.target_token_mask is not None
            )
            tgt_mask = None
            if apply_ctx:
                tgt_mask = installer._target_mask_for_current_forward(
                    routing_weights.shape[0],
                    routing_weights.device,
                )
                if tgt_mask is None or tgt_mask.shape[0] != routing_weights.shape[0]:
                    apply_ctx = False
                    tgt_mask = None
                elif tgt_mask.device != routing_weights.device:
                    tgt_mask = tgt_mask.to(routing_weights.device)
            if apply_ctx and not tgt_mask.any():
                apply_ctx = False

            did_mask = False
            keep_mask = None
            layer_stats = None
            if apply_ctx:
                rows = tgt_mask.nonzero(as_tuple=True)[0]
                if installer._calibrating:
                    installer._collect_calibration(layer_idx, routing_weights, rows, top_k)
                elif installer.thresholds_ready:
                    chosen_m = installer._choose_top_m(layer_idx, routing_weights, rows, top_k)
                    if installer.context.task_min_top_m is not None:
                        chosen_m = torch.clamp(chosen_m, min=int(installer.context.task_min_top_m), max=top_k)
                    audio_floor, audio_meta = installer._audio_condition_floor(
                        layer_idx=layer_idx,
                        routing_weights=routing_weights,
                        top_k=top_k,
                    )
                    if audio_floor is not None:
                        chosen_m = torch.clamp(chosen_m, min=int(audio_floor), max=top_k)
                    if installer.compact_dispatch:
                        keep_mask = torch.ones_like(selected_experts, dtype=torch.bool)
                        pos = torch.arange(top_k, device=routing_weights.device).unsqueeze(0)
                        keep_mask[rows] = pos < chosen_m.unsqueeze(1)
                    installer._apply_dynamic_mask(routing_weights, rows, chosen_m, top_k)
                    layer_stats = installer._record_stats(layer_idx, routing_weights.shape[0], rows, chosen_m, top_k)
                    if audio_meta is not None:
                        installer._record_audio_condition(layer_stats, audio_meta)
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

            if installer.compact_dispatch and did_mask and keep_mask is not None:
                final_hidden_states, compact_metrics = compact_moe_forward(
                    experts=module.experts,
                    flat_hidden_states=flat,
                    selected_experts=selected_experts,
                    routing_weights=routing_weights,
                    num_experts=num_experts,
                    keep_mask=keep_mask,
                )
                if layer_stats is not None:
                    installer._record_compact_metrics(layer_stats, compact_metrics)
                if (
                    layer_stats is not None
                    and installer.config.compact_debug_compare
                    and installer._compact_debug_calls < int(installer.config.compact_debug_max_calls)
                ):
                    reference = _mask_only_moe_forward(
                        experts=module.experts,
                        flat_hidden_states=flat,
                        selected_experts=selected_experts,
                        routing_weights=routing_weights,
                        num_experts=num_experts,
                    )
                    installer._record_compact_debug_diff(layer_stats, final_hidden_states, reference)
                    installer._compact_debug_calls += 1
            else:
                final_hidden_states = _mask_only_moe_forward(
                    experts=module.experts,
                    flat_hidden_states=flat,
                    selected_experts=selected_experts,
                    routing_weights=routing_weights,
                    num_experts=num_experts,
                )
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states, router_logits

        module.forward = wrapped_forward

    def _collect_calibration(
        self,
        layer_idx: int,
        routing_weights: torch.Tensor,
        rows: torch.Tensor,
        top_k: int,
    ) -> None:
        n = int(rows.numel())
        if n == 0:
            return
        if n > _CALIB_MAX_ROWS_PER_CALL:
            perm = torch.randperm(n, device=rows.device)[:_CALIB_MAX_ROWS_PER_CALL]
            rows = rows[perm]
        w = routing_weights[rows]
        margin12 = (w[:, 0] - w[:, 1]).detach().to("cpu", dtype=torch.float32).numpy()
        self._calib_margin12.append(margin12)
        if self.threshold_granularity == "per_layer":
            self._calib_margin12_by_layer.setdefault(layer_idx, []).append(margin12)
        boundary_parts = []
        for m in (self.m_small, self.m_mid):
            if m < top_k:
                boundary_parts.append((w[:, m - 1] - w[:, m]).detach().to("cpu", dtype=torch.float32).numpy())
        if boundary_parts:
            boundary = np.concatenate(boundary_parts)
            self._calib_boundary.append(boundary)
            if self.threshold_granularity == "per_layer":
                self._calib_boundary_by_layer.setdefault(layer_idx, []).append(boundary)

    def _choose_top_m(
        self,
        layer_idx: int,
        routing_weights: torch.Tensor,
        rows: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        w = routing_weights[rows]
        if self.local_prob_mode:
            keep_counts = (w >= self.local_prob_threshold).sum(dim=1).clamp(min=self.m_small, max=self.m_large)
            chosen = torch.full((w.shape[0],), self.m_large, device=w.device, dtype=torch.long)
            chosen[keep_counts <= self.m_mid] = self.m_mid
            chosen[keep_counts <= self.m_small] = self.m_small
            return chosen
        margin12 = w[:, 0] - w[:, 1]
        b_small = (
            w[:, self.m_small - 1] - w[:, self.m_small]
            if self.m_small < top_k else torch.full_like(margin12, float("inf"))
        )
        b_mid = (
            w[:, self.m_mid - 1] - w[:, self.m_mid]
            if self.m_mid < top_k else torch.full_like(margin12, float("inf"))
        )
        high_thr, medium_thr, boundary_thr = self._thresholds_for_layer(layer_idx)
        choose_small = (margin12 >= high_thr) & (b_small >= boundary_thr)
        choose_mid = (~choose_small) & (margin12 >= medium_thr) & (b_mid >= boundary_thr)
        chosen = torch.full((w.shape[0],), self.m_large, device=w.device, dtype=torch.long)
        chosen[choose_mid] = self.m_mid
        chosen[choose_small] = self.m_small
        return chosen

    def _audio_condition_floor(
        self,
        layer_idx: int,
        routing_weights: torch.Tensor,
        top_k: int,
    ) -> tuple[int | None, dict[str, Any] | None]:
        """Return an optional video top-m floor from same-layer audio routing."""
        spec = self.audio_router_conditioning or {}
        if not self.audio_conditioning_enabled:
            return None, None
        if self.target_code != MODALITY_CODES["video"]:
            return None, None
        modalities = self.context.modalities
        if modalities is None or len(modalities) != routing_weights.shape[0]:
            return None, None

        audio_np = modalities == MODALITY_CODES["audio"]
        audio_count = int(audio_np.sum())
        min_audio_tokens = int(spec.get("min_audio_tokens", 1))
        default_floor = spec.get("default_floor")
        if audio_count < min_audio_tokens:
            floor = spec.get("insufficient_audio_min_top_m", default_floor)
            meta = {
                "state": "insufficient_audio",
                "audio_tokens": audio_count,
                "margin12_mean": 0.0,
                "boundary_mean": 0.0,
                "floor": floor,
            }
            return (int(floor) if floor is not None else None), meta

        audio_rows = torch.from_numpy(np.flatnonzero(audio_np)).to(device=routing_weights.device, dtype=torch.long)
        w = routing_weights[audio_rows]
        margin12 = w[:, 0] - w[:, 1]
        boundary_top_m = int(spec.get("boundary_top_m", self.m_small))
        if 0 < boundary_top_m < top_k:
            boundary = w[:, boundary_top_m - 1] - w[:, boundary_top_m]
        else:
            boundary = torch.full_like(margin12, float("inf"))
        margin_mean = float(margin12.detach().float().mean().item()) if margin12.numel() else 0.0
        boundary_mean = float(boundary.detach().float().mean().item()) if boundary.numel() else 0.0

        fragile_margin = spec.get("fragile_margin_threshold")
        fragile_boundary = spec.get("fragile_boundary_threshold")
        stable_margin = spec.get("stable_margin_threshold")
        stable_boundary = spec.get("stable_boundary_threshold")

        fragile = False
        if fragile_margin is not None and margin_mean < float(fragile_margin):
            fragile = True
        if fragile_boundary is not None and boundary_mean < float(fragile_boundary):
            fragile = True

        stable = True
        if stable_margin is not None and margin_mean < float(stable_margin):
            stable = False
        if stable_boundary is not None and boundary_mean < float(stable_boundary):
            stable = False

        if fragile:
            state = "fragile"
            floor = spec.get("fragile_min_top_m", default_floor)
        elif stable and (stable_margin is not None or stable_boundary is not None):
            state = "stable"
            floor = spec.get("stable_min_top_m", default_floor)
        else:
            state = "neutral"
            floor = spec.get("neutral_min_top_m", default_floor)

        meta = {
            "state": state,
            "audio_tokens": audio_count,
            "margin12_mean": margin_mean,
            "boundary_mean": boundary_mean,
            "floor": floor,
        }
        return (int(floor) if floor is not None else None), meta

    def _thresholds_for_layer(self, layer_idx: int) -> tuple[float, float, float]:
        if self.threshold_granularity == "per_layer":
            layer_thr = self.per_layer_thresholds.get(layer_idx)
            if layer_thr is not None:
                return (
                    float(layer_thr["top1_top2_high"]),
                    float(layer_thr["top1_top2_medium"]),
                    float(layer_thr["boundary"]),
                )
        return (
            float(self.high_thr if self.high_thr is not None else 0.0),
            float(self.medium_thr if self.medium_thr is not None else 0.0),
            float(self.boundary_thr if self.boundary_thr is not None else 0.0),
        )

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
    ) -> _DynCallStats:
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
        return ls

    @staticmethod
    def _record_audio_condition(layer_stats: _DynCallStats, meta: dict[str, Any]) -> None:
        layer_stats.audio_condition_calls += 1
        layer_stats.audio_condition_audio_tokens_sum += int(meta.get("audio_tokens", 0) or 0)
        layer_stats.audio_condition_margin12_sum += float(meta.get("margin12_mean", 0.0) or 0.0)
        layer_stats.audio_condition_boundary_sum += float(meta.get("boundary_mean", 0.0) or 0.0)
        floor = meta.get("floor")
        if floor is not None:
            layer_stats.audio_condition_floor_sum += int(floor)
        state = str(meta.get("state", "unknown"))
        layer_stats.audio_condition_state_counts[state] = layer_stats.audio_condition_state_counts.get(state, 0) + 1

    @staticmethod
    def _record_compact_metrics(layer_stats: _DynCallStats, metrics: Any) -> None:
        layer_stats.compact_assignment_count_before += metrics.assignment_count_before
        layer_stats.compact_assignment_count_after += metrics.assignment_count_after
        layer_stats.compact_active_experts_before_sum += metrics.active_experts_before
        layer_stats.compact_active_experts_after_sum += metrics.active_experts_after
        layer_stats.compact_hot_to_mean_before_sum += metrics.hot_to_mean_before
        layer_stats.compact_hot_to_mean_after_sum += metrics.hot_to_mean_after
        layer_stats.compact_calls += 1
        layer_stats.compact_dispatch_time_ms += metrics.dispatch_time_ms
        layer_stats.compact_expert_mlp_time_ms += metrics.expert_mlp_time_ms
        layer_stats.compact_combine_time_ms += metrics.combine_time_ms

    @staticmethod
    def _record_compact_debug_diff(
        layer_stats: _DynCallStats,
        compact_output: torch.Tensor,
        reference_output: torch.Tensor,
    ) -> None:
        diff = (compact_output.detach() - reference_output.detach()).abs().float()
        layer_stats.compact_debug_max_abs_diff = max(
            layer_stats.compact_debug_max_abs_diff,
            float(diff.max().item()) if diff.numel() else 0.0,
        )
        layer_stats.compact_debug_mean_abs_diff_sum += float(diff.mean().item()) if diff.numel() else 0.0
        layer_stats.compact_debug_calls += 1


def _mask_only_moe_forward(
    *,
    experts: Any,
    flat_hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    num_tokens, hidden_dim = flat_hidden_states.shape
    final_hidden_states = torch.zeros(
        (num_tokens, hidden_dim),
        dtype=flat_hidden_states.dtype,
        device=flat_hidden_states.device,
    )
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        expert_layer = experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
        current_state = flat_hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(flat_hidden_states.dtype))
    return final_hidden_states

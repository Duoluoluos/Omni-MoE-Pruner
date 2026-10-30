#!/usr/bin/env python3
"""CPU/GPU parity smoke for the ARC-SCEB sparse TopKRouter patch.

The production plugin is imported through Megatron-SWIFT and is not importable
on a normal development host.  This test installs a deliberately small MCore
shim, then loads the real plugin file and compares its patched sparse path to
an independent frozen implementation of the historical dense mask path.

It covers all three ARC-SCEB budgets in packed input: stable audio -> top-4,
fragile audio -> top-6, and insufficient audio -> top-8.  It also reruns a
same-length second microbatch so static layout caching cannot accidentally
reuse positions from the preceding batch.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_PATH = ROOT / "scripts" / "megatron_arc_sceb_router.py"
TOP_K = 8
VIDEO_TOKEN_ID = 900
AUDIO_TOKEN_ID = 901
IMAGE_TOKEN_ID = 902


def _register_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


def _install_fake_mcore() -> type[torch.nn.Module]:
    """Supply exactly the import surface used by the production plugin."""
    megatron = _register_module("megatron")
    megatron.__path__ = []
    core = _register_module("megatron.core")
    core.__path__ = []
    transformer = _register_module("megatron.core.transformer")
    transformer.__path__ = []
    moe = _register_module("megatron.core.transformer.moe")
    moe.__path__ = []
    moe_layer_module = _register_module("megatron.core.transformer.moe.moe_layer")
    moe_utils_module = _register_module("megatron.core.transformer.moe.moe_utils")
    router_module = _register_module("megatron.core.transformer.moe.router")

    class _AttachAux(torch.autograd.Function):
        """Mimic MCore's identity-forward auxiliary-loss attachment."""

        @staticmethod
        def forward(ctx: Any, output: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
            ctx.save_for_backward(aux_loss)
            return output

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            (aux_loss,) = ctx.saved_tensors
            return grad_output, torch.ones_like(aux_loss)

    class FakeMoELayer(torch.nn.Module):
        pass

    class FakeTopKRouter(torch.nn.Module):
        def __init__(self, config: Any, layer_number: int = 20) -> None:
            super().__init__()
            self.config = config
            self.layer_number = layer_number
            self.topk = TOP_K
            self.routing_type = "aux_loss"
            self.score_function = "softmax"
            self.expert_bias = None
            self.router_replay = None
            self.training = bool(config.enable_aux)

        def apply_z_loss(
            self, logits: torch.Tensor, padding_mask: torch.Tensor | None = None
        ) -> torch.Tensor:
            del padding_mask
            return logits

        def is_aux_loss_enabled(self) -> bool:
            return bool(self.config.enable_aux)

        def _apply_aux_loss(
            self,
            probs: torch.Tensor,
            scores_for_aux_loss: torch.Tensor,
            routing_map: torch.Tensor,
            **_: Any,
        ) -> torch.Tensor:
            # The exact scalar is immaterial; it must depend on router scores
            # so the test detects a wrongly ordered aux-loss attachment.
            del routing_map
            return _AttachAux.apply(probs, scores_for_aux_loss.mean())

        def _apply_seq_aux_loss(self, probs: torch.Tensor, *_: Any, **__: Any) -> torch.Tensor:
            return probs

        def _apply_global_aux_loss(self, probs: torch.Tensor, *_: Any, **__: Any) -> torch.Tensor:
            return probs

        def _apply_expert_bias(
            self, routing_map: torch.Tensor, padding_mask: torch.Tensor | None = None
        ) -> None:
            del routing_map, padding_mask

        def routing(
            self, logits: torch.Tensor, padding_mask: torch.Tensor | None = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del padding_mask
            flat_logits = logits.view(-1, self.config.num_moe_experts)
            probs, routing_map = _topk_routing(flat_logits, self.topk, dense_output=False)
            if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled():
                aux_map, aux_scores = _compute_routing_scores_for_aux_loss(flat_logits, self.topk)
                probs = self._apply_aux_loss(probs, aux_scores, aux_map)
            return probs, routing_map

    def _topk_routing(
        logits: torch.Tensor,
        topk: int,
        *,
        dense_output: bool = False,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values, indices = torch.topk(logits, k=topk, dim=1, sorted=torch.is_grad_enabled())
        values = torch.softmax(values, dim=-1).type_as(logits)
        if dense_output:
            return values, indices
        probs = torch.zeros_like(logits).scatter(1, indices, values)
        routing_map = torch.zeros_like(logits).int().scatter(1, indices, 1).bool()
        return probs, routing_map

    def _compute_routing_scores_for_aux_loss(
        logits: torch.Tensor, topk: int, *_: Any, **__: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.softmax(logits, dim=-1)
        _, indices = torch.topk(scores, k=topk, dim=-1)
        routing_map = torch.zeros_like(logits).int().scatter(1, indices, 1).bool()
        return routing_map, scores

    moe_layer_module.MoELayer = FakeMoELayer
    moe_utils_module.topk_routing_with_score_function = _topk_routing
    moe_utils_module.compute_routing_scores_for_aux_loss = _compute_routing_scores_for_aux_loss
    router_module.TopKRouter = FakeTopKRouter

    mcore_bridge = _register_module("mcore_bridge")
    mcore_bridge.__path__ = []
    model = _register_module("mcore_bridge.model")
    model.__path__ = []
    mm_gpt_model = _register_module("mcore_bridge.model.mm_gpt_model")

    class FakeMultimodalGPTModel(torch.nn.Module):
        def forward(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            return None

    mm_gpt_model.MultimodalGPTModel = FakeMultimodalGPTModel
    return FakeTopKRouter


def _load_plugin() -> tuple[types.ModuleType, type[torch.nn.Module]]:
    fake_router = _install_fake_mcore()
    spec = importlib.util.spec_from_file_location("arc_sceb_router_under_test", PLUGIN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load plugin: {PLUGIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules while the module is
    # executing, so this registration is required before exec_module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, fake_router


def _frozen_legacy_mask(
    plugin: types.ModuleType,
    probs: torch.Tensor,
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Independent frozen oracle for the old dense post-router implementation."""
    flat_ids = input_ids.transpose(0, 1).contiguous().reshape(-1)
    positions = torch.arange(flat_ids.numel(), device=flat_ids.device)
    segment_ids = torch.bucketize(positions, cu_seqlens[1:], right=True)
    segment_count = int(cu_seqlens.numel() - 1)
    video_rows = flat_ids.eq(VIDEO_TOKEN_ID).nonzero(as_tuple=True)[0]
    audio_rows = flat_ids.eq(AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]
    if video_rows.numel() == 0:
        return probs

    top_values, top_indices = torch.topk(probs, k=TOP_K, dim=-1, sorted=True)
    target_values = top_values[video_rows]
    threshold = float(plugin._BOUNDARY_THRESHOLD)
    chosen_m = torch.full(
        (video_rows.numel(),), plugin._FULL_TOP_M, dtype=torch.long, device=probs.device
    )
    choose_small = target_values[:, plugin._SMALL_TOP_M - 1] - target_values[:, plugin._SMALL_TOP_M]
    choose_mid = target_values[:, plugin._MID_TOP_M - 1] - target_values[:, plugin._MID_TOP_M]
    small = choose_small >= threshold
    mid = (~small) & (choose_mid >= threshold)
    chosen_m[mid] = plugin._MID_TOP_M
    chosen_m[small] = plugin._SMALL_TOP_M

    audio_segment_ids = segment_ids[audio_rows]
    audio_count = torch.bincount(audio_segment_ids, minlength=segment_count)
    audio_margin = (top_values[audio_rows, 0] - top_values[audio_rows, 1]).detach().float()
    audio_margin_sum = torch.zeros(segment_count, dtype=torch.float32, device=probs.device)
    audio_margin_sum.scatter_add_(0, audio_segment_ids, audio_margin)
    audio_margin = audio_margin_sum / audio_count.clamp_min(1)
    video_segment_ids = segment_ids[video_rows]
    insufficient = audio_count[video_segment_ids] < int(plugin._ARC["min_audio_tokens"])
    fragile = (~insufficient) & (
        audio_margin[video_segment_ids] < float(plugin._ARC["fragile_margin_threshold"])
    )
    chosen_m = torch.where(
        insufficient,
        torch.full_like(chosen_m, int(plugin._ARC["insufficient_audio_min_top_m"])),
        chosen_m,
    )
    chosen_m = torch.where(
        fragile,
        chosen_m.clamp(min=int(plugin._ARC["fragile_min_top_m"]), max=plugin._FULL_TOP_M),
        chosen_m,
    )

    keep = torch.arange(TOP_K, device=probs.device).unsqueeze(0) < chosen_m.unsqueeze(1)
    kept = target_values * keep.to(dtype=target_values.dtype)
    kept = kept / kept.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    updated = top_values.clone()
    updated[video_rows] = kept
    return probs.scatter(1, top_indices, updated)


def _make_input(device: torch.device, alternate_layout: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Three packed segments: stable, fragile, then insufficient audio."""
    length = 240
    ids = torch.ones((1, length), dtype=torch.long, device=device)
    starts = (0, 80, 160)
    audio_lengths = (70, 70, 20)
    video_offsets = (70, 70, 20)
    if alternate_layout:
        # Same length and per-segment modality counts, different positions.
        video_offsets = (79, 79, 79)
    for start, audio_length, video_offset in zip(starts, audio_lengths, video_offsets, strict=True):
        ids[0, start : start + audio_length] = AUDIO_TOKEN_ID
        ids[0, start + video_offset] = VIDEO_TOKEN_ID
    cu_seqlens = torch.tensor([0, 80, 160, 240], dtype=torch.long, device=device)
    return ids, cu_seqlens


def _make_logits(device: torch.device, dtype: torch.dtype, alternate_layout: bool) -> torch.Tensor:
    torch.manual_seed(1234 if not alternate_layout else 5678)
    logits = torch.randn((240, 1, 128), device=device, dtype=dtype)
    ids, _ = _make_input(device, alternate_layout)
    flat_ids = ids.transpose(0, 1).reshape(-1)
    video_rows = flat_ids.eq(VIDEO_TOKEN_ID).nonzero(as_tuple=True)[0]
    audio_rows = flat_ids.eq(AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]
    logits_flat = logits[:, 0]

    # Video rows are safely inside the top-4 threshold; SCEB then changes the
    # latter two rows solely through fragile/insufficient audio safeguards.
    video_pattern = torch.tensor([8.0, 7.0, 6.0, 5.0, 1.0, 0.0, -1.0, -2.0], device=device, dtype=dtype)
    logits_flat[video_rows] = -12
    logits_flat[video_rows, :TOP_K] = video_pattern

    stable_audio = audio_rows[(audio_rows < 80)]
    fragile_audio = audio_rows[(audio_rows >= 80) & (audio_rows < 160)]
    logits_flat[stable_audio] = -12
    logits_flat[stable_audio, 0] = 8
    logits_flat[stable_audio, 1] = 0
    logits_flat[fragile_audio] = -12
    logits_flat[fragile_audio, 0] = 1.0
    logits_flat[fragile_audio, 1] = 0.999
    logits_flat[fragile_audio, 2:TOP_K] = -3
    return logits


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, *, dtype: torch.dtype) -> None:
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)


def _run_case(
    plugin: types.ModuleType,
    fake_router: type[torch.nn.Module],
    device: torch.device,
    dtype: torch.dtype,
    alternate_layout: bool,
    enable_aux: bool,
) -> None:
    input_ids, cu_seqlens = _make_input(device, alternate_layout)
    bridge = SimpleNamespace(
        hf_config=SimpleNamespace(
            thinker_config=SimpleNamespace(
                video_token_id=VIDEO_TOKEN_ID,
                audio_token_id=AUDIO_TOKEN_ID,
                image_token_id=IMAGE_TOKEN_ID,
            )
        )
    )
    plugin._capture_modalities(bridge, input_ids, SimpleNamespace(cu_seqlens_q=cu_seqlens))
    config = SimpleNamespace(
        num_moe_experts=128,
        num_layers=48,
        moe_router_pre_softmax=False,
        moe_router_num_groups=None,
        moe_router_group_topk=None,
        moe_router_topk_scaling_factor=None,
        moe_expert_capacity_factor=None,
        moe_router_fusion=False,
        enable_aux=enable_aux,
    )
    router = fake_router(config, layer_number=20)
    sparse_logits = _make_logits(device, dtype, alternate_layout).detach().requires_grad_(True)
    dense_logits = sparse_logits.detach().clone().requires_grad_(True)

    sparse_probs, sparse_map = router.routing(sparse_logits)
    dense_probs, dense_map = plugin._ORIGINAL_TOPK_ROUTING(router, dense_logits)
    expected = _frozen_legacy_mask(plugin, dense_probs, input_ids, cu_seqlens)
    _assert_close(sparse_probs, expected, dtype=dtype)
    assert torch.equal(sparse_map, dense_map), "ARC must never alter routing_map."

    flat_ids = input_ids.transpose(0, 1).reshape(-1)
    video_rows = flat_ids.eq(VIDEO_TOKEN_ID).nonzero(as_tuple=True)[0]
    protected_rows = ~flat_ids.eq(VIDEO_TOKEN_ID)
    _assert_close(sparse_probs[protected_rows], dense_probs[protected_rows], dtype=dtype)
    expected_budgets = (4, 6, 8)
    for row, budget in zip(video_rows.tolist(), expected_budgets, strict=True):
        assert int(sparse_map[row].sum()) == TOP_K
        assert int(torch.count_nonzero(sparse_probs[row])) == budget
        torch.testing.assert_close(
            sparse_probs[row].sum(),
            torch.ones((), device=device, dtype=dtype),
            rtol=1e-6 if dtype == torch.float32 else 3e-3,
            atol=1e-6 if dtype == torch.float32 else 3e-3,
        )

    torch.manual_seed(999)
    loss_weights = torch.randn_like(sparse_probs)
    (sparse_probs * loss_weights).sum().backward()
    (expected * loss_weights).sum().backward()
    assert sparse_logits.grad is not None and dense_logits.grad is not None
    _assert_close(sparse_logits.grad, dense_logits.grad, dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    dtype = getattr(torch, args.dtype)
    if device.type == "cpu" and dtype == torch.bfloat16:
        raise SystemExit("Use --dtype float32 for the CPU parity smoke.")

    plugin, fake_router = _load_plugin()
    _run_case(plugin, fake_router, device, dtype, alternate_layout=False, enable_aux=False)
    _run_case(plugin, fake_router, device, dtype, alternate_layout=True, enable_aux=False)
    _run_case(plugin, fake_router, device, dtype, alternate_layout=False, enable_aux=True)
    print(f"ARC-SCEB sparse-router parity passed ({device.type}, {dtype}).")


if __name__ == "__main__":
    main()

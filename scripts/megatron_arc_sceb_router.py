"""Apply the repository's ARC-SCEB mask-only policy to Megatron-Core training.

The original policy wraps HuggingFace Qwen3-Omni SparseMoeBlock inference
modules.  Megatron-SWIFT uses MCore's distributed ``TopKRouter`` instead, so
this plugin implements the same *mask-only* behavior at its router output:

* protect text, audio, image, and special tokens;
* dynamically retain top-4/top-6/top-8 only for video placeholders in the
  middle 25%--67% of the 48 text layers;
* use the original boundary-only SCEB rule; and
* use same-layer audio-router margin as the ARC floor (6 or 8).

The routing map intentionally remains unchanged.  That mirrors the existing
``compact_dispatch = false`` ARC-SCEB experiment: tail weights are zeroed and
the retained weights are renormalized, while MCore still executes the normal
EP token-dispatch path.  It is therefore a controlled *training-dynamics*
comparison, not a training-throughput optimization.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import (
    compute_routing_scores_for_aux_loss,
    topk_routing_with_score_function,
)
from megatron.core.transformer.moe.router import TopKRouter
from mcore_bridge.model.mm_gpt_model import MultimodalGPTModel


_POLICY_PATH = Path(
    os.environ.get(
        "ARC_SCEB_POLICY",
        Path(__file__).resolve().parents[1]
        / "moe_expert_budget"
        / "configs"
        / "arc_video_boundary_audio_floor6_full_v1.json",
    )
)
with _POLICY_PATH.open("r", encoding="utf-8") as _handle:
    _POLICY: dict[str, Any] = json.load(_handle)

_TOP_K = int(_POLICY["original_top_k"])
_SMALL_TOP_M, _MID_TOP_M, _FULL_TOP_M = sorted(int(v) for v in _POLICY["top_m_options"])
_BOUNDARY_THRESHOLD = float(_POLICY["thresholds"]["boundary_threshold"])
_LAYER_START_RATIO = float(_POLICY["layer_start_ratio"])
_LAYER_END_RATIO = float(_POLICY["layer_end_ratio"])
_ARC = dict(_POLICY["audio_router_conditioning"])

# ``dense`` is deliberately retained as an A/B benchmark control.  It uses
# the same non-fused MCore router configuration as ``sparse``, but applies the
# historical dense-probabilities mask after the upstream routing call.  Do not
# force this through router fusion/capacity settings: those change the recipe
# itself rather than only the ARC implementation.
_ROUTER_MODE = os.environ.get("ARC_SCEB_ROUTER_MODE", "sparse").strip().lower()
if _ROUTER_MODE not in {"sparse", "dense"}:
    raise RuntimeError(
        "ARC_SCEB_ROUTER_MODE must be either 'sparse' or 'dense', got "
        f"{_ROUTER_MODE!r}."
    )

# Production training must not repeatedly synchronize the GPU merely to print
# policy telemetry.  Keep one early health signal and make later summaries
# explicitly opt-in / sparse.  ``0`` means first summary only.
try:
    _DEBUG_EVERY = int(os.environ.get("ARC_SCEB_DEBUG_EVERY", "20000"))
except ValueError as exc:
    raise RuntimeError("ARC_SCEB_DEBUG_EVERY must be a non-negative integer.") from exc
if _DEBUG_EVERY < 0:
    raise RuntimeError("ARC_SCEB_DEBUG_EVERY must be a non-negative integer.")

# Qwen3-Omni IDs are fixed by the model config, but we resolve them from the
# visual bridge on each forward to avoid embedding model constants here.


def _is_rank_zero() -> bool:
    return os.environ.get("RANK", "0") == "0"


@dataclass
class _ArcScebContext:
    # MCore routers flatten logits from [sequence, batch, experts], whereas
    # input IDs arrive as [batch, sequence].  The following index tensors are
    # compiled once per packed microbatch in precisely that router order.  They
    # are static across all ARC layers (and activation recomputation) for the
    # same forward, so never rediscover them from input IDs in every router.
    token_count: int = 0
    segment_count: int = 0
    video_rows: torch.Tensor | None = None
    audio_rows: torch.Tensor | None = None
    video_segment_ids: torch.Tensor | None = None
    audio_segment_ids: torch.Tensor | None = None
    audio_count_by_segment: torch.Tensor | None = None
    topk_positions: torch.Tensor | None = None
    video_token_id: int | None = None
    audio_token_id: int | None = None
    image_token_id: int | None = None
    captures: int = 0
    forward_hook_calls: int = 0
    route_calls: int = 0
    applied_calls: int = 0
    skipped_shape_calls: int = 0


_CONTEXT = _ArcScebContext()


def _capture_modalities(
    bridge: Any,
    input_ids: torch.Tensor | None,
    packed_seq_params: Any | None,
) -> None:
    """Capture placeholder positions immediately before the text model forward."""
    if input_ids is None:
        return
    if input_ids.ndim != 2:
        raise RuntimeError(f"[ARC-SCEB] expected input_ids [batch, sequence], got {tuple(input_ids.shape)}")

    batch_size, sequence_length = input_ids.shape
    config = bridge.hf_config.thinker_config
    flat_input_ids = input_ids.detach().transpose(0, 1).contiguous().reshape(-1)

    cu_seqlens = getattr(packed_seq_params, "cu_seqlens_q", None)
    if cu_seqlens is None:
        # The normal (unpacked) layout is [sequence, batch] after flattening,
        # so sample indices are interleaved rather than contiguous.
        segment_ids = torch.arange(batch_size, device=input_ids.device).repeat(sequence_length)
        segment_count = batch_size
    else:
        if batch_size != 1:
            raise RuntimeError(
                "[ARC-SCEB] packed sequence mode expects micro_batch_size=1; "
                f"got input_ids shape={tuple(input_ids.shape)}"
            )
        cu_seqlens = cu_seqlens.detach().to(device=input_ids.device, dtype=torch.long).reshape(-1)
        if cu_seqlens.numel() < 2:
            raise RuntimeError("[ARC-SCEB] packed cu_seqlens_q must contain at least one segment.")
        # Fail early if MCore changes the packed layout.  This synchronizes only
        # on the first capture, not every training microbatch.
        if _CONTEXT.captures == 0 and (
            int(cu_seqlens[0].item()) != 0 or int(cu_seqlens[-1].item()) != flat_input_ids.numel()
        ):
            raise RuntimeError(
                "[ARC-SCEB] packed cu_seqlens_q does not span input_ids: "
                f"first={int(cu_seqlens[0].item())} last={int(cu_seqlens[-1].item())} "
                f"tokens={flat_input_ids.numel()}"
            )
        # Boundaries excluding zero assign positions [cu[i], cu[i + 1]) to
        # segment i.  ``right=True`` places an exact boundary in its next
        # sample, as required.
        positions = torch.arange(flat_input_ids.numel(), device=input_ids.device)
        segment_ids = torch.bucketize(positions, cu_seqlens[1:], right=True)
        segment_count = int(cu_seqlens.numel() - 1)

    _CONTEXT.video_token_id = int(config.video_token_id)
    _CONTEXT.audio_token_id = int(config.audio_token_id)
    _CONTEXT.image_token_id = int(config.image_token_id)
    video_rows = flat_input_ids.eq(_CONTEXT.video_token_id).nonzero(as_tuple=True)[0]
    audio_rows = flat_input_ids.eq(_CONTEXT.audio_token_id).nonzero(as_tuple=True)[0]

    # These tensors only encode the packed input layout, not router scores.
    # Caching them removes repeated eq/nonzero/gather/bincount work from every
    # ARC layer while preserving exactly the same per-sample audio condition.
    _CONTEXT.token_count = int(flat_input_ids.numel())
    _CONTEXT.segment_count = segment_count
    _CONTEXT.video_rows = video_rows
    _CONTEXT.audio_rows = audio_rows
    _CONTEXT.video_segment_ids = segment_ids[video_rows]
    _CONTEXT.audio_segment_ids = segment_ids[audio_rows]
    _CONTEXT.audio_count_by_segment = torch.bincount(
        _CONTEXT.audio_segment_ids, minlength=segment_count
    )
    _CONTEXT.topk_positions = torch.arange(_TOP_K, device=input_ids.device, dtype=torch.long)
    _CONTEXT.captures += 1
    if _is_rank_zero() and _CONTEXT.captures == 1:
        print(
            "[ARC-SCEB] captured input layout "
            f"input_ids={tuple(input_ids.shape)} tokens={flat_input_ids.numel()} "
            f"packed_segments={segment_count}",
            flush=True,
        )


_ORIGINAL_MULTIMODAL_FORWARD = MultimodalGPTModel.forward


def _arc_sceb_multimodal_forward(
    model: MultimodalGPTModel, *args: Any, **kwargs: Any
) -> Any:
    """Capture input IDs at the model forward boundary.

    Megatron-SWIFT can supply a prebuilt ``decoder_input`` during training,
    which legitimately bypasses ``_patch_word_embeddings``.  The model forward
    boundary always retains both the raw packed input IDs and
    ``packed_seq_params``, so it is the reliable policy capture point.
    """
    input_ids = kwargs.get("input_ids", args[0] if args else None)
    packed_seq_params = kwargs.get("packed_seq_params", args[6] if len(args) > 6 else None)
    bridge = getattr(model, "visual", None)
    config_holder = bridge if hasattr(getattr(bridge, "hf_config", None), "thinker_config") else model.config
    if hasattr(getattr(config_holder, "hf_config", None), "thinker_config"):
        _capture_modalities(config_holder, input_ids, packed_seq_params)
    _CONTEXT.forward_hook_calls += 1
    if _CONTEXT.forward_hook_calls == 1:
        print(
            "[ARC-SCEB] multimodal forward hook "
            f"rank={os.environ.get('RANK', 'unset')} model={type(model).__name__} "
            f"pre_process={getattr(model, 'pre_process', None)} "
            f"input_ids={'none' if input_ids is None else tuple(input_ids.shape)}",
            flush=True,
        )
    return _ORIGINAL_MULTIMODAL_FORWARD(model, *args, **kwargs)


def _middle_layer(moe_layer: Any) -> bool:
    """Return whether this MCore router corresponds to SCEB's middle layers."""
    num_layers = int(getattr(moe_layer.config, "num_layers", 0))
    layer_number = getattr(moe_layer, "layer_number", None)
    if num_layers <= 0 or layer_number is None:
        return False

    # Megatron assigns TransformerLayer.layer_number starting from one.  The
    # original HF policy uses zero-based layers [floor(.25L), floor(.67L)).
    layer_zero_based = int(layer_number) - 1
    start = int(num_layers * _LAYER_START_RATIO)
    end = int(num_layers * _LAYER_END_RATIO)
    return start <= layer_zero_based < end


def _move_layout_to_device(device: torch.device) -> _ArcScebContext:
    """Move cached static layout only on an unexpected router-device change."""
    for field_name in (
        "video_rows",
        "audio_rows",
        "video_segment_ids",
        "audio_segment_ids",
        "audio_count_by_segment",
        "topk_positions",
    ):
        value = getattr(_CONTEXT, field_name)
        if value is None:
            raise RuntimeError(f"[ARC-SCEB] missing cached layout field: {field_name}")
        if value.device != device:
            setattr(_CONTEXT, field_name, value.to(device=device))
    return _CONTEXT


def _require_context(
    moe_layer: Any, token_count: int, device: torch.device
) -> _ArcScebContext | None:
    if _CONTEXT.token_count <= 0 or _CONTEXT.video_rows is None or _CONTEXT.audio_rows is None:
        if _is_rank_zero() and _CONTEXT.skipped_shape_calls == 0:
            print("[ARC-SCEB] no input-id context yet; leaving this router call unchanged.", flush=True)
        _CONTEXT.skipped_shape_calls += 1
        return None
    if _CONTEXT.token_count != int(token_count):
        # This is deliberately fail-closed: silently masking mismatched tokens
        # would no longer be the video-only ARC policy.  With the requested
        # EP=2 / TP=1 / CP=1 recipe these values must agree exactly.
        raise RuntimeError(
            "[ARC-SCEB] router/input-id token mismatch: "
            f"router={token_count}, input_ids={_CONTEXT.token_count}, "
            f"layer={getattr(moe_layer, 'layer_number', None)}"
        )
    if _CONTEXT.segment_count <= 0:
        raise RuntimeError("[ARC-SCEB] no packed-sample segment context was captured.")
    return _move_layout_to_device(device)


def _mask_selected_router_probs(router: Any, top_values: torch.Tensor) -> torch.Tensor:
    """Mask sorted [tokens, top-k] router weights without changing assignments."""
    if int(getattr(router, "topk", -1)) != _TOP_K:
        raise RuntimeError(
            f"[ARC-SCEB] policy expects router top-k={_TOP_K}, got {getattr(router, 'topk', None)}"
        )

    context = _require_context(router, top_values.shape[0], top_values.device)
    if context is None:
        return top_values
    if _CONTEXT.video_token_id is None or _CONTEXT.audio_token_id is None:
        raise RuntimeError("[ARC-SCEB] Qwen3-Omni placeholder IDs were not captured.")

    target_rows = context.video_rows
    audio_rows = context.audio_rows
    audio_segment_ids = context.audio_segment_ids
    video_segment_ids = context.video_segment_ids
    audio_count_by_segment = context.audio_count_by_segment
    positions = context.topk_positions
    assert target_rows is not None
    assert audio_rows is not None
    assert audio_segment_ids is not None
    assert video_segment_ids is not None
    assert audio_count_by_segment is not None
    assert positions is not None
    if target_rows.numel() == 0:
        return top_values
    target_values = top_values[target_rows]

    boundary_small = target_values[:, _SMALL_TOP_M - 1] - target_values[:, _SMALL_TOP_M]
    boundary_mid = target_values[:, _MID_TOP_M - 1] - target_values[:, _MID_TOP_M]
    chosen_m = torch.full(
        (target_rows.numel(),), _FULL_TOP_M, dtype=torch.long, device=top_values.device
    )
    choose_small = boundary_small >= _BOUNDARY_THRESHOLD
    choose_mid = (~choose_small) & (boundary_mid >= _BOUNDARY_THRESHOLD)
    chosen_m[choose_mid] = _MID_TOP_M
    chosen_m[choose_small] = _SMALL_TOP_M

    # ARC: condition video budgets on same-layer audio router confidence while
    # keeping audio token expert assignments themselves untouched.  When
    # packing is enabled, do this separately for every original sample.  A
    # sample with many stable audio tokens must not weaken the top-8 floor for
    # a different packed sample that has little or fragile audio context.
    # Keep the routing path asynchronous: the policy can be expressed entirely
    # as tensor operations.  We only transfer counts to Python on sparse debug
    # prints, rather than synchronizing every dynamic layer.
    audio_margin_by_token = (
        top_values[audio_rows, 0] - top_values[audio_rows, 1]
    ).detach().float()
    audio_margin_sum_by_segment = torch.zeros(
        _CONTEXT.segment_count, dtype=torch.float32, device=top_values.device
    )
    audio_margin_sum_by_segment.scatter_add_(
        0, audio_segment_ids, audio_margin_by_token
    )
    audio_margin_by_segment = audio_margin_sum_by_segment / audio_count_by_segment.clamp_min(1)
    video_audio_count = audio_count_by_segment[video_segment_ids]
    video_audio_margin = audio_margin_by_segment[video_segment_ids]
    insufficient_audio = video_audio_count < int(_ARC["min_audio_tokens"])
    fragile_audio = (~insufficient_audio) & (
        video_audio_margin < float(_ARC["fragile_margin_threshold"])
    )
    chosen_m = torch.where(
        insufficient_audio,
        torch.full_like(chosen_m, int(_ARC["insufficient_audio_min_top_m"])),
        chosen_m,
    )
    chosen_m = torch.where(
        fragile_audio,
        chosen_m.clamp(min=int(_ARC["fragile_min_top_m"]), max=_FULL_TOP_M),
        chosen_m,
    )

    keep = positions.unsqueeze(0) < chosen_m.unsqueeze(1)
    kept_values = target_values * keep.to(dtype=target_values.dtype)
    if bool(_POLICY["renormalize_weights"]):
        kept_values = kept_values / kept_values.sum(dim=-1, keepdim=True).clamp_min(1e-20)

    updated_top_values = top_values.clone()
    updated_top_values[target_rows] = kept_values

    _CONTEXT.applied_calls += 1
    if _is_rank_zero() and (
        _CONTEXT.applied_calls == 1
        or (_DEBUG_EVERY > 0 and _CONTEXT.applied_calls % _DEBUG_EVERY == 0)
    ):
        current_hist = {
            budget: int((chosen_m == budget).sum().item())
            for budget in (_SMALL_TOP_M, _MID_TOP_M, _FULL_TOP_M)
        }
        print(
            "[ARC-SCEB] applied "
            f"layer={getattr(router, 'layer_number', None)} tokens={top_values.shape[0]} "
            f"video={target_rows.numel()} audio={audio_rows.numel()} "
            f"packed_segments={_CONTEXT.segment_count} "
            f"audio_floor8_video={int(insufficient_audio.sum().item())} "
            f"audio_floor6_video={int(fragile_audio.sum().item())} "
            f"top_m={current_hist}",
            flush=True,
        )
    return updated_top_values


def _mask_dense_router_probs(router: Any, probs: torch.Tensor) -> torch.Tensor:
    """Legacy dense ARC path, retained as a parity fallback for unsupported routers."""
    if not _middle_layer(router):
        return probs
    top_values, top_indices = torch.topk(probs, k=_TOP_K, dim=-1, sorted=True)
    masked_top_values = _mask_selected_router_probs(router, top_values)
    return probs.scatter(1, top_indices, masked_top_values)


def _build_routing_map(logits: torch.Tensor, top_indices: torch.Tensor) -> torch.Tensor:
    """Build MCore's dense bool dispatch map from its already-selected indices."""
    if torch.are_deterministic_algorithms_enabled():
        routing_map = torch.zeros_like(logits, dtype=logits.dtype)
        rows = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1)
        routing_map.index_put_(
            (rows, top_indices),
            torch.ones_like(top_indices, dtype=routing_map.dtype),
            accumulate=False,
        )
        return routing_map.bool()
    return torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()


def _scatter_selected_probs(
    logits: torch.Tensor, top_indices: torch.Tensor, top_values: torch.Tensor
) -> torch.Tensor:
    """Use MCore's original dense layout only after ARC has changed top-k weights."""
    if torch.are_deterministic_algorithms_enabled():
        routing_probs = torch.zeros_like(logits)
        rows = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1)
        routing_probs.index_put_((rows, top_indices), top_values, accumulate=False)
        return routing_probs
    return torch.zeros_like(logits).scatter(1, top_indices, top_values)


def _sparse_arc_supported(router: TopKRouter) -> bool:
    """Whether this MCore router can preserve exact semantics on the sparse path."""
    config = router.config
    return (
        int(getattr(router, "topk", -1)) == _TOP_K
        and getattr(router, "routing_type", None) != "sinkhorn"
        and getattr(config, "moe_expert_capacity_factor", None) is None
        # Current TE fused top-k does not expose the selected values/indices
        # contract needed here.  Fall back to the exact legacy path instead of
        # guessing at its return layout.
        and not bool(getattr(config, "moe_router_fusion", False))
    )


def _sort_sparse_topk(
    top_values: torch.Tensor, top_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the legacy ``torch.topk(..., sorted=True)`` rank convention.

    MCore deliberately disables sorting for some no-grad paths.  ARC's
    boundary rule is rank-sensitive, whereas the old dense wrapper always
    recovered a sorted top-k.  Sorting only the already-selected eight values
    retains that behavior without revisiting every expert.
    """
    sorted_values, rank_order = torch.topk(top_values, k=_TOP_K, dim=-1, sorted=True)
    return sorted_values, torch.gather(top_indices, dim=1, index=rank_order)


def _arc_sceb_sparse_topk_routing(
    router: TopKRouter, logits: torch.Tensor, padding_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """MCore TopKRouter.routing with ARC applied before dense probs expansion.

    The only semantic change from the upstream method is that ARC masks the
    selected video-token weights before the sole dense probabilities scatter.
    The selected expert indices and dense routing_map follow the upstream path
    unchanged.  The auxiliary-loss autograd attachment is deliberately kept
    before ARC, matching the historical dense wrapper.
    """
    seq_length, batch_size = logits.shape[:2]
    logits = logits.view(-1, router.config.num_moe_experts)
    if padding_mask is not None:
        padding_mask = padding_mask.reshape(-1)

    logits = router.apply_z_loss(logits, padding_mask=padding_mask)
    top_values, top_indices = topk_routing_with_score_function(
        logits,
        router.topk,
        use_pre_softmax=router.config.moe_router_pre_softmax,
        num_groups=router.config.moe_router_num_groups,
        group_topk=router.config.moe_router_group_topk,
        scaling_factor=router.config.moe_router_topk_scaling_factor,
        score_function=router.score_function,
        expert_bias=router.expert_bias,
        fused=False,
        router_replay=router.router_replay,
        dense_output=True,
    )
    top_values, top_indices = _sort_sparse_topk(top_values, top_indices)
    routing_map = _build_routing_map(logits, top_indices)

    # This is copied in operation order from MCore's TopKRouter.routing.  The
    # aux-loss scaler is shape-agnostic, so attaching it to [tokens, top-k]
    # before the final scatter gives the same backward contribution as first
    # attaching it to sparse-in-dense [tokens, experts] probabilities.
    if router.training and torch.is_grad_enabled() and router.is_aux_loss_enabled():
        routing_map_for_aux_loss, scores_for_aux_loss = compute_routing_scores_for_aux_loss(
            logits,
            router.topk,
            router.score_function,
            fused=False,
            padding_mask=padding_mask,
        )
        top_values = router._apply_aux_loss(
            top_values,
            scores_for_aux_loss,
            routing_map_for_aux_loss,
            with_padding_mask=padding_mask is not None,
        )
        top_values = router._apply_seq_aux_loss(
            top_values,
            scores_for_aux_loss,
            routing_map_for_aux_loss,
            seq_length,
            batch_size,
            with_padding_mask=padding_mask is not None,
        )
        top_values = router._apply_global_aux_loss(
            top_values,
            scores_for_aux_loss,
            routing_map_for_aux_loss,
            with_padding_mask=padding_mask is not None,
        )

    router._apply_expert_bias(routing_map, padding_mask=padding_mask)
    top_values = _mask_selected_router_probs(router, top_values)
    return _scatter_selected_probs(logits, top_indices, top_values), routing_map


def _note_router_hook(router: TopKRouter, token_count: int) -> None:
    _CONTEXT.route_calls += 1
    if _CONTEXT.route_calls == 1:
        print(
            "[ARC-SCEB] router hook "
            f"rank={os.environ.get('RANK', 'unset')} class={type(router).__name__} "
            f"layer={getattr(router, 'layer_number', None)} "
            f"num_layers={getattr(router.config, 'num_layers', None)} "
            f"tokens={token_count}",
            flush=True,
        )


_ORIGINAL_TOPK_ROUTING = TopKRouter.routing


def _arc_sceb_topk_routing(
    router: TopKRouter, logits: torch.Tensor, padding_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ARC in MCore's sparse top-k representation when the router allows it."""
    token_count = logits.numel() // int(router.config.num_moe_experts)
    _note_router_hook(router, token_count)
    if not _middle_layer(router):
        return _ORIGINAL_TOPK_ROUTING(router, logits, padding_mask=padding_mask)

    context = _require_context(router, token_count, logits.device)
    if context is None:
        return _ORIGINAL_TOPK_ROUTING(router, logits, padding_mask=padding_mask)

    if _ROUTER_MODE == "sparse" and _sparse_arc_supported(router):
        return _arc_sceb_sparse_topk_routing(router, logits, padding_mask=padding_mask)

    # Unsupported router features (for example capacity token dropping) stay
    # on the original MCore implementation and retain the old dense masking
    # path.  This is slower but exact and keeps the plugin version-safe.
    probs, routing_map = _ORIGINAL_TOPK_ROUTING(router, logits, padding_mask=padding_mask)
    return _mask_dense_router_probs(router, probs), routing_map


if not getattr(TopKRouter, "_arc_sceb_training_patch", False):
    MultimodalGPTModel.forward = _arc_sceb_multimodal_forward
    TopKRouter.routing = _arc_sceb_topk_routing
    TopKRouter._arc_sceb_training_patch = True
    print(
        "[ARC-SCEB] installed sparse MCore mask-only router patch "
        f"policy={_POLICY.get('policy_version')} mode={_ROUTER_MODE} "
        f"debug_every={_DEBUG_EVERY} path={_POLICY_PATH}",
        flush=True,
    )

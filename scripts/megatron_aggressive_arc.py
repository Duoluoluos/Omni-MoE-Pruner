"""Aggressive ARC-SCEB: per-layer *static* top-k reduction for real speedup.

Motivation
----------
The mask-only ARC-SCEB is a deployment quality tool with ~0% training speedup
(it only zeros routing weights; the dispatcher still processes top-8 per token).
The removed compact-dispatch tried per-token variable-length dispatch and was
*slower* because it forced the dropless static ``num_out_tokens`` optimization
off, triggering a GPU->CPU sync every MoE layer.

The only mechanism that actually speeds MoE up on this recipe is a **static**
top-k reduction: MCore's alltoall dispatcher sizes the grouped-GEMM as
``num_out_tokens = num_tokens * config.moe_router_topk`` -- a compile-time
scalar with no sync (token_dispatcher.py, dropless branch).  The global-topk
probe proved this: top-8 -> top-4 gave a measured 24.9% end-to-end speedup.

This plugin keeps that sync-free static property but makes the reduction
**layer-selective**, mirroring ARC-SCEB's key insight that video information is
concentrated in the middle layers.  Concretely it lowers top-k only for the
middle layer band ``[start, end)`` and leaves the semantic early layers and the
output-facing late layers at the native top-k.  Cutting only ~half the layers
(instead of all 48) buys most of the speedup at a fraction of the quality cost:

    effective avg top-k = ((L - n_mid) * native + n_mid * mid_topk) / L

With native=8, L=48, band [10,34) (24 layers), mid_topk=4:
    eff_avg = (24*8 + 24*4)/48 = 6.0  =>  ~12% end-to-end (KPI target 10%).

Because static top-k can only be set *per layer* (the dispatcher reads one
scalar off the layer's config), true per-token modality gradients would require
variable-length dispatch and reintroduce the sync penalty.  We therefore
approximate the "modality gradient" via **layer depth**: the middle band is
where video routing dominates (see the interpretability study), so restricting
cuts to it implicitly protects the text/audio processing that lives in the
early/late layers.

Mechanism
---------
``TransformerConfig`` is a (non-frozen) dataclass shared by all layers, and both
the router (``self.topk`` copied at init) and the alltoall dispatcher
(``self.config.moe_router_topk`` read at forward) consult it.  MCore constructs
``MoELayer`` with no layer number and injects the real number afterwards through
``set_layer_number``.  We patch both lifecycle points and, once the number is
known, install a shallow-copied config carrying the reduced
``moe_router_topk`` onto the layer, its router, and its token dispatcher, and set
``router.topk`` to match.  Every consumer then sees a consistent, static reduced
top-k for that layer only.

Config (environment variables)
------------------------------
``AGGR_ARC_MID_TOPK``     reduced top-k for the middle band (default 0, transparent; set 4 for the aggressive recipe).
``AGGR_ARC_LAYER_START``  middle band start layer, zero-based inclusive (default 10).
``AGGR_ARC_LAYER_END``    middle band end layer, zero-based exclusive (default 34).

Load it as an external plugin exactly like the ARC-SCEB router patch.  With
``AGGR_ARC_MID_TOPK=0`` (default) it is fully transparent.
"""

from __future__ import annotations

import copy
import os
from typing import Any

from megatron.core.transformer.moe.moe_layer import MoELayer


def _is_rank_zero() -> bool:
    return os.environ.get("RANK", "0") == "0"


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive config guard
        raise RuntimeError(f"{name} must be an integer, got {raw!r}.") from exc


_MID_TOPK = _int_env("AGGR_ARC_MID_TOPK", 0)
_LAYER_START = _int_env("AGGR_ARC_LAYER_START", 10)
_LAYER_END = _int_env("AGGR_ARC_LAYER_END", 34)

if _MID_TOPK < 0:
    raise RuntimeError("AGGR_ARC_MID_TOPK must be a non-negative integer.")
if _LAYER_START < 0 or _LAYER_END < _LAYER_START:
    raise RuntimeError(
        f"AGGR_ARC layer band invalid: start={_LAYER_START} end={_LAYER_END}."
    )

# ``0`` (default) means transparent: leave native top-k everywhere.
_ENABLED = _MID_TOPK > 0

# Preserve the pristine constructor across re-imports / importlib.reload so the
# wrapper never wraps itself (infinite recursion guard).
if not hasattr(MoELayer, "_aggr_arc_orig_init"):
    MoELayer._aggr_arc_orig_init = MoELayer.__init__
_ORIGINAL_MOE_LAYER_INIT = MoELayer._aggr_arc_orig_init

# ``TransformerLayer`` constructs its MLP without forwarding ``layer_number``
# and injects the real number afterwards through ``set_layer_number``.  Saving
# that pristine method is therefore just as important as saving ``__init__``:
# patching only the constructor sees ``None`` for every layer and silently
# leaves the whole model at native top-k.
if not hasattr(MoELayer, "_aggr_arc_orig_set_layer_number"):
    MoELayer._aggr_arc_orig_set_layer_number = MoELayer.set_layer_number
_ORIGINAL_MOE_LAYER_SET_LAYER_NUMBER = MoELayer._aggr_arc_orig_set_layer_number


def _middle_layer(layer_number: int | None) -> bool:
    """Return whether a 1-based MCore layer_number falls in the reduced band."""
    if layer_number is None:
        return False
    # MCore assigns TransformerLayer.layer_number starting at 1; the band is
    # expressed zero-based like the ARC-SCEB policy ([start, end)).
    zero_based = int(layer_number) - 1
    return _LAYER_START <= zero_based < _LAYER_END


def _install_layer_topk(self: MoELayer) -> None:
    """Install the static per-layer config once the real layer number is known."""
    layer_number = getattr(self, "layer_number", None)
    if not _middle_layer(layer_number):
        return

    native_topk = int(self.config.moe_router_topk)
    if _MID_TOPK > native_topk:
        raise RuntimeError(
            f"[AGGR-ARC] AGGR_ARC_MID_TOPK={_MID_TOPK} exceeds native "
            f"moe_router_topk={native_topk}; refusing to add experts."
        )
    if _MID_TOPK == native_topk:
        return

    # One shallow copy shared by this layer's router + dispatcher keeps every
    # consumer consistent (router selection AND dispatcher num_out_tokens) while
    # leaving all other layers' shared config untouched.  Shallow copy is safe:
    # we only rebind the scalar ``moe_router_topk`` on the copy.
    layer_config = copy.copy(self.config)
    layer_config.moe_router_topk = _MID_TOPK
    self.config = layer_config
    if getattr(self, "router", None) is not None:
        self.router.config = layer_config
        # ``TopKRouter.__init__`` cached ``self.topk`` from the old config.
        self.router.topk = _MID_TOPK
    if getattr(self, "token_dispatcher", None) is not None:
        self.token_dispatcher.config = layer_config
        # The alltoall dispatcher also caches router_topk in some code paths.
        if hasattr(self.token_dispatcher, "router_topk"):
            self.token_dispatcher.router_topk = _MID_TOPK

    if _is_rank_zero() and not getattr(MoELayer, "_aggr_arc_logged", False):
        print(
            f"[AGGR-ARC] middle-band top-k reduced {native_topk} -> {_MID_TOPK} "
            f"for layers [{_LAYER_START}, {_LAYER_END}) "
            f"(first hit layer_number={layer_number}); "
            f"early/late layers keep top-{native_topk}",
            flush=True,
        )
        MoELayer._aggr_arc_logged = True


def _aggr_arc_moe_layer_init(self: MoELayer, *args: Any, **kwargs: Any) -> None:
    """Handle callers that do pass ``layer_number`` directly at construction."""
    _ORIGINAL_MOE_LAYER_INIT(self, *args, **kwargs)
    _install_layer_topk(self)


def _aggr_arc_set_layer_number(self: MoELayer, layer_number: int) -> None:
    """Apply the layer policy at MCore's actual post-construction injection point."""
    _ORIGINAL_MOE_LAYER_SET_LAYER_NUMBER(self, layer_number)
    _install_layer_topk(self)


if _ENABLED and not getattr(MoELayer, "_aggr_arc_patch", False):
    MoELayer.__init__ = _aggr_arc_moe_layer_init
    MoELayer.set_layer_number = _aggr_arc_set_layer_number
    MoELayer._aggr_arc_patch = True
    if _is_rank_zero():
        print(
            f"[AGGR-ARC] installed aggressive per-layer static top-k patch "
            f"mid_topk={_MID_TOPK} band=[{_LAYER_START},{_LAYER_END})",
            flush=True,
        )
elif not _ENABLED and _is_rank_zero():
    print(
        "[AGGR-ARC] AGGR_ARC_MID_TOPK unset/0; leaving native router top-k unchanged.",
        flush=True,
    )

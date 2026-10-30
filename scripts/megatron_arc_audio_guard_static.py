"""ARC audio-router-confidence protected, static per-layer MoE top-k.

This is the throughput-oriented counterpart to ``megatron_arc_sceb_router``.
The original mask-only plugin applies ARC's audio confidence guard online but
still dispatches top-8 experts, so it cannot save expert FFN compute.  Making
that decision variable per sample would force Megatron's synchronized dynamic
dispatch path and was measured slower than native.

Here the exact same audio signal is calibrated offline on real profiles.  The
middle layers where ARC's audio guard fires most often receive static top-6;
the remaining original ARC middle layers receive static top-4.  Early and late
layers retain native top-8.  One scalar top-k per layer preserves Megatron's
dropless dispatcher and therefore turns the budget into real speedup.

The schedule and its profile evidence live in
``moe_expert_budget/configs/arc_audio_guard_static_topk_v1.json``.  Set
``ARC_AUDIO_GUARD_POLICY`` to use another derived policy.  Merely importing
this external plugin enables it; do not load it in the native benchmark arm.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from megatron.core.transformer.moe.moe_layer import MoELayer


_DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "moe_expert_budget/configs/arc_audio_guard_static_topk_v1.json"
)
_POLICY_PATH = Path(os.environ.get("ARC_AUDIO_GUARD_POLICY", _DEFAULT_POLICY_PATH))


def _is_rank_zero() -> bool:
    return os.environ.get("RANK", "0") == "0"


def _load_and_validate_policy(path: Path) -> tuple[dict[str, Any], dict[int, int]]:
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"[ARC-AUDIO-GUARD] cannot load policy {path}: {exc}") from exc
    if policy.get("mode") != "static_per_layer_topk":
        raise RuntimeError("[ARC-AUDIO-GUARD] policy mode must be static_per_layer_topk")
    audio = policy.get("audio_router_confidence_protection") or {}
    if not audio.get("enabled"):
        raise RuntimeError("[ARC-AUDIO-GUARD] audio confidence protection must be enabled")
    start = int(policy["middle_layer_start"])
    end = int(policy["middle_layer_end"])
    total_layers = int(policy["model_num_layers"])
    native_top_k = int(policy["original_top_k"])
    if not (0 <= start < end <= total_layers):
        raise RuntimeError(
            f"[ARC-AUDIO-GUARD] invalid middle window [{start},{end})/{total_layers}"
        )
    raw_schedule = policy.get("layer_top_k") or {}
    schedule = {int(layer): int(top_k) for layer, top_k in raw_schedule.items()}
    expected_layers = set(range(start, end))
    if set(schedule) != expected_layers:
        missing = sorted(expected_layers - set(schedule))
        outside = sorted(set(schedule) - expected_layers)
        raise RuntimeError(
            "[ARC-AUDIO-GUARD] schedule must cover exactly the middle window; "
            f"missing={missing}, outside={outside}"
        )
    invalid = {
        layer: top_k
        for layer, top_k in schedule.items()
        if not (1 <= top_k <= native_top_k)
    }
    if invalid:
        raise RuntimeError(
            f"[ARC-AUDIO-GUARD] invalid/increasing layer top-k values: {invalid}"
        )
    protected_top_k = int(audio["fragile_audio_target_top_k"])
    protected = {int(layer) for layer in audio["protected_layers_ranked"]}
    actual_protected = {layer for layer, top_k in schedule.items() if top_k == protected_top_k}
    if protected != actual_protected:
        raise RuntimeError(
            "[ARC-AUDIO-GUARD] protected layer list does not match the top-k schedule: "
            f"declared={sorted(protected)} actual={sorted(actual_protected)}"
        )
    speed = policy.get("speed_model") or {}
    if float(speed.get("predicted_speedup_percent", 0.0)) < float(
        speed.get("target_speedup_percent", 0.0)
    ):
        raise RuntimeError("[ARC-AUDIO-GUARD] policy speed model misses its own target")
    guards = policy.get("runtime_guards") or {}
    if not guards.get("only_middle_layers_may_change"):
        raise RuntimeError("[ARC-AUDIO-GUARD] only-middle-layers guard must be enabled")
    if guards.get("dispatcher_mode") != "static_dropless":
        raise RuntimeError("[ARC-AUDIO-GUARD] dynamic dispatcher policies are forbidden")
    return policy, schedule


_POLICY, _LAYER_TOP_K = _load_and_validate_policy(_POLICY_PATH)
_NATIVE_TOP_K = int(_POLICY["original_top_k"])
_MIDDLE_START = int(_POLICY["middle_layer_start"])
_MIDDLE_END = int(_POLICY["middle_layer_end"])

# MCore builds MoELayer with ``layer_number=None`` and injects its real 1-based
# number through set_layer_number later.  Preserve and patch both lifecycle
# methods.  Class-level pristine methods also make importlib reload safe.
if not hasattr(MoELayer, "_arc_audio_guard_orig_init"):
    MoELayer._arc_audio_guard_orig_init = MoELayer.__init__
_ORIGINAL_MOE_LAYER_INIT = MoELayer._arc_audio_guard_orig_init

if not hasattr(MoELayer, "_arc_audio_guard_orig_set_layer_number"):
    MoELayer._arc_audio_guard_orig_set_layer_number = MoELayer.set_layer_number
_ORIGINAL_MOE_LAYER_SET_LAYER_NUMBER = MoELayer._arc_audio_guard_orig_set_layer_number


def _install_layer_top_k(self: MoELayer) -> None:
    layer_number = getattr(self, "layer_number", None)
    if layer_number is None:
        return
    zero_based_layer = int(layer_number) - 1
    target_top_k = _LAYER_TOP_K.get(zero_based_layer)
    if target_top_k is None:
        return
    if getattr(self, "_arc_audio_guard_installed_top_k", None) == target_top_k:
        return
    current_top_k = int(self.config.moe_router_topk)
    if current_top_k != _NATIVE_TOP_K:
        raise RuntimeError(
            "[ARC-AUDIO-GUARD] expected native top-k "
            f"{_NATIVE_TOP_K} before patching layer {zero_based_layer}, got {current_top_k}; "
            "do not compose this plugin with another top-k patch"
        )

    # A single shallow-copied config is shared by the layer, router, and
    # dispatcher.  This changes both selection and the dispatcher's static
    # num_out_tokens scalar without mutating the model-wide shared config.
    layer_config = copy.copy(self.config)
    layer_config.moe_router_topk = target_top_k
    self.config = layer_config
    if getattr(self, "router", None) is not None:
        self.router.config = layer_config
        self.router.topk = target_top_k
    if getattr(self, "token_dispatcher", None) is not None:
        self.token_dispatcher.config = layer_config
        if hasattr(self.token_dispatcher, "router_topk"):
            self.token_dispatcher.router_topk = target_top_k
    self._arc_audio_guard_installed_top_k = target_top_k

    if _is_rank_zero():
        logged = getattr(MoELayer, "_arc_audio_guard_logged_topks", set())
        if target_top_k not in logged:
            label = "audio-confidence protected" if target_top_k > min(_LAYER_TOP_K.values()) else "reduced"
            print(
                f"[ARC-AUDIO-GUARD] {label} static top-{target_top_k} applied "
                f"(first layer={zero_based_layer}, MCore layer_number={layer_number})",
                flush=True,
            )
            logged = set(logged)
            logged.add(target_top_k)
            MoELayer._arc_audio_guard_logged_topks = logged


def _arc_audio_guard_moe_layer_init(
    self: MoELayer, *args: Any, **kwargs: Any
) -> None:
    _ORIGINAL_MOE_LAYER_INIT(self, *args, **kwargs)
    _install_layer_top_k(self)


def _arc_audio_guard_set_layer_number(self: MoELayer, layer_number: int) -> None:
    _ORIGINAL_MOE_LAYER_SET_LAYER_NUMBER(self, layer_number)
    _install_layer_top_k(self)


if not getattr(MoELayer, "_arc_audio_guard_patch", False):
    MoELayer.__init__ = _arc_audio_guard_moe_layer_init
    MoELayer.set_layer_number = _arc_audio_guard_set_layer_number
    MoELayer._arc_audio_guard_patch = True
    if _is_rank_zero():
        audio = _POLICY["audio_router_confidence_protection"]
        topk_histogram = {
            top_k: sum(value == top_k for value in _LAYER_TOP_K.values())
            for top_k in sorted(set(_LAYER_TOP_K.values()))
        }
        speed = _POLICY["speed_model"]
        print(
            "[ARC-AUDIO-GUARD] installed calibrated static policy "
            f"{_POLICY['policy_version']} middle=[{_MIDDLE_START},{_MIDDLE_END}) "
            f"topk_layers={topk_histogram} protected={audio['protected_layers_ranked']} "
            f"threshold={audio['fragile_margin_threshold']} "
            f"profile_samples={audio['complete_profile_samples']} "
            f"predicted_speedup={speed['predicted_speedup_percent']:.3f}%",
            flush=True,
        )

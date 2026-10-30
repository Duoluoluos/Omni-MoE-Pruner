#!/usr/bin/env python3
"""Derive a sync-free ARC audio-confidence layer schedule from router profiles.

The original ARC-SCEB policy computes the mean audio top1-top2 router margin
for every sample and middle layer.  A margin below 0.0082 protects that
sample's video tokens with a top-6 floor; fewer than 64 audio tokens gives a
top-8 floor.  That per-sample policy is useful for quality, but a variable
number of dispatched experts forces Megatron's slow dynamic dispatch path.

This tool converts the same signal into a *static per-layer calibration*:

* only the original ARC middle window [12, 32) is eligible;
* layers with the highest audio guard activation rate receive static top-6;
* remaining middle layers use static top-4;
* early and late layers remain native top-8;
* the number of protected layers is maximized subject to the measured speed
  model retaining the requested target plus a small prediction headroom.

The resulting schedule keeps the dropless ``tokens * layer_topk`` dispatcher
fast path, while making the protected layers traceable to ARC's actual audio
router confidence rule rather than an arbitrary depth band.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

try:  # ujson cuts the 2.1 GiB profile scan from minutes to seconds.
    import ujson as _json_loader
except ImportError:  # pragma: no cover - standard-library fallback
    _json_loader = json


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = (
    REPO_ROOT
    / "logs/omnivideobench_qwen3_profile/20260612_144602/profile_samples.jsonl"
)
DEFAULT_ARC_POLICY = (
    REPO_ROOT / "moe_expert_budget/configs/arc_video_boundary_audio_floor6_full_v1.json"
)
DEFAULT_OUTPUT_POLICY = (
    REPO_ROOT / "moe_expert_budget/configs/arc_audio_guard_static_topk_v1.json"
)


def _weighted_audio_margin(audio: dict[str, Any]) -> float | None:
    segments = audio.get("segments") or []
    denominator = sum(int(segment.get("num_tokens") or 0) for segment in segments)
    if denominator <= 0:
        return None
    numerator = sum(
        int(segment.get("num_tokens") or 0)
        * float(segment.get("top1_top2_margin_mean") or 0.0)
        for segment in segments
    )
    return numerator / denominator


def profile_audio_guard_rates(
    profile_path: Path,
    *,
    layer_start: int,
    layer_end: int,
    min_audio_tokens: int,
    fragile_margin_threshold: float,
) -> tuple[int, dict[int, dict[str, int | float]]]:
    """Reconstruct ARC's sample-level audio guard decisions for each layer."""
    accumulators = {
        layer: {
            "observed_samples": 0,
            "sufficient_audio_samples": 0,
            "insufficient_audio_samples": 0,
            "fragile_audio_samples": 0,
            "margin_sum": 0.0,
        }
        for layer in range(layer_start, layer_end)
    }
    complete_samples = 0
    with profile_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = _json_loader.loads(line)
            layers = record.get("prefill_layers") or []
            if len(layers) < layer_end:
                continue
            complete_samples += 1
            for layer in range(layer_start, layer_end):
                stats = accumulators[layer]
                stats["observed_samples"] += 1
                audio = ((layers[layer].get("token_trajectory") or {}).get("audio") or {})
                token_count = int(audio.get("token_count") or 0)
                margin = _weighted_audio_margin(audio)
                if token_count < min_audio_tokens or margin is None:
                    stats["insufficient_audio_samples"] += 1
                    continue
                stats["sufficient_audio_samples"] += 1
                stats["margin_sum"] += margin
                if margin < fragile_margin_threshold:
                    stats["fragile_audio_samples"] += 1

    result: dict[int, dict[str, int | float]] = {}
    for layer, raw in accumulators.items():
        observed = int(raw["observed_samples"])
        sufficient = int(raw["sufficient_audio_samples"])
        insufficient = int(raw["insufficient_audio_samples"])
        fragile = int(raw["fragile_audio_samples"])
        result[layer] = {
            "observed_samples": observed,
            "sufficient_audio_samples": sufficient,
            "insufficient_audio_samples": insufficient,
            "fragile_audio_samples": fragile,
            "fragile_rate_among_sufficient": round(fragile / sufficient, 6)
            if sufficient
            else 0.0,
            "audio_guard_activation_rate": round((fragile + insufficient) / observed, 6)
            if observed
            else 0.0,
            "mean_audio_margin": round(float(raw["margin_sum"]) / sufficient, 8)
            if sufficient
            else None,
        }
    return complete_samples, result


def select_static_schedule(
    layer_stats: dict[int, dict[str, int | float]],
    *,
    native_top_k: int,
    reduced_top_k: int,
    protected_top_k: int,
    total_layers: int,
    target_speedup_percent: float,
    prediction_headroom_percent: float,
    anchor_cut_units: int,
    anchor_speedup_percent: float,
) -> tuple[dict[int, int], dict[str, float | int | list[int]]]:
    """Maximize audio-protected layers while preserving the speed target."""
    schedule = {layer: reduced_top_k for layer in layer_stats}
    ranked = sorted(
        layer_stats,
        key=lambda layer: (
            -float(layer_stats[layer]["audio_guard_activation_rate"]),
            layer,
        ),
    )

    def cut_units(candidate: dict[int, int]) -> int:
        return sum(native_top_k - top_k for top_k in candidate.values())

    def predicted_speed(candidate: dict[int, int]) -> float:
        return anchor_speedup_percent * cut_units(candidate) / anchor_cut_units

    minimum_prediction = target_speedup_percent + prediction_headroom_percent
    protected: list[int] = []
    for layer in ranked:
        candidate = dict(schedule)
        candidate[layer] = protected_top_k
        if predicted_speed(candidate) + 1e-12 < minimum_prediction:
            break
        schedule = candidate
        protected.append(layer)

    cuts = cut_units(schedule)
    effective_average = (
        total_layers * native_top_k - cuts
    ) / total_layers
    metadata: dict[str, float | int | list[int]] = {
        "protected_layers_ranked": protected,
        "cut_topk_layer_units": cuts,
        "effective_average_top_k": round(effective_average, 8),
        "expert_pair_reduction_fraction": round(
            cuts / (total_layers * native_top_k), 8
        ),
        "predicted_speedup_percent": round(predicted_speed(schedule), 8),
    }
    return schedule, metadata


def derive_policy(
    profile_path: Path,
    arc_policy_path: Path,
    *,
    total_layers: int = 48,
    target_speedup_percent: float = 8.0,
    prediction_headroom_percent: float = 0.2,
    anchor_cut_units: int = 96,
    anchor_speedup_percent: float = 10.7,
) -> dict[str, Any]:
    arc = json.loads(arc_policy_path.read_text(encoding="utf-8"))
    native_top_k = int(arc["original_top_k"])
    layer_start = math.floor(total_layers * float(arc["layer_start_ratio"]))
    layer_end = math.floor(total_layers * float(arc["layer_end_ratio"]))
    audio_rule = arc["audio_router_conditioning"]
    sample_count, stats = profile_audio_guard_rates(
        profile_path,
        layer_start=layer_start,
        layer_end=layer_end,
        min_audio_tokens=int(audio_rule["min_audio_tokens"]),
        fragile_margin_threshold=float(audio_rule["fragile_margin_threshold"]),
    )
    schedule, speed = select_static_schedule(
        stats,
        native_top_k=native_top_k,
        reduced_top_k=int(audio_rule["boundary_top_m"]),
        protected_top_k=int(audio_rule["fragile_min_top_m"]),
        total_layers=total_layers,
        target_speedup_percent=target_speedup_percent,
        prediction_headroom_percent=prediction_headroom_percent,
        anchor_cut_units=anchor_cut_units,
        anchor_speedup_percent=anchor_speedup_percent,
    )
    return {
        "policy_version": "arc_audio_guard_static_topk_v1",
        "mode": "static_per_layer_topk",
        "source_arc_policy": arc["policy_version"],
        "model_num_layers": total_layers,
        "original_top_k": native_top_k,
        "middle_layer_start": layer_start,
        "middle_layer_end": layer_end,
        "layer_top_k": {str(layer): schedule[layer] for layer in sorted(schedule)},
        "audio_router_confidence_protection": {
            "enabled": True,
            "calibration_scope": "same-layer audio mean top1-top2 router margin",
            "min_audio_tokens": int(audio_rule["min_audio_tokens"]),
            "fragile_margin_threshold": float(audio_rule["fragile_margin_threshold"]),
            "insufficient_audio_target_top_k": int(
                audio_rule["insufficient_audio_min_top_m"]
            ),
            "fragile_audio_target_top_k": int(audio_rule["fragile_min_top_m"]),
            "profile_path": str(profile_path.relative_to(REPO_ROOT)),
            "complete_profile_samples": sample_count,
            "selection_rule": (
                "rank middle layers by audio guard activation rate; maximize top-6 "
                "layers while predicted speed remains target + headroom"
            ),
            "protected_layers_ranked": speed["protected_layers_ranked"],
            "per_layer_stats": {str(layer): stats[layer] for layer in sorted(stats)},
        },
        "speed_model": {
            "target_speedup_percent": target_speedup_percent,
            "prediction_headroom_percent": prediction_headroom_percent,
            "empirical_anchor": {
                "schedule": "top-4 on [10,34), top-8 elsewhere",
                "cut_topk_layer_units": anchor_cut_units,
                "measured_speedup_percent": anchor_speedup_percent,
            },
            **{key: value for key, value in speed.items() if key != "protected_layers_ranked"},
        },
        "runtime_guards": {
            "only_middle_layers_may_change": True,
            "dispatcher_mode": "static_dropless",
            "per_forward_gpu_cpu_sync": False,
        },
        "note": (
            "Offline-calibrated ARC audio guard. Re-derive this policy if the data "
            "distribution, base model, native top-k, or ARC threshold changes."
        ),
    }


def _core_signature(policy: dict[str, Any]) -> dict[str, Any]:
    audio = policy["audio_router_confidence_protection"]
    return {
        "layer_top_k": policy["layer_top_k"],
        "protected_layers_ranked": audio["protected_layers_ranked"],
        "complete_profile_samples": audio["complete_profile_samples"],
        "per_layer_stats": audio["per_layer_stats"],
        "speed_model": policy["speed_model"],
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--arc-policy", type=Path, default=DEFAULT_ARC_POLICY)
    parser.add_argument("--check", type=Path, help="Fail if this policy differs from a fresh derivation")
    parser.add_argument("--json", action="store_true", help="Print the full derived policy")
    args = parser.parse_args(argv)
    policy = derive_policy(args.profile, args.arc_policy)
    if args.check:
        existing = json.loads(args.check.read_text(encoding="utf-8"))
        if _core_signature(existing) != _core_signature(policy):
            raise SystemExit(f"STALE: {args.check} differs from profile-derived schedule")
        print(f"PASS: {args.check} matches {policy['audio_router_confidence_protection']['complete_profile_samples']} profile samples")
    if args.json:
        print(json.dumps(policy, indent=2))
    elif not args.check:
        audio = policy["audio_router_confidence_protection"]
        speed = policy["speed_model"]
        print(f"protected layers (risk-ranked): {audio['protected_layers_ranked']}")
        print(f"layer schedule: {policy['layer_top_k']}")
        print(f"predicted speedup: {speed['predicted_speedup_percent']:.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

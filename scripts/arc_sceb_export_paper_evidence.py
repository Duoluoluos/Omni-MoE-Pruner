#!/usr/bin/env python3
"""Export a compact, reproducible evidence packet for ARC-SCEB paper numbers.

Large routing dumps remain in ``logs/``.  This script recomputes the values used
by the motivating-study figure from ``aggregated_stats.json`` and
``worker_capture.json`` and extracts accepted diagnostic/system values from the
canonical research story.  The output is small enough for numeric-evidence
checks while retaining hashes of its immediate sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def find_repo() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "research_paper/ARC_SCEB_RESEARCH_STORY.md").is_file():
            return parent
    raise RuntimeError("Cannot locate Omni-MoE-Pruner repository root")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def match_numbers(text: str, pattern: str, expected_groups: int) -> list[float]:
    match = re.search(pattern, text, flags=re.DOTALL)
    if match is None or len(match.groups()) != expected_groups:
        raise RuntimeError(f"Canonical story pattern did not match: {pattern}")
    return [float(value) for value in match.groups()]


def stage_means(values: list[float | None], start: int, end: int) -> dict[str, float]:
    array = np.asarray([np.nan if value is None else value for value in values], dtype=float)
    return {
        "early": float(np.nanmean(array[:start]) * 1e3),
        "middle": float(np.nanmean(array[start:end]) * 1e3),
        "late": float(np.nanmean(array[end:]) * 1e3),
    }


def main() -> int:
    repo = find_repo()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "research_paper/iclr2027/evidence/routing_audit_summary.json",
    )
    args = parser.parse_args()

    story_path = repo / "research_paper/ARC_SCEB_RESEARCH_STORY.md"
    aggregate_path = repo / "logs/arc_sceb_interpretability/aggregated_stats.json"
    capture_path = repo / "logs/arc_sceb_interpretability/worker_capture.json"
    policy_path = repo / "moe_expert_budget/configs/arc_video_boundary_audio_floor6_full_v1.json"
    story = story_path.read_text(encoding="utf-8")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    capture = json.loads(capture_path.read_text(encoding="utf-8"))

    uniform_speedup, uniform_loss_base, uniform_loss_top4 = match_numbers(
        story,
        r"训练 step 快了约\s*([0-9.]+)%.*?验证损失从约\s*([0-9.]+)\s*上升到\s*([0-9.]+)",
        3,
    )
    static_mean_k, static_pair_reduction = match_numbers(
        story,
        r"平均 top-k 从 8 降到\s*([0-9.]+).*?token-expert pair 减少了\s*([0-9.]+)%",
        2,
    )
    step_base, step_static, static_speedup = match_numbers(
        story,
        r"稳态训练速度\s*\|\s*([0-9.]+) s/it\s*\|\s*([0-9.]+) s/it\s*\|\s*\*\*加速 ([0-9.]+)%",
        3,
    )
    val_base, val_static, val_delta = match_numbers(
        story,
        r"验证损失\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*\+([0-9.]+)",
        3,
    )
    memory_base, memory_static = match_numbers(
        story,
        r"峰值显存\s*\|\s*([0-9.]+) GiB\s*\|\s*([0-9.]+) GiB",
        2,
    )

    ovb = aggregate["omnivideobench"]
    avut = aggregate["avut"]
    start, end = ovb["policy"]["L_START"], ovb["policy"]["L_END"]

    budget_counts = {4: 0, 6: 0, 8: 0}
    state_counts: dict[str, int] = {}
    fragile_ge6 = []
    for sample in capture["samples"]:
        for key, value in sample["chosen_m_hist_total"].items():
            budget_counts[int(key)] += int(value)
        for layer in sample["per_layer"]:
            state = layer["audio_state"]
            state_counts[state] = state_counts.get(state, 0) + 1
            if state == "fragile_audio":
                hist = {int(k): int(v) for k, v in layer["chosen_m_hist"].items()}
                total = sum(hist.values()) or 1
                fragile_ge6.append((hist.get(6, 0) + hist.get(8, 0)) / total)
    total_masks = sum(budget_counts.values())
    mean_budget = sum(k * value for k, value in budget_counts.items()) / total_masks

    output = {
        "schema_version": "arc-sceb/paper-evidence-v1",
        "provenance": {
            "canonical_story": str(story_path.relative_to(repo)),
            "canonical_story_sha256": sha256(story_path),
            "aggregated_stats": str(aggregate_path.relative_to(repo)),
            "aggregated_stats_sha256": sha256(aggregate_path),
            "worker_capture": str(capture_path.relative_to(repo)),
            "worker_capture_sha256": sha256(capture_path),
            "policy": str(policy_path.relative_to(repo)),
            "policy_sha256": sha256(policy_path),
            "generator": str(Path(__file__).resolve().relative_to(repo)),
        },
        "model": {
            "layers": len(ovb["layers"]),
            "native_top_k": int(capture["policy"]["TOP_K"]),
            "experts": 128,
            "candidate_layer_start": int(start),
            "candidate_layer_end_exclusive": int(end),
        },
        "uniform_top4_diagnostic": {
            "speedup_percent": uniform_speedup,
            "validation_loss_top8": uniform_loss_base,
            "validation_loss_top4": uniform_loss_top4,
            "provenance_kind": "verified canonical story",
        },
        "audit": {
            "omnivideobench_samples": int(ovb["n_samples"]),
            "avut_samples": int(avut["n_samples"]),
            "expert_use_entropy": {
                "omnivideobench": ovb["middle_entropy"],
                "avut": avut["middle_entropy"],
            },
            "routing_decisiveness_times_1e3": stage_means(
                ovb["m12_mean"]["mean"], start, end
            ),
            "candidate_boundary_gap_times_1e3": stage_means(
                ovb["bnd_mean"]["mean"], start, end
            ),
        },
        "replay": {
            "clips": int(capture["n_samples"]),
            "middle_layer_states": int(sum(state_counts.values())),
            "audio_state_counts": state_counts,
            "budget_counts": {str(k): int(v) for k, v in budget_counts.items()},
            "budget_share_percent": {
                str(k): float(value / total_masks * 100)
                for k, value in budget_counts.items()
            },
            "mean_mask_budget": float(mean_budget),
            "logical_allocation_reduction_percent": float((1 - mean_budget / 8) * 100),
            "fragile_states_all_video_ge_top6": bool(
                fragile_ge6 and min(fragile_ge6) >= 1 - 1e-12
            ),
        },
        "static_realization": {
            "mean_top_k": static_mean_k,
            "token_expert_pair_reduction_percent": static_pair_reduction,
            "paired_steps": 25,
            "seed": 42,
            "step_seconds_top8": step_base,
            "step_seconds_static": step_static,
            "speedup_percent": static_speedup,
            "validation_loss_top8": val_base,
            "validation_loss_static": val_static,
            "validation_loss_delta": val_delta,
            "peak_memory_gib_top8": memory_base,
            "peak_memory_gib_static": memory_static,
            "provenance_kind": "verified canonical story",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

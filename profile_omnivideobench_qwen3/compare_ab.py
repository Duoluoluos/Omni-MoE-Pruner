#!/usr/bin/env python3
"""Paired A/B comparison between baseline and budget runs.

Reports the metrics required by step 01 of the implementation plan:
- overall accuracy (baseline / treatment / delta)
- baseline-correct flip rate
- per task-type accuracy (focus on temporal / spatial / reference / counting)
- per task-type answer flip rate
- average active experts/token (effective vs original)
- MoE FFN FLOPs estimate reduction
- pass/fail thresholds from the plan

Usage:
    python profile_omnivideobench_qwen3/compare_ab.py \\
        --baseline logs/omnivideobench_qwen3_profile/<ts> \\
        --treatment logs/omnivideobench_qwen3_budget_topm/<ts>
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


STRESS_TASKS = (
    "temporal understanding",
    "spatial understanding",
    "reference reasoning",
    "counting",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def index_by(items: list[dict[str, Any]], key: str) -> dict[Any, dict[str, Any]]:
    return {it[key]: it for it in items if key in it}


def task_accuracy(predictions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_task: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in predictions:
        t = r.get("task_type") or "Unknown"
        by_task[t]["total"] += 1
        by_task[t]["correct"] += int(bool(r.get("is_correct")))
    return {
        t: {"total": v["total"], "correct": v["correct"], "accuracy": (v["correct"] / v["total"]) if v["total"] else 0.0}
        for t, v in sorted(by_task.items())
    }


def expert_budget_summary(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    total_orig = 0
    total_eff = 0
    target_tokens = 0
    layers_seen: set[int] = set()
    for p in profiles:
        eb = p.get("expert_budget") or {}
        if not eb.get("enabled"):
            continue
        total_orig += int(eb.get("assignment_count_original_est", 0) or 0)
        total_eff += int(eb.get("assignment_count_effective", 0) or 0)
        target_tokens += int(eb.get("target_token_count", 0) or 0)
        for ls in eb.get("layer_summaries", []) or []:
            layers_seen.add(int(ls.get("layer_idx", -1)))
    if total_orig == 0:
        return {"enabled": False}
    return {
        "enabled": True,
        "assignments_original_total": total_orig,
        "assignments_effective_total": total_eff,
        "assignments_reduction_fraction": 1.0 - (total_eff / total_orig),
        "avg_active_experts_per_target_token_effective": (total_eff / target_tokens) if target_tokens else None,
        "target_token_count_total": target_tokens,
        "target_layers_seen": sorted(layers_seen),
    }


def moe_flops_summary(profiles: list[dict[str, Any]]) -> float | None:
    vals = []
    for p in profiles:
        ps = p.get("prefill_summary") or {}
        if ps.get("moe_ffn_flops_estimate") is not None:
            vals.append(int(ps["moe_ffn_flops_estimate"]))
    if not vals:
        return None
    return statistics.mean(vals)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="Directory with baseline predictions.jsonl + profile_samples.jsonl")
    p.add_argument("--treatment", required=True, help="Directory with treatment predictions.jsonl + profile_samples.jsonl")
    p.add_argument("--output", default=None, help="Optional path to write the comparison JSON. Defaults to <treatment>/ab_compare.json.")
    p.add_argument("--accuracy-drop-threshold-pp", type=float, default=5.0)
    p.add_argument("--baseline-flip-threshold", type=float, default=0.10)
    p.add_argument("--stress-drop-threshold-pp", type=float, default=8.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    baseline_dir = Path(args.baseline)
    treatment_dir = Path(args.treatment)

    base_pred = load_jsonl(baseline_dir / "predictions.jsonl")
    base_prof = load_jsonl(baseline_dir / "profile_samples.jsonl")
    treat_pred = load_jsonl(treatment_dir / "predictions.jsonl")
    treat_prof = load_jsonl(treatment_dir / "profile_samples.jsonl")

    base_by_idx = index_by(base_pred, "sample_index")
    treat_by_idx = index_by(treat_pred, "sample_index")
    common = sorted(set(base_by_idx) & set(treat_by_idx))

    overall_base = sum(1 for r in base_pred if r.get("is_correct")) / max(len(base_pred), 1)
    overall_treat = sum(1 for r in treat_pred if r.get("is_correct")) / max(len(treat_pred), 1)

    paired_base_correct = 0
    paired_treat_correct = 0
    flips_total = 0
    base_correct_count = 0
    base_correct_flipped = 0
    by_task_paired: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "base_correct": 0, "treat_correct": 0, "flip_count": 0}
    )
    for idx in common:
        b = base_by_idx[idx]
        t = treat_by_idx[idx]
        task = b.get("task_type") or t.get("task_type") or "Unknown"
        by_task_paired[task]["total"] += 1
        if b.get("is_correct"):
            paired_base_correct += 1
            base_correct_count += 1
            by_task_paired[task]["base_correct"] += 1
            if not t.get("is_correct"):
                base_correct_flipped += 1
        if t.get("is_correct"):
            paired_treat_correct += 1
            by_task_paired[task]["treat_correct"] += 1
        if (b.get("predicted_answer") or "") != (t.get("predicted_answer") or ""):
            flips_total += 1
            by_task_paired[task]["flip_count"] += 1

    paired_base_acc = paired_base_correct / max(len(common), 1)
    paired_treat_acc = paired_treat_correct / max(len(common), 1)
    paired_flip_rate = flips_total / max(len(common), 1)
    base_correct_flip_rate = base_correct_flipped / max(base_correct_count, 1)

    task_summary: dict[str, dict[str, Any]] = {}
    for task, v in sorted(by_task_paired.items()):
        n = v["total"]
        base_acc = v["base_correct"] / n if n else 0.0
        treat_acc = v["treat_correct"] / n if n else 0.0
        task_summary[task] = {
            "paired_total": n,
            "baseline_accuracy": base_acc,
            "treatment_accuracy": treat_acc,
            "delta_pp": (treat_acc - base_acc) * 100.0,
            "answer_flip_rate": v["flip_count"] / n if n else 0.0,
        }

    base_eb = expert_budget_summary(base_prof)
    treat_eb = expert_budget_summary(treat_prof)

    base_moe_flops_mean = moe_flops_summary(base_prof)
    treat_moe_flops_mean = moe_flops_summary(treat_prof)
    moe_flops_reduction = None
    if base_moe_flops_mean and treat_moe_flops_mean:
        moe_flops_reduction = 1.0 - (treat_moe_flops_mean / base_moe_flops_mean)

    overall_drop_pp = (paired_base_acc - paired_treat_acc) * 100.0
    pass_overall = overall_drop_pp <= args.accuracy_drop_threshold_pp
    pass_baseline_flip = base_correct_flip_rate <= args.baseline_flip_threshold
    stress_drops = []
    stress_status = {}
    for st in STRESS_TASKS:
        match = next((task_summary[k] for k in task_summary if k.lower() == st.lower()), None)
        if match is None:
            stress_status[st] = {"present": False}
            continue
        drop_pp = (match["baseline_accuracy"] - match["treatment_accuracy"]) * 100.0
        stress_drops.append(drop_pp)
        stress_status[st] = {
            "present": True,
            "drop_pp": drop_pp,
            "passes": drop_pp <= args.stress_drop_threshold_pp,
        }
    pass_stress = all(v.get("passes", True) for v in stress_status.values() if v.get("present"))

    report = {
        "baseline_dir": str(baseline_dir),
        "treatment_dir": str(treatment_dir),
        "samples": {
            "baseline_predictions": len(base_pred),
            "treatment_predictions": len(treat_pred),
            "paired_common": len(common),
        },
        "accuracy": {
            "baseline_overall": overall_base,
            "treatment_overall": overall_treat,
            "paired_baseline": paired_base_acc,
            "paired_treatment": paired_treat_acc,
            "paired_overall_delta_pp": (paired_treat_acc - paired_base_acc) * 100.0,
            "paired_overall_drop_pp": overall_drop_pp,
            "paired_answer_flip_rate": paired_flip_rate,
            "baseline_correct_flip_rate": base_correct_flip_rate,
        },
        "task_type": task_summary,
        "task_type_baseline_overall": task_accuracy(base_pred),
        "task_type_treatment_overall": task_accuracy(treat_pred),
        "expert_budget": {
            "baseline": base_eb,
            "treatment": treat_eb,
        },
        "moe_ffn_flops": {
            "baseline_mean": base_moe_flops_mean,
            "treatment_mean": treat_moe_flops_mean,
            "reduction_fraction": moe_flops_reduction,
        },
        "thresholds": {
            "accuracy_drop_pp": args.accuracy_drop_threshold_pp,
            "baseline_flip_rate": args.baseline_flip_threshold,
            "stress_drop_pp": args.stress_drop_threshold_pp,
        },
        "pass": {
            "overall": pass_overall,
            "baseline_correct_flip": pass_baseline_flip,
            "stress_tasks": pass_stress,
            "stress_breakdown": stress_status,
            "all": bool(pass_overall and pass_baseline_flip and pass_stress),
        },
    }

    out_path = Path(args.output) if args.output else (treatment_dir / "ab_compare.json")
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "paired_n": len(common),
        "paired_baseline_accuracy": paired_base_acc,
        "paired_treatment_accuracy": paired_treat_acc,
        "paired_drop_pp": overall_drop_pp,
        "baseline_correct_flip_rate": base_correct_flip_rate,
        "moe_flops_reduction": moe_flops_reduction,
        "pass_all": report["pass"]["all"],
        "report_path": str(out_path),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

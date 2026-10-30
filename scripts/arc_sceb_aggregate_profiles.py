#!/usr/bin/env python3
"""Aggregate large-scale Qwen3-Omni router-profiling evidence for ARC-SCEB
interpretability figures.

Reads the existing profile_samples.jsonl dumps (no GPU, no rerun) and produces
one compact aggregated_stats.json capturing the routing geometry behind the
ARC-SCEB design pillars:

  P2  video expert use is more diffuse (a target-selection diagnostic)
  P3  candidate separation varies with depth (a layer-selection diagnostic)
  P4  boundary margin selects the per-token budget (top-4/6/8)

Per-modality expert-use dispersion is measured as the normalized entropy of the
per-modality expert-activation histogram.  This population-level diagnostic is
not, by itself, evidence that any expert can be removed safely.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

def find_repo() -> Path:
    """Find the repository from either the canonical or archived script path."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "moe_expert_budget").is_dir() and (parent / "logs").is_dir():
            return parent
    raise RuntimeError("Cannot locate Omni-MoE-Pruner repository root")


REPO = find_repo()
NUM_LAYERS = 48
NUM_EXPERTS = 128
LOGE = math.log(NUM_EXPERTS)
# ARC-SCEB middle-layer window and boundary threshold (from the deployed policy)
POLICY = json.loads(
    (REPO / "moe_expert_budget/configs/arc_video_boundary_audio_floor6_full_v1.json").read_text()
)
L_START = int(NUM_LAYERS * POLICY["layer_start_ratio"])   # 12
L_END = int(NUM_LAYERS * POLICY["layer_end_ratio"])       # 32
BOUNDARY_THR = float(POLICY["thresholds"]["boundary_threshold"])

SOURCES = {
    "omnivideobench": REPO / "logs/omnivideobench_qwen3_profile/20260612_144602/profile_samples.jsonl",
    "avut": REPO / "logs/avut_qwen3_profile/full_20260611_192836/profile_samples.jsonl",
}


def norm_entropy(counts: np.ndarray) -> float:
    s = counts.sum()
    if s <= 0:
        return float("nan")
    p = counts[counts > 0] / s
    return float(-(p * np.log(p)).sum() / LOGE)


def frac_experts_for_mass(counts: np.ndarray, mass: float = 0.5) -> float:
    s = counts.sum()
    if s <= 0:
        return float("nan")
    p = np.sort(counts / s)[::-1]
    c = np.cumsum(p)
    k = int(np.searchsorted(c, mass) + 1)
    return k / len(counts)


def aggregate(path: Path) -> dict:
    # Accumulators shaped [NUM_LAYERS]
    acc = {
        "n_samples": 0,
        # per-layer, per-modality normalized entropy (list of per-sample values)
        "ent_video": [[] for _ in range(NUM_LAYERS)],
        "ent_audio": [[] for _ in range(NUM_LAYERS)],
        "ent_text": [[] for _ in range(NUM_LAYERS)],
        # per-layer margins (top1-top2 and selected boundary), per-sample means
        "m12_mean": [[] for _ in range(NUM_LAYERS)],
        "m12_p50": [[] for _ in range(NUM_LAYERS)],
        "bnd_mean": [[] for _ in range(NUM_LAYERS)],
        "bnd_p50": [[] for _ in range(NUM_LAYERS)],
        "bnd_p05": [[] for _ in range(NUM_LAYERS)],
        "bnd_p95": [[] for _ in range(NUM_LAYERS)],
        # per-layer normalized expert-use entropy (activation histogram)
        "ent_all": [[] for _ in range(NUM_LAYERS)],
        # per-layer fraction of experts to cover 50% mass (video)
        "spread50_video": [[] for _ in range(NUM_LAYERS)],
        "spread50_audio": [[] for _ in range(NUM_LAYERS)],
        "spread50_text": [[] for _ in range(NUM_LAYERS)],
        # compute / token share
        "moe_ms": [[] for _ in range(NUM_LAYERS)],
        "tok_video": 0, "tok_audio": 0, "tok_text": 0,
        # One middle-layer mean per sample.  These support honest sample-level
        # error bars instead of treating 20 correlated layers as independent.
        "middle_ent_video": [],
        "middle_ent_audio": [],
        "middle_ent_text": [],
    }

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            layers = r.get("prefill_layers")
            if not layers or len(layers) < NUM_LAYERS:
                continue
            acc["n_samples"] += 1
            tmc = r.get("token_modality_counts", {})
            acc["tok_video"] += int(tmc.get("video", 0))
            acc["tok_audio"] += int(tmc.get("audio", 0))
            acc["tok_text"] += int(tmc.get("text", 0))
            sample_middle_entropy = {"video": [], "audio": [], "text": []}
            for li in range(NUM_LAYERS):
                L = layers[li]
                tm = L.get("topk_margin", {})
                acc["m12_mean"][li].append(tm.get("top1_top2_mean", float("nan")))
                acc["m12_p50"][li].append(tm.get("top1_top2_p50", float("nan")))
                acc["bnd_mean"][li].append(tm.get("selected_boundary_mean", float("nan")))
                acc["bnd_p50"][li].append(tm.get("selected_boundary_p50", float("nan")))
                acc["bnd_p05"][li].append(tm.get("selected_boundary_p05", float("nan")))
                acc["bnd_p95"][li].append(tm.get("selected_boundary_p95", float("nan")))
                acc["ent_all"][li].append(L.get("routing_entropy_normalized", float("nan")))
                acc["moe_ms"][li].append(L.get("moe_forward_time_ms", float("nan")))
                med = L.get("modality_expert_distribution", {})
                for mod, key_e, key_s in (
                    ("video", "ent_video", "spread50_video"),
                    ("audio", "ent_audio", "spread50_audio"),
                    ("text", "ent_text", "spread50_text"),
                ):
                    d = med.get(mod)
                    if not d or not d.get("tokens"):
                        continue
                    counts = np.asarray(d.get("activation_counts", []), dtype=float)
                    if counts.size != NUM_EXPERTS or counts.sum() <= 0:
                        continue
                    entropy = norm_entropy(counts)
                    acc[key_e][li].append(entropy)
                    acc[key_s][li].append(frac_experts_for_mass(counts, 0.5))
                    if L_START <= li < L_END:
                        sample_middle_entropy[mod].append(entropy)
            for mod, values in sample_middle_entropy.items():
                if values:
                    acc[f"middle_ent_{mod}"].append(float(np.mean(values)))
    return acc


def summarize(acc: dict) -> dict:
    def stat(list_of_lists):
        out_mean, out_lo, out_hi = [], [], []
        for vals in list_of_lists:
            a = np.asarray([v for v in vals if v == v], dtype=float)  # drop nan
            if a.size == 0:
                out_mean.append(None); out_lo.append(None); out_hi.append(None); continue
            out_mean.append(float(a.mean()))
            out_lo.append(float(np.percentile(a, 25)))
            out_hi.append(float(np.percentile(a, 75)))
        return {"mean": out_mean, "p25": out_lo, "p75": out_hi}

    def scalar_stat(values):
        a = np.asarray([v for v in values if v == v], dtype=float)
        if a.size == 0:
            return {"mean": None, "p25": None, "p75": None, "n": 0}
        return {
            "mean": float(a.mean()),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
            "n": int(a.size),
        }

    total_tok = acc["tok_video"] + acc["tok_audio"] + acc["tok_text"]
    return {
        "n_samples": acc["n_samples"],
        "definitions": {
            "expert_use_entropy": (
                "normalized entropy of the 128-expert activation-count histogram"
            ),
            "middle_entropy_summary": (
                "distribution across samples after averaging each sample over layers [12,32)"
            ),
        },
        "policy": {"L_START": L_START, "L_END": L_END, "boundary_threshold": BOUNDARY_THR},
        "layers": list(range(NUM_LAYERS)),
        "ent_video": stat(acc["ent_video"]),
        "ent_audio": stat(acc["ent_audio"]),
        "ent_text": stat(acc["ent_text"]),
        "ent_all": stat(acc["ent_all"]),
        "spread50_video": stat(acc["spread50_video"]),
        "spread50_audio": stat(acc["spread50_audio"]),
        "spread50_text": stat(acc["spread50_text"]),
        "middle_entropy": {
            "video": scalar_stat(acc["middle_ent_video"]),
            "audio": scalar_stat(acc["middle_ent_audio"]),
            "text": scalar_stat(acc["middle_ent_text"]),
        },
        "m12_mean": stat(acc["m12_mean"]),
        "m12_p50": stat(acc["m12_p50"]),
        "bnd_mean": stat(acc["bnd_mean"]),
        "bnd_p50": stat(acc["bnd_p50"]),
        "bnd_p05": stat(acc["bnd_p05"]),
        "bnd_p95": stat(acc["bnd_p95"]),
        "moe_ms": stat(acc["moe_ms"]),
        "token_share": {
            "video": acc["tok_video"] / total_tok if total_tok else 0,
            "audio": acc["tok_audio"] / total_tok if total_tok else 0,
            "text": acc["tok_text"] / total_tok if total_tok else 0,
            "total_tokens": total_tok,
        },
    }


def main():
    out = {}
    for name, path in SOURCES.items():
        if not path.exists():
            print(f"[skip] {name}: {path} missing", file=sys.stderr)
            continue
        print(f"[agg] {name}: {path}", file=sys.stderr)
        out[name] = summarize(aggregate(path))
        print(f"      n_samples={out[name]['n_samples']} "
              f"video_share={out[name]['token_share']['video']:.3f}", file=sys.stderr)
    dst = REPO / "logs/arc_sceb_interpretability/aggregated_stats.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(f"[done] wrote {dst}", file=sys.stderr)


if __name__ == "__main__":
    main()

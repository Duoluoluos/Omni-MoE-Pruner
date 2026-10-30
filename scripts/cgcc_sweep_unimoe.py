#!/usr/bin/env python3
"""CG-CC (CG-Skip) hyperparameter sweep for Uni-MoE-2.0-Omni across benchmarks.

Single model load, multiple configs sequentially. Reuses the verified
CGSkipController + generate path from run_mb_new2_cgskip_unimoe2.py, and the
benchmark-agnostic load_benchmark from run_mb_new2_cgskip_qwen3omni.py.

Uni-MoE CG-Skip policy (leakage-free, 5-way dynamic softmax = 4 real + 1 null):
  if p1 >= tau_skip or null_mass >= tau_null : budget 0 (shared-only)
  elif p1 <= tau_hard                        : budget = boost_budget
  else                                       : budget = base_budget

Arms: native, cgskip, uniform top-1..4. Compares cgskip vs uniform top-2
(the near-lossless efficient point on Uni-MoE).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/home/tiger/Uni-MoE/Uni-MoE-2")
sys.path.insert(0, str(REPO / "scripts"))

from run_mb_new2_cgskip_unimoe2 import (  # noqa: E402
    CGSkipController,
    build_conversation,
    build_inputs,
    move_inputs,
    generate_once,
    _ensure_matplotlib_stub,
)
from run_mb_new2_cgskip_qwen3omni import load_benchmark  # noqa: E402
from run_mb_new2_cgskip_qwen3omni import (  # noqa: E402
    append_jsonl,
    atomic_json,
    load_jsonl,
)

SEED = 20260907
FIXED = {"top1": 1, "top2": 2, "top3": 3, "top4": 4}


def log(*p):
    print("[CGCC-SWEEP-UNIMOE]", *p, file=sys.stderr, flush=True)


def calibrate(records, video_dir, model, processor, pmi, controller, policy, n_samples):
    p1_pool, null_pool = [], []
    controller._record_stats = True
    controller._stat_records = []
    seen = 0
    for i in range(len(records)):
        if seen >= n_samples:
            break
        vp = (video_dir / str(records[i]["video"])).resolve()
        if not vp.exists():
            continue
        conv = build_conversation(vp, records[i])
        inputs = move_inputs(build_inputs(processor, pmi, conv), model)
        controller.set_thresholds({"tau_skip": 2.0, "tau_null": 2.0, "tau_hard": -1.0})
        with controller.arm("cgskip"):
            generate_once(model, processor, inputs, 1)
        for rec in controller._stat_records:
            p1_pool.append(rec["p1"])
            null_pool.append(rec["null_mass"])
        controller._stat_records = []
        seen += 1
    controller._record_stats = False
    p1 = torch.cat(p1_pool).numpy().astype(np.float64)
    nm = torch.cat(null_pool).numpy().astype(np.float64)
    tau = {
        "tau_skip": float(np.percentile(p1, 100.0 - policy["skip_percentile"])),
        "tau_null": float(np.percentile(nm, 100.0 - policy["null_percentile"])),
        "tau_hard": float(np.percentile(p1, policy["hard_percentile"])),
    }
    log(f"calibrated {tau} over {seen} samples (p1 mean {p1.mean():.3f})")
    return tau


def pairci(a, b, seed):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)])
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def run_config(
    *,
    tag,
    policy,
    band,
    records,
    video_dir,
    selected,
    model,
    processor,
    pmi,
    output_dir,
    max_new_tokens,
    calibrate_samples,
    total_layers,
    real_expert_num,
):
    cfg_dir = output_dir / tag
    cfg_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cfg_dir / "raw_results.jsonl"
    summary_path = cfg_dir / "summary.json"
    if summary_path.exists():
        log(f"[{tag}] done already, skip")
        return

    controller = CGSkipController(
        model,
        expected_blocks=total_layers,
        real_expert_num=real_expert_num,
        layer_band=band,
        policy=policy,
    )
    tau = calibrate(
        records, video_dir, model, processor, pmi, controller, policy, calibrate_samples
    )
    controller.set_thresholds(tau)

    rows = load_jsonl(raw_path)
    done = {int(r["sample_index"]) for r in rows}

    for ordinal, si in enumerate(selected, 1):
        if si in done:
            continue
        rec = records[si]
        vp = (video_dir / str(rec["video"])).resolve()
        if not vp.exists():
            continue
        try:
            conv = build_conversation(vp, rec)
            inputs = move_inputs(build_inputs(processor, pmi, conv), model)
            correct = str(rec["correct_option"]).upper()
            native = generate_once(model, processor, inputs, max_new_tokens)
            with controller.arm("cgskip"):
                cg = generate_once(model, processor, inputs, max_new_tokens)
                cg_bud = controller.executed_mean_budget()
                cg_disp = controller.budget_dispersion()
            arms = {}
            for kn, kv in FIXED.items():
                with controller.arm(f"uniform_{kn}", fixed_budget=kv):
                    res = generate_once(model, processor, inputs, max_new_tokens)
                arms[kn] = {
                    "answer": res.answer,
                    "is_correct": res.answer == correct,
                    "answer_flip_vs_native": res.answer != native.answer,
                    "end_to_end_latency_ms": res.e2e_seconds * 1000.0,
                }
            row = {
                "sample_index": si,
                "correct_option": correct,
                "native": {
                    "answer": native.answer,
                    "is_correct": native.answer == correct,
                    "end_to_end_latency_ms": native.e2e_seconds * 1000.0,
                },
                "cgskip": {
                    "answer": cg.answer,
                    "is_correct": cg.answer == correct,
                    "answer_flip_vs_native": cg.answer != native.answer,
                    "end_to_end_latency_ms": cg.e2e_seconds * 1000.0,
                    "mean_executed_budget": cg_bud,
                    "budget_dispersion": cg_disp,
                },
                "arms": arms,
            }
            append_jsonl(raw_path, row)
            rows.append(row)
            done.add(si)
            if ordinal % 20 == 0 or ordinal <= 2:
                log(
                    f"[{tag}] [{ordinal}/{len(selected)}] s{si}: cg_bud={cg_bud:.2f} "
                    f"cg_ok={cg.answer == correct} nat_ok={native.answer == correct}"
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log(f"[{tag}] s{si} OOM skip")
        except Exception as e:
            log(f"[{tag}] s{si} err: {e}")

    controller.close()
    n = len(rows)
    if n == 0:
        log(f"[{tag}] no rows")
        return
    warmup = min(5, n)
    nat_acc = np.mean([r["native"]["is_correct"] for r in rows])
    cg_acc = np.mean([r["cgskip"]["is_correct"] for r in rows])
    cg_bud = np.nanmean([r["cgskip"]["mean_executed_budget"] for r in rows])
    t2_acc = np.mean([r["arms"]["top2"]["is_correct"] for r in rows])
    t1_acc = np.mean([r["arms"]["top1"]["is_correct"] for r in rows])
    cg_a = [r["cgskip"]["is_correct"] for r in rows]
    t2_a = [r["arms"]["top2"]["is_correct"] for r in rows]
    cg_l = [
        r["cgskip"]["end_to_end_latency_ms"] for i, r in enumerate(rows) if i >= warmup
    ]
    t2_l = [
        r["arms"]["top2"]["end_to_end_latency_ms"]
        for i, r in enumerate(rows)
        if i >= warmup
    ]
    acc_m, acc_lo, acc_hi = pairci(cg_a, t2_a, 1)
    lat_m, lat_lo, lat_hi = pairci(cg_l, t2_l, 2)
    summary = {
        "tag": tag,
        "policy": policy,
        "band": list(band) if band else None,
        "tau": tau,
        "n": n,
        "native_acc": round(nat_acc, 4),
        "cgskip_acc": round(cg_acc, 4),
        "cgskip_budget": round(float(cg_bud), 3),
        "top1_acc": round(t1_acc, 4),
        "top2_acc": round(t2_acc, 4),
        "cgskip_vs_top2": {
            "acc_diff": round(acc_m, 4),
            "acc_ci": [round(acc_lo, 4), round(acc_hi, 4)],
            "lat_diff_ms": round(lat_m, 1),
            "lat_ci_ms": [round(lat_lo, 1), round(lat_hi, 1)],
            "matched_acc_speedup": bool(acc_lo <= 0 <= acc_hi and lat_hi < 0),
        },
    }
    atomic_json(summary_path, summary)
    st = "✓ MAS" if summary["cgskip_vs_top2"]["matched_acc_speedup"] else "✗"
    log(
        f"[{tag}] DONE n={n} cg_acc={cg_acc:.3f} t2_acc={t2_acc:.3f} bud={cg_bud:.2f} "
        f"acc_diff={acc_m:+.3f}[{acc_lo:+.3f},{acc_hi:+.3f}] "
        f"lat_diff={lat_m:+.0f}[{lat_lo:+.0f},{lat_hi:+.0f}] {st}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--limit", type=int, required=True)
    ap.add_argument("--skip-indices", type=str, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--configs", type=str, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--calibrate-samples", type=int, default=16)
    args = ap.parse_args()

    configs = json.loads(args.configs)
    records, video_dir = load_benchmark(args.benchmark)
    n = min(len(records), args.limit)
    skip = (
        set(int(x) for x in args.skip_indices.split(",") if x.strip())
        if args.skip_indices
        else set()
    )
    selected = [i for i in range(n) if i not in skip]
    log(
        f"benchmark={args.benchmark} records={len(records)} selected={len(selected)} configs={len(configs)}"
    )

    import random

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    log("loading Uni-MoE-2 model (one-time)")
    _ensure_matplotlib_stub()
    from uni_moe.model.processing_qwen2_vl import Qwen2VLProcessor
    from uni_moe.model.modeling_qwen_grin_moe import GrinQwen2VLForConditionalGeneration
    from uni_moe.qwen_vl_utils import process_mm_info
    from uni_moe.model import deepspeed_moe_inference_utils  # noqa: F401

    model_path = REPO / "models/Uni-MoE-2.0-Omni"
    processor = Qwen2VLProcessor.from_pretrained(str(model_path))
    model = GrinQwen2VLForConditionalGeneration.from_pretrained(
        str(model_path),
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ).cuda()
    model.eval()
    processor.data_args = model.config
    cfg = json.loads((model_path / "config.json").read_text())
    total_layers = int(cfg["num_hidden_layers"])
    real_expert_num = int(cfg["mlp_dynamic_expert_num"])
    log(f"model loaded: {total_layers} layers, {real_expert_num} dynamic experts")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for c in configs:
        band = None
        if c.get("band"):
            lo, hi = (float(x) for x in c["band"].split(","))
            band = (int(np.floor(lo * total_layers)), int(np.floor(hi * total_layers)))
        policy = {
            "base_budget": int(c.get("base_budget", 2)),
            "boost_budget": int(c.get("boost_budget", 3)),
            "skip_percentile": float(c["skip_percentile"]),
            "hard_percentile": float(c.get("hard_percentile", 15)),
            "null_percentile": float(c.get("null_percentile", 8)),
        }
        log(f"=== config {c['tag']}: policy={policy} band={band} ===")
        try:
            run_config(
                tag=c["tag"],
                policy=policy,
                band=band,
                records=records,
                video_dir=video_dir,
                selected=selected,
                model=model,
                processor=processor,
                pmi=process_mm_info,
                output_dir=output_dir,
                max_new_tokens=args.max_new_tokens,
                calibrate_samples=args.calibrate_samples,
                total_layers=total_layers,
                real_expert_num=real_expert_num,
            )
        except Exception as e:
            log(f"[{c['tag']}] FAILED: {e}")
            import traceback

            traceback.print_exc()

    log("=== ALL CONFIGS DONE ===")
    for c in configs:
        sp = output_dir / c["tag"] / "summary.json"
        if sp.exists():
            s = json.loads(sp.read_text())
            vs = s["cgskip_vs_top2"]
            st = "✓ MAS" if vs["matched_acc_speedup"] else "✗"
            log(
                f"  {c['tag']}: cg={s['cgskip_acc']:.3f} t2={s['top2_acc']:.3f} bud={s['cgskip_budget']:.2f} "
                f"Δacc={vs['acc_diff']:+.3f}{vs['acc_ci']} Δlat={vs['lat_diff_ms']:+.0f}{vs['lat_ci_ms']} {st}"
            )


if __name__ == "__main__":
    main()

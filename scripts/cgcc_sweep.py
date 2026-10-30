#!/usr/bin/env python3
"""CG-CC hyperparameter sweep: single model load, multiple configs sequentially.

Usage:
  python scripts/cgcc_sweep.py \
    --benchmark avut --limit 173 --skip-indices 1495 \
    --output-dir /tmp/cgcc_sweep_avut \
    --configs '[ \
      {"tag":"rb6_rp30","reduce_budget":6,"reduce_percentile":30,"boost_percentile":0,"base_budget":8,"boost_budget":8,"band":"0.25,0.67"}, \
      {"tag":"rb6_rp50","reduce_budget":6,"reduce_percentile":50,"boost_percentile":0,"base_budget":8,"boost_budget":8,"band":"0.25,0.67"}, \
      {"tag":"rb4_rp30","reduce_budget":4,"reduce_percentile":30,"boost_percentile":0,"base_budget":8,"boost_budget":8,"band":"0.25,0.67"} \
    ]'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "qwen-omni-utils/src"))
sys.path.insert(0, str(REPO / "scripts"))

from run_mb_new2_cgskip_qwen3omni import (
    CGSkipIntervention,
    BENCHMARKS,
    MODEL_PATH,
    NATIVE_TOP_K,
    TOTAL_MOE_LAYERS,
    SEED,
    build_conversation,
    calibrate_thresholds,
    confidence_signal,
    cg_reduce_budget,
    distribution_js,
    generate_once,
    load_benchmark,
    parse_answer,
    append_jsonl,
    atomic_json,
    load_jsonl,
)


def log(*parts):
    print("[CGCC-SWEEP]", *parts, file=sys.stderr, flush=True)


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
    process_mm_info,
    output_dir,
    max_new_tokens,
    calibrate_samples,
):
    cfg_dir = output_dir / tag
    cfg_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cfg_dir / "raw_results.jsonl"
    summary_path = cfg_dir / "summary.json"

    if summary_path.exists():
        log(f"[{tag}] already has summary.json, skipping")
        return

    intervention = CGSkipIntervention(model, NATIVE_TOP_K, band)
    intervention.policy = policy

    tau = calibrate_thresholds(
        records=records,
        video_dir=video_dir,
        model=model,
        processor=processor,
        process_mm_info=process_mm_info,
        intervention=intervention,
        policy=policy,
        n_samples=calibrate_samples,
    )
    intervention.set_thresholds(tau)

    completed_rows = load_jsonl(raw_path)
    completed = {int(r["sample_index"]) for r in completed_rows}
    n_done_before = len(completed)

    for ordinal, si in enumerate(selected, 1):
        if si in completed:
            continue
        record = records[si]
        video_path = (video_dir / str(record["video"])).resolve()
        if not video_path.exists():
            log(f"[{tag}] sample {si} video missing, skip")
            continue
        try:
            conv = build_conversation(video_path, record)
            text = processor.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=False
            )
            from qwen_omni_utils import process_mm_info as pmi

            audios, images, videos = pmi(conv, use_audio_in_video=True)
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=True,
            )
            inputs = inputs.to(model.device).to(model.dtype)
            correct = str(record["correct_option"]).upper()
            intervention.prepare_sample(None)

            native = generate_once(model, processor, inputs, max_new_tokens)

            with intervention.arm("cgskip"):
                cg = generate_once(model, processor, inputs, max_new_tokens)
                cg_budget = intervention.executed_mean_budget()
                cg_disp = intervention.budget_dispersion()

            arms = {}
            for k_name, k_val in [("top4", 4), ("top6", 6), ("top8", 8)]:
                with intervention.arm(f"fixed:{k_val}"):
                    res = generate_once(model, processor, inputs, max_new_tokens)
                    fb = intervention.executed_mean_budget()
                arms[k_name] = {
                    "answer": res["answer"],
                    "is_correct": res["answer"] == correct,
                    "answer_flip_vs_native": res["answer"] != native["answer"],
                    "end_to_end_latency_ms": res["end_to_end_latency_ms"],
                    "mean_executed_budget": fb,
                }

            row = {
                "sample_index": si,
                "correct_option": correct,
                "native": {
                    "answer": native["answer"],
                    "is_correct": native["answer"] == correct,
                    "end_to_end_latency_ms": native["end_to_end_latency_ms"],
                },
                "cgskip": {
                    "answer": cg["answer"],
                    "is_correct": cg["answer"] == correct,
                    "answer_flip_vs_native": cg["answer"] != native["answer"],
                    "end_to_end_latency_ms": cg["end_to_end_latency_ms"],
                    "mean_executed_budget": cg_budget,
                    "budget_dispersion": cg_disp,
                },
                "arms": arms,
            }
            append_jsonl(raw_path, row)
            completed_rows.append(row)
            completed.add(si)

            if ordinal % 20 == 0 or ordinal <= 2:
                log(
                    f"[{tag}] [{ordinal}/{len(selected)}] s{si}: "
                    f"cg_bud={cg_budget:.2f} cg_ok={cg['answer'] == correct} "
                    f"nat_ok={native['answer'] == correct}"
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log(f"[{tag}] sample {si} OOM, skip")
        except Exception as e:
            log(f"[{tag}] sample {si} error: {e}")

    intervention.close()

    n = len(completed_rows)
    if n == 0:
        log(f"[{tag}] no samples completed")
        return

    nat_acc = np.mean([r["native"]["is_correct"] for r in completed_rows])
    cg_acc = np.mean([r["cgskip"]["is_correct"] for r in completed_rows])
    cg_flip = np.mean([r["cgskip"]["answer_flip_vs_native"] for r in completed_rows])
    cg_bud = np.nanmean([r["cgskip"]["mean_executed_budget"] for r in completed_rows])
    warmup = min(5, n)
    cg_lat = np.mean(
        [
            r["cgskip"]["end_to_end_latency_ms"]
            for i, r in enumerate(completed_rows)
            if i >= warmup
        ]
    )
    t8_lat = np.mean(
        [
            r["arms"]["top8"]["end_to_end_latency_ms"]
            for i, r in enumerate(completed_rows)
            if i >= warmup
        ]
    )
    t6_acc = np.mean([r["arms"]["top6"]["is_correct"] for r in completed_rows])
    t4_acc = np.mean([r["arms"]["top4"]["is_correct"] for r in completed_rows])
    t8_acc = np.mean([r["arms"]["top8"]["is_correct"] for r in completed_rows])

    # Paired diff CIs
    def pairci(a, b, seed):
        d = np.asarray(a, float) - np.asarray(b, float)
        rng = np.random.default_rng(seed)
        m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)])
        return (
            float(d.mean()),
            float(np.quantile(m, 0.025)),
            float(np.quantile(m, 0.975)),
        )

    cg_a = [r["cgskip"]["is_correct"] for r in completed_rows]
    t8_a = [r["arms"]["top8"]["is_correct"] for r in completed_rows]
    cg_l = [
        r["cgskip"]["end_to_end_latency_ms"]
        for i, r in enumerate(completed_rows)
        if i >= warmup
    ]
    t8_l = [
        r["arms"]["top8"]["end_to_end_latency_ms"]
        for i, r in enumerate(completed_rows)
        if i >= warmup
    ]
    acc_m, acc_lo, acc_hi = pairci(cg_a, t8_a, 1)
    lat_m, lat_lo, lat_hi = pairci(cg_l, t8_l, 2)

    summary = {
        "tag": tag,
        "policy": policy,
        "band": list(band) if band else None,
        "tau": tau,
        "n": n,
        "native_acc": round(nat_acc, 4),
        "cgskip_acc": round(cg_acc, 4),
        "cgskip_flip": round(cg_flip, 4),
        "cgskip_budget": round(cg_bud, 3),
        "top4_acc": round(t4_acc, 4),
        "top6_acc": round(t6_acc, 4),
        "top8_acc": round(t8_acc, 4),
        "cgskip_vs_top8": {
            "acc_diff": round(acc_m, 4),
            "acc_ci": [round(acc_lo, 4), round(acc_hi, 4)],
            "lat_diff_ms": round(lat_m, 1),
            "lat_ci_ms": [round(lat_lo, 1), round(lat_hi, 1)],
            "matched_acc_speedup": bool(acc_lo <= 0 <= acc_hi and lat_hi < 0),
        },
    }
    atomic_json(summary_path, summary)
    status = "✓ MAS" if summary["cgskip_vs_top8"]["matched_acc_speedup"] else "✗"
    log(
        f"[{tag}] DONE n={n} cg_acc={cg_acc:.3f} t8_acc={t8_acc:.3f} "
        f"acc_diff={acc_m:+.3f}[{acc_lo:+.3f},{acc_hi:+.3f}] "
        f"lat_diff={lat_m:+.0f}[{lat_lo:+.0f},{lat_hi:+.0f}] {status}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARKS.keys()))
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First dataset index (inclusive); use with --limit for held-out splits.",
    )
    parser.add_argument("--skip-indices", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--configs", type=str, required=True, help="JSON array of config dicts"
    )
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--calibrate-samples", type=int, default=12)
    args = parser.parse_args()

    configs = json.loads(args.configs)
    records, video_dir = load_benchmark(args.benchmark)
    start = max(0, int(args.start_index))
    stop = min(len(records), start + int(args.limit))
    skip = (
        set(int(x) for x in args.skip_indices.split(",") if x.strip())
        if args.skip_indices
        else set()
    )
    selected = [i for i in range(start, stop) if i not in skip]
    log(
        f"benchmark={args.benchmark} records={len(records)} selected={len(selected)} configs={len(configs)}"
    )

    import random

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    log("loading Qwen3-Omni model (one-time)")
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(str(MODEL_PATH), use_fast=False)
    log("model loaded")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        tag = cfg["tag"]
        band_str = cfg.get("band")
        band = None
        if band_str:
            lo, hi = (float(x) for x in band_str.split(","))
            band = (
                int(np.floor(lo * TOTAL_MOE_LAYERS)),
                int(np.floor(hi * TOTAL_MOE_LAYERS)),
            )
        policy = {
            "base_budget": int(cfg.get("base_budget", 8)),
            "reduce_budget": int(cfg["reduce_budget"]),
            "boost_budget": int(cfg.get("boost_budget", 8)),
            "reduce_percentile": float(cfg["reduce_percentile"]),
            "boost_percentile": float(cfg.get("boost_percentile", 0)),
            "gate_mode": str(cfg.get("gate_mode", "token")),
        }
        log(f"=== config {tag}: policy={policy} band={band} ===")
        try:
            run_config(
                tag=tag,
                policy=policy,
                band=band,
                records=records,
                video_dir=video_dir,
                selected=selected,
                model=model,
                processor=processor,
                process_mm_info=process_mm_info,
                output_dir=output_dir,
                max_new_tokens=args.max_new_tokens,
                calibrate_samples=args.calibrate_samples,
            )
        except Exception as e:
            log(f"[{tag}] FAILED: {e}")
            import traceback

            traceback.print_exc()

    log("=== ALL CONFIGS DONE ===")
    for cfg in configs:
        sp = output_dir / cfg["tag"] / "summary.json"
        if sp.exists():
            s = json.loads(sp.read_text())
            vs = s["cgskip_vs_top8"]
            status = "✓ MAS" if vs["matched_acc_speedup"] else "✗"
            log(
                f"  {cfg['tag']}: cg={s['cgskip_acc']:.3f} t8={s['top8_acc']:.3f} "
                f"Δacc={vs['acc_diff']:+.3f}{vs['acc_ci']} Δlat={vs['lat_diff_ms']:+.0f}{vs['lat_ci_ms']} {status}"
            )
        else:
            log(f"  {cfg['tag']}: NO SUMMARY")


if __name__ == "__main__":
    main()

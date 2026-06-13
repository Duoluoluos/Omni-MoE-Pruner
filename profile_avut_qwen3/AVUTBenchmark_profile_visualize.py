#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BENCH = "AVUTBenchmark"


def parse_args():
    parser = argparse.ArgumentParser(description="Create AVUTBenchmark profile snapshot plots.")
    parser.add_argument("--run-dir", default="logs/avut_qwen3_profile/full_20260611_192836")
    parser.add_argument("--output-dir", default=f"profile_avut_qwen3/{BENCH}_profile_snapshot")
    parser.add_argument("--top-task-count", type=int, default=12)
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    skipped = 0
    if not path.exists():
        return rows, skipped
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
    return rows, skipped


def pct(values, q):
    if len(values) == 0:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def safe_mean(values):
    return float(np.mean(values)) if len(values) else None


def safe_median(values):
    return float(np.median(values)) if len(values) else None


def ensure_2d(arr, rows, cols, dtype=np.float64):
    if arr is None:
        return np.zeros((rows, cols), dtype=dtype)
    return arr


def iter_jsonl(path):
    skipped = 0
    loaded = 0
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                loaded += 1
                yield json.loads(line), loaded, skipped
            except json.JSONDecodeError:
                skipped += 1


def accumulate_profiles(profile_path):
    samples = []
    layer_counts = None
    modality_layer_counts = {}
    task_expert_counts = defaultdict(lambda: None)
    layer_entropy = defaultdict(list)
    layer_hot_ratio = defaultdict(list)
    layer_empty = defaultdict(list)
    layer_moe_ms = defaultdict(list)
    num_layers = 0
    num_experts = 0
    loaded = 0
    skipped = 0

    for p, loaded, skipped in iter_jsonl(profile_path):
        ps = p.get("prefill_summary", {})
        token_counts = p.get("token_modality_counts", {})
        row = {
            "sample_index": p.get("sample_index"),
            "task_type": p.get("task_type", "Unknown"),
            "input_seq_len": p.get("input_seq_len"),
            "prefill_latency_ms": ps.get("prefill_latency_ms"),
            "attention_flops": ps.get("attention_flops_estimate"),
            "moe_ffn_flops": ps.get("moe_ffn_flops_estimate"),
            "kv_cache_write_bytes": ps.get("kv_cache_write_bytes_estimate"),
            "generate_latency_ms": p.get("generate_latency_ms"),
            "flash_attention_requested": ps.get("flash_attention_requested"),
            "chunked_prefill_observed": ps.get("chunked_prefill_observed"),
            "text_tokens": token_counts.get("text", 0),
            "video_tokens": token_counts.get("video", 0),
            "audio_tokens": token_counts.get("audio", 0),
            "image_tokens": token_counts.get("image", 0),
            "decoded_video_frames": p.get("input_meta", {}).get("decoded_video_frames", 0),
            "use_audio_in_video": p.get("input_meta", {}).get("use_audio_in_video"),
        }
        samples.append(row)

        for layer in p.get("prefill_layers", []):
            layer_idx = int(layer["layer_idx"])
            counts = np.asarray(layer.get("activation_counts", []), dtype=np.float64)
            if counts.size == 0:
                continue
            num_layers = max(num_layers, layer_idx + 1)
            num_experts = max(num_experts, counts.shape[0])
            layer_counts = ensure_2d(layer_counts, max(num_layers, 48), max(num_experts, 128))
            if layer_counts.shape[0] <= layer_idx or layer_counts.shape[1] < counts.shape[0]:
                grown = np.zeros((max(layer_counts.shape[0], layer_idx + 1), max(layer_counts.shape[1], counts.shape[0])))
                grown[: layer_counts.shape[0], : layer_counts.shape[1]] = layer_counts
                layer_counts = grown
            layer_counts[layer_idx, : counts.shape[0]] += counts

            task = p.get("task_type", "Unknown")
            if task_expert_counts[task] is None:
                task_expert_counts[task] = np.zeros(counts.shape[0], dtype=np.float64)
            if task_expert_counts[task].shape[0] < counts.shape[0]:
                grown = np.zeros(counts.shape[0], dtype=np.float64)
                grown[: task_expert_counts[task].shape[0]] = task_expert_counts[task]
                task_expert_counts[task] = grown
            task_expert_counts[task][: counts.shape[0]] += counts

            if layer.get("routing_entropy_normalized") is not None:
                layer_entropy[layer_idx].append(float(layer["routing_entropy_normalized"]))
            lb = layer.get("load_balance", {})
            if lb.get("hot_to_mean_ratio") is not None:
                layer_hot_ratio[layer_idx].append(float(lb["hot_to_mean_ratio"]))
            if lb.get("empty_experts") is not None:
                layer_empty[layer_idx].append(float(lb["empty_experts"]))
            if layer.get("moe_forward_time_ms") is not None:
                layer_moe_ms[layer_idx].append(float(layer["moe_forward_time_ms"]))

            for modality, md in layer.get("modality_expert_distribution", {}).items():
                m_counts = np.asarray(md.get("activation_counts", []), dtype=np.float64)
                if m_counts.size == 0:
                    continue
                cur = modality_layer_counts.get(modality)
                cur = ensure_2d(cur, max(num_layers, 48), max(num_experts, 128))
                if cur.shape[0] <= layer_idx or cur.shape[1] < m_counts.shape[0]:
                    grown = np.zeros((max(cur.shape[0], layer_idx + 1), max(cur.shape[1], m_counts.shape[0])))
                    grown[: cur.shape[0], : cur.shape[1]] = cur
                    cur = grown
                cur[layer_idx, : m_counts.shape[0]] += m_counts
                modality_layer_counts[modality] = cur

    if layer_counts is None:
        layer_counts = np.zeros((0, 0), dtype=np.float64)
    return {
        "samples": samples,
        "layer_counts": layer_counts,
        "modality_layer_counts": modality_layer_counts,
        "task_expert_counts": dict(task_expert_counts),
        "layer_entropy": layer_entropy,
        "layer_hot_ratio": layer_hot_ratio,
        "layer_empty": layer_empty,
        "layer_moe_ms": layer_moe_ms,
        "profiles_loaded": loaded,
        "skipped_profile_lines": skipped,
    }


def merge_predictions(samples, predictions):
    pred_by_index = {p.get("sample_index"): p for p in predictions}
    for row in samples:
        pred = pred_by_index.get(row.get("sample_index"), {})
        row["correct_answer"] = pred.get("correct_answer")
        row["predicted_answer"] = pred.get("predicted_answer")
        row["is_correct"] = pred.get("is_correct")
        row["error"] = pred.get("error")
    return samples


def task_stats(samples):
    stats = defaultdict(lambda: defaultdict(list))
    acc = defaultdict(lambda: [0, 0])
    for row in samples:
        task = row.get("task_type", "Unknown")
        for key in ["prefill_latency_ms", "input_seq_len", "video_tokens", "audio_tokens", "generate_latency_ms"]:
            val = row.get(key)
            if val is not None:
                stats[task][key].append(float(val))
        if row.get("is_correct") is not None:
            acc[task][1] += 1
            acc[task][0] += int(bool(row.get("is_correct")))
    out = {}
    for task, values in stats.items():
        out[task] = {
            "count": len(values.get("prefill_latency_ms", [])),
            "prefill_mean_ms": safe_mean(values.get("prefill_latency_ms", [])),
            "prefill_p50_ms": safe_median(values.get("prefill_latency_ms", [])),
            "prefill_p95_ms": pct(values.get("prefill_latency_ms", []), 95),
            "input_seq_len_mean": safe_mean(values.get("input_seq_len", [])),
            "video_tokens_mean": safe_mean(values.get("video_tokens", [])),
            "audio_tokens_mean": safe_mean(values.get("audio_tokens", [])),
            "accuracy": acc[task][0] / acc[task][1] if acc[task][1] else None,
            "correct": acc[task][0],
            "total": acc[task][1],
        }
    return out


def write_csv(samples, path):
    keys = [
        "sample_index",
        "task_type",
        "input_seq_len",
        "prefill_latency_ms",
        "generate_latency_ms",
        "attention_flops",
        "moe_ffn_flops",
        "kv_cache_write_bytes",
        "text_tokens",
        "video_tokens",
        "audio_tokens",
        "decoded_video_frames",
        "is_correct",
        "correct_answer",
        "predicted_answer",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in samples:
            writer.writerow({k: row.get(k) for k in keys})


def plot_progress(samples, output):
    xs = np.array([s["sample_index"] for s in samples], dtype=np.float64)
    correct = np.array([1 if s.get("is_correct") else 0 for s in samples], dtype=np.float64)
    lat = np.array([s.get("prefill_latency_ms") or np.nan for s in samples], dtype=np.float64)
    cumulative_acc = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    window = min(50, max(1, len(lat)))
    kernel = np.ones(window) / window
    valid = np.where(np.isfinite(lat), lat, np.nanmedian(lat))
    rolling_lat = np.convolve(valid, kernel, mode="same")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(xs, cumulative_acc, color="#2563eb", linewidth=2)
    axes[0].set_ylabel("Cumulative accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.25)
    axes[1].scatter(xs, lat, s=9, alpha=0.35, color="#64748b", label="sample")
    axes[1].plot(xs, rolling_lat, color="#dc2626", linewidth=2, label=f"rolling mean ({window})")
    axes[1].set_ylabel("Prefill latency (ms)")
    axes[1].set_xlabel("Sample index")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle(f"{BENCH}: progress, accuracy, and prefill latency")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_task_bars(stats, output, top_count):
    tasks = sorted(stats, key=lambda k: stats[k]["count"], reverse=True)[:top_count]
    counts = np.array([stats[t]["count"] for t in tasks])
    lat = np.array([stats[t]["prefill_mean_ms"] or 0 for t in tasks])
    acc = np.array([stats[t]["accuracy"] if stats[t]["accuracy"] is not None else np.nan for t in tasks])
    y = np.arange(len(tasks))

    fig, axes = plt.subplots(1, 2, figsize=(15, max(6, len(tasks) * 0.45)), sharey=True)
    axes[0].barh(y, lat, color="#0f766e")
    axes[0].set_xlabel("Mean prefill latency (ms)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(tasks)
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.25)
    for i, c in enumerate(counts):
        axes[0].text(lat[i], i, f" n={int(c)}", va="center", fontsize=8)
    axes[1].barh(y, acc, color="#7c3aed")
    axes[1].set_xlabel("Accuracy so far")
    axes[1].set_xlim(0, 1)
    axes[1].grid(axis="x", alpha=0.25)
    fig.suptitle(f"{BENCH}: task-level latency and accuracy")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_scatter(samples, output):
    seq = np.array([s.get("input_seq_len") or np.nan for s in samples], dtype=np.float64)
    lat = np.array([s.get("prefill_latency_ms") or np.nan for s in samples], dtype=np.float64)
    video_tokens = np.array([s.get("video_tokens") or 0 for s in samples], dtype=np.float64)
    audio_tokens = np.array([s.get("audio_tokens") or 0 for s in samples], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(seq, lat, s=10, alpha=0.45, c=video_tokens, cmap="viridis")
    axes[0].set_xlabel("Input sequence length")
    axes[0].set_ylabel("Prefill latency (ms)")
    axes[0].set_title("Latency vs input length")
    axes[0].grid(alpha=0.25)
    cbar = fig.colorbar(axes[0].collections[0], ax=axes[0])
    cbar.set_label("Video tokens")
    axes[1].scatter(audio_tokens, lat, s=10, alpha=0.45, color="#ea580c")
    axes[1].set_xlabel("Audio tokens")
    axes[1].set_ylabel("Prefill latency (ms)")
    axes[1].set_title("Latency vs audio tokens")
    axes[1].grid(alpha=0.25)
    fig.suptitle(f"{BENCH}: prefill scaling signals")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_costs(samples, output):
    attn = np.array([s.get("attention_flops") or np.nan for s in samples], dtype=np.float64) / 1e12
    moe = np.array([s.get("moe_ffn_flops") or np.nan for s in samples], dtype=np.float64) / 1e12
    kv = np.array([s.get("kv_cache_write_bytes") or np.nan for s in samples], dtype=np.float64) / (1024**3)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist(attn[np.isfinite(attn)], bins=30, color="#2563eb", alpha=0.85)
    axes[0].set_xlabel("Attention FLOPs estimate (TFLOPs)")
    axes[0].set_ylabel("Samples")
    axes[1].hist(moe[np.isfinite(moe)], bins=30, color="#16a34a", alpha=0.85)
    axes[1].set_xlabel("MoE FFN FLOPs estimate (TFLOPs)")
    axes[2].hist(kv[np.isfinite(kv)], bins=30, color="#9333ea", alpha=0.85)
    axes[2].set_xlabel("KV cache writes estimate (GiB)")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(f"{BENCH}: prefill cost distributions")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_moe_layer(metrics, output):
    layers = sorted(set(metrics["layer_entropy"]) | set(metrics["layer_hot_ratio"]) | set(metrics["layer_empty"]))
    x = np.array(layers)
    entropy = np.array([safe_mean(metrics["layer_entropy"].get(i, [])) or np.nan for i in layers])
    hot = np.array([safe_mean(metrics["layer_hot_ratio"].get(i, [])) or np.nan for i in layers])
    empty = np.array([safe_mean(metrics["layer_empty"].get(i, [])) or np.nan for i in layers])
    moe_ms = np.array([safe_mean(metrics["layer_moe_ms"].get(i, [])) or np.nan for i in layers])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes[0, 0].plot(x, entropy, marker="o", linewidth=1.5, color="#2563eb")
    axes[0, 0].set_ylabel("Normalized entropy")
    axes[0, 1].plot(x, hot, marker="o", linewidth=1.5, color="#dc2626")
    axes[0, 1].set_ylabel("Hot/mean expert batch")
    axes[1, 0].plot(x, empty, marker="o", linewidth=1.5, color="#7c3aed")
    axes[1, 0].set_ylabel("Empty experts")
    axes[1, 1].plot(x, moe_ms, marker="o", linewidth=1.5, color="#0f766e")
    axes[1, 1].set_ylabel("MoE block time (ms)")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
        ax.set_xlabel("Layer")
    fig.suptitle(f"{BENCH}: MoE routing and system overhead by layer")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_expert_heatmap(layer_counts, output):
    if layer_counts.size == 0:
        return
    row_sums = layer_counts.sum(axis=1, keepdims=True)
    freq = np.divide(layer_counts, np.maximum(row_sums, 1.0))
    fig, ax = plt.subplots(figsize=(15, 7))
    im = ax.imshow(np.log10(freq + 1e-5), aspect="auto", cmap="magma")
    ax.set_xlabel("Expert id")
    ax.set_ylabel("Layer")
    ax.set_title(f"{BENCH}: aggregate expert activation frequency by layer (log10)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(freq + 1e-5)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_modality_heatmap(modality_layer_counts, output):
    collapsed = {}
    for modality, arr in modality_layer_counts.items():
        collapsed[modality] = arr.sum(axis=0)
    if not collapsed:
        return []
    total = sum(collapsed.values())
    top_experts = np.argsort(total)[::-1][:32]
    modalities = [m for m in ["text", "video", "audio", "image"] if m in collapsed]
    mat = []
    for m in modalities:
        vals = collapsed[m][top_experts]
        vals = vals / max(vals.sum(), 1.0)
        mat.append(vals)
    mat = np.asarray(mat)
    fig, ax = plt.subplots(figsize=(14, max(3.5, len(modalities) * 1.1)))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(top_experts)))
    ax.set_xticklabels([str(int(x)) for x in top_experts], rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(modalities)))
    ax.set_yticklabels(modalities)
    ax.set_xlabel("Top aggregate expert ids")
    ax.set_title(f"{BENCH}: modality expert distribution, top 32 experts")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Within-modality activation share")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return [int(x) for x in top_experts]


def plot_task_overlap(task_expert_counts, task_stats_map, output, top_task_count):
    tasks = sorted(task_expert_counts, key=lambda t: task_stats_map.get(t, {}).get("count", 0), reverse=True)[:top_task_count]
    top_sets = {}
    for task in tasks:
        counts = task_expert_counts[task]
        if counts is None or counts.size == 0:
            continue
        top_sets[task] = set(int(i) for i in np.argsort(counts)[::-1][:32])
    tasks = [t for t in tasks if t in top_sets]
    if not tasks:
        return
    mat = np.zeros((len(tasks), len(tasks)), dtype=np.float64)
    for i, left in enumerate(tasks):
        for j, right in enumerate(tasks):
            union = top_sets[left] | top_sets[right]
            inter = top_sets[left] & top_sets[right]
            mat[i, j] = len(inter) / len(union) if union else 0
    fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 0.7), max(7, len(tasks) * 0.65)))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=8)
    ax.set_title(f"{BENCH}: task expert overlap, top-32 Jaccard")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Jaccard overlap")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def summarize(samples, predictions, skipped_pred, task_stats_map, metrics, top_modality_experts):
    total = len(samples)
    correct = sum(1 for s in samples if s.get("is_correct"))
    lat = [float(s["prefill_latency_ms"]) for s in samples if s.get("prefill_latency_ms") is not None]
    seq = [float(s["input_seq_len"]) for s in samples if s.get("input_seq_len") is not None]
    attn = [float(s["attention_flops"]) for s in samples if s.get("attention_flops") is not None]
    moe = [float(s["moe_ffn_flops"]) for s in samples if s.get("moe_ffn_flops") is not None]
    kv = [float(s["kv_cache_write_bytes"]) for s in samples if s.get("kv_cache_write_bytes") is not None]
    chunked = sum(1 for s in samples if s.get("chunked_prefill_observed"))
    flash = sum(1 for s in samples if s.get("flash_attention_requested"))
    layer_counts = metrics["layer_counts"]
    if layer_counts.size:
        total_counts = layer_counts.sum(axis=0)
        global_top = [
            {"expert_id": int(i), "count": int(total_counts[i]), "frequency": float(total_counts[i] / max(total_counts.sum(), 1.0))}
            for i in np.argsort(total_counts)[::-1][:16]
        ]
    else:
        global_top = []
    return {
        "snapshot_at": dt.datetime.now().isoformat(timespec="seconds"),
        "samples_profiled": total,
        "predictions_loaded": len(predictions),
        "profiles_loaded": metrics.get("profiles_loaded", total),
        "skipped_prediction_lines": skipped_pred,
        "skipped_profile_lines": metrics.get("skipped_profile_lines", 0),
        "accuracy_so_far": correct / total if total else None,
        "correct_so_far": correct,
        "prefill_latency_ms": {
            "mean": safe_mean(lat),
            "p50": safe_median(lat),
            "p95": pct(lat, 95),
            "max": max(lat) if lat else None,
        },
        "input_seq_len": {"mean": safe_mean(seq), "p50": safe_median(seq), "p95": pct(seq, 95), "max": max(seq) if seq else None},
        "attention_flops_estimate": {"mean": safe_mean(attn), "p50": safe_median(attn), "p95": pct(attn, 95)},
        "moe_ffn_flops_estimate": {"mean": safe_mean(moe), "p50": safe_median(moe), "p95": pct(moe, 95)},
        "kv_cache_write_bytes_estimate": {"mean": safe_mean(kv), "p50": safe_median(kv), "p95": pct(kv, 95)},
        "flash_attention_requested_samples": flash,
        "chunked_prefill_observed_samples": chunked,
        "task_stats": task_stats_map,
        "global_top_experts": global_top,
        "modality_top_expert_ids": top_modality_experts,
    }


def write_report(summary, output_dir, image_names):
    def num(value, digits=2):
        if value is None:
            return "n/a"
        if isinstance(value, int):
            return str(value)
        return f"{value:.{digits}f}"

    lines = [
        f"# {BENCH} Profile Snapshot",
        "",
        f"- Snapshot time: `{summary['snapshot_at']}`",
        f"- Samples profiled: `{summary['samples_profiled']}`",
        f"- Accuracy so far: `{summary['correct_so_far']}/{summary['samples_profiled']}` = `{num(summary['accuracy_so_far'], 4)}`",
        f"- Prefill latency mean / p50 / p95 / max: `{num(summary['prefill_latency_ms']['mean'])}` / `{num(summary['prefill_latency_ms']['p50'])}` / `{num(summary['prefill_latency_ms']['p95'])}` / `{num(summary['prefill_latency_ms']['max'])}` ms",
        f"- Input seq len mean / p50 / p95 / max: `{num(summary['input_seq_len']['mean'])}` / `{num(summary['input_seq_len']['p50'])}` / `{num(summary['input_seq_len']['p95'])}` / `{num(summary['input_seq_len']['max'])}`",
        f"- FlashAttention requested samples: `{summary['flash_attention_requested_samples']}`",
        f"- Chunked prefill observed samples: `{summary['chunked_prefill_observed_samples']}`",
        "",
        "## Figures",
        "",
    ]
    for name in image_names:
        lines.append(f"![{name}]({name})")
        lines.append("")
    lines.extend(
        [
            "## Global Top Experts",
            "",
            "| rank | expert_id | activation_count | frequency |",
            "|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(summary.get("global_top_experts", []), 1):
        lines.append(f"| {rank} | {row['expert_id']} | {row['count']} | {row['frequency']:.5f} |")
    lines.extend(["", "## Task Stats", "", "| task | n | acc | prefill mean ms | prefill p95 ms | input len mean |", "|---|---:|---:|---:|---:|---:|"])
    for task, stats in sorted(summary["task_stats"].items(), key=lambda item: item[1]["count"], reverse=True):
        lines.append(
            f"| {task} | {stats['count']} | {num(stats['accuracy'], 4)} | "
            f"{num(stats['prefill_mean_ms'])} | {num(stats['prefill_p95_ms'])} | {num(stats['input_seq_len_mean'])} |"
        )
    (output_dir / f"{BENCH}_profile_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions, skipped_pred = load_jsonl(run_dir / "predictions.jsonl")
    metrics = accumulate_profiles(run_dir / "profile_samples.jsonl")
    samples = merge_predictions(metrics["samples"], predictions)
    samples.sort(key=lambda row: row.get("sample_index") if row.get("sample_index") is not None else -1)
    stats = task_stats(samples)
    write_csv(samples, output_dir / f"{BENCH}_profile_samples.csv")

    images = [
        f"{BENCH}_progress_accuracy_prefill.png",
        f"{BENCH}_task_latency_accuracy.png",
        f"{BENCH}_prefill_scaling_scatter.png",
        f"{BENCH}_prefill_cost_distributions.png",
        f"{BENCH}_moe_layer_routing_overhead.png",
        f"{BENCH}_expert_activation_heatmap.png",
        f"{BENCH}_modality_expert_distribution.png",
        f"{BENCH}_task_expert_overlap.png",
    ]
    if samples:
        plot_progress(samples, output_dir / images[0])
        plot_task_bars(stats, output_dir / images[1], args.top_task_count)
        plot_scatter(samples, output_dir / images[2])
        plot_costs(samples, output_dir / images[3])
    plot_moe_layer(metrics, output_dir / images[4])
    plot_expert_heatmap(metrics["layer_counts"], output_dir / images[5])
    top_modality_experts = plot_modality_heatmap(metrics["modality_layer_counts"], output_dir / images[6])
    plot_task_overlap(metrics["task_expert_counts"], stats, output_dir / images[7], args.top_task_count)

    existing_images = [name for name in images if (output_dir / name).exists()]
    summary = summarize(samples, predictions, skipped_pred, stats, metrics, top_modality_experts)
    (output_dir / f"{BENCH}_profile_snapshot_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(summary, output_dir, existing_images)
    print(json.dumps({
        "output_dir": str(output_dir),
        "samples_profiled": summary["samples_profiled"],
        "accuracy_so_far": summary["accuracy_so_far"],
        "prefill_latency_mean_ms": summary["prefill_latency_ms"]["mean"],
        "images": existing_images,
    }, indent=2))


if __name__ == "__main__":
    main()

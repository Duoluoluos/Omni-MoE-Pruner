#!/usr/bin/env python3
"""E16-A: frozen, label-free routing-audit-to-settings selector.

The run intentionally uses no assistant answers and excludes videos whose
basenames occur in the paper's evaluation datasets.  It captures only compact
router summaries (never raw logits) and derives one deterministic configuration
plus sample-level bootstrap stability diagnostics.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import sys
import time
import traceback
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "qwen-omni-utils" / "src"))

SCHEMA = "arc-sceb/e16-a-auto-selector-v1"
MODALITIES = ("video", "audio", "text")
VIDEO_ID = 151656
AUDIO_ID = 151675
IMAGE_ID = 151655
NUM_LAYERS = 48
NUM_EXPERTS = 128
TOP_K = 8
GAP_KINDS = ("top4_boundary", "top6_boundary", "either_boundary")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def pack_array(array: np.ndarray) -> dict[str, Any]:
    value = np.ascontiguousarray(array)
    raw = value.tobytes(order="C")
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "raw_sha256": sha256_bytes(raw),
        "codec": "zlib+base64",
        "data": base64.b64encode(zlib.compress(raw, level=6)).decode("ascii"),
    }


def unpack_array(record: dict[str, Any]) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(record["data"]))
    if sha256_bytes(raw) != record["raw_sha256"]:
        raise ValueError("packed array checksum mismatch")
    value = np.frombuffer(raw, dtype=np.dtype(record["dtype"]))
    return value.reshape(record["shape"]).copy()


def text_content(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                pieces.append(str(item.get("text", "")))
        return "\n".join(pieces)
    return str(content)


def prompt_parts(record: dict[str, Any]) -> tuple[str, str]:
    system = []
    user = []
    for message in record.get("messages", []):
        role = message.get("role")
        if role == "system":
            system.append(text_content(message))
        elif role == "user":
            user.append(text_content(message))
        # The assistant branch is deliberately never read.
    system_text = "\n".join(x for x in system if x).strip()
    user_text = "\n".join(x for x in user if x).replace("<video>", "").strip()
    if not system_text:
        system_text = "You are a precise audio-visual assistant."
    if not user_text:
        raise ValueError("record has no usable user prompt")
    return system_text, user_text


def eval_video_basenames(directories: list[Path]) -> set[str]:
    names: set[str] = set()
    for directory in directories:
        if not directory.exists():
            continue
        for _root, child_directories, filenames in os.walk(directory):
            child_directories[:] = [
                name
                for name in child_directories
                if name not in {".cache", ".git", "__pycache__"}
            ]
            for filename in filenames:
                if Path(filename).suffix.lower() in VIDEO_EXTENSIONS:
                    names.add(filename)
    return names


def prepare_samples(protocol: dict[str, Any]) -> dict[str, Any]:
    data = protocol["data"]
    source = REPO / data["source_jsonl"]
    source_hash = sha256_file(source)
    expected_source_hash = data.get("source_sha256")
    if expected_source_hash and source_hash != expected_source_hash:
        raise ValueError(
            f"source checksum changed: expected {expected_source_hash}, got {source_hash}"
        )

    excluded_dirs = [REPO / value for value in data["excluded_evaluation_directories"]]
    eval_names = eval_video_basenames(excluded_dirs)
    candidates: list[dict[str, Any]] = []
    seen_video: set[str] = set()
    counters = Counter()
    with source.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            counters["source_records"] += 1
            record = json.loads(line)
            videos = record.get("videos") or []
            if not videos:
                counters["missing_video_field"] += 1
                continue
            video_path = Path(videos[0])
            if not video_path.is_absolute():
                video_path = REPO / video_path
            basename = video_path.name
            if basename in eval_names:
                counters["evaluation_overlap_excluded"] += 1
                continue
            if basename in seen_video:
                counters["duplicate_video_excluded"] += 1
                continue
            if not video_path.exists():
                counters["missing_video_file"] += 1
                continue
            system_text, user_text = prompt_parts(record)
            prompt_hash = sha256_bytes(
                canonical_json({"system": system_text, "user": user_text}).encode()
            )
            stat = video_path.stat()
            stable_key = sha256_bytes(
                (
                    data["selection_salt"]
                    + "\0"
                    + basename
                    + "\0"
                    + str(line_index)
                ).encode()
            )
            candidates.append(
                {
                    "line_index": line_index,
                    "video_basename": basename,
                    "video_path": str(video_path),
                    "video_size_bytes": stat.st_size,
                    "prompt_sha256": prompt_hash,
                    "stable_key": stable_key,
                }
            )
            seen_video.add(basename)

    candidates.sort(key=lambda value: value["stable_key"])
    selected = candidates[: int(data["sample_limit"])]
    if len(selected) < int(protocol["stability_gates"]["minimum_completed_samples"]):
        raise ValueError(
            f"only {len(selected)} eligible records; fewer than frozen minimum"
        )
    selection_hash = sha256_bytes(canonical_json(selected).encode())
    expected_selection_hash = data.get("selected_samples_sha256")
    if expected_selection_hash and selection_hash != expected_selection_hash:
        raise ValueError(
            f"selection checksum changed: expected {expected_selection_hash}, got {selection_hash}"
        )
    return {
        "schema_version": SCHEMA,
        "created_at": utc_now(),
        "source_jsonl": str(source.relative_to(REPO)),
        "source_sha256": source_hash,
        "excluded_evaluation_directories": data["excluded_evaluation_directories"],
        "evaluation_video_basename_count": len(eval_names),
        "labels_used": False,
        "roles_used": ["system", "user"],
        "selection_rule": "deduplicate basename; exclude evaluation basenames; sort SHA256(salt, basename, line index); take first N",
        "selection_counters": dict(counters),
        "eligible_count": len(candidates),
        "selected_count": len(selected),
        "selected_samples_sha256": selection_hash,
        "samples": selected,
    }


def load_source_record(source: Path, line_index: int) -> dict[str, Any]:
    with source.open(encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == line_index:
                return json.loads(line)
    raise IndexError(f"source line {line_index} not found")


def build_conversation(record: dict[str, Any], selected: dict[str, Any]) -> list[dict[str, Any]]:
    system_text, user_text = prompt_parts(record)
    observed_hash = sha256_bytes(
        canonical_json({"system": system_text, "user": user_text}).encode()
    )
    if observed_hash != selected["prompt_sha256"]:
        raise ValueError("prompt changed after sample freeze")
    videos = record.get("videos") or []
    video_path = Path(videos[0])
    if not video_path.is_absolute():
        video_path = REPO / video_path
    if video_path.name != selected["video_basename"]:
        raise ValueError("video basename changed after sample freeze")
    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": user_text},
            ],
        },
    ]


class RouterSummaryCapture:
    """Capture small sufficient statistics from each prefill gate output."""

    def __init__(self, model: Any, gap_boundaries: np.ndarray):
        import torch

        self.torch = torch
        self.gap_boundaries_np = np.asarray(gap_boundaries, dtype=np.float32)
        self.handles = []
        self.layer_modules: dict[int, str] = {}
        for name, module in model.thinker.named_modules():
            cls = module.__class__.__name__.lower()
            if "sparsemoeblock" not in cls or not hasattr(module, "gate"):
                continue
            layer_index = self._layer_index(name)
            if layer_index is None:
                continue
            self.layer_modules[layer_index] = name
            self.handles.append(
                module.gate.register_forward_hook(self._make_hook(layer_index))
            )
        if sorted(self.layer_modules) != list(range(NUM_LAYERS)):
            raise RuntimeError(
                f"expected {NUM_LAYERS} MoE gates, found {sorted(self.layer_modules)}"
            )
        self.reset()

    @staticmethod
    def _layer_index(name: str) -> int | None:
        parts = name.split(".")
        for index, part in enumerate(parts[:-1]):
            if part == "layers" and parts[index + 1].isdigit():
                return int(parts[index + 1])
        return None

    def reset(self) -> None:
        self.seq_len = 0
        self.masks: tuple[Any, ...] = ()
        self.gap_boundaries = None
        self.token_counts = np.zeros(len(MODALITIES), dtype=np.int32)
        self.activation_counts = np.zeros(
            (NUM_LAYERS, len(MODALITIES), NUM_EXPERTS), dtype=np.uint32
        )
        self.margin12_sums = np.zeros(
            (NUM_LAYERS, len(MODALITIES)), dtype=np.float64
        )
        self.gap_any_sums = np.zeros(
            (NUM_LAYERS, len(MODALITIES)), dtype=np.float64
        )
        self.gap_histograms = np.zeros(
            (
                NUM_LAYERS,
                len(MODALITIES),
                len(GAP_KINDS),
                len(self.gap_boundaries_np) + 1,
            ),
            dtype=np.uint32,
        )
        self.seen_layers: set[int] = set()

    def begin(self, input_ids: Any) -> None:
        torch = self.torch
        self.reset()
        ids = input_ids.reshape(-1)
        self.seq_len = int(ids.numel())
        video = ids.eq(VIDEO_ID)
        audio = ids.eq(AUDIO_ID)
        text = ~(video | audio | ids.eq(IMAGE_ID))
        self.masks = (video, audio, text)
        self.token_counts = np.asarray(
            [int(mask.sum().item()) for mask in self.masks], dtype=np.int32
        )
        self.gap_boundaries = torch.as_tensor(
            self.gap_boundaries_np, device=ids.device, dtype=torch.float32
        )

    def _make_hook(self, layer_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            torch = self.torch
            if not self.masks or layer_index in self.seen_layers:
                return
            logits = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(logits):
                return
            if logits.dim() == 3:
                logits = logits.reshape(-1, logits.shape[-1])
            if logits.dim() != 2 or logits.shape[0] != self.seq_len:
                return
            if logits.shape[1] != NUM_EXPERTS:
                raise RuntimeError(
                    f"layer {layer_index}: expected {NUM_EXPERTS} experts, got {logits.shape[1]}"
                )
            probs = torch.softmax(logits.detach(), dim=-1, dtype=torch.float32)
            top_values, top_indices = torch.topk(
                probs, TOP_K, dim=-1, largest=True, sorted=True
            )
            for modality_index, mask in enumerate(self.masks):
                count = int(mask.sum().item())
                if count == 0:
                    continue
                values = top_values[mask]
                indices = top_indices[mask]
                activations = torch.bincount(
                    indices.reshape(-1), minlength=NUM_EXPERTS
                )
                margin12 = values[:, 0] - values[:, 1]
                gap4 = values[:, 3] - values[:, 4]
                gap6 = values[:, 5] - values[:, 6]
                gap_any = torch.maximum(gap4, gap6)
                self.activation_counts[layer_index, modality_index] = (
                    activations.to(torch.int64).cpu().numpy().astype(np.uint32)
                )
                self.margin12_sums[layer_index, modality_index] = float(
                    margin12.sum().item()
                )
                self.gap_any_sums[layer_index, modality_index] = float(
                    gap_any.sum().item()
                )
                for gap_index, gap in enumerate((gap4, gap6, gap_any)):
                    buckets = torch.bucketize(
                        gap, self.gap_boundaries, right=True
                    )
                    histogram = torch.bincount(
                        buckets, minlength=len(self.gap_boundaries_np) + 1
                    )
                    self.gap_histograms[
                        layer_index, modality_index, gap_index
                    ] = histogram.to(torch.int64).cpu().numpy().astype(np.uint32)
            self.seen_layers.add(layer_index)

        return hook

    def export(self) -> dict[str, Any]:
        missing = sorted(set(range(NUM_LAYERS)) - self.seen_layers)
        if missing:
            raise RuntimeError(f"missing prefill summaries for layers {missing}")
        return {
            "seq_len": self.seq_len,
            "token_counts": {
                modality: int(self.token_counts[index])
                for index, modality in enumerate(MODALITIES)
            },
            "arrays": {
                "activation_counts": pack_array(self.activation_counts),
                "margin12_sums": pack_array(self.margin12_sums),
                "gap_any_sums": pack_array(self.gap_any_sums),
                "gap_histograms": pack_array(self.gap_histograms),
            },
        }

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def read_raw_records(path: Path) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    samples: dict[int, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    if not path.exists():
        return samples, errors
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            payload = dict(record)
            expected = payload.pop("record_sha256", None)
            if expected != sha256_bytes(canonical_json(payload).encode()):
                raise ValueError("raw JSONL record checksum mismatch")
            if record.get("kind") == "sample":
                samples[int(record["line_index"])] = record
            elif record.get("kind") == "error":
                errors.append(record)
    return samples, errors


def norm_entropy(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=-1, keepdims=True)
    probs = np.divide(
        counts,
        totals,
        out=np.zeros_like(counts, dtype=np.float64),
        where=totals > 0,
    )
    logs = np.zeros_like(probs)
    np.log(probs, out=logs, where=probs > 0)
    entropy = -(probs * logs).sum(axis=-1) / math.log(NUM_EXPERTS)
    entropy[totals[..., 0] <= 0] = np.nan
    return entropy


def modality_scores(
    activation_counts: np.ndarray, token_counts: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    aggregate = activation_counts[indices].sum(axis=0, dtype=np.float64)
    entropy = norm_entropy(aggregate)
    mean_entropy = np.nanmean(entropy, axis=0)
    token_volume = token_counts[indices].sum(axis=0, dtype=np.float64)
    return token_volume * mean_entropy, mean_entropy


def best_layer_window(layer_scores: np.ndarray, width: int) -> tuple[int, int, float]:
    candidates = []
    for start in range(NUM_LAYERS - width + 1):
        score = float(np.mean(layer_scores[start : start + width]))
        candidates.append((score, start))
    # Earliest start is the deterministic tie-break.
    score, start = max(candidates, key=lambda value: (value[0], -value[1]))
    return start, start + width, score


def threshold_configuration(
    gap_histograms: np.ndarray,
    indices: np.ndarray,
    modality_index: int,
    layer_start: int,
    layer_end: int,
    boundaries: np.ndarray,
    target_mean_budget: float,
) -> dict[str, Any]:
    aggregate = gap_histograms[
        indices, layer_start:layer_end, modality_index
    ].sum(axis=(0, 1), dtype=np.int64)
    hist4 = aggregate[0]
    hist_any = aggregate[2]
    total = int(hist_any.sum())
    if total <= 0 or int(hist4.sum()) != total:
        raise ValueError("invalid boundary histograms")
    candidates = []
    for boundary_index, threshold in enumerate(boundaries):
        count4 = int(hist4[boundary_index + 1 :].sum())
        count_any = int(hist_any[boundary_index + 1 :].sum())
        count6 = max(0, count_any - count4)
        count8 = total - count_any
        mean_budget = (4 * count4 + 6 * count6 + 8 * count8) / total
        candidates.append(
            {
                "threshold": float(threshold),
                "mean_budget": float(mean_budget),
                "counts": {"4": count4, "6": count6, "8": count8},
            }
        )
    # Prefer a safer (larger) mean budget on an exact distance tie.
    chosen = min(
        candidates,
        key=lambda value: (
            abs(value["mean_budget"] - target_mean_budget),
            -value["mean_budget"],
            value["threshold"],
        ),
    )
    chosen["shares"] = {
        key: float(value / total) for key, value in chosen["counts"].items()
    }
    chosen["target_mean_budget"] = target_mean_budget
    chosen["tokens_times_layers"] = total
    return chosen


def quantile(values: np.ndarray, q: float, method: str = "linear") -> float:
    return float(np.quantile(values, q, method=method))


def gap_threshold_grid(selector: dict[str, Any]) -> np.ndarray:
    spec = selector["gap_threshold_grid"]
    positive = np.geomspace(
        float(spec["positive_min"]),
        float(spec["positive_max"]),
        int(spec["positive_points"]),
        dtype=np.float64,
    )
    return np.concatenate((np.asarray([0.0], dtype=np.float64), positive))


def derive_configuration(
    activation_counts: np.ndarray,
    token_counts: np.ndarray,
    margin12_sums: np.ndarray,
    gap_any_sums: np.ndarray,
    gap_histograms: np.ndarray,
    indices: np.ndarray,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    selector = protocol["selector"]
    scores, entropies = modality_scores(activation_counts, token_counts, indices)
    modality_index = max(
        range(len(MODALITIES)), key=lambda index: (scores[index], -index)
    )
    denominator = float(token_counts[indices, modality_index].sum())
    if denominator <= 0:
        raise ValueError("selected modality has no tokens")
    layer_scores = gap_any_sums[indices, :, modality_index].sum(axis=0) / denominator
    layer_start, layer_end, window_score = best_layer_window(
        layer_scores, int(selector["layer_window_width"])
    )
    boundaries = gap_threshold_grid(selector)
    cutoff = threshold_configuration(
        gap_histograms,
        indices,
        modality_index,
        layer_start,
        layer_end,
        boundaries,
        float(selector["target_mean_budget"]),
    )

    audio_index = MODALITIES.index("audio")
    positive_audio = token_counts[indices, audio_index]
    positive_audio = positive_audio[positive_audio > 0]
    if positive_audio.size:
        min_audio_tokens = int(
            quantile(
                positive_audio.astype(np.float64),
                float(selector["minimum_audio_token_quantile"]),
                method="lower",
            )
        )
        qualifying = indices[token_counts[indices, audio_index] >= min_audio_tokens]
        denominators = token_counts[qualifying, audio_index, None]
        audio_layer_means = np.divide(
            margin12_sums[qualifying, :, audio_index],
            denominators,
            out=np.zeros_like(margin12_sums[qualifying, :, audio_index]),
            where=denominators > 0,
        ).reshape(-1)
        uncertainty_threshold = quantile(
            audio_layer_means,
            float(selector["audio_uncertainty_quantile"]),
        )
    else:
        min_audio_tokens = 0
        uncertainty_threshold = 0.0

    return {
        "target_modality": MODALITIES[modality_index],
        "modality_opportunity_scores": {
            modality: float(scores[index])
            for index, modality in enumerate(MODALITIES)
        },
        "modality_mean_expert_use_entropy": {
            modality: float(entropies[index])
            for index, modality in enumerate(MODALITIES)
        },
        "eligible_layer_start": layer_start,
        "eligible_layer_end_exclusive": layer_end,
        "layer_window_width": layer_end - layer_start,
        "selected_window_mean_either_boundary_gap": window_score,
        "per_layer_either_boundary_gap_mean": [float(x) for x in layer_scores],
        "boundary_threshold": cutoff["threshold"],
        "implied_mean_budget": cutoff["mean_budget"],
        "target_mean_budget": cutoff["target_mean_budget"],
        "budget_counts": cutoff["counts"],
        "budget_shares": cutoff["shares"],
        "minimum_audio_tokens": min_audio_tokens,
        "audio_uncertainty_threshold": uncertainty_threshold,
        "insufficient_audio_min_budget": int(
            selector["insufficient_audio_min_budget"]
        ),
        "uncertain_audio_min_budget": int(selector["uncertain_audio_min_budget"]),
    }


def config_digest(config: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(config).encode())


def interval_jaccard(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return float(intersection / union) if union else 1.0


def analyze_run(
    selected_packet: dict[str, Any],
    raw_path: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    raw_samples, errors = read_raw_records(raw_path)
    ordered = [
        raw_samples[sample["line_index"]]
        for sample in selected_packet["samples"]
        if sample["line_index"] in raw_samples
    ]
    if not ordered:
        raise ValueError("no completed sample records")
    token_counts = np.asarray(
        [
            [record["token_counts"][modality] for modality in MODALITIES]
            for record in ordered
        ],
        dtype=np.int64,
    )
    activation_counts = np.stack(
        [unpack_array(record["arrays"]["activation_counts"]) for record in ordered]
    )
    margin12_sums = np.stack(
        [unpack_array(record["arrays"]["margin12_sums"]) for record in ordered]
    )
    gap_any_sums = np.stack(
        [unpack_array(record["arrays"]["gap_any_sums"]) for record in ordered]
    )
    gap_histograms = np.stack(
        [unpack_array(record["arrays"]["gap_histograms"]) for record in ordered]
    )
    n = len(ordered)
    all_indices = np.arange(n, dtype=np.int64)
    full_config = derive_configuration(
        activation_counts,
        token_counts,
        margin12_sums,
        gap_any_sums,
        gap_histograms,
        all_indices,
        protocol,
    )
    repeated_config = derive_configuration(
        activation_counts,
        token_counts,
        margin12_sums,
        gap_any_sums,
        gap_histograms,
        all_indices,
        protocol,
    )
    repeat_match = config_digest(full_config) == config_digest(repeated_config)

    bootstrap = protocol["bootstrap"]
    rng = np.random.default_rng(int(bootstrap["seed"]))
    target_modalities = []
    windows = []
    thresholds = []
    mean_budgets = []
    for _ in range(int(bootstrap["draws"])):
        indices = rng.integers(0, n, size=n, dtype=np.int64)
        config = derive_configuration(
            activation_counts,
            token_counts,
            margin12_sums,
            gap_any_sums,
            gap_histograms,
            indices,
            protocol,
        )
        target_modalities.append(config["target_modality"])
        windows.append(
            (
                config["eligible_layer_start"],
                config["eligible_layer_end_exclusive"],
            )
        )
        thresholds.append(config["boundary_threshold"])
        mean_budgets.append(config["implied_mean_budget"])

    target_counts = Counter(target_modalities)
    expected_target = protocol["selector"]["expected_target_modality"]
    expected_target_frequency = target_counts[expected_target] / len(target_modalities)
    jaccards = np.asarray(
        [
            interval_jaccard(
                start,
                end,
                full_config["eligible_layer_start"],
                full_config["eligible_layer_end_exclusive"],
            )
            for start, end in windows
        ],
        dtype=np.float64,
    )
    thresholds_array = np.asarray(thresholds, dtype=np.float64)
    threshold_p05, threshold_p50, threshold_p95 = np.quantile(
        thresholds_array, [0.05, 0.5, 0.95]
    )
    relative_width = float(
        (threshold_p95 - threshold_p05) / max(abs(threshold_p50), 1e-12)
    )
    budget_shares = sorted(full_config["budget_shares"].values(), reverse=True)
    second_budget_share = budget_shares[1] if len(budget_shares) > 1 else 0.0

    gates_spec = protocol["stability_gates"]
    gates = {
        "minimum_completed_samples": {
            "value": n,
            "threshold": int(gates_spec["minimum_completed_samples"]),
            "pass": n >= int(gates_spec["minimum_completed_samples"]),
        },
        "maximum_error_records": {
            "value": len(errors),
            "threshold": int(gates_spec["maximum_error_records"]),
            "pass": len(errors) <= int(gates_spec["maximum_error_records"]),
        },
        "zero_evaluation_overlap": {
            "value": selected_packet["selection_counters"].get(
                "evaluation_overlap_excluded", 0
            ),
            "threshold": 0,
            "pass": all(
                sample["video_basename"]
                not in eval_video_basenames(
                    [REPO / x for x in protocol["data"]["excluded_evaluation_directories"]]
                )
                for sample in selected_packet["samples"]
            ),
            "note": "value is excluded overlap count; pass checks selected overlap is zero",
        },
        "labels_not_used": {"value": False, "threshold": False, "pass": True},
        "deterministic_repeat_hash_match": {
            "value": repeat_match,
            "threshold": True,
            "pass": repeat_match,
        },
        "expected_target_selected": {
            "value": full_config["target_modality"],
            "threshold": expected_target,
            "pass": full_config["target_modality"] == expected_target,
        },
        "target_modality_bootstrap_frequency": {
            "value": expected_target_frequency,
            "threshold": float(
                gates_spec["minimum_target_modality_bootstrap_frequency"]
            ),
            "pass": expected_target_frequency
            >= float(gates_spec["minimum_target_modality_bootstrap_frequency"]),
        },
        "layer_window_jaccard_median": {
            "value": float(np.median(jaccards)),
            "threshold": float(gates_spec["minimum_layer_window_jaccard_median"]),
            "pass": float(np.median(jaccards))
            >= float(gates_spec["minimum_layer_window_jaccard_median"]),
        },
        "cutoff_relative_90_width": {
            "value": relative_width,
            "threshold": float(gates_spec["maximum_cutoff_relative_90_width"]),
            "pass": relative_width
            <= float(gates_spec["maximum_cutoff_relative_90_width"]),
        },
        "mean_budget_absolute_error": {
            "value": abs(
                full_config["implied_mean_budget"]
                - full_config["target_mean_budget"]
            ),
            "threshold": float(gates_spec["maximum_mean_budget_absolute_error"]),
            "pass": abs(
                full_config["implied_mean_budget"]
                - full_config["target_mean_budget"]
            )
            <= float(gates_spec["maximum_mean_budget_absolute_error"]),
        },
        "at_least_two_nontrivial_budget_levels": {
            "value": second_budget_share,
            "threshold": float(gates_spec["minimum_budget_share_for_two_levels"]),
            "pass": second_budget_share
            >= float(gates_spec["minimum_budget_share_for_two_levels"]),
        },
    }
    supports_claim = all(value["pass"] for value in gates.values())
    window_counts = Counter(f"{start}:{end}" for start, end in windows)
    return {
        "schema_version": SCHEMA,
        "experiment_id": protocol["experiment_id"],
        "completed_at": utc_now(),
        "result_status": "verified"
        if n >= int(gates_spec["minimum_completed_samples"])
        else "placeholder",
        "supports_e16_a_claim": supports_claim,
        "conclusion": (
            "The frozen label-free selector satisfies the deliberately loose E16-A stability gates."
            if supports_claim
            else "The run is informative, but the frozen selector does not satisfy every E16-A stability gate."
        ),
        "scope": "selector reproducibility/stability only; no task quality, training speed, or wall-clock claim",
        "sample_count": n,
        "error_record_count": len(errors),
        "selected_samples_sha256": selected_packet["selected_samples_sha256"],
        "full_configuration": full_config,
        "full_configuration_sha256": config_digest(full_config),
        "bootstrap": {
            "draws": int(bootstrap["draws"]),
            "seed": int(bootstrap["seed"]),
            "target_modality_counts": dict(target_counts),
            "expected_target_modality_frequency": expected_target_frequency,
            "layer_window_counts": dict(window_counts.most_common()),
            "layer_window_jaccard": {
                "p05": quantile(jaccards, 0.05),
                "median": quantile(jaccards, 0.50),
                "p95": quantile(jaccards, 0.95),
            },
            "boundary_threshold": {
                "p05": float(threshold_p05),
                "median": float(threshold_p50),
                "p95": float(threshold_p95),
                "relative_90_width": relative_width,
            },
            "implied_mean_budget": {
                "p05": quantile(np.asarray(mean_budgets), 0.05),
                "median": quantile(np.asarray(mean_budgets), 0.50),
                "p95": quantile(np.asarray(mean_budgets), 0.95),
            },
        },
        "stability_gates": gates,
        "protocol_sha256": sha256_bytes(canonical_json(protocol).encode()),
        "runner_sha256": sha256_file(Path(__file__)),
        "raw_results_sha256": sha256_file(raw_path),
    }


def log(message: str) -> None:
    print(f"[e16-a {utc_now()}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    output_dir = args.output_dir.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol["schema_version"] != SCHEMA:
        raise ValueError("unexpected protocol schema")
    expected_runner_hash = protocol["provenance"].get("runner_sha256")
    observed_runner_hash = sha256_file(Path(__file__))
    if expected_runner_hash and observed_runner_hash != expected_runner_hash:
        raise ValueError(
            f"runner checksum changed: expected {expected_runner_hash}, got {observed_runner_hash}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_dir / "launcher.pid", f"{os.getpid()}\n")
    selected_packet = prepare_samples(protocol)
    selected_path = output_dir / "selected_samples.json"
    if selected_path.exists():
        previous = json.loads(selected_path.read_text())
        if (
            previous["selected_samples_sha256"]
            != selected_packet["selected_samples_sha256"]
        ):
            raise ValueError("existing selected sample packet disagrees with frozen selector")
    atomic_write_json(selected_path, selected_packet)
    log(
        f"prepared {selected_packet['selected_count']} label-free samples; "
        f"selection_sha256={selected_packet['selected_samples_sha256']}"
    )
    if args.prepare_only:
        return
    if not protocol["data"].get("selected_samples_sha256"):
        raise ValueError("formal run requires a frozen selected_samples_sha256")

    import torch
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    model_path = REPO / protocol["model"]["path"]
    log(f"loading model from {model_path}")
    load_start = time.time()
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map=protocol["model"]["device_map"],
        attn_implementation=protocol["model"]["attention_implementation"],
    ).eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(str(model_path))
    log(f"model loaded in {time.time() - load_start:.1f}s")

    boundaries = gap_threshold_grid(protocol["selector"]).astype(np.float32)
    capture = RouterSummaryCapture(model, boundaries)
    log(f"hooked {len(capture.handles)} MoE gates")
    raw_path = output_dir / "raw_results.jsonl"
    completed, errors = read_raw_records(raw_path)
    source = REPO / protocol["data"]["source_jsonl"]
    started_at = utc_now()
    try:
        for ordinal, selected in enumerate(selected_packet["samples"], start=1):
            line_index = int(selected["line_index"])
            if line_index in completed:
                log(f"[{ordinal}/{selected_packet['selected_count']}] already complete; skipping")
                continue
            sample_start = time.time()
            try:
                record = load_source_record(source, line_index)
                conversation = build_conversation(record, selected)
                prompt = processor.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False
                )
                audios, images, videos = process_mm_info(
                    conversation, use_audio_in_video=True
                )
                inputs = processor(
                    text=prompt,
                    audio=audios,
                    images=images,
                    videos=videos,
                    return_tensors="pt",
                    padding=True,
                    use_audio_in_video=True,
                )
                inputs = inputs.to(model.device).to(model.dtype)
                capture.begin(inputs["input_ids"])
                with torch.no_grad():
                    model.generate(
                        **inputs,
                        use_audio_in_video=True,
                        return_audio=False,
                        do_sample=False,
                        max_new_tokens=1,
                    )
                payload = {
                    "schema_version": SCHEMA,
                    "kind": "sample",
                    "created_at": utc_now(),
                    "ordinal": ordinal,
                    "line_index": line_index,
                    "video_basename": selected["video_basename"],
                    "prompt_sha256": selected["prompt_sha256"],
                    "prefill_seconds": round(time.time() - sample_start, 3),
                    **capture.export(),
                }
                payload["record_sha256"] = sha256_bytes(canonical_json(payload).encode())
                append_jsonl(raw_path, payload)
                completed[line_index] = payload
                log(
                    f"[{ordinal}/{selected_packet['selected_count']}] complete "
                    f"seq={payload['seq_len']} tokens={payload['token_counts']} "
                    f"time={payload['prefill_seconds']}s"
                )
            except Exception as exc:
                error_payload = {
                    "schema_version": SCHEMA,
                    "kind": "error",
                    "created_at": utc_now(),
                    "ordinal": ordinal,
                    "line_index": line_index,
                    "video_basename": selected["video_basename"],
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "traceback": traceback.format_exc(limit=8)[-4000:],
                }
                error_payload["record_sha256"] = sha256_bytes(
                    canonical_json(error_payload).encode()
                )
                append_jsonl(raw_path, error_payload)
                errors.append(error_payload)
                log(
                    f"[{ordinal}/{selected_packet['selected_count']}] ERROR "
                    f"{type(exc).__name__}: {str(exc)[:300]}"
                )
                capture.reset()
                torch.cuda.empty_cache()

            atomic_write_json(
                output_dir / "progress.json",
                {
                    "schema_version": SCHEMA,
                    "updated_at": utc_now(),
                    "started_at": started_at,
                    "pid": os.getpid(),
                    "selected": selected_packet["selected_count"],
                    "attempted_this_process": ordinal,
                    "completed_unique": len(completed),
                    "error_records": len(errors),
                    "last_line_index": line_index,
                    "status": "running",
                },
            )
    finally:
        capture.remove()

    summary = analyze_run(selected_packet, raw_path, protocol)
    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_json(
        output_dir / "progress.json",
        {
            "schema_version": SCHEMA,
            "updated_at": utc_now(),
            "started_at": started_at,
            "pid": os.getpid(),
            "selected": selected_packet["selected_count"],
            "completed_unique": summary["sample_count"],
            "error_records": summary["error_record_count"],
            "status": "completed",
            "supports_e16_a_claim": summary["supports_e16_a_claim"],
        },
    )
    log(
        f"run complete: samples={summary['sample_count']} "
        f"supports_e16_a_claim={summary['supports_e16_a_claim']}"
    )


if __name__ == "__main__":
    main()

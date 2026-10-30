#!/usr/bin/env python3
"""Run E7: budget-matched cutoff-gap output-sensitivity controls.

The protocol is frozen in
research_paper/iclr2027/evidence/e7_cutoff_gap_protocol.json.  This runner uses
one native top-8 routing pass per sample to freeze per-layer intervention maps,
then compares high-gap, low-gap, and three random assignments with exactly the
same top-4/top-6/top-8 counts in every sample and eligible layer.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import random
import re
import socket
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO / "research_paper/iclr2027/evidence/e7_cutoff_gap_protocol.json"
sys.path.insert(0, str(REPO / "qwen-omni-utils" / "src"))
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def log(*parts: Any) -> None:
    print("[E7]", *parts, file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_rank(seed: int, *parts: Any) -> int:
    payload = ":".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def proportional_stratified_indices(
    records: list[dict[str, Any]], target: int, seed: int
) -> list[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        groups[str(record.get("question_type", "unknown"))].append(idx)
    if target > len(records):
        raise ValueError(f"target={target} exceeds dataset size={len(records)}")

    exact = {name: target * len(indices) / len(records) for name, indices in groups.items()}
    allocation = {name: int(math.floor(value)) for name, value in exact.items()}
    remainder = target - sum(allocation.values())
    order = sorted(
        groups,
        key=lambda name: (-(exact[name] - allocation[name]), stable_rank(seed, "stratum", name)),
    )
    for name in order[:remainder]:
        allocation[name] += 1

    selected: list[int] = []
    for name, indices in sorted(groups.items()):
        ranked = sorted(indices, key=lambda idx: stable_rank(seed, "sample", idx))
        selected.extend(ranked[: allocation[name]])
    return sorted(selected)


def parse_answer(text: str) -> str | None:
    upper = str(text).strip().upper()
    match = re.search(r"(?:^|[^A-Z])([A-D])(?:[^A-Z]|$)", upper)
    if match:
        return match.group(1)
    for char in upper:
        if char in "ABCD":
            return char
    return None


def build_conversation(video_path: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    options = "\n".join(str(option) for option in record["options"])
    prompt = (
        f"{record['question']}\nOptions:\n{options}\n"
        "Answer with only the option letter."
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def _moe_mask_forward(
    module: Any,
    hidden_states: torch.Tensor,
    chosen_budget: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = module.gate(flat)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, module.top_k, dim=-1)

    chosen_budget = chosen_budget.to(device=routing_weights.device, dtype=torch.long)
    positions = torch.arange(module.top_k, device=routing_weights.device).unsqueeze(0)
    routing_weights = routing_weights.masked_fill(positions >= chosen_budget.unsqueeze(1), 0.0)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    routing_weights = routing_weights.to(flat.dtype)

    final = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=flat.dtype,
        device=flat.device,
    )
    expert_mask = F.one_hot(selected_experts, num_classes=module.num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        expert_layer = module.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
        current_state = flat[None, top_x].reshape(-1, hidden_dim)
        current = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final.index_add_(0, top_x, current.to(flat.dtype))
    return final.reshape(batch_size, sequence_length, hidden_dim), router_logits


class FrozenBudgetIntervention:
    """Capture native gaps and apply precomputed per-position budget maps."""

    def __init__(self, model: Any, eligible_layers: list[int], top_k: int) -> None:
        self.model = model
        self.eligible_layers = set(int(layer) for layer in eligible_layers)
        self.top_k = int(top_k)
        self.active_maps: dict[int, torch.Tensor] | None = None
        self.capture_enabled = False
        self.expected_seq_len = 0
        self.video_mask: torch.Tensor | None = None
        self.captured_gaps: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._wrapped: list[tuple[Any, Callable[..., Any]]] = []
        self._hooks: list[Any] = []

        installed = 0
        for name, module in model.thinker.named_modules():
            if "sparsemoeblock" not in module.__class__.__name__.lower():
                continue
            if not all(hasattr(module, attr) for attr in ("gate", "experts", "top_k", "num_experts")):
                continue
            layer_idx = self._parse_layer_idx(name)
            if layer_idx is None or layer_idx not in self.eligible_layers:
                continue
            self._install_layer(module, layer_idx)
            installed += 1
        if installed != len(self.eligible_layers):
            raise RuntimeError(
                f"Expected {len(self.eligible_layers)} eligible MoE blocks, installed {installed}"
            )
        log(f"installed frozen intervention on {installed} MoE layers")

    @staticmethod
    def _parse_layer_idx(name: str) -> int | None:
        parts = name.split(".")
        for i, part in enumerate(parts[:-1]):
            if part == "layers" and parts[i + 1].isdigit():
                return int(parts[i + 1])
        return None

    def _install_layer(self, module: Any, layer_idx: int) -> None:
        original = module.forward
        self._wrapped.append((module, original))

        def gate_hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            if not self.capture_enabled or output.ndim != 2:
                return
            if int(output.shape[0]) != self.expected_seq_len or self.video_mask is None:
                return
            mask = self.video_mask.to(output.device)
            probs = F.softmax(output.detach(), dim=1, dtype=torch.float)
            top_values = torch.topk(probs, self.top_k, dim=-1, sorted=True).values[mask]
            g4 = (top_values[:, 3] - top_values[:, 4]).to("cpu", dtype=torch.float32).numpy()
            g6 = (top_values[:, 5] - top_values[:, 6]).to("cpu", dtype=torch.float32).numpy()
            self.captured_gaps[layer_idx] = (g4, g6)

        self._hooks.append(module.gate.register_forward_hook(gate_hook))

        def wrapped(hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            maps = self.active_maps
            if maps is None or layer_idx not in maps:
                return original(hidden_states)
            budget = maps[layer_idx]
            flat_tokens = int(hidden_states.shape[0] * hidden_states.shape[1])
            if flat_tokens != int(budget.numel()):
                return original(hidden_states)
            return _moe_mask_forward(module, hidden_states, budget)

        module.forward = wrapped

    def begin_capture(self, input_ids: torch.Tensor, video_token_id: int) -> None:
        flat_ids = input_ids.detach().reshape(-1).to("cpu")
        self.expected_seq_len = int(flat_ids.numel())
        self.video_mask = flat_ids.eq(int(video_token_id))
        self.captured_gaps = {}
        self.active_maps = None
        self.capture_enabled = True

    def end_capture(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        self.capture_enabled = False
        missing = sorted(self.eligible_layers - set(self.captured_gaps))
        if missing:
            raise RuntimeError(f"Native routing capture missed eligible layers: {missing}")
        return {layer: self.captured_gaps[layer] for layer in sorted(self.captured_gaps)}

    @contextlib.contextmanager
    def use_maps(self, maps: dict[int, torch.Tensor] | None):
        previous = self.active_maps
        self.active_maps = maps
        try:
            yield
        finally:
            self.active_maps = previous

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        for module, original in self._wrapped:
            module.forward = original
        self._hooks.clear()
        self._wrapped.clear()


def full_budget_map(seq_len: int, video_positions: np.ndarray, video_budgets: np.ndarray) -> torch.Tensor:
    result = torch.full((seq_len,), 8, dtype=torch.int16)
    result[torch.from_numpy(video_positions)] = torch.from_numpy(video_budgets.astype(np.int16))
    return result


def build_maps(
    gaps: dict[int, tuple[np.ndarray, np.ndarray]],
    video_positions: np.ndarray,
    seq_len: int,
    threshold: float,
    sample_index: int,
    random_seeds: list[int],
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, Any]]:
    arms: dict[str, dict[int, torch.Tensor]] = {
        "high_gap": {},
        "low_gap_matched": {},
        **{f"random_matched_{seed}": {} for seed in random_seeds},
    }
    audit: dict[str, Any] = {"layers": {}, "maximum_abs_count_difference": 0}
    for layer_idx, (g4, g6) in sorted(gaps.items()):
        if len(g4) != len(video_positions) or len(g6) != len(video_positions):
            raise RuntimeError(
                f"Layer {layer_idx} gap/video mismatch: {len(g4)}/{len(g6)} vs {len(video_positions)}"
            )
        high = np.full(len(g4), 8, dtype=np.int16)
        choose4 = g4 >= threshold
        choose6 = (~choose4) & (g6 >= threshold)
        high[choose4] = 4
        high[choose6] = 6
        counts = {str(q): int(np.count_nonzero(high == q)) for q in (4, 6, 8)}
        n4, n6 = counts["4"], counts["6"]

        low = np.full(len(g4), 8, dtype=np.int16)
        order4 = np.argsort(g4, kind="mergesort")
        low4 = order4[:n4]
        low[low4] = 4
        remaining = np.flatnonzero(low == 8)
        order6 = remaining[np.argsort(g6[remaining], kind="mergesort")]
        low[order6[:n6]] = 6

        video_maps: dict[str, np.ndarray] = {"high_gap": high, "low_gap_matched": low}
        for seed in random_seeds:
            rng = np.random.default_rng(stable_rank(seed, sample_index, layer_idx))
            order = rng.permutation(len(g4))
            random_budget = np.full(len(g4), 8, dtype=np.int16)
            random_budget[order[:n4]] = 4
            random_budget[order[n4 : n4 + n6]] = 6
            video_maps[f"random_matched_{seed}"] = random_budget

        layer_counts: dict[str, dict[str, int]] = {}
        layer_hashes: dict[str, str] = {}
        for arm, video_budget in video_maps.items():
            arm_counts = {str(q): int(np.count_nonzero(video_budget == q)) for q in (4, 6, 8)}
            layer_counts[arm] = arm_counts
            difference = max(abs(arm_counts[str(q)] - counts[str(q)]) for q in (4, 6, 8))
            audit["maximum_abs_count_difference"] = max(
                int(audit["maximum_abs_count_difference"]), difference
            )
            layer_hashes[arm] = hashlib.sha256(video_budget.tobytes()).hexdigest()
            arms[arm][layer_idx] = full_budget_map(seq_len, video_positions, video_budget)

        audit["layers"][str(layer_idx)] = {
            "video_tokens": int(len(g4)),
            "reference_counts": counts,
            "arm_counts": layer_counts,
            "allocation_sha256": layer_hashes,
            "g4_quantiles": np.quantile(g4, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist(),
            "g6_quantiles": np.quantile(g6, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist(),
        }
    if audit["maximum_abs_count_difference"] != 0:
        raise AssertionError("Budget matching failed")
    return arms, audit


@dataclass
class GenerationResult:
    first_logits: torch.Tensor
    first_top1: int
    output_text: str
    answer: str | None
    elapsed_seconds: float


def generate_once(
    model: Any,
    processor: Any,
    inputs: Any,
    use_audio_in_video: bool,
    max_new_tokens: int,
) -> GenerationResult:
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        raw = model.generate(
            **inputs,
            use_audio_in_video=use_audio_in_video,
            return_audio=False,
            thinker_max_new_tokens=max_new_tokens,
            thinker_do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            thinker_eos_token_id=processor.tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    thinker = raw[0] if isinstance(raw, tuple) else raw
    if not getattr(thinker, "scores", None):
        raise RuntimeError("Generation returned no score tensor")
    first_logits = thinker.scores[0][0].detach().float().cpu()
    input_len = int(inputs["input_ids"].shape[1])
    generated_ids = thinker.sequences[0, input_len:].detach().cpu()
    output_text = processor.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return GenerationResult(
        first_logits=first_logits,
        first_top1=int(first_logits.argmax().item()),
        output_text=output_text,
        answer=parse_answer(output_text),
        elapsed_seconds=elapsed,
    )


def distribution_distances(native_logits: torch.Tensor, arm_logits: torch.Tensor) -> dict[str, float]:
    native_logp = torch.log_softmax(native_logits.double(), dim=-1)
    arm_logp = torch.log_softmax(arm_logits.double(), dim=-1)
    native_p = native_logp.exp()
    arm_p = arm_logp.exp()
    mixture = 0.5 * (native_p + arm_p)
    mixture_log = mixture.clamp_min(torch.finfo(torch.float64).tiny).log()
    js = 0.5 * (
        torch.sum(native_p * (native_logp - mixture_log))
        + torch.sum(arm_p * (arm_logp - mixture_log))
    )
    tv = 0.5 * torch.sum(torch.abs(native_p - arm_p))
    # Roundoff can make an analytically non-negative divergence a few ulps
    # below zero for an exact identity comparison.
    return {
        "js_divergence_nats": max(0.0, float(js.item())),
        "total_variation": max(0.0, float(tv.item())),
    }


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL, timeout=20
        ).strip()
    except Exception:
        return "unavailable"


def environment_record(protocol: dict[str, Any], dataset_path: Path, model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    shards = sorted(model_path.glob("*.safetensors"))
    return {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "requested_worker": os.environ.get("E7_WORKER_ID", "unknown"),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_device_count": torch.cuda.device_count(),
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status_short": git_value("status", "--short"),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "model_config_sha256": sha256_file(config_path) if config_path.exists() else None,
        "model_index_sha256": sha256_file(index_path) if index_path.exists() else None,
        "model_shards": [
            {"name": path.name, "bytes": path.stat().st_size} for path in shards
        ],
        "protocol": protocol,
    }


def bootstrap_mean_ci(values: np.ndarray, seed: int, draws: int = 10000) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    chunk = 1000
    for start in range(0, draws, chunk):
        stop = min(draws, start + chunk)
        sample_idx = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[sample_idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def summarize(
    rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    random_names = [name for name in protocol["arms"] if name.startswith("random_matched_")]
    metrics: dict[str, list[float]] = defaultdict(list)
    identity_values: list[float] = []
    max_budget_diff = 0
    arm_accuracy: dict[str, list[int]] = defaultdict(list)
    arm_flips: dict[str, list[int]] = defaultdict(list)
    arm_top1: dict[str, list[int]] = defaultdict(list)

    for row in rows:
        max_budget_diff = max(max_budget_diff, int(row["budget_audit"]["maximum_abs_count_difference"]))
        if row.get("identity") is not None:
            identity_values.append(float(row["identity"]["js_divergence_nats"]))
        for arm_name, arm in row["arms"].items():
            metrics[f"{arm_name}.js"].append(float(arm["js_divergence_nats"]))
            metrics[f"{arm_name}.tv"].append(float(arm["total_variation"]))
            arm_accuracy[arm_name].append(int(bool(arm["is_correct"])))
            arm_flips[arm_name].append(int(bool(arm["answer_flip_vs_native"])))
            arm_top1[arm_name].append(int(bool(arm["native_top1_retained"])))

    n = len(rows)
    seed = int(protocol["sample_selection"]["seed"])
    high = np.asarray(metrics["high_gap.js"], dtype=np.float64)
    low = np.asarray(metrics["low_gap_matched.js"], dtype=np.float64)
    random_matrix = np.asarray(
        [[row["arms"][name]["js_divergence_nats"] for name in random_names] for row in rows],
        dtype=np.float64,
    ) if rows else np.empty((0, len(random_names)))
    random_mean = random_matrix.mean(axis=1) if n else np.asarray([], dtype=np.float64)

    primary: dict[str, Any] = {}
    if n:
        primary = {
            "high_gap": bootstrap_mean_ci(high, seed + 1),
            "low_gap_matched": bootstrap_mean_ci(low, seed + 2),
            "random_matched_within_sample_mean": bootstrap_mean_ci(random_mean, seed + 3),
            "effect_low_minus_high": bootstrap_mean_ci(low - high, seed + 4),
            "effect_random_minus_high": bootstrap_mean_ci(random_mean - high, seed + 5),
        }

    minimum_n = int(protocol["sample_selection"]["minimum_paired_samples"])
    identity_limit = float(
        protocol["implementation_validation"]["maximum_identity_js_divergence_nats"]
    )
    identity_required = int(protocol["implementation_validation"]["identity_arm_samples"])
    max_identity = max(identity_values, default=float("inf"))
    evidence_verified = (
        n >= minimum_n
        and len(identity_values) >= identity_required
        and max_identity <= identity_limit
        and max_budget_diff == 0
    )
    predictive_supported = False
    if evidence_verified and primary:
        predictive_supported = (
            primary["effect_low_minus_high"]["ci95_low"] > 0.0
            and primary["effect_random_minus_high"]["ci95_low"] > 0.0
        )

    arm_summary: dict[str, Any] = {}
    for arm_name in ["high_gap", "low_gap_matched", *random_names]:
        if not metrics[f"{arm_name}.js"]:
            continue
        arm_summary[arm_name] = {
            "js_divergence_nats": bootstrap_mean_ci(
                np.asarray(metrics[f"{arm_name}.js"]), seed + stable_rank(seed, arm_name) % 100000
            ),
            "total_variation_mean": float(np.mean(metrics[f"{arm_name}.tv"])),
            "native_top1_retention": float(np.mean(arm_top1[arm_name])),
            "answer_flip_rate": float(np.mean(arm_flips[arm_name])),
            "accuracy": float(np.mean(arm_accuracy[arm_name])),
            "correct": int(sum(arm_accuracy[arm_name])),
            "samples": n,
        }

    native_correct = int(sum(int(bool(row["native"]["is_correct"])) for row in rows))
    return {
        "schema_version": "arc-sceb/e7-results-v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "experiment_id": "E7",
        "evidence_status": "verified" if evidence_verified else "placeholder",
        "predictive_validity_supported": predictive_supported,
        "paired_samples": n,
        "failed_samples": len(errors),
        "requested_samples": int(protocol["sample_selection"]["target_samples"]),
        "primary_endpoint": primary,
        "native": {
            "accuracy": native_correct / n if n else None,
            "correct": native_correct,
            "samples": n,
        },
        "arms": arm_summary,
        "implementation_gates": {
            "identity_samples": len(identity_values),
            "maximum_identity_js_divergence_nats": max_identity if identity_values else None,
            "identity_limit": identity_limit,
            "maximum_abs_budget_count_difference": max_budget_diff,
            "minimum_paired_samples": minimum_n,
            "passed": evidence_verified,
        },
        "interpretation_gate": {
            "required": "Both paired 95% CIs for control minus high_gap are strictly positive.",
            "passed": predictive_supported,
        },
        "scope": protocol["claim_gate"]["scope"],
        "errors": errors,
    }


def save_gap_archive(path: Path, gaps: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for layer, (g4, g6) in gaps.items():
        arrays[f"layer_{layer:02d}_g4"] = g4.astype(np.float32, copy=False)
        arrays[f"layer_{layer:02d}_g6"] = g6.astype(np.float32, copy=False)
    np.savez_compressed(path, **arrays)


def run_sample(
    *,
    sample_index: int,
    record: dict[str, Any],
    model: Any,
    processor: Any,
    process_mm_info: Callable[..., Any],
    intervention: FrozenBudgetIntervention,
    protocol: dict[str, Any],
    output_dir: Path,
    identity: bool,
    max_new_tokens: int,
) -> dict[str, Any]:
    video_path = (REPO / "eval_data/OmniVideoBench" / str(record["video"])).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    conversation = build_conversation(video_path, record)
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
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
    input_ids = inputs["input_ids"]
    seq_len = int(input_ids.shape[1])
    video_token_id = int(model.thinker.config.video_token_id)
    video_positions = np.flatnonzero(
        input_ids.detach().reshape(-1).to("cpu").numpy() == video_token_id
    ).astype(np.int64)
    if not len(video_positions):
        raise RuntimeError("No video tokens detected")

    intervention.begin_capture(input_ids, video_token_id)
    native = generate_once(model, processor, inputs, True, max_new_tokens)
    gaps = intervention.end_capture()
    save_gap_archive(output_dir / "routing_gaps" / f"sample_{sample_index:04d}.npz", gaps)

    random_seeds = [
        int(name.rsplit("_", 1)[1])
        for name in protocol["arms"]
        if name.startswith("random_matched_")
    ]
    maps, budget_audit = build_maps(
        gaps=gaps,
        video_positions=video_positions,
        seq_len=seq_len,
        threshold=float(protocol["policy"]["cutoff_gap_threshold"]),
        sample_index=sample_index,
        random_seeds=random_seeds,
    )

    identity_result: dict[str, Any] | None = None
    if identity:
        identity_maps = {
            layer: torch.full((seq_len,), 8, dtype=torch.int16)
            for layer in protocol["policy"]["eligible_moe_layers"]
        }
        with intervention.use_maps(identity_maps):
            identity_output = generate_once(model, processor, inputs, True, max_new_tokens)
        identity_result = {
            **distribution_distances(native.first_logits, identity_output.first_logits),
            "native_top1_retained": identity_output.first_top1 == native.first_top1,
            "output_text_equal": identity_output.output_text == native.output_text,
        }

    arm_results: dict[str, Any] = {}
    for arm_name in ["high_gap", "low_gap_matched", *sorted(name for name in maps if name.startswith("random_matched_"))]:
        with intervention.use_maps(maps[arm_name]):
            output = generate_once(model, processor, inputs, True, max_new_tokens)
        arm_results[arm_name] = {
            **distribution_distances(native.first_logits, output.first_logits),
            "first_top1_token_id": output.first_top1,
            "native_top1_retained": output.first_top1 == native.first_top1,
            "output_text": output.output_text,
            "answer": output.answer,
            "answer_flip_vs_native": output.answer != native.answer,
            "is_correct": output.answer == str(record["correct_option"]).upper(),
            "elapsed_seconds": output.elapsed_seconds,
        }

    high_counts = Counter()
    for layer in budget_audit["layers"].values():
        high_counts.update({int(q): int(count) for q, count in layer["reference_counts"].items()})
    total_decisions = sum(high_counts.values())
    mean_budget = sum(q * count for q, count in high_counts.items()) / total_decisions
    return {
        "sample_index": sample_index,
        "video": record["video"],
        "question_type": record.get("question_type"),
        "correct_option": str(record["correct_option"]).upper(),
        "input_sequence_length": seq_len,
        "video_tokens": int(len(video_positions)),
        "native": {
            "first_top1_token_id": native.first_top1,
            "output_text": native.output_text,
            "answer": native.answer,
            "is_correct": native.answer == str(record["correct_option"]).upper(),
            "elapsed_seconds": native.elapsed_seconds,
        },
        "identity": identity_result,
        "arms": arm_results,
        "budget": {
            "top_m_counts": {str(q): int(high_counts[q]) for q in (4, 6, 8)},
            "mean_expert_budget": mean_budget,
            "total_video_token_layer_decisions": total_decisions,
        },
        "budget_audit": budget_audit,
        "video_file": {
            "path": str(video_path.relative_to(REPO)),
            "bytes": video_path.stat().st_size,
            "mtime_ns": video_path.stat().st_mtime_ns,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Debug-only override; changes the frozen run.")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> None:
    gaps = {
        12: (
            np.asarray([0.9, 0.1, 0.8, 0.2, 0.7, 0.3], dtype=np.float32),
            np.asarray([0.4, 0.6, 0.3, 0.5, 0.2, 0.1], dtype=np.float32),
        )
    }
    maps, audit = build_maps(
        gaps=gaps,
        video_positions=np.arange(6, dtype=np.int64),
        seq_len=6,
        threshold=0.5,
        sample_index=9,
        random_seeds=[1729, 2718, 3141],
    )
    assert audit["maximum_abs_count_difference"] == 0
    assert maps["high_gap"][12].tolist() == [4, 6, 4, 6, 4, 8]
    for arm in maps:
        assert Counter(maps[arm][12].tolist()) == Counter({4: 3, 6: 2, 8: 1})
    print("E7 map construction self-test passed.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --self-test is used")

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    dataset_path = REPO / protocol["dataset"]
    model_path = REPO / protocol["model"]
    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    target = int(protocol["sample_selection"]["target_samples"])
    if args.limit is not None:
        target = int(args.limit)
    selected = proportional_stratified_indices(
        records, target, int(protocol["sample_selection"]["seed"])
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "routing_gaps").mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_results.jsonl"
    error_path = output_dir / "errors.jsonl"
    selected_path = output_dir / "selected_samples.json"
    summary_path = output_dir / "summary.json"

    selection_record = {
        "selection_method": protocol["sample_selection"],
        "dataset_sha256": sha256_file(dataset_path),
        "indices": selected,
        "samples": [
            {
                "sample_index": idx,
                "video": records[idx]["video"],
                "question_type": records[idx].get("question_type"),
                "correct_option": records[idx].get("correct_option"),
            }
            for idx in selected
        ],
    }
    if selected_path.exists():
        existing = json.loads(selected_path.read_text(encoding="utf-8"))
        if existing != selection_record:
            raise RuntimeError("Existing selected_samples.json conflicts with this run")
    else:
        atomic_json(selected_path, selection_record)

    random.seed(int(protocol["sample_selection"]["seed"]))
    np.random.seed(int(protocol["sample_selection"]["seed"]))
    torch.manual_seed(int(protocol["sample_selection"]["seed"]))
    torch.cuda.manual_seed_all(int(protocol["sample_selection"]["seed"]))

    log("loading Qwen3-Omni model")
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(str(model_path), use_fast=False)
    atomic_json(output_dir / "environment.json", environment_record(protocol, dataset_path, model_path))

    intervention = FrozenBudgetIntervention(
        model,
        eligible_layers=[int(v) for v in protocol["policy"]["eligible_moe_layers"]],
        top_k=int(protocol["policy"]["original_top_k"]),
    )
    completed_rows = load_jsonl(raw_path)
    completed = {int(row["sample_index"]) for row in completed_rows}
    errors = load_jsonl(error_path)
    identity_done = sum(row.get("identity") is not None for row in completed_rows)
    identity_required = int(protocol["implementation_validation"]["identity_arm_samples"])
    identity_limit = float(
        protocol["implementation_validation"]["maximum_identity_js_divergence_nats"]
    )

    try:
        for ordinal, sample_index in enumerate(selected, 1):
            if sample_index in completed:
                log(f"[{ordinal}/{len(selected)}] sample {sample_index} already complete")
                continue
            log(f"[{ordinal}/{len(selected)}] sample {sample_index} starting")
            try:
                row = run_sample(
                    sample_index=sample_index,
                    record=records[sample_index],
                    model=model,
                    processor=processor,
                    process_mm_info=process_mm_info,
                    intervention=intervention,
                    protocol=protocol,
                    output_dir=output_dir,
                    identity=identity_done < identity_required,
                    max_new_tokens=args.max_new_tokens,
                )
                if row.get("identity") is not None:
                    identity_done += 1
                    identity_js = float(row["identity"]["js_divergence_nats"])
                    if identity_js > identity_limit or not row["identity"]["native_top1_retained"]:
                        raise RuntimeError(
                            f"Identity gate failed: JS={identity_js:.6g}, "
                            f"limit={identity_limit:.6g}, top1={row['identity']['native_top1_retained']}"
                        )
                append_jsonl(raw_path, row)
                completed_rows.append(row)
                completed.add(sample_index)
                log(
                    f"[{ordinal}/{len(selected)}] sample {sample_index} complete; "
                    f"mean_budget={row['budget']['mean_expert_budget']:.4f}; "
                    f"high_js={row['arms']['high_gap']['js_divergence_nats']:.6g}; "
                    f"low_js={row['arms']['low_gap_matched']['js_divergence_nats']:.6g}"
                )
                atomic_json(
                    output_dir / "progress.json",
                    {
                        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
                        "paired_samples": len(completed_rows),
                        "requested_samples": len(selected),
                        "error_records": len(errors),
                        "last_completed_sample_index": sample_index,
                    },
                )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                error = {
                    "sample_index": sample_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(error_path, error)
                errors.append(error)
                log(f"sample {sample_index} OOM; recorded and continuing")
            except Exception as exc:
                error = {
                    "sample_index": sample_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(error_path, error)
                errors.append(error)
                log(f"sample {sample_index} failed: {exc!r}")
                if "Identity gate failed" in str(exc):
                    atomic_json(summary_path, summarize(completed_rows, errors, protocol))
                    return 2
        final_summary = summarize(completed_rows, errors, protocol)
        atomic_json(summary_path, final_summary)
        log(
            f"finished: status={final_summary['evidence_status']} "
            f"paired={final_summary['paired_samples']} "
            f"predictive_supported={final_summary['predictive_validity_supported']}"
        )
        return 0 if final_summary["evidence_status"] == "verified" else 3
    finally:
        intervention.close()


if __name__ == "__main__":
    raise SystemExit(main())

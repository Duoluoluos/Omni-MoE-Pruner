#!/usr/bin/env python3
"""E16-B: held-out validation of the configuration frozen by E16-A.

The E16-A output is never re-fit here.  On evaluation questions not used by
E7/E8, this runner compares native top-8, the previous author-curated ARC-SCEB
configuration, and the automatic E16-A configuration.  It preserves compact
per-question quality, timing, and exact logical-allocation evidence.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import random
import socket
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "qwen-omni-utils" / "src"))
import run_e7_cutoff_gap_control as common  # noqa: E402


SCHEMA = "arc-sceb/e16-b-heldout-validation-v1"
NUM_LAYERS = 48
NUM_EXPERTS = 128
TOP_K = 8
ARMS = ("native_top8", "author_curated_arc_sceb", "automatic_arc_sceb")


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def log(*parts: Any) -> None:
    print("[E16-B]", *parts, file=sys.stderr, flush=True)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def validate_file_hash(relative_path: str, expected: str) -> Path:
    path = REPO / relative_path
    observed = common.sha256_file(path)
    if observed != expected:
        raise ValueError(f"checksum mismatch for {relative_path}: {observed} != {expected}")
    return path


def load_excluded_indices(protocol: dict[str, Any]) -> tuple[set[int], dict[str, int]]:
    exclusions: set[int] = set()
    counts: dict[str, int] = {}
    for item in protocol["sample_selection"]["exclude_prior_runs"]:
        path = validate_file_hash(item["path"], item["sha256"])
        name = item["name"]
        if item["format"] == "selected_samples_json":
            indices = {int(value) for value in json.loads(path.read_text())["indices"]}
        elif item["format"] == "raw_jsonl_sample_index_only":
            indices = set()
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        indices.add(int(json.loads(line)["sample_index"]))
        else:
            raise ValueError(f"unsupported exclusion format: {item['format']}")
        exclusions.update(indices)
        counts[name] = len(indices)
    counts["union"] = len(exclusions)
    return exclusions, counts


def allocate_strata(
    candidates: list[tuple[int, dict[str, Any]]], target: int, seed: int
) -> list[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in candidates:
        groups[str(record.get("question_type", "unknown"))].append(index)
    exact = {name: target * len(values) / len(candidates) for name, values in groups.items()}
    allocation = {name: int(np.floor(value)) for name, value in exact.items()}
    remainder = target - sum(allocation.values())
    order = sorted(
        groups,
        key=lambda name: (
            -(exact[name] - allocation[name]),
            common.stable_rank(seed, "stratum", name),
        ),
    )
    for name in order[:remainder]:
        allocation[name] += 1
    selected = []
    for name, values in groups.items():
        ranked = sorted(values, key=lambda idx: common.stable_rank(seed, "sample", idx))
        if allocation[name] > len(ranked):
            raise ValueError(f"stratum {name} lacks candidates")
        selected.extend(ranked[: allocation[name]])
    return sorted(selected, key=lambda idx: common.stable_rank(seed, "run-order", idx))


def prepare_selection(protocol: dict[str, Any]) -> dict[str, Any]:
    dataset = validate_file_hash(protocol["dataset"]["path"], protocol["dataset"]["sha256"])
    records = json.loads(dataset.read_text())
    exclusions, exclusion_counts = load_excluded_indices(protocol)
    seed = int(protocol["sample_selection"]["seed"])

    # Freeze at most one question per video before stratification.  This makes
    # the paired question bootstrap also a video-level bootstrap.
    by_video: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, record in enumerate(records):
        if index not in exclusions:
            by_video[str(record["video"])].append((index, record))
    representatives = []
    for video, values in by_video.items():
        representatives.append(
            min(values, key=lambda pair: common.stable_rank(seed, "within-video", video, pair[0]))
        )
    target = int(protocol["sample_selection"]["target_samples"])
    if len(representatives) < target:
        raise ValueError(f"only {len(representatives)} unique-video candidates for target {target}")
    indices = allocate_strata(representatives, target, seed)
    samples = [
        {
            "sample_index": index,
            "video": records[index]["video"],
            "question_type": records[index].get("question_type"),
            "stable_order": ordinal,
        }
        for ordinal, index in enumerate(indices)
    ]
    if len({item["video"] for item in samples}) != len(samples):
        raise AssertionError("selected samples are not video-unique")
    selection_hash = sha256_value(samples)
    expected = protocol["sample_selection"].get("selected_samples_sha256")
    if expected and selection_hash != expected:
        raise ValueError(f"selection changed: {selection_hash} != {expected}")
    return {
        "schema_version": SCHEMA,
        "created_at": now(),
        "selection_rule": (
            "exclude E7/E8 sample indices; choose one SHA256-ranked question per video; "
            "proportionally stratify by question_type; SHA256 run order"
        ),
        "dataset_sha256": protocol["dataset"]["sha256"],
        "labels_used_for_selection": False,
        "excluded_prior_run_counts": exclusion_counts,
        "remaining_questions": len(records) - len(exclusions),
        "unique_video_candidates": len(representatives),
        "selected_count": len(samples),
        "selected_samples_sha256": selection_hash,
        "samples": samples,
    }


def validate_frozen_configs(protocol: dict[str, Any]) -> None:
    validate_file_hash(
        f"{protocol['model']['path']}/config.json",
        protocol["model"]["config_sha256"],
    )
    auto_source = validate_file_hash(
        protocol["automatic_configuration"]["source_summary"],
        protocol["automatic_configuration"]["source_summary_sha256"],
    )
    auto_summary = json.loads(auto_source.read_text())
    source_config = auto_summary["full_configuration"]
    if source_config["target_modality"] != "video":
        raise ValueError("E16-A selected a non-video target; E16-B protocol is invalid")
    if sha256_value(source_config) != protocol["automatic_configuration"]["configuration_sha256"]:
        raise ValueError("E16-A configuration digest does not match frozen E16-B protocol")
    expected = protocol["automatic_configuration"]["policy"]
    observed = {
        "eligible_moe_layers": list(
            range(
                int(source_config["eligible_layer_start"]),
                int(source_config["eligible_layer_end_exclusive"]),
            )
        ),
        "cutoff_gap_threshold": float(source_config["boundary_threshold"]),
        "minimum_audio_tokens": int(source_config["minimum_audio_tokens"]),
        "audio_uncertainty_threshold": float(source_config["audio_uncertainty_threshold"]),
        "insufficient_audio_min_budget": int(source_config["insufficient_audio_min_budget"]),
        "uncertain_audio_min_budget": int(source_config["uncertain_audio_min_budget"]),
    }
    if canonical_json(observed) != canonical_json(expected):
        raise ValueError("E16-B automatic policy fields differ from E16-A output")

    manual_source = validate_file_hash(
        protocol["author_curated_configuration"]["source_policy"],
        protocol["author_curated_configuration"]["source_policy_sha256"],
    )
    manual = json.loads(manual_source.read_text())
    if manual["target_modality"] != "video":
        raise ValueError("retained manual policy is not video-targeted")
    manual_observed = {
        "eligible_moe_layers": list(
            range(
                int(NUM_LAYERS * float(manual["layer_start_ratio"])),
                int(NUM_LAYERS * float(manual["layer_end_ratio"])),
            )
        ),
        "cutoff_gap_threshold": float(manual["thresholds"]["boundary_threshold"]),
        "minimum_audio_tokens": int(manual["audio_router_conditioning"]["min_audio_tokens"]),
        "audio_uncertainty_threshold": float(
            manual["audio_router_conditioning"]["fragile_margin_threshold"]
        ),
        "insufficient_audio_min_budget": int(
            manual["audio_router_conditioning"]["insufficient_audio_min_top_m"]
        ),
        "uncertain_audio_min_budget": int(
            manual["audio_router_conditioning"]["fragile_min_top_m"]
        ),
    }
    if canonical_json(manual_observed) != canonical_json(
        protocol["author_curated_configuration"]["policy"]
    ):
        raise ValueError("manual policy fields differ from retained configuration")


def physical_budget_forward(
    module: Any, hidden_states: torch.Tensor, chosen_budget: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = module.gate(flat)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(
        routing_weights, module.top_k, dim=-1
    )
    chosen_budget = chosen_budget.to(device=flat.device, dtype=torch.long)
    positions = torch.arange(module.top_k, device=flat.device).unsqueeze(0)
    active = positions < chosen_budget.unsqueeze(1)
    routing_weights = routing_weights.masked_fill(~active, 0.0)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    routing_weights = routing_weights.to(flat.dtype)

    final = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=flat.dtype, device=flat.device
    )
    assignments = F.one_hot(selected_experts, num_classes=module.num_experts).bool()
    assignments &= active.unsqueeze(-1)
    expert_mask = assignments.permute(2, 1, 0)
    expert_hit = expert_mask.sum(dim=(-1, -2)).gt(0).nonzero(as_tuple=False).flatten()
    for expert_index_tensor in expert_hit:
        expert_index = int(expert_index_tensor.item())
        rank_index, token_index = torch.where(expert_mask[expert_index])
        current_state = flat[token_index]
        current = module.experts[expert_index](current_state)
        current = current * routing_weights[token_index, rank_index, None]
        final.index_add_(0, token_index, current.to(flat.dtype))
    return final.reshape(batch_size, sequence_length, hidden_dim), router_logits


class PairedPolicyIntervention:
    def __init__(self, model: Any, union_layers: list[int]) -> None:
        self.union_layers = set(int(value) for value in union_layers)
        self.active_maps: dict[int, torch.Tensor] | None = None
        self.capture_enabled = False
        self.expected_seq_len = 0
        self.video_mask: torch.Tensor | None = None
        self.audio_mask: torch.Tensor | None = None
        self.captured: dict[int, dict[str, Any]] = {}
        self.wrapped: list[tuple[Any, Callable[..., Any]]] = []
        self.hooks = []
        installed = 0
        for name, module in model.thinker.named_modules():
            if "sparsemoeblock" not in module.__class__.__name__.lower():
                continue
            if not all(hasattr(module, key) for key in ("gate", "experts", "top_k", "num_experts")):
                continue
            layer = self._layer(name)
            if layer is None or layer not in self.union_layers:
                continue
            original = module.forward
            self.wrapped.append((module, original))

            def gate_hook(_module: Any, _inputs: Any, output: torch.Tensor, *, layer_index: int = layer) -> None:
                if not self.capture_enabled or output.ndim != 2:
                    return
                if int(output.shape[0]) != self.expected_seq_len:
                    return
                if self.video_mask is None or self.audio_mask is None:
                    return
                probabilities = F.softmax(output.detach(), dim=1, dtype=torch.float)
                top_values = torch.topk(probabilities, TOP_K, dim=-1, sorted=True).values
                video_values = top_values[self.video_mask.to(output.device)]
                audio_values = top_values[self.audio_mask.to(output.device)]
                if not int(video_values.shape[0]):
                    raise RuntimeError("no video routing rows")
                self.captured[layer_index] = {
                    "g4": (video_values[:, 3] - video_values[:, 4]).cpu().numpy().astype(np.float32),
                    "g6": (video_values[:, 5] - video_values[:, 6]).cpu().numpy().astype(np.float32),
                    "audio_tokens": int(audio_values.shape[0]),
                    "audio_margin": float(
                        (audio_values[:, 0] - audio_values[:, 1]).mean().item()
                    )
                    if int(audio_values.shape[0])
                    else 0.0,
                }

            self.hooks.append(module.gate.register_forward_hook(gate_hook))

            def wrapped(hidden_states: torch.Tensor, *, original_forward: Callable[..., Any] = original, layer_index: int = layer, block: Any = module):
                maps = self.active_maps
                if maps is None or layer_index not in maps:
                    return original_forward(hidden_states)
                budget = maps[layer_index]
                if int(hidden_states.shape[0] * hidden_states.shape[1]) != int(budget.numel()):
                    return original_forward(hidden_states)
                return physical_budget_forward(block, hidden_states, budget)

            module.forward = wrapped
            installed += 1
        if installed != len(self.union_layers):
            raise RuntimeError(f"expected {len(self.union_layers)} union layers, installed {installed}")
        log(f"installed physical budget intervention on {installed} MoE layers")

    @staticmethod
    def _layer(name: str) -> int | None:
        parts = name.split(".")
        for index, part in enumerate(parts[:-1]):
            if part == "layers" and parts[index + 1].isdigit():
                return int(parts[index + 1])
        return None

    def begin_capture(self, input_ids: torch.Tensor, video_id: int, audio_id: int) -> None:
        ids = input_ids.detach().reshape(-1).cpu()
        self.expected_seq_len = int(ids.numel())
        self.video_mask = ids.eq(video_id)
        self.audio_mask = ids.eq(audio_id)
        self.captured = {}
        self.active_maps = None
        self.capture_enabled = True

    def end_capture(self) -> dict[int, dict[str, Any]]:
        self.capture_enabled = False
        missing = sorted(self.union_layers - set(self.captured))
        if missing:
            raise RuntimeError(f"native capture missed layers {missing}")
        return self.captured

    @contextlib.contextmanager
    def use_maps(self, maps: dict[int, torch.Tensor] | None):
        previous = self.active_maps
        self.active_maps = maps
        try:
            yield
        finally:
            self.active_maps = previous

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()
        for module, original in self.wrapped:
            module.forward = original


def build_policy_maps(
    capture: dict[int, dict[str, Any]],
    video_positions: np.ndarray,
    seq_len: int,
    policy: dict[str, Any],
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    maps: dict[int, torch.Tensor] = {}
    total_counts = Counter()
    audio_states = Counter()
    layer_records = {}
    allocation_bytes = []
    for layer in policy["eligible_moe_layers"]:
        item = capture[int(layer)]
        g4 = item["g4"]
        g6 = item["g6"]
        values = np.full(len(g4), 8, dtype=np.int16)
        choose4 = g4 >= float(policy["cutoff_gap_threshold"])
        choose6 = (~choose4) & (g6 >= float(policy["cutoff_gap_threshold"]))
        values[choose4] = 4
        values[choose6] = 6
        audio_tokens = int(item["audio_tokens"])
        audio_margin = float(item["audio_margin"])
        if audio_tokens < int(policy["minimum_audio_tokens"]):
            floor = int(policy["insufficient_audio_min_budget"])
            state = "insufficient"
        elif audio_margin < float(policy["audio_uncertainty_threshold"]):
            floor = int(policy["uncertain_audio_min_budget"])
            state = "uncertain"
        else:
            floor = 0
            state = "confident"
        if floor:
            values = np.maximum(values, floor).astype(np.int16)
        counts = {str(q): int(np.count_nonzero(values == q)) for q in (4, 6, 8)}
        total_counts.update({int(q): count for q, count in counts.items()})
        audio_states[state] += 1
        full = torch.full((seq_len,), 8, dtype=torch.long, device=device)
        full[torch.as_tensor(video_positions, device=device)] = torch.as_tensor(
            values, dtype=torch.long, device=device
        )
        maps[int(layer)] = full
        allocation_bytes.append(values.tobytes())
        layer_records[str(layer)] = {
            "counts": counts,
            "audio_tokens": audio_tokens,
            "audio_margin": audio_margin,
            "audio_state": state,
            "audio_floor": floor,
        }
    total = sum(total_counts.values())
    mean_budget = sum(q * count for q, count in total_counts.items()) / total
    saved_pairs = sum((8 - q) * count for q, count in total_counts.items())
    return maps, {
        "budget_counts": {str(q): int(total_counts[q]) for q in (4, 6, 8)},
        "mean_eligible_video_budget": float(mean_budget),
        "eligible_video_pair_reduction_fraction": float(saved_pairs / (8 * total)),
        "saved_token_expert_pairs": int(saved_pairs),
        "audio_state_layers": dict(audio_states),
        "layers": layer_records,
        "allocation_sha256": hashlib.sha256(b"".join(allocation_bytes)).hexdigest(),
    }


def identity_maps(
    union_layers: list[int], seq_len: int, device: torch.device
) -> dict[int, torch.Tensor]:
    return {
        int(layer): torch.full((seq_len,), 8, dtype=torch.long, device=device)
        for layer in union_layers
    }


def generation_record(
    output: common.GenerationResult,
    native: common.GenerationResult,
    correct_option: str,
) -> dict[str, Any]:
    return {
        **common.distribution_distances(native.first_logits, output.first_logits),
        "first_top1_token_id": output.first_top1,
        "native_top1_retained": output.first_top1 == native.first_top1,
        "output_text": output.output_text,
        "answer": output.answer,
        "is_correct": output.answer == correct_option,
        "elapsed_seconds": output.elapsed_seconds,
    }


def run_sample(
    sample_index: int,
    record: dict[str, Any],
    model: Any,
    processor: Any,
    process_mm_info: Callable[..., Any],
    intervention: PairedPolicyIntervention,
    protocol: dict[str, Any],
    identity: bool,
) -> dict[str, Any]:
    video_path = (REPO / "eval_data/OmniVideoBench" / str(record["video"])).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    conversation = common.build_conversation(video_path, record)
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
    ).to(model.device).to(model.dtype)
    ids = inputs["input_ids"]
    seq_len = int(ids.shape[1])
    video_id = int(model.thinker.config.video_token_id)
    audio_id = int(model.thinker.config.audio_token_id)
    flat = ids.detach().reshape(-1).cpu().numpy()
    video_positions = np.flatnonzero(flat == video_id).astype(np.int64)
    audio_positions = np.flatnonzero(flat == audio_id).astype(np.int64)
    if not len(video_positions):
        raise RuntimeError("no video tokens detected")

    intervention.begin_capture(ids, video_id, audio_id)
    capture_native = common.generate_once(
        model, processor, inputs, True, int(protocol["generation"]["max_new_tokens"])
    )
    capture = intervention.end_capture()
    policies = {
        "author_curated_arc_sceb": protocol["author_curated_configuration"]["policy"],
        "automatic_arc_sceb": protocol["automatic_configuration"]["policy"],
    }
    maps = {}
    allocations = {}
    for arm, policy in policies.items():
        maps[arm], allocations[arm] = build_policy_maps(
            capture, video_positions, seq_len, policy, model.device
        )

    identity_result = None
    if identity:
        with intervention.use_maps(
            identity_maps(protocol["implementation_validation"]["union_layers"], seq_len, model.device)
        ):
            identity_output = common.generate_once(
                model, processor, inputs, True, int(protocol["generation"]["max_new_tokens"])
            )
        identity_result = {
            **common.distribution_distances(capture_native.first_logits, identity_output.first_logits),
            "output_text_equal": capture_native.output_text == identity_output.output_text,
            "native_top1_retained": capture_native.first_top1 == identity_output.first_top1,
        }

    arm_order = list(ARMS)
    offset = common.stable_rank(
        int(protocol["sample_selection"]["seed"]), sample_index, "arm-order"
    ) % len(arm_order)
    arm_order = arm_order[offset:] + arm_order[:offset]
    outputs = {}
    for arm in arm_order:
        with intervention.use_maps(None if arm == "native_top8" else maps[arm]):
            outputs[arm] = common.generate_once(
                model, processor, inputs, True, int(protocol["generation"]["max_new_tokens"])
            )
    native = outputs["native_top8"]
    native_repeat = common.distribution_distances(
        capture_native.first_logits, native.first_logits
    )
    correct = str(record["correct_option"]).upper()
    arms = {
        arm: generation_record(output, native, correct)
        for arm, output in outputs.items()
    }
    total_model_pairs = seq_len * NUM_LAYERS * TOP_K
    for arm in allocations:
        allocations[arm]["full_input_logical_pair_reduction_fraction"] = float(
            allocations[arm]["saved_token_expert_pairs"] / total_model_pairs
        )
    return {
        "schema_version": SCHEMA,
        "sample_index": sample_index,
        "video": record["video"],
        "question_type": record.get("question_type"),
        "correct_option": correct,
        "input_sequence_length": seq_len,
        "video_tokens": int(len(video_positions)),
        "audio_tokens": int(len(audio_positions)),
        "arm_order": arm_order,
        "capture_native_elapsed_seconds_not_in_timing_endpoint": capture_native.elapsed_seconds,
        "native_repeat": native_repeat,
        "identity": identity_result,
        "arms": arms,
        "allocation": allocations,
        "video_file": {
            "path": str(video_path.relative_to(REPO)),
            "bytes": video_path.stat().st_size,
            "mtime_ns": video_path.stat().st_mtime_ns,
        },
    }


def bootstrap_mean(values: np.ndarray, seed: int, draws: int) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 1000):
        stop = min(draws, start + 1000)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def aggregate_allocation(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    counts = Counter()
    saved = 0
    total_model_pairs = 0
    for row in rows:
        counts.update(
            {
                int(q): int(count)
                for q, count in row["allocation"][arm]["budget_counts"].items()
            }
        )
        saved += int(row["allocation"][arm]["saved_token_expert_pairs"])
        total_model_pairs += int(row["input_sequence_length"]) * NUM_LAYERS * TOP_K
    total = sum(counts.values())
    shares = {str(q): float(counts[q] / total) for q in (4, 6, 8)}
    return {
        "budget_counts": {str(q): int(counts[q]) for q in (4, 6, 8)},
        "budget_shares": shares,
        "mean_eligible_video_budget": float(
            sum(q * counts[q] for q in (4, 6, 8)) / total
        ),
        "eligible_video_pair_reduction_fraction": float(saved / (8 * total)),
        "full_input_logical_pair_reduction_fraction": float(saved / total_model_pairs),
    }


def summarize(
    rows: list[dict[str, Any]], errors: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    bootstrap = protocol["bootstrap"]
    draws = int(bootstrap["draws"])
    seed = int(bootstrap["seed"])
    n = len(rows)
    arm_summary = {}
    for arm_index, arm in enumerate(ARMS):
        correct = np.asarray([float(row["arms"][arm]["is_correct"]) for row in rows])
        elapsed = np.asarray([float(row["arms"][arm]["elapsed_seconds"]) for row in rows])
        arm_summary[arm] = {
            "accuracy": bootstrap_mean(correct, seed + 10 + arm_index, draws),
            "correct": int(correct.sum()),
            "elapsed_seconds": bootstrap_mean(elapsed, seed + 20 + arm_index, draws),
            "median_elapsed_seconds": float(np.median(elapsed)),
        }
        if arm != "native_top8":
            js = np.asarray([row["arms"][arm]["js_divergence_nats"] for row in rows])
            retained = np.asarray([float(row["arms"][arm]["native_top1_retained"]) for row in rows])
            arm_summary[arm].update(
                {
                    "first_token_js_divergence_nats": bootstrap_mean(
                        js, seed + 30 + arm_index, draws
                    ),
                    "native_first_top1_retention": float(retained.mean()),
                    "allocation": aggregate_allocation(rows, arm),
                }
            )

    def vector(arm: str, field: str) -> np.ndarray:
        return np.asarray([float(row["arms"][arm][field]) for row in rows])

    native_correct = vector("native_top8", "is_correct")
    manual_correct = vector("author_curated_arc_sceb", "is_correct")
    auto_correct = vector("automatic_arc_sceb", "is_correct")
    native_time = vector("native_top8", "elapsed_seconds")
    manual_time = vector("author_curated_arc_sceb", "elapsed_seconds")
    auto_time = vector("automatic_arc_sceb", "elapsed_seconds")
    effects = {
        "automatic_minus_native_accuracy": bootstrap_mean(
            auto_correct - native_correct, seed + 101, draws
        ),
        "automatic_minus_manual_accuracy": bootstrap_mean(
            auto_correct - manual_correct, seed + 102, draws
        ),
        "automatic_vs_native_latency_fraction": bootstrap_mean(
            (auto_time - native_time) / native_time, seed + 103, draws
        ),
        "manual_vs_native_latency_fraction": bootstrap_mean(
            (manual_time - native_time) / native_time, seed + 104, draws
        ),
        "automatic_manual_answer_agreement": float(
            np.mean(
                [
                    row["arms"]["automatic_arc_sceb"]["answer"]
                    == row["arms"]["author_curated_arc_sceb"]["answer"]
                    for row in rows
                ]
            )
        ),
    }

    identity = [row["identity"] for row in rows if row.get("identity")]
    max_identity_js = max(
        (float(value["js_divergence_nats"]) for value in identity), default=float("inf")
    )
    max_repeat_js = max(
        (float(row["native_repeat"]["js_divergence_nats"]) for row in rows),
        default=float("inf"),
    )
    implementation = protocol["implementation_validation"]
    implementation_gates = {
        "minimum_paired_samples": {
            "value": n,
            "threshold": int(implementation["minimum_paired_samples"]),
            "pass": n >= int(implementation["minimum_paired_samples"]),
        },
        "maximum_error_records": {
            "value": len(errors),
            "threshold": int(implementation["maximum_error_records"]),
            "pass": len(errors) <= int(implementation["maximum_error_records"]),
        },
        "identity_samples": {
            "value": len(identity),
            "threshold": int(implementation["identity_arm_samples"]),
            "pass": len(identity) >= int(implementation["identity_arm_samples"]),
        },
        "maximum_identity_js": {
            "value": max_identity_js,
            "threshold": float(implementation["maximum_identity_js_divergence_nats"]),
            "pass": max_identity_js
            <= float(implementation["maximum_identity_js_divergence_nats"]),
        },
        "maximum_native_repeat_js": {
            "value": max_repeat_js,
            "threshold": float(implementation["maximum_identity_js_divergence_nats"]),
            "pass": max_repeat_js
            <= float(implementation["maximum_identity_js_divergence_nats"]),
        },
        "unique_video_per_question": {
            "value": len({row["video"] for row in rows}),
            "threshold": n,
            "pass": len({row["video"] for row in rows}) == n,
        },
    }
    implementation_passed = all(item["pass"] for item in implementation_gates.values())

    claim_spec = protocol["claim_gates"]
    auto_alloc = arm_summary["automatic_arc_sceb"]["allocation"]
    shares = sorted(auto_alloc["budget_shares"].values(), reverse=True)
    claim_gates = {
        "accuracy_noninferior_to_native": {
            "value": effects["automatic_minus_native_accuracy"]["ci95_low"],
            "threshold": -float(claim_spec["native_accuracy_noninferiority_margin"]),
            "pass": effects["automatic_minus_native_accuracy"]["ci95_low"]
            >= -float(claim_spec["native_accuracy_noninferiority_margin"]),
        },
        "accuracy_noninferior_to_manual": {
            "value": effects["automatic_minus_manual_accuracy"]["ci95_low"],
            "threshold": -float(claim_spec["manual_accuracy_noninferiority_margin"]),
            "pass": effects["automatic_minus_manual_accuracy"]["ci95_low"]
            >= -float(claim_spec["manual_accuracy_noninferiority_margin"]),
        },
        "no_large_latency_regression": {
            "value": effects["automatic_vs_native_latency_fraction"]["ci95_high"],
            "threshold": float(claim_spec["maximum_latency_regression_fraction"]),
            "pass": effects["automatic_vs_native_latency_fraction"]["ci95_high"]
            <= float(claim_spec["maximum_latency_regression_fraction"]),
        },
        "heldout_mean_budget_near_target": {
            "value": abs(
                auto_alloc["mean_eligible_video_budget"]
                - float(protocol["automatic_configuration"]["audit_target_mean_budget"])
            ),
            "threshold": float(claim_spec["maximum_mean_budget_absolute_error"]),
            "pass": abs(
                auto_alloc["mean_eligible_video_budget"]
                - float(protocol["automatic_configuration"]["audit_target_mean_budget"])
            )
            <= float(claim_spec["maximum_mean_budget_absolute_error"]),
        },
        "minimum_logical_reduction": {
            "value": auto_alloc["eligible_video_pair_reduction_fraction"],
            "threshold": float(claim_spec["minimum_eligible_video_pair_reduction_fraction"]),
            "pass": auto_alloc["eligible_video_pair_reduction_fraction"]
            >= float(claim_spec["minimum_eligible_video_pair_reduction_fraction"]),
        },
        "at_least_two_nontrivial_budget_levels": {
            "value": shares[1],
            "threshold": float(claim_spec["minimum_budget_share_for_two_levels"]),
            "pass": shares[1] >= float(claim_spec["minimum_budget_share_for_two_levels"]),
        },
    }
    supports = implementation_passed and all(item["pass"] for item in claim_gates.values())
    return {
        "schema_version": SCHEMA,
        "experiment_id": "E16-B",
        "generated_at": now(),
        "result_status": "verified" if implementation_passed else "placeholder",
        "supports_e16_b_claim": supports,
        "paired_samples": n,
        "failed_samples": len(errors),
        "arms": arm_summary,
        "paired_effects": effects,
        "implementation_gates": implementation_gates,
        "claim_gates": claim_gates,
        "scope": protocol["scope"],
        "errors": errors,
    }


def environment_record(
    protocol: dict[str, Any], protocol_path: Path, dataset_path: Path, model_path: Path
) -> dict[str, Any]:
    return {
        "created_at": now(),
        "hostname": socket.gethostname(),
        "requested_worker": os.environ.get("E16B_WORKER_ID", "unknown"),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "git_head": common.git_value("rev-parse", "HEAD"),
        "git_status_short": common.git_value("status", "--short"),
        "protocol_sha256": common.sha256_file(protocol_path),
        "runner_sha256": common.sha256_file(Path(__file__).resolve()),
        "dataset_sha256": common.sha256_file(dataset_path),
        "model_config_sha256": common.sha256_file(model_path / "config.json"),
        "physical_dispatch_note": (
            "custom prefill path removes ranks above each chosen budget from the expert "
            "dispatch mask; decode tokens retain native top-8"
        ),
        "protocol": protocol,
    }


def self_test(protocol: dict[str, Any]) -> None:
    capture = {
        layer: {
            "g4": np.asarray([0.1, 0.001, 0.0]),
            "g6": np.asarray([0.0, 0.1, 0.0]),
            "audio_tokens": 390,
            "audio_margin": 0.01,
        }
        for layer in protocol["automatic_configuration"]["policy"]["eligible_moe_layers"]
    }
    maps, audit = build_policy_maps(
        capture,
        np.arange(3),
        3,
        protocol["automatic_configuration"]["policy"],
        torch.device("cpu"),
    )
    assert len(maps) == 20
    assert sum(audit["budget_counts"].values()) == 60
    assert audit["budget_counts"]["4"] == 40
    assert audit["budget_counts"]["8"] == 20
    log("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol["schema_version"] != SCHEMA:
        raise ValueError("unexpected protocol schema")
    expected_runner = protocol["provenance"].get("runner_sha256")
    observed_runner = common.sha256_file(Path(__file__).resolve())
    if expected_runner and expected_runner != observed_runner:
        raise ValueError(f"runner checksum changed: {observed_runner} != {expected_runner}")
    validate_frozen_configs(protocol)
    if args.self_test:
        self_test(protocol)
        return 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = prepare_selection(protocol)
    selected_path = output_dir / "selected_samples.json"
    if selected_path.exists():
        previous = json.loads(selected_path.read_text())
        if previous["selected_samples_sha256"] != selection["selected_samples_sha256"]:
            raise ValueError("existing selection conflicts with frozen selection")
    common.atomic_json(selected_path, selection)
    log(
        f"prepared {selection['selected_count']} held-out unique-video questions; "
        f"sha256={selection['selected_samples_sha256']}"
    )
    if args.prepare_only:
        return 0
    if not protocol["sample_selection"].get("selected_samples_sha256"):
        raise ValueError("formal run requires frozen selected_samples_sha256")
    common.atomic_json(output_dir / "launcher.json", {"pid": os.getpid(), "started_at": now()})

    dataset_path = REPO / protocol["dataset"]["path"]
    model_path = REPO / protocol["model"]["path"]
    records = json.loads(dataset_path.read_text())
    selected_indices = [int(item["sample_index"]) for item in selection["samples"]]
    seed = int(protocol["sample_selection"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    log("loading Qwen3-Omni model")
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(str(model_path), use_fast=False)
    common.atomic_json(
        output_dir / "environment.json",
        environment_record(protocol, protocol_path, dataset_path, model_path),
    )
    union_layers = [int(value) for value in protocol["implementation_validation"]["union_layers"]]
    intervention = PairedPolicyIntervention(model, union_layers)

    raw_path = output_dir / "raw_results.jsonl"
    error_path = output_dir / "errors.jsonl"
    summary_path = output_dir / "summary.json"
    rows = common.load_jsonl(raw_path)
    errors = common.load_jsonl(error_path)
    completed = {int(row["sample_index"]) for row in rows}
    identity_done = sum(row.get("identity") is not None for row in rows)
    identity_required = int(protocol["implementation_validation"]["identity_arm_samples"])
    identity_limit = float(
        protocol["implementation_validation"]["maximum_identity_js_divergence_nats"]
    )
    try:
        for ordinal, sample_index in enumerate(selected_indices, 1):
            if sample_index in completed:
                continue
            log(f"[{ordinal}/{len(selected_indices)}] sample {sample_index} starting")
            try:
                row = run_sample(
                    sample_index,
                    records[sample_index],
                    model,
                    processor,
                    process_mm_info,
                    intervention,
                    protocol,
                    identity_done < identity_required,
                )
                if row.get("identity"):
                    identity_done += 1
                    identity_js = float(row["identity"]["js_divergence_nats"])
                    if (
                        identity_js > identity_limit
                        or not row["identity"]["native_top1_retained"]
                        or not row["identity"]["output_text_equal"]
                    ):
                        raise RuntimeError(f"identity gate failed: {row['identity']}")
                common.append_jsonl(raw_path, row)
                rows.append(row)
                completed.add(sample_index)
                auto = row["allocation"]["automatic_arc_sceb"]
                answer_trace = ", ".join(
                    f"{arm}:{row['arms'][arm]['answer']}" for arm in ARMS
                )
                log(
                    f"[{ordinal}/{len(selected_indices)}] sample {sample_index} complete; "
                    f"auto_budget={auto['mean_eligible_video_budget']:.4f}; "
                    f"answers={{{answer_trace}}}"
                )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                error = {
                    "sample_index": sample_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                common.append_jsonl(error_path, error)
                errors.append(error)
                log(f"sample {sample_index} OOM; recorded and continuing")
            except Exception as exc:
                error = {
                    "sample_index": sample_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                common.append_jsonl(error_path, error)
                errors.append(error)
                log(f"sample {sample_index} failed: {exc!r}")
                if "identity gate failed" in str(exc):
                    return 2
            common.atomic_json(
                output_dir / "progress.json",
                {
                    "updated_at": now(),
                    "status": "running",
                    "pid": os.getpid(),
                    "paired_samples": len(rows),
                    "requested_samples": len(selected_indices),
                    "error_records": len(errors),
                    "last_attempted_sample_index": sample_index,
                },
            )
        final = summarize(rows, errors, protocol)
        final.update(
            {
                "protocol_sha256": common.sha256_file(protocol_path),
                "runner_sha256": common.sha256_file(Path(__file__).resolve()),
                "raw_results_sha256": common.sha256_file(raw_path),
                "selected_samples_sha256": selection["selected_samples_sha256"],
            }
        )
        common.atomic_json(summary_path, final)
        common.atomic_json(
            output_dir / "progress.json",
            {
                "updated_at": now(),
                "status": "completed",
                "pid": os.getpid(),
                "paired_samples": len(rows),
                "requested_samples": len(selected_indices),
                "error_records": len(errors),
                "supports_e16_b_claim": final["supports_e16_b_claim"],
            },
        )
        log(
            f"finished status={final['result_status']} paired={len(rows)} "
            f"supports_e16_b_claim={final['supports_e16_b_claim']}"
        )
        return 0 if final["result_status"] == "verified" else 3
    finally:
        intervention.close()


if __name__ == "__main__":
    raise SystemExit(main())

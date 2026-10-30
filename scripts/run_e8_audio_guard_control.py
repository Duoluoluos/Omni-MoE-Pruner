#!/usr/bin/env python3
"""Run E8: true same-layer audio versus budget-matched layer-shuffled audio."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
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
PROTOCOL_PATH = REPO / "research_paper/iclr2027/evidence/e8_audio_guard_protocol.json"
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "qwen-omni-utils" / "src"))
import run_e7_cutoff_gap_control as common  # noqa: E402


def log(*parts: Any) -> None:
    print("[E8]", *parts, file=sys.stderr, flush=True)


class AudioGuardIntervention:
    def __init__(self, model: Any, eligible_layers: list[int], top_k: int) -> None:
        self.eligible_layers = set(eligible_layers)
        self.top_k = top_k
        self.active_maps: dict[int, torch.Tensor] | None = None
        self.capture_enabled = False
        self.expected_seq_len = 0
        self.video_mask: torch.Tensor | None = None
        self.audio_mask: torch.Tensor | None = None
        self.captured: dict[int, dict[str, Any]] = {}
        self._wrapped: list[tuple[Any, Callable[..., Any]]] = []
        self._hooks: list[Any] = []
        installed = 0
        for name, module in model.thinker.named_modules():
            if "sparsemoeblock" not in module.__class__.__name__.lower():
                continue
            if not all(hasattr(module, key) for key in ("gate", "experts", "top_k", "num_experts")):
                continue
            layer = common.FrozenBudgetIntervention._parse_layer_idx(name)
            if layer is None or layer not in self.eligible_layers:
                continue
            self._install(module, layer)
            installed += 1
        if installed != len(self.eligible_layers):
            raise RuntimeError(f"expected {len(self.eligible_layers)} MoE blocks, installed {installed}")
        log(f"installed intervention on {installed} MoE layers")

    def _install(self, module: Any, layer: int) -> None:
        original = module.forward
        self._wrapped.append((module, original))

        def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            if not self.capture_enabled or output.ndim != 2 or output.shape[0] != self.expected_seq_len:
                return
            assert self.video_mask is not None and self.audio_mask is not None
            probs = F.softmax(output.detach(), dim=1, dtype=torch.float)
            video_values = torch.topk(probs, self.top_k, dim=-1, sorted=True).values[
                self.video_mask.to(output.device)
            ]
            audio_values = torch.topk(probs, 2, dim=-1, sorted=True).values[
                self.audio_mask.to(output.device)
            ]
            audio_count = int(audio_values.shape[0])
            audio_margin = (
                float((audio_values[:, 0] - audio_values[:, 1]).mean().item())
                if audio_count else float("nan")
            )
            self.captured[layer] = {
                "g4": (video_values[:, 3] - video_values[:, 4]).cpu().float().numpy(),
                "g6": (video_values[:, 5] - video_values[:, 6]).cpu().float().numpy(),
                "audio_margin": audio_margin,
                "audio_tokens": audio_count,
            }

        self._hooks.append(module.gate.register_forward_hook(hook))

        def wrapped(hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if self.active_maps is None or layer not in self.active_maps:
                return original(hidden_states)
            budget = self.active_maps[layer]
            if hidden_states.shape[0] * hidden_states.shape[1] != budget.numel():
                return original(hidden_states)
            return common._moe_mask_forward(module, hidden_states, budget)

        module.forward = wrapped

    def begin_capture(
        self, input_ids: torch.Tensor, video_token_id: int, audio_token_id: int
    ) -> None:
        ids = input_ids.detach().reshape(-1).cpu()
        self.expected_seq_len = ids.numel()
        self.video_mask = ids.eq(video_token_id)
        self.audio_mask = ids.eq(audio_token_id)
        self.captured = {}
        self.active_maps = None
        self.capture_enabled = True

    def end_capture(self) -> dict[int, dict[str, Any]]:
        self.capture_enabled = False
        missing = sorted(self.eligible_layers - set(self.captured))
        if missing:
            raise RuntimeError(f"routing capture missed layers: {missing}")
        return {layer: self.captured[layer] for layer in sorted(self.captured)}

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


def budget_counts(maps: dict[int, np.ndarray]) -> dict[str, int]:
    merged = np.concatenate([maps[layer] for layer in sorted(maps)])
    return {str(q): int(np.count_nonzero(merged == q)) for q in (4, 6, 8)}


def transition_counts(base: dict[int, np.ndarray], arm: dict[int, np.ndarray]) -> Counter:
    counts: Counter = Counter()
    for layer in sorted(base):
        for source, target in zip(base[layer].tolist(), arm[layer].tolist()):
            if source != target:
                counts[(int(source), int(target))] += 1
    return counts


def build_maps(
    capture: dict[int, dict[str, Any]], video_positions: np.ndarray, seq_len: int,
    protocol: dict[str, Any], sample_index: int,
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, Any]]:
    policy = protocol["policy"]
    threshold = float(policy["cutoff_gap_threshold"])
    audio_threshold = float(policy["audio_uncertainty_threshold"])
    min_audio = int(policy["minimum_audio_tokens"])
    base: dict[int, np.ndarray] = {}
    true: dict[int, np.ndarray] = {}
    layer_audio: dict[int, dict[str, Any]] = {}
    for layer, item in sorted(capture.items()):
        g4, g6 = item["g4"], item["g6"]
        if len(g4) != len(video_positions):
            raise RuntimeError(f"layer {layer} video count mismatch")
        values = np.full(len(g4), 8, dtype=np.int16)
        choose4 = g4 >= threshold
        choose6 = (~choose4) & (g6 >= threshold)
        values[choose4] = 4
        values[choose6] = 6
        base[layer] = values
        guarded = values.copy()
        audio_tokens = int(item["audio_tokens"])
        margin = float(item["audio_margin"])
        floor = 8 if audio_tokens < min_audio else 6 if margin < audio_threshold else 0
        if floor:
            guarded = np.maximum(guarded, floor).astype(np.int16)
        true[layer] = guarded
        layer_audio[layer] = {
            "audio_tokens": audio_tokens,
            "audio_margin": margin,
            "true_floor": floor,
        }

    true_transitions = transition_counts(base, true)
    true_counts = budget_counts(true)
    arms_np: dict[str, dict[int, np.ndarray]] = {"true_same_layer_audio": true}
    controls: dict[str, Any] = {}
    layers = sorted(base)
    for arm_name in protocol["arms"]:
        if not arm_name.startswith("layer_shuffled_budget_matched_"):
            continue
        seed = int(arm_name.rsplit("_", 1)[1])
        rng = np.random.default_rng(common.stable_rank(seed, sample_index, "audio-layer-shuffle"))
        source_layers = list(np.asarray(layers)[rng.permutation(len(layers))])
        assigned = {target: int(source) for target, source in zip(layers, source_layers)}
        control = {layer: base[layer].copy() for layer in layers}

        def candidates(source_budget: int, target_budget: int) -> list[tuple[Any, ...]]:
            rows: list[tuple[Any, ...]] = []
            for layer in layers:
                source_layer = assigned[layer]
                audio = layer_audio[source_layer]
                insufficient = int(audio["audio_tokens"] < min_audio)
                margin = float(audio["audio_margin"])
                target_signal = insufficient if target_budget == 8 else int(margin < audio_threshold)
                for token_pos in np.flatnonzero(control[layer] == source_budget):
                    rows.append((
                        -target_signal,
                        margin if np.isfinite(margin) else float("-inf"),
                        common.stable_rank(seed, sample_index, layer, int(token_pos), target_budget),
                        layer,
                        int(token_pos),
                    ))
            return sorted(rows)

        for source_budget, target_budget in ((4, 8), (6, 8), (4, 6)):
            required = int(true_transitions[(source_budget, target_budget)])
            ranked = candidates(source_budget, target_budget)
            if len(ranked) < required:
                raise RuntimeError(
                    f"control {arm_name} lacks candidates for {source_budget}->{target_budget}"
                )
            for *_priority, layer, token_pos in ranked[:required]:
                control[layer][token_pos] = target_budget
        if transition_counts(base, control) != true_transitions:
            raise AssertionError(f"transition matching failed for {arm_name}")
        if budget_counts(control) != true_counts:
            raise AssertionError(f"budget matching failed for {arm_name}")
        arms_np[arm_name] = control
        controls[arm_name] = {
            "layer_permutation": {str(k): v for k, v in assigned.items()},
            "budget_counts": budget_counts(control),
            "transition_counts": {
                f"{a}_to_{b}": n for (a, b), n in sorted(transition_counts(base, control).items())
            },
        }

    arms: dict[str, dict[int, torch.Tensor]] = {}
    hashes: dict[str, str] = {}
    for arm_name, layer_maps in arms_np.items():
        arms[arm_name] = {
            layer: common.full_budget_map(seq_len, video_positions, values)
            for layer, values in layer_maps.items()
        }
        joined = b"".join(layer_maps[layer].tobytes() for layer in sorted(layer_maps))
        hashes[arm_name] = hashlib.sha256(joined).hexdigest()
    max_diff = max(
        abs(budget_counts(arms_np[name])[str(q)] - true_counts[str(q)])
        for name in arms_np for q in (4, 6, 8)
    )
    return arms, {
        "base_budget_counts": budget_counts(base),
        "true_budget_counts": true_counts,
        "true_transition_counts": {
            f"{a}_to_{b}": n for (a, b), n in sorted(true_transitions.items())
        },
        "audio_by_layer": {str(k): v for k, v in layer_audio.items()},
        "controls": controls,
        "allocation_sha256": hashes,
        "maximum_abs_budget_count_difference": max_diff,
    }


def save_capture(path: Path, capture: dict[int, dict[str, Any]]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for layer, item in capture.items():
        arrays[f"layer_{layer:02d}_g4"] = item["g4"].astype(np.float32, copy=False)
        arrays[f"layer_{layer:02d}_g6"] = item["g6"].astype(np.float32, copy=False)
        arrays[f"layer_{layer:02d}_audio"] = np.asarray(
            [item["audio_margin"], item["audio_tokens"]], dtype=np.float64
        )
    np.savez_compressed(path, **arrays)


def run_sample(
    sample_index: int, record: dict[str, Any], model: Any, processor: Any,
    process_mm_info: Callable[..., Any], intervention: AudioGuardIntervention,
    protocol: dict[str, Any], routing_path: Path | None, identity: bool,
    max_new_tokens: int,
) -> dict[str, Any]:
    video_path = (REPO / "eval_data/OmniVideoBench" / str(record["video"])).resolve()
    conversation = common.build_conversation(video_path, record)
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos, return_tensors="pt",
        padding=True, use_audio_in_video=True,
    ).to(model.device).to(model.dtype)
    ids = inputs["input_ids"]
    seq_len = int(ids.shape[1])
    video_id = int(model.thinker.config.video_token_id)
    audio_id = int(model.thinker.config.audio_token_id)
    flat_ids = ids.detach().reshape(-1).cpu().numpy()
    video_positions = np.flatnonzero(flat_ids == video_id).astype(np.int64)
    audio_positions = np.flatnonzero(flat_ids == audio_id).astype(np.int64)
    if not len(video_positions):
        raise RuntimeError("no video tokens detected")
    intervention.begin_capture(ids, video_id, audio_id)
    native = common.generate_once(model, processor, inputs, True, max_new_tokens)
    capture = intervention.end_capture()
    if routing_path is not None:
        save_capture(routing_path, capture)
    maps, audit = build_maps(capture, video_positions, seq_len, protocol, sample_index)

    identity_result = None
    if identity:
        identity_maps = {
            layer: torch.full((seq_len,), 8, dtype=torch.int16)
            for layer in protocol["policy"]["eligible_moe_layers"]
        }
        with intervention.use_maps(identity_maps):
            output = common.generate_once(model, processor, inputs, True, max_new_tokens)
        identity_result = {
            **common.distribution_distances(native.first_logits, output.first_logits),
            "native_top1_retained": output.first_top1 == native.first_top1,
            "output_text_equal": output.output_text == native.output_text,
        }

    arm_results: dict[str, Any] = {}
    for arm_name in protocol["arms"]:
        if arm_name == "native_top8":
            continue
        with intervention.use_maps(maps[arm_name]):
            output = common.generate_once(model, processor, inputs, True, max_new_tokens)
        arm_results[arm_name] = {
            **common.distribution_distances(native.first_logits, output.first_logits),
            "first_top1_token_id": output.first_top1,
            "native_top1_retained": output.first_top1 == native.first_top1,
            "output_text": output.output_text,
            "answer": output.answer,
            "is_correct": output.answer == str(record["correct_option"]).upper(),
            "elapsed_seconds": output.elapsed_seconds,
        }
    return {
        "sample_index": sample_index,
        "video": record["video"],
        "question_type": record.get("question_type"),
        "correct_option": str(record["correct_option"]).upper(),
        "input_sequence_length": seq_len,
        "video_tokens": int(len(video_positions)),
        "audio_tokens": int(len(audio_positions)),
        "native": {
            "first_top1_token_id": native.first_top1,
            "output_text": native.output_text,
            "answer": native.answer,
            "is_correct": native.answer == str(record["correct_option"]).upper(),
            "elapsed_seconds": native.elapsed_seconds,
        },
        "identity": identity_result,
        "arms": arm_results,
        "budget_audit": audit,
    }


def summarize(rows: list[dict[str, Any]], errors: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    true_name = "true_same_layer_audio"
    shuffled = [name for name in protocol["arms"] if name.startswith("layer_shuffled_")]
    true_correct = np.asarray([float(row["arms"][true_name]["is_correct"]) for row in rows])
    shuffle_correct = np.asarray([
        [float(row["arms"][name]["is_correct"]) for name in shuffled] for row in rows
    ]) if rows else np.empty((0, len(shuffled)))
    true_js = np.asarray([row["arms"][true_name]["js_divergence_nats"] for row in rows])
    shuffle_js = np.asarray([
        [row["arms"][name]["js_divergence_nats"] for name in shuffled] for row in rows
    ]) if rows else np.empty((0, len(shuffled)))
    seed = int(protocol["sample_selection"]["seed"])
    primary = {}
    if rows:
        primary = {
            "accuracy_true": common.bootstrap_mean_ci(true_correct, seed + 1),
            "accuracy_shuffled_within_question_mean": common.bootstrap_mean_ci(
                shuffle_correct.mean(axis=1), seed + 2
            ),
            "accuracy_effect_true_minus_shuffled": common.bootstrap_mean_ci(
                true_correct - shuffle_correct.mean(axis=1), seed + 3
            ),
            "js_true": common.bootstrap_mean_ci(true_js, seed + 4),
            "js_shuffled_within_question_mean": common.bootstrap_mean_ci(
                shuffle_js.mean(axis=1), seed + 5
            ),
            "js_effect_true_minus_shuffled": common.bootstrap_mean_ci(
                true_js - shuffle_js.mean(axis=1), seed + 6
            ),
        }
    identity = [row["identity"]["js_divergence_nats"] for row in rows if row.get("identity")]
    max_budget_diff = max(
        (int(row["budget_audit"]["maximum_abs_budget_count_difference"]) for row in rows),
        default=0,
    )
    verified = (
        len(rows) >= int(protocol["sample_selection"]["minimum_paired_samples"])
        and len(identity) >= int(protocol["implementation_validation"]["identity_arm_samples"])
        and max(identity, default=float("inf"))
        <= float(protocol["implementation_validation"]["maximum_identity_js_divergence_nats"])
        and max_budget_diff == 0
    )
    return {
        "schema_version": "arc-sceb/e8-results-v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "experiment_id": "E8",
        "evidence_status": "verified" if verified else "placeholder",
        "paired_samples": len(rows),
        "failed_samples": len(errors),
        "requested_samples": int(protocol["sample_selection"]["target_samples"]),
        "primary_endpoint": primary,
        "audio_alignment_supported": bool(
            verified and primary
            and primary["accuracy_effect_true_minus_shuffled"]["ci95_low"] > 0
        ),
        "secondary_js_supported": bool(
            verified and primary and primary["js_effect_true_minus_shuffled"]["ci95_high"] < 0
        ),
        "implementation_gates": {
            "identity_samples": len(identity),
            "maximum_identity_js_divergence_nats": max(identity) if identity else None,
            "maximum_abs_budget_count_difference": max_budget_diff,
            "passed": verified,
        },
        "scope": protocol["claim_gate"]["scope"],
        "errors": errors,
    }


def self_test() -> None:
    capture = {
        12: {"g4": np.asarray([.9, .8, .1]), "g6": np.asarray([.2, .1, .1]), "audio_margin": .001, "audio_tokens": 100},
        13: {"g4": np.asarray([.9, .1, .1]), "g6": np.asarray([.2, .9, .1]), "audio_margin": .02, "audio_tokens": 100},
    }
    protocol = json.loads(PROTOCOL_PATH.read_text())
    protocol["policy"]["eligible_moe_layers"] = [12, 13]
    arms, audit = build_maps(capture, np.arange(3), 3, protocol, 7)
    assert audit["maximum_abs_budget_count_difference"] == 0
    expected = audit["true_budget_counts"]
    for name, maps in arms.items():
        counts = Counter(v for layer in maps.values() for v in layer.tolist())
        assert {str(q): counts[q] for q in (4, 6, 8)} == expected, name
    print("E8 map construction self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--routing-retention-limit",
        type=int,
        help=(
            "Persist full routing tensors only for sample indices below this "
            "exclusive limit. All samples retain allocation hashes, exact budget "
            "counts, and layer-level audio summaries in raw_results.jsonl."
        ),
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.output_dir is None:
        raise SystemExit("--output-dir is required")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    dataset_path = REPO / protocol["dataset"]
    model_path = REPO / protocol["model"]
    records = json.loads(dataset_path.read_text())
    target = int(args.limit or protocol["sample_selection"]["target_samples"])
    selected = list(range(min(target, len(records))))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "routing").mkdir(exist_ok=True)
    raw_path = output_dir / "raw_results.jsonl"
    error_path = output_dir / "errors.jsonl"
    summary_path = output_dir / "summary.json"
    selection = {
        "selection_method": protocol["sample_selection"],
        "dataset_sha256": common.sha256_file(dataset_path),
        "indices": selected,
    }
    selected_path = output_dir / "selected_samples.json"
    if selected_path.exists() and json.loads(selected_path.read_text()) != selection:
        raise RuntimeError("existing selection conflicts with this run")
    common.atomic_json(selected_path, selection)
    seed = int(protocol["sample_selection"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    log("loading Qwen3-Omni model")
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        str(model_path), dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(str(model_path), use_fast=False)
    requested_worker = os.environ.get("E8_WORKER_ID", "unknown")
    environment = common.environment_record(protocol, dataset_path, model_path)
    environment.update({
        "requested_worker": requested_worker,
        "protocol_sha256": common.sha256_file(PROTOCOL_PATH),
        "runner_sha256": common.sha256_file(Path(__file__).resolve()),
        "routing_retention": {
            "full_tensor_sample_index_exclusive_limit": args.routing_retention_limit,
            "per_sample_raw_audit_retained": True,
        },
    })
    environment_path = output_dir / "environment.json"
    if environment_path.exists():
        raw_rows_before_resume = len(common.load_jsonl(raw_path))
        environment["resume_from"] = {
            "environment_sha256": common.sha256_file(environment_path),
            "raw_rows_before_resume": raw_rows_before_resume,
        }
        common.atomic_json(
            output_dir
            / f"environment_resume_worker{requested_worker}_from_{raw_rows_before_resume:04d}.json",
            environment,
        )
    else:
        common.atomic_json(environment_path, environment)
    intervention = AudioGuardIntervention(
        model, [int(v) for v in protocol["policy"]["eligible_moe_layers"]],
        int(protocol["policy"]["original_top_k"]),
    )
    rows = common.load_jsonl(raw_path)
    done = {int(row["sample_index"]) for row in rows}
    errors = common.load_jsonl(error_path)
    identity_done = sum(row.get("identity") is not None for row in rows)
    identity_required = int(protocol["implementation_validation"]["identity_arm_samples"])
    identity_limit = float(protocol["implementation_validation"]["maximum_identity_js_divergence_nats"])
    try:
        for ordinal, sample_index in enumerate(selected, 1):
            if sample_index in done:
                continue
            log(f"[{ordinal}/{len(selected)}] sample {sample_index} starting")
            try:
                routing_path = None
                if (
                    args.routing_retention_limit is None
                    or sample_index < args.routing_retention_limit
                ):
                    routing_path = output_dir / "routing" / f"sample_{sample_index:04d}.npz"
                row = run_sample(
                    sample_index, records[sample_index], model, processor, process_mm_info,
                    intervention, protocol, routing_path,
                    identity_done < identity_required, args.max_new_tokens,
                )
                if row.get("identity"):
                    identity_done += 1
                    if (
                        row["identity"]["js_divergence_nats"] > identity_limit
                        or not row["identity"]["native_top1_retained"]
                        or not row["identity"]["output_text_equal"]
                    ):
                        raise RuntimeError(f"identity gate failed: {row['identity']}")
                common.append_jsonl(raw_path, row)
                rows.append(row); done.add(sample_index)
                common.atomic_json(output_dir / "progress.json", {
                    "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
                    "paired_samples": len(rows), "requested_samples": len(selected),
                    "error_records": len(errors), "last_completed_sample_index": sample_index,
                })
                log(
                    f"[{ordinal}/{len(selected)}] sample {sample_index} complete; "
                    f"true_js={row['arms']['true_same_layer_audio']['js_divergence_nats']:.6g}; "
                    f"budget_diff={row['budget_audit']['maximum_abs_budget_count_difference']}"
                )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                error = {"sample_index": sample_index, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
                common.append_jsonl(error_path, error); errors.append(error)
                log(f"sample {sample_index} OOM; continuing")
            except Exception as exc:
                error = {"sample_index": sample_index, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
                common.append_jsonl(error_path, error); errors.append(error)
                log(f"sample {sample_index} failed: {exc!r}")
                if "identity gate failed" in str(exc):
                    common.atomic_json(summary_path, summarize(rows, errors, protocol))
                    return 2
        result = summarize(rows, errors, protocol)
        common.atomic_json(summary_path, result)
        log(f"finished status={result['evidence_status']} paired={result['paired_samples']}")
        return 0 if result["evidence_status"] == "verified" else 3
    finally:
        intervention.close()


if __name__ == "__main__":
    raise SystemExit(main())

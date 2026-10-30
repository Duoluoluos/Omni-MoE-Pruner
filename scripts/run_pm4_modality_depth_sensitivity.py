#!/usr/bin/env python3
"""Run P-M4: matched-budget modality x depth expert sensitivity.

For each OmniVideoBench sample, choose the same number of audio and video
tokens.  In six matched arms, lower those tokens from the native top-8 expert
budget to top-4 in one equal-sized depth band.  The intervention is confined to
prefill.  Outcomes are never used to construct an arm.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO
    / "research_paper/iclr2027/evidence/pm4_modality_depth_sensitivity_protocol.json"
)
sys.path.insert(0, str(REPO / "qwen-omni-utils" / "src"))

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
OPTION_LETTERS = ("A", "B", "C", "D")


def log(*parts: Any) -> None:
    print("[P-M4]", *parts, file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


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
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
    return rows


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=20,
        ).strip()
    except Exception:
        return "unavailable"


def parse_answer(text: str) -> str | None:
    upper = str(text).strip().upper()
    match = re.search(r"(?:^|[^A-Z])([A-D])(?:[^A-Z]|$)", upper)
    if match:
        return match.group(1)
    for character in upper:
        if character in OPTION_LETTERS:
            return character
    return None


def build_conversation(
    video_path: Path, record: dict[str, Any]
) -> list[dict[str, Any]]:
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


def spread_select(positions: np.ndarray, count: int) -> np.ndarray:
    """Select `count` deterministic, approximately uniform positions."""
    positions = np.asarray(positions, dtype=np.int64)
    count = int(count)
    if count < 0 or count > len(positions):
        raise ValueError((count, len(positions)))
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    if count == len(positions):
        return positions.copy()
    indices = ((np.arange(count, dtype=np.float64) + 0.5) * len(positions) / count).astype(
        np.int64
    )
    selected = positions[indices]
    if len(np.unique(selected)) != count:
        raise AssertionError("spread_select produced duplicate positions")
    return selected


def arm_names(protocol: dict[str, Any]) -> list[str]:
    return list(protocol["factorial_intervention"]["arms"])


def build_plans(
    input_ids: np.ndarray, protocol: dict[str, Any]
) -> tuple[dict[str, dict[int, torch.Tensor] | None], dict[str, Any]]:
    cfg = protocol["factorial_intervention"]
    native_k = int(cfg["original_top_k"])
    reduced_k = int(cfg["intervened_top_k"])
    video_id = int(cfg["modalities"]["video"]["input_token_id"])
    audio_id = int(cfg["modalities"]["audio"]["input_token_id"])
    video_positions = np.nonzero(input_ids == video_id)[0].astype(np.int64)
    audio_positions = np.nonzero(input_ids == audio_id)[0].astype(np.int64)
    matched_count = min(len(video_positions), len(audio_positions))
    selected = {
        "video": spread_select(video_positions, matched_count),
        "audio": spread_select(audio_positions, matched_count),
    }
    sequence_length = int(len(input_ids))
    plans: dict[str, dict[int, torch.Tensor] | None] = {"native_top8": None}
    removed_by_arm: dict[str, int] = {}
    for modality in ("video", "audio"):
        for depth_name, bounds in cfg["depth_bands"].items():
            start = int(bounds["start_inclusive"])
            end = int(bounds["end_exclusive"])
            name = f"{modality}_{depth_name}_top4"
            plan: dict[int, torch.Tensor] = {}
            for layer in range(start, end):
                budget = torch.full((sequence_length,), native_k, dtype=torch.long)
                if matched_count:
                    budget[torch.as_tensor(selected[modality], dtype=torch.long)] = reduced_k
                plan[layer] = budget
            plans[name] = plan
            removed_by_arm[name] = matched_count * (end - start) * (native_k - reduced_k)

    expected_arms = set(arm_names(protocol))
    if set(plans) != expected_arms:
        raise AssertionError((sorted(plans), sorted(expected_arms)))
    non_native_removed = list(removed_by_arm.values())
    budget_match = len(set(non_native_removed)) <= 1
    metadata = {
        "video_token_count": int(len(video_positions)),
        "audio_token_count": int(len(audio_positions)),
        "matched_token_count": int(matched_count),
        "selected_video_positions": selected["video"].tolist(),
        "selected_audio_positions": selected["audio"].tolist(),
        "removed_token_expert_pairs_by_arm": removed_by_arm,
        "budget_match_ok": bool(budget_match),
    }
    return plans, metadata


class TargetedBudgetIntervention:
    """Execute fewer experts for selected prefill tokens and layers."""

    def __init__(self, model: Any, native_top_k: int) -> None:
        self.native_top_k = int(native_top_k)
        self.active_plan: dict[int, torch.Tensor] | None = None
        self._wrapped: list[tuple[Any, Callable[..., Any]]] = []
        for name, module in model.thinker.named_modules():
            if "sparsemoeblock" not in module.__class__.__name__.lower():
                continue
            if not all(
                hasattr(module, attribute)
                for attribute in ("gate", "experts", "top_k", "num_experts")
            ):
                continue
            if int(module.top_k) != self.native_top_k:
                raise RuntimeError(
                    f"{name}: top_k={module.top_k}, expected {self.native_top_k}"
                )
            layer = self._layer_index(name)
            if layer is None:
                continue
            self._install(module, layer)
        if not self._wrapped:
            raise RuntimeError("no SparseMoeBlock modules found")
        log(f"installed targeted intervention on {len(self._wrapped)} MoE layers")

    @staticmethod
    def _layer_index(name: str) -> int | None:
        parts = name.split(".")
        for index, part in enumerate(parts[:-1]):
            if part == "layers" and parts[index + 1].isdigit():
                return int(parts[index + 1])
        return None

    def _install(self, module: Any, layer: int) -> None:
        original = module.forward

        def wrapped(hidden_states: torch.Tensor):
            budget = None if self.active_plan is None else self.active_plan.get(layer)
            if budget is None:
                return original(hidden_states)
            return self._budgeted_forward(module, hidden_states, budget)

        module.forward = wrapped
        self._wrapped.append((module, original))

    def _budgeted_forward(
        self, module: Any, hidden_states: torch.Tensor, budget_vector: torch.Tensor
    ):
        batch, sequence_length, hidden_size = hidden_states.shape
        flat = hidden_states.view(-1, hidden_size)
        router_logits = module.gate(flat)
        routing_weights = torch.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, module.top_k, dim=-1
        )
        token_count = flat.shape[0]
        if budget_vector.shape[0] == token_count:
            budgets = budget_vector.to(routing_weights.device)
        else:
            # The dense vector describes the input prefill.  Decode calls have
            # one token and deliberately retain the native budget.
            budgets = torch.full(
                (token_count,),
                module.top_k,
                dtype=torch.long,
                device=routing_weights.device,
            )
        ranks = torch.arange(module.top_k, device=routing_weights.device).unsqueeze(0)
        keep = ranks < budgets.unsqueeze(1)
        routing_weights = routing_weights.masked_fill(~keep, 0.0)
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-20)
        routing_weights = routing_weights.to(flat.dtype)

        final = torch.zeros(
            (token_count, hidden_size), dtype=flat.dtype, device=flat.device
        )
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=module.num_experts
        ).permute(2, 1, 0)
        active_experts = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_index in active_experts:
            rank_index, token_index = torch.where(expert_mask[expert_index].squeeze(0))
            if token_index.numel() == 0:
                continue
            nonzero = routing_weights[token_index, rank_index] > 0
            if not bool(nonzero.any()):
                continue
            token_index = token_index[nonzero]
            rank_index = rank_index[nonzero]
            expert_output = module.experts[expert_index](
                flat[None, token_index].reshape(-1, hidden_size)
            )
            expert_output = expert_output * routing_weights[
                token_index, rank_index, None
            ]
            final.index_add_(0, token_index, expert_output.to(flat.dtype))
        return final.reshape(batch, sequence_length, hidden_size), router_logits

    @contextlib.contextmanager
    def use_plan(self, plan: dict[int, torch.Tensor] | None):
        previous = self.active_plan
        self.active_plan = plan
        try:
            yield
        finally:
            self.active_plan = previous

    def close(self) -> None:
        for module, original in self._wrapped:
            module.forward = original
        self._wrapped.clear()


def distribution_js(native_logits: torch.Tensor, arm_logits: torch.Tensor) -> float:
    native_logp = torch.log_softmax(native_logits.double(), dim=-1)
    arm_logp = torch.log_softmax(arm_logits.double(), dim=-1)
    native_p, arm_p = native_logp.exp(), arm_logp.exp()
    mixture_log = (0.5 * (native_p + arm_p)).clamp_min(
        torch.finfo(torch.float64).tiny
    ).log()
    js = 0.5 * (
        torch.sum(native_p * (native_logp - mixture_log))
        + torch.sum(arm_p * (arm_logp - mixture_log))
    )
    return max(0.0, float(js.item()))


def gold_logprob(
    first_logits: torch.Tensor, option_token_ids: dict[str, int], correct: str
) -> float:
    token_id = option_token_ids.get(correct)
    if token_id is None:
        return float("nan")
    return float(torch.log_softmax(first_logits.double(), dim=-1)[token_id].item())


def generate_once(
    model: Any,
    processor: Any,
    inputs: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        raw = model.generate(
            **inputs,
            use_audio_in_video=True,
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
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    thinker = raw[0] if isinstance(raw, tuple) else raw
    first_logits = thinker.scores[0][0].detach().float().cpu()
    input_length = int(inputs["input_ids"].shape[1])
    generated = thinker.sequences[0, input_length:].detach().cpu()
    output_text = processor.tokenizer.decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    return {
        "first_logits": first_logits,
        "first_top1_token_id": int(first_logits.argmax().item()),
        "output_text": output_text,
        "answer": parse_answer(output_text),
        "end_to_end_latency_ms": elapsed_ms,
    }


def run_sample(
    *,
    sample_index: int,
    record: dict[str, Any],
    model: Any,
    processor: Any,
    process_mm_info: Callable[..., Any],
    intervention: TargetedBudgetIntervention,
    protocol: dict[str, Any],
    option_token_ids: dict[str, int],
    identity_check: bool,
) -> dict[str, Any]:
    video_path = (REPO / "eval_data/OmniVideoBench" / str(record["video"])).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    conversation = build_conversation(video_path, record)
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
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
    input_ids = inputs["input_ids"].reshape(-1).detach().cpu().numpy()
    plans, budget_metadata = build_plans(input_ids, protocol)
    if not budget_metadata["budget_match_ok"]:
        raise AssertionError("non-native arms are not budget matched")

    seed = int(protocol["decoding"]["seed"]) + int(sample_index)
    order = arm_names(protocol)
    np.random.default_rng(seed).shuffle(order)
    max_new_tokens = int(protocol["decoding"]["thinker_max_new_tokens"])
    correct = str(record["correct_option"]).upper()
    transient: dict[str, dict[str, Any]] = {}
    for arm in order:
        with intervention.use_plan(plans[arm]):
            generated = generate_once(model, processor, inputs, max_new_tokens)
        generated["gold_option_logprob"] = gold_logprob(
            generated["first_logits"], option_token_ids, correct
        )
        generated["is_correct"] = bool(generated["answer"] == correct)
        transient[arm] = generated

    native = transient["native_top8"]
    native_logits = native["first_logits"]
    native_top1 = native["first_top1_token_id"]
    native_answer = native["answer"]
    arms: dict[str, Any] = {}
    for arm in arm_names(protocol):
        value = transient[arm]
        first_logits = value.pop("first_logits")
        arms[arm] = {
            **value,
            "first_token_js_vs_native": 0.0
            if arm == "native_top8"
            else distribution_js(native_logits, first_logits),
            "native_top1_retained": bool(
                value["first_top1_token_id"] == native_top1
            ),
            "answer_flip_vs_native": bool(value["answer"] != native_answer),
        }

    identity: dict[str, Any] | None = None
    if identity_check:
        native_k = int(protocol["factorial_intervention"]["original_top_k"])
        identity_plan = {
            layer: torch.full((len(input_ids),), native_k, dtype=torch.long)
            for layer in range(48)
        }
        with intervention.use_plan(identity_plan):
            identity_generated = generate_once(
                model, processor, inputs, max_new_tokens
            )
        identity = {
            "first_token_js_vs_native": distribution_js(
                native_logits, identity_generated["first_logits"]
            ),
            "native_top1_retained": bool(
                identity_generated["first_top1_token_id"] == native_top1
            ),
            "output_text_equal": bool(
                identity_generated["output_text"] == native["output_text"]
            ),
        }

    return {
        "sample_index": int(sample_index),
        "video": str(record["video"]),
        "question_type": record.get("question_type"),
        "correct_option": correct,
        "input_sequence_length": int(len(input_ids)),
        "arm_order": order,
        "budget": budget_metadata,
        "identity": identity,
        "arms": arms,
    }


def cluster_bootstrap_difference(
    rows: list[dict[str, Any]],
    arm_a: str,
    arm_b: str,
    metric: str,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    by_cluster: dict[str, list[float]] = defaultdict(list)
    observations = 0
    for row in rows:
        if int(row["budget"]["matched_token_count"]) == 0:
            continue
        value_a = row["arms"][arm_a].get(metric)
        value_b = row["arms"][arm_b].get(metric)
        if value_a is None or value_b is None:
            continue
        difference = float(value_a) - float(value_b)
        if np.isnan(difference):
            continue
        by_cluster[str(row["video"])].append(difference)
        observations += 1
    if not by_cluster:
        return {
            "mean_difference_a_minus_b": None,
            "ci95_low": None,
            "ci95_high": None,
            "samples": 0,
            "clusters": 0,
        }
    cluster_ids = sorted(by_cluster)
    cluster_means = np.asarray(
        [np.mean(by_cluster[cluster]) for cluster in cluster_ids], dtype=np.float64
    )
    observed = float(
        np.mean([value for values in by_cluster.values() for value in values])
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sample = rng.integers(0, len(cluster_ids), size=len(cluster_ids))
        bootstrap[draw] = cluster_means[sample].mean()
    return {
        "mean_difference_a_minus_b": observed,
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "samples": observations,
        "clusters": len(cluster_ids),
    }


def bootstrap_mean(values: list[float], seed: int, draws: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[~np.isnan(array)]
    if not len(array):
        return {"mean": None, "ci95_low": None, "ci95_high": None, "samples": 0}
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(draws, dtype=np.float64)
    chunk = 500
    for start in range(0, draws, chunk):
        stop = min(draws, start + chunk)
        indices = rng.integers(0, len(array), size=(stop - start, len(array)))
        bootstrap[start:stop] = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "samples": int(len(array)),
    }


def summarize(
    rows: list[dict[str, Any]], errors: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    seed = int(protocol["decoding"]["seed"])
    draws = int(protocol["statistics"]["uncertainty"].split("-")[0])
    nonzero_rows = [
        row for row in rows if int(row["budget"]["matched_token_count"]) > 0
    ]
    budget_match_all = all(
        bool(row["budget"]["budget_match_ok"]) for row in nonzero_rows
    )

    identity_rows = [row["identity"] for row in rows if row.get("identity")]
    maximum_identity_js = max(
        (float(value["first_token_js_vs_native"]) for value in identity_rows),
        default=None,
    )
    validation = protocol["implementation_validation"]
    identity_pass = (
        len(identity_rows) >= int(validation["identity_check_samples"])
        and maximum_identity_js is not None
        and maximum_identity_js
        <= float(validation["maximum_identity_js_divergence_nats"])
        and all(bool(value["native_top1_retained"]) for value in identity_rows)
    )

    summaries: dict[str, Any] = {}
    native_name = "native_top8"
    for index, arm in enumerate(arm_names(protocol)):
        accuracy = [float(row["arms"][arm]["is_correct"]) for row in nonzero_rows]
        gold = [float(row["arms"][arm]["gold_option_logprob"]) for row in nonzero_rows]
        js = [float(row["arms"][arm]["first_token_js_vs_native"]) for row in nonzero_rows]
        flips = [float(row["arms"][arm]["answer_flip_vs_native"]) for row in nonzero_rows]
        c_to_w = [
            float(
                bool(row["arms"][native_name]["is_correct"])
                and not bool(row["arms"][arm]["is_correct"])
            )
            for row in nonzero_rows
        ]
        w_to_c = [
            float(
                not bool(row["arms"][native_name]["is_correct"])
                and bool(row["arms"][arm]["is_correct"])
            )
            for row in nonzero_rows
        ]
        top1 = [float(row["arms"][arm]["native_top1_retained"]) for row in nonzero_rows]
        summaries[arm] = {
            "accuracy": bootstrap_mean(accuracy, seed + 100 + index, draws),
            "gold_option_logprob": bootstrap_mean(gold, seed + 200 + index, draws),
            "gold_logprob_difference_vs_native": cluster_bootstrap_difference(
                nonzero_rows,
                arm,
                native_name,
                "gold_option_logprob",
                seed + 300 + index,
                draws,
            ),
            "first_token_js_vs_native": bootstrap_mean(
                js, seed + 400 + index, draws
            ),
            "answer_flip_rate": float(np.mean(flips)) if flips else None,
            "correct_to_wrong_rate": float(np.mean(c_to_w)) if c_to_w else None,
            "wrong_to_correct_rate": float(np.mean(w_to_c)) if w_to_c else None,
            "native_top1_retention": float(np.mean(top1)) if top1 else None,
        }

    contrasts: dict[str, Any] = {}
    for index, pair in enumerate(protocol["primary_contrasts"]):
        arm_a, arm_b = pair
        contrasts[f"{arm_a}_minus_{arm_b}"] = cluster_bootstrap_difference(
            nonzero_rows,
            arm_a,
            arm_b,
            "gold_option_logprob",
            seed + 1000 + index,
            draws,
        )

    selection = protocol["sample_selection"]
    complete_gate = len(rows) >= int(selection["minimum_completed_samples"])
    nonzero_gate = len(nonzero_rows) >= int(selection["minimum_nonzero_matched_samples"])
    verified = complete_gate and nonzero_gate and budget_match_all and identity_pass
    return {
        "schema_version": "my-budgeting/pm4-results-v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "experiment_id": "P-M4",
        "evidence_status": "verified" if verified else "placeholder",
        "completed_samples": len(rows),
        "nonzero_matched_samples": len(nonzero_rows),
        "failed_samples": len(errors),
        "requested_samples": int(selection["target_samples"]),
        "budget_match_all_nonzero_samples": bool(budget_match_all),
        "arms": summaries,
        "primary_gold_logprob_contrasts": contrasts,
        "implementation_gates": {
            "identity_samples": len(identity_rows),
            "maximum_identity_js_divergence_nats": maximum_identity_js,
            "identity_top1_all_retained": all(
                bool(value["native_top1_retained"]) for value in identity_rows
            )
            if identity_rows
            else False,
            "identity_passed": bool(identity_pass),
            "completion_passed": bool(complete_gate),
            "nonzero_matched_passed": bool(nonzero_gate),
            "budget_matching_passed": bool(budget_match_all),
            "passed": bool(verified),
        },
        "scope": protocol["scope"],
        "errors": errors,
    }


def environment_record(
    protocol: dict[str, Any], dataset_path: Path, model_path: Path
) -> dict[str, Any]:
    model_config = model_path / "config.json"
    return {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "requested_worker": os.environ.get("PM4_WORKER_ID", "unknown"),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status_short": git_value("status", "--short"),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "model_config_sha256": sha256_file(model_config)
        if model_config.exists()
        else None,
        "protocol": protocol,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> None:
    positions = np.arange(100, dtype=np.int64)
    selected = spread_select(positions, 17)
    assert len(selected) == 17 and len(np.unique(selected)) == 17
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    ids = np.asarray([151656] * 40 + [151675] * 10 + [1] * 5, dtype=np.int64)
    plans, metadata = build_plans(ids, protocol)
    assert metadata["matched_token_count"] == 10
    assert metadata["budget_match_ok"] is True
    assert len(set(metadata["removed_token_expert_pairs_by_arm"].values())) == 1
    assert set(plans) == set(arm_names(protocol))
    assert plans["video_early_top4"][0][metadata["selected_video_positions"]].eq(4).all()
    assert 16 not in plans["video_early_top4"]
    assert plans["video_middle_top4"][16][metadata["selected_video_positions"]].eq(4).all()
    logits = torch.tensor([0.0, 2.0, 1.0, -1.0])
    assert distribution_js(logits, logits.clone()) < 1e-12
    assert gold_logprob(logits, {"A": 0, "B": 1, "C": 2, "D": 3}, "B") > -1.0
    print("P-M4 matched-budget/self-test passed.")


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
    requested = int(protocol["sample_selection"]["target_samples"])
    requested = min(requested, len(records))
    if args.limit is not None:
        requested = min(requested, int(args.limit))
    selected_indices = list(range(requested))

    seed = int(protocol["decoding"]["seed"])
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_results.jsonl"
    error_path = output_dir / "errors.jsonl"

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
    option_token_ids: dict[str, int] = {}
    for letter in OPTION_LETTERS:
        token_ids = processor.tokenizer.encode(letter, add_special_tokens=False)
        if len(token_ids) == 1:
            option_token_ids[letter] = int(token_ids[0])
    if set(option_token_ids) != set(OPTION_LETTERS):
        raise RuntimeError(f"option letters are not single tokens: {option_token_ids}")

    atomic_json(
        output_dir / "environment.json",
        environment_record(protocol, dataset_path, model_path),
    )
    intervention = TargetedBudgetIntervention(
        model, int(protocol["factorial_intervention"]["original_top_k"])
    )
    completed_rows = load_jsonl(raw_path)
    completed = {int(row["sample_index"]) for row in completed_rows}
    errors = load_jsonl(error_path)
    identity_done = sum(row.get("identity") is not None for row in completed_rows)
    identity_required = int(
        protocol["implementation_validation"]["identity_check_samples"]
    )

    try:
        for ordinal, sample_index in enumerate(selected_indices, 1):
            if sample_index in completed:
                continue
            try:
                row = run_sample(
                    sample_index=sample_index,
                    record=records[sample_index],
                    model=model,
                    processor=processor,
                    process_mm_info=process_mm_info,
                    intervention=intervention,
                    protocol=protocol,
                    option_token_ids=option_token_ids,
                    identity_check=identity_done < identity_required,
                )
                if row.get("identity") is not None:
                    identity_done += 1
                append_jsonl(raw_path, row)
                completed_rows.append(row)
                completed.add(sample_index)
                if ordinal <= 3 or ordinal % 20 == 0:
                    log(
                        f"[{ordinal}/{len(selected_indices)}] sample={sample_index} "
                        f"q={row['budget']['matched_token_count']} "
                        f"match={row['budget']['budget_match_ok']}"
                    )
                atomic_json(
                    output_dir / "progress.json",
                    {
                        "updated_at": datetime.now(timezone.utc)
                        .astimezone()
                        .isoformat(),
                        "completed_samples": len(completed_rows),
                        "requested_samples": len(selected_indices),
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
        summary = summarize(completed_rows, errors, protocol)
        atomic_json(output_dir / "summary.json", summary)
        log(
            f"finished status={summary['evidence_status']} "
            f"completed={summary['completed_samples']} failed={summary['failed_samples']}"
        )
        return 0 if summary["evidence_status"] == "verified" else 3
    finally:
        intervention.close()


if __name__ == "__main__":
    raise SystemExit(main())

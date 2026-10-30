#!/usr/bin/env python3
"""Run P-M2 on Uni-MoE-2.0-Omni: equal-layer uniform expert-budget frontier.

The protocol is frozen in
research_paper/iclr2027/evidence/pm2_unimoe2_uniform_frontier_protocol.json.

For every OmniVideoBench example this runner runs greedy generation under the
native GRIN reference and under uniform dynamic-expert budgets applied to EVERY
token at EVERY MoE layer.  Uni-MoE-2's GRINMoESparseMoeBlock selects dynamic
experts with a top-p rule (mlp_dynamic_top_p=0.7) over four real dynamic experts
plus two always-on shared experts.  Each arm switches every block to fixed
dynamic budgeting (top_p=0) with a uniform mlp_dynamic_top_k, leaving the two
shared experts untouched.  It records multiple-choice accuracy, first-token JS
divergence from native, end-to-end latency, and peak GPU memory.  The
uniform_dynamic_top4 arm is a near-identity/wiring check.  This is a
single-checkpoint, single-H20 deterministic inference counterpart, not a
multi-seed training frontier and not a systems-speed claim.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO / "research_paper/iclr2027/evidence/pm2_unimoe2_uniform_frontier_protocol.json"
)
sys.path.insert(0, "/home/tiger/Uni-MoE/Uni-MoE-2")


def log(*parts: Any) -> None:
    print("[P-M2-UniMoE2]", *parts, file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
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


def parse_answer(text: str) -> str | None:
    upper = str(text).strip().upper()
    match = re.search(r"(?:^|[^A-Z])([A-D])(?:[^A-Z]|$)", upper)
    if match:
        return match.group(1)
    for char in upper:
        if char in "ABCD":
            return char
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
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(video_path)},
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": f"<video>\n<audio>\n{prompt}"},
            ],
        }
    ]


def expand_placeholders(texts: str) -> str:
    return (
        texts.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        .replace("<audio>", "<|audio_start|><|audio_pad|><|audio_end|>")
        .replace("<video>", "<|vision_start|><|video_pad|><|vision_end|>")
    )


class UniformDynamicBudgetIntervention:
    """Uniformly fix every GRIN MoE block's dynamic expert budget.

    Native routing uses a per-token top-p rule.  Each arm switches every block
    to fixed budgeting (``mlp_dynamic_top_p = 0.0``) with a shared
    ``mlp_dynamic_top_k``, so every token selects exactly the requested number
    of dynamic real experts at every MoE layer.  The two always-on shared
    experts (``mlp_fixed_expert_num``) are never touched.  Values are restored
    to the model's native configuration afterward.
    """

    def __init__(self, model: Any, expected_blocks: int) -> None:
        self.model = model
        self._blocks: list[Any] = []
        for name, module in model.named_modules():
            if module.__class__.__name__ == "GRINMoESparseMoeBlock":
                if not all(
                    hasattr(module, attr)
                    for attr in ("mlp_dynamic_top_k", "mlp_dynamic_top_p")
                ):
                    continue
                self._blocks.append(module)
        if len(self._blocks) != expected_blocks:
            raise RuntimeError(
                f"Found {len(self._blocks)} GRIN MoE blocks, expected {expected_blocks}"
            )
        self._native_top_p = [float(b.mlp_dynamic_top_p) for b in self._blocks]
        self._native_top_k = [int(b.mlp_dynamic_top_k) for b in self._blocks]
        self.installed = len(self._blocks)
        log(f"installed uniform dynamic-budget control on {self.installed} MoE blocks")

    @contextlib.contextmanager
    def use_budget(self, budget: int):
        for module in self._blocks:
            module.mlp_dynamic_top_p = 0.0
            module.mlp_dynamic_top_k = int(budget)
        try:
            yield
        finally:
            self.restore()

    def restore(self) -> None:
        for module, top_p, top_k in zip(
            self._blocks, self._native_top_p, self._native_top_k
        ):
            module.mlp_dynamic_top_p = top_p
            module.mlp_dynamic_top_k = top_k

    def close(self) -> None:
        self.restore()


@dataclass
class GenerationResult:
    first_logits: torch.Tensor
    first_top1: int
    output_text: str
    answer: str | None
    e2e_seconds: float
    peak_memory_bytes: int


def build_inputs(processor: Any, process_mm_info: Callable[..., Any], conversation):
    texts = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    texts = expand_placeholders(texts)
    image_inputs, video_inputs, audio_inputs = process_mm_info(conversation)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        padding=True,
        return_tensors="pt",
    )
    if "second_grid_ts" in inputs:
        inputs["second_per_grid_ts"] = inputs["second_grid_ts"]
        del inputs["second_grid_ts"]
    inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
    return inputs


def move_inputs(inputs: Any, model: Any) -> Any:
    inputs = inputs.to(device=model.device)
    for k, v in list(inputs.items()):
        if k in ("pixel_values", "pixel_values_videos", "audio_features"):
            inputs[k] = v.to(dtype=torch.bfloat16)
    return inputs


def generate_once(
    model: Any, processor: Any, inputs: Any, max_new_tokens: int
) -> GenerationResult:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        raw = model.generate(
            **inputs,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )
    torch.cuda.synchronize()
    e2e = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated())
    if not getattr(raw, "scores", None):
        raise RuntimeError("Generation returned no score tensor")
    first_logits = raw.scores[0][0].detach().float().cpu()
    input_len = int(inputs["input_ids"].shape[-1])
    generated_ids = raw.sequences[0, input_len:].detach().cpu()
    output_text = processor.tokenizer.decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    return GenerationResult(
        first_logits=first_logits,
        first_top1=int(first_logits.argmax().item()),
        output_text=output_text,
        answer=parse_answer(output_text),
        e2e_seconds=e2e,
        peak_memory_bytes=peak,
    )


def distribution_distances(
    native_logits: torch.Tensor, arm_logits: torch.Tensor
) -> dict[str, float]:
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


def environment_record(
    protocol: dict[str, Any], dataset_path: Path, model_path: Path
) -> dict[str, Any]:
    config_path = model_path / "config.json"
    return {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "requested_worker": os.environ.get("PM2_WORKER_ID", "unknown"),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status_short": git_value("status", "--short"),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "model_config_sha256": sha256_file(config_path)
        if config_path.exists()
        else None,
        "protocol": protocol,
    }


def bootstrap_mean_ci(
    values: np.ndarray, seed: int, draws: int = 10000
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None}
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


def run_sample(
    *,
    sample_index: int,
    record: dict[str, Any],
    model: Any,
    processor: Any,
    process_mm_info: Callable[..., Any],
    intervention: UniformDynamicBudgetIntervention,
    protocol: dict[str, Any],
    max_new_tokens: int,
    identity: bool,
) -> dict[str, Any]:
    video_path = (REPO / "eval_data/OmniVideoBench" / str(record["video"])).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    conversation = build_conversation(video_path, record)
    inputs = build_inputs(processor, process_mm_info, conversation)
    seq_len = int(inputs["input_ids"].shape[-1])
    inputs = move_inputs(inputs, model)
    correct = str(record["correct_option"]).upper()

    native = generate_once(model, processor, inputs, max_new_tokens)

    identity_result: dict[str, Any] | None = None
    identity_arm_budget = int(protocol["policy"]["dynamic_real_expert_num"])
    if identity:
        with intervention.use_budget(identity_arm_budget):
            identity_out = generate_once(model, processor, inputs, max_new_tokens)
        identity_result = {
            **distribution_distances(native.first_logits, identity_out.first_logits),
            "native_top1_retained": identity_out.first_top1 == native.first_top1,
            "output_text_equal": identity_out.output_text == native.output_text,
            "budget": identity_arm_budget,
        }

    arms: dict[str, Any] = {}
    budgets = list(protocol["policy"]["uniform_dynamic_budgets"])
    for budget in budgets:
        arm_name = f"uniform_dynamic_top{budget}"
        with intervention.use_budget(budget):
            out = generate_once(model, processor, inputs, max_new_tokens)
        arms[arm_name] = {
            **distribution_distances(native.first_logits, out.first_logits),
            "budget": int(budget),
            "first_top1_token_id": out.first_top1,
            "native_top1_retained": out.first_top1 == native.first_top1,
            "output_text": out.output_text,
            "answer": out.answer,
            "answer_flip_vs_native": out.answer != native.answer,
            "is_correct": out.answer == correct,
            "end_to_end_latency_ms": out.e2e_seconds * 1000.0,
            "peak_gpu_memory_bytes": out.peak_memory_bytes,
        }

    return {
        "sample_index": sample_index,
        "video": record["video"],
        "question_type": record.get("question_type"),
        "correct_option": correct,
        "input_sequence_length": seq_len,
        "native": {
            "first_top1_token_id": native.first_top1,
            "output_text": native.output_text,
            "answer": native.answer,
            "is_correct": native.answer == correct,
            "end_to_end_latency_ms": native.e2e_seconds * 1000.0,
            "peak_gpu_memory_bytes": native.peak_memory_bytes,
        },
        "identity": identity_result,
        "arms": arms,
    }


def summarize(
    rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    protocol: dict[str, Any],
    controller_blocks: int,
    controller_expected: int,
) -> dict[str, Any]:
    controller_gate_passed = controller_blocks == controller_expected
    n = len(rows)
    seed = int(protocol["decoding"]["seed"])
    warmup = int(protocol["timing"]["warmup_samples"])
    budgets = list(protocol["policy"]["uniform_dynamic_budgets"])
    arm_names = [f"uniform_dynamic_top{b}" for b in budgets]
    steady_rows = rows[warmup:] if len(rows) > warmup else rows

    identity_js = [
        float(row["identity"]["js_divergence_nats"])
        for row in rows
        if row.get("identity") is not None
    ]
    identity_top1 = [
        bool(row["identity"]["native_top1_retained"])
        for row in rows
        if row.get("identity") is not None
    ]

    arm_summary: dict[str, Any] = {}
    native_acc = np.asarray(
        [int(bool(r["native"]["is_correct"])) for r in rows], dtype=np.float64
    )
    native_e2e = np.asarray(
        [float(r["native"]["end_to_end_latency_ms"]) for r in steady_rows],
        dtype=np.float64,
    )
    arm_summary["native"] = {
        "accuracy": bootstrap_mean_ci(native_acc, seed + 1),
        "end_to_end_latency_ms": bootstrap_mean_ci(native_e2e, seed + 2),
        "samples": n,
    }
    for i, arm_name in enumerate(arm_names):
        acc = np.asarray(
            [int(bool(r["arms"][arm_name]["is_correct"])) for r in rows],
            dtype=np.float64,
        )
        e2e = np.asarray(
            [float(r["arms"][arm_name]["end_to_end_latency_ms"]) for r in steady_rows],
            dtype=np.float64,
        )
        js = np.asarray(
            [float(r["arms"][arm_name]["js_divergence_nats"]) for r in rows],
            dtype=np.float64,
        )
        flip = np.asarray(
            [int(bool(r["arms"][arm_name]["answer_flip_vs_native"])) for r in rows],
            dtype=np.float64,
        )
        top1 = np.asarray(
            [int(bool(r["arms"][arm_name]["native_top1_retained"])) for r in rows],
            dtype=np.float64,
        )
        peak = np.asarray(
            [int(r["arms"][arm_name]["peak_gpu_memory_bytes"]) for r in steady_rows],
            dtype=np.float64,
        )
        arm_summary[arm_name] = {
            "budget": int(budgets[i]),
            "accuracy": bootstrap_mean_ci(acc, seed + 10 + i),
            "end_to_end_latency_ms": bootstrap_mean_ci(e2e, seed + 20 + i),
            "first_token_js_divergence_nats": bootstrap_mean_ci(js, seed + 40 + i),
            "answer_flip_rate": float(flip.mean()) if len(flip) else None,
            "native_top1_retention": float(top1.mean()) if len(top1) else None,
            "peak_gpu_memory_bytes_mean": float(peak.mean()) if len(peak) else None,
            "samples": n,
        }

    minimum = int(protocol["gate"]["minimum_completed_samples"])
    identity_required = int(
        protocol["implementation_validation"]["identity_check_samples"]
    )
    max_identity = max(identity_js, default=None)
    mean_identity = float(np.mean(identity_js)) if identity_js else None
    controller_ok = bool(controller_gate_passed)
    verified = n >= minimum and controller_ok

    return {
        "schema_version": "unimoe2/pm2-results-v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "experiment_id": "P-M2-UniMoE2",
        "backbone": protocol["backbone"],
        "evidence_status": "verified" if verified else "placeholder",
        "measurement_mode": protocol["measurement_mode"],
        "completed_samples": n,
        "failed_samples": len(errors),
        "requested_samples": int(protocol["sample_selection"]["target_samples"]),
        "steady_state_samples": len(steady_rows),
        "warmup_samples_excluded": warmup,
        "arms": arm_summary,
        "implementation_gates": {
            "controller_blocks_toggled": controller_blocks,
            "controller_blocks_expected": controller_expected,
            "controller_wiring_passed": controller_ok,
            "identity_arm": protocol["implementation_validation"]["identity_arm"],
            "identity_samples": len(identity_js),
            "identity_arm_js_divergence_from_native_max": max_identity,
            "identity_arm_js_divergence_from_native_mean": mean_identity,
            "identity_arm_top1_retention": (
                float(np.mean([1.0 if x else 0.0 for x in identity_top1]))
                if identity_top1
                else None
            ),
            "identity_note": protocol["implementation_validation"][
                "identity_semantics"
            ],
            "passed": controller_ok,
        },
        "gates": {"minimum_completed_samples": minimum, "passed": verified},
        "scope": protocol["scope"],
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Debug-only override; changes the frozen run.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> None:
    native = torch.tensor([2.0, 1.0, 0.5, 0.1])
    same = distribution_distances(native, native.clone())
    assert same["js_divergence_nats"] < 1e-9, same
    other = distribution_distances(native, torch.tensor([0.1, 0.5, 1.0, 2.0]))
    assert other["js_divergence_nats"] > 0.0
    ci = bootstrap_mean_ci(np.array([1.0, 0.0, 1.0, 1.0]), 7)
    assert ci["ci95_low"] <= ci["mean"] <= ci["ci95_high"]
    assert parse_answer("The answer is C.") == "C"
    print("P-M2-UniMoE2 distance/bootstrap self-test passed.")


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
    n_records = len(records)
    if args.limit is not None:
        n_records = min(n_records, int(args.limit))
    else:
        n_records = min(n_records, target)
    selected = list(range(n_records))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_results.jsonl"
    error_path = output_dir / "errors.jsonl"
    summary_path = output_dir / "summary.json"

    import random as _random

    _random.seed(int(protocol["decoding"]["seed"]))
    np.random.seed(int(protocol["decoding"]["seed"]))
    torch.manual_seed(int(protocol["decoding"]["seed"]))
    torch.cuda.manual_seed_all(int(protocol["decoding"]["seed"]))

    log("loading Uni-MoE-2 model")
    from uni_moe.model.processing_qwen2_vl import Qwen2VLProcessor
    from uni_moe.model.modeling_qwen_grin_moe import (
        GrinQwen2VLForConditionalGeneration,
    )
    from uni_moe.qwen_vl_utils import process_mm_info
    from uni_moe.model import deepspeed_moe_inference_utils  # noqa: F401

    processor = Qwen2VLProcessor.from_pretrained(str(model_path))
    model = GrinQwen2VLForConditionalGeneration.from_pretrained(
        str(model_path),
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ).cuda()
    model.eval()
    processor.data_args = model.config
    atomic_json(
        output_dir / "environment.json",
        environment_record(protocol, dataset_path, model_path),
    )

    total_layers = int(
        json.loads((model_path / "config.json").read_text())["num_hidden_layers"]
    )
    intervention = UniformDynamicBudgetIntervention(model, expected_blocks=total_layers)

    completed_rows = load_jsonl(raw_path)
    completed = {int(row["sample_index"]) for row in completed_rows}
    errors = load_jsonl(error_path)
    identity_done = sum(row.get("identity") is not None for row in completed_rows)
    identity_required = int(
        protocol["implementation_validation"]["identity_check_samples"]
    )

    try:
        for ordinal, sample_index in enumerate(selected, 1):
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
                    max_new_tokens=args.max_new_tokens,
                    identity=identity_done < identity_required,
                )
                if row.get("identity") is not None:
                    identity_done += 1
                append_jsonl(raw_path, row)
                completed_rows.append(row)
                completed.add(sample_index)
                if ordinal % 20 == 0 or ordinal <= 3:
                    log(
                        f"[{ordinal}/{len(selected)}] sample {sample_index} done; "
                        f"native_e2e={row['native']['end_to_end_latency_ms']:.1f}ms "
                        f"top2_e2e={row['arms']['uniform_dynamic_top2']['end_to_end_latency_ms']:.1f}ms "
                        f"top1_e2e={row['arms']['uniform_dynamic_top1']['end_to_end_latency_ms']:.1f}ms"
                    )
                atomic_json(
                    output_dir / "progress.json",
                    {
                        "updated_at": datetime.now(timezone.utc)
                        .astimezone()
                        .isoformat(),
                        "completed_samples": len(completed_rows),
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
        final_summary = summarize(
            completed_rows,
            errors,
            protocol,
            controller_blocks=intervention.installed,
            controller_expected=total_layers,
        )
        atomic_json(summary_path, final_summary)
        log(
            f"finished: status={final_summary['evidence_status']} "
            f"completed={final_summary['completed_samples']} "
            f"failed={final_summary['failed_samples']}"
        )
        return 0 if final_summary["evidence_status"] == "verified" else 3
    finally:
        intervention.close()


if __name__ == "__main__":
    raise SystemExit(main())

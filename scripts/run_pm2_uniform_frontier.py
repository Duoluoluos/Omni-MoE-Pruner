#!/usr/bin/env python3
"""Run P-M2: uniform expert-budget speed-quality frontier.

The protocol is frozen in
research_paper/iclr2027/evidence/pm2_uniform_frontier_protocol.json.

For every OmniVideoBench example this runner runs greedy generation under four
arms: native top-8, and a uniform budget of 8, 6, and 4 experts applied to
EVERY token at EVERY MoE layer during both prefill and decode.  It records
multiple-choice accuracy, first-generated-token JS divergence from native,
prefill and end-to-end latency, and peak GPU memory.  uniform_top8 is an
identity check on the masking path.  This traces the global (unstructured)
reduction frontier that motivates structured ARC-SCEB budgeting; it is a
single-checkpoint, single-H20 deterministic inference counterpart, not a
multi-seed training frontier.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO / "research_paper/iclr2027/evidence/pm2_uniform_frontier_protocol.json"
)
sys.path.insert(0, str(REPO / "qwen-omni-utils" / "src"))
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def log(*parts: Any) -> None:
    print("[P-M2]", *parts, file=sys.stderr, flush=True)


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
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": prompt},
            ],
        },
    ]


class UniformBudgetIntervention:
    """Uniformly lower every MoE block's router top-k so fewer experts run.

    This does NOT mask a full top-8 selection.  It mutates each
    ``SparseMoeBlock.top_k`` in place, so the block's native forward calls
    ``torch.topk(router_weights, top_k)`` with the smaller budget and its
    per-expert loop only executes the selected token--expert pairs.  Expert
    computation is genuinely skipped -- the same "lower the static top-k so the
    grouped compute processes fewer rows" trick used by the Megatron global
    top-k probe -- with no mask, no renormalization overhead, and no
    GPU->CPU sync.  The renormalization over the retained weights is the block's
    own (``routing_weights /= routing_weights.sum(...)``), so smaller budgets
    stay probability-normalized exactly as native top-8 does.
    """

    def __init__(self, model: Any, top_k: int) -> None:
        self.model = model
        self.top_k = int(top_k)
        self._blocks: list[Any] = []

        for name, module in model.thinker.named_modules():
            if "sparsemoeblock" not in module.__class__.__name__.lower():
                continue
            if not all(
                hasattr(module, attr)
                for attr in ("gate", "experts", "top_k", "num_experts")
            ):
                continue
            if int(module.top_k) != self.top_k:
                raise RuntimeError(
                    f"MoE block {name} has top_k={module.top_k}, expected {self.top_k}"
                )
            self._blocks.append(module)
        if not self._blocks:
            raise RuntimeError("No MoE blocks found for uniform intervention")
        self.installed = len(self._blocks)
        log(f"installed uniform top-k control on {self.installed} MoE layers")

    @contextlib.contextmanager
    def use_budget(self, budget: int | None):
        target = self.top_k if budget is None else int(budget)
        for module in self._blocks:
            module.top_k = target
        try:
            yield
        finally:
            for module in self._blocks:
                module.top_k = self.top_k

    def close(self) -> None:
        for module in self._blocks:
            module.top_k = self.top_k


@dataclass
class GenerationResult:
    first_logits: torch.Tensor
    first_top1: int
    output_text: str
    answer: str | None
    prefill_seconds: float
    e2e_seconds: float
    peak_memory_bytes: int


def generate_once(
    model: Any,
    processor: Any,
    inputs: Any,
    max_new_tokens: int,
) -> GenerationResult:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
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
    e2e = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated())
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
        prefill_seconds=0.0,
        e2e_seconds=e2e,
        peak_memory_bytes=peak,
    )


def prefill_once(model: Any, processor: Any, inputs: Any) -> float:
    """Measure prefill + first-token latency via the model's own generate path.

    Calling model.thinker directly omits internal multimodal rope plumbing
    (e.g. video_second_per_grid), so instead we time a single-token generate,
    which uses the exact same prefill code path the frontier arms use.
    """
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        model.generate(
            **inputs,
            use_audio_in_video=True,
            return_audio=False,
            thinker_max_new_tokens=1,
            thinker_do_sample=False,
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            thinker_eos_token_id=processor.tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    return time.perf_counter() - started


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
) -> dict[str, float]:
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
    intervention: UniformBudgetIntervention,
    protocol: dict[str, Any],
    max_new_tokens: int,
    identity: bool,
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
    seq_len = int(inputs["input_ids"].shape[1])
    correct = str(record["correct_option"]).upper()

    native = generate_once(model, processor, inputs, max_new_tokens)

    identity_result: dict[str, Any] | None = None
    if identity:
        with intervention.use_budget(8):
            identity_out = generate_once(model, processor, inputs, max_new_tokens)
        identity_result = {
            **distribution_distances(native.first_logits, identity_out.first_logits),
            "native_top1_retained": identity_out.first_top1 == native.first_top1,
            "output_text_equal": identity_out.output_text == native.output_text,
        }

    arms: dict[str, Any] = {}
    budgets = list(protocol["policy"]["uniform_budgets"])
    for budget in budgets:
        arm_name = f"uniform_top{budget}"
        with intervention.use_budget(budget):
            prefill_s = prefill_once(model, processor, inputs)
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
            "prefill_latency_ms": prefill_s * 1000.0,
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
    rows: list[dict[str, Any]], errors: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    n = len(rows)
    seed = int(protocol["decoding"]["seed"])
    warmup = int(protocol["timing"]["warmup_samples"])
    budgets = list(protocol["policy"]["uniform_budgets"])
    arm_names = [f"uniform_top{b}" for b in budgets]

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
    arm_summary["native_top8"] = {
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
        prefill = np.asarray(
            [float(r["arms"][arm_name]["prefill_latency_ms"]) for r in steady_rows],
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
            "prefill_latency_ms": bootstrap_mean_ci(prefill, seed + 30 + i),
            "first_token_js_divergence_nats": bootstrap_mean_ci(js, seed + 40 + i),
            "answer_flip_rate": float(flip.mean()) if len(flip) else None,
            "native_top1_retention": float(top1.mean()) if len(top1) else None,
            "peak_gpu_memory_bytes_mean": float(peak.mean()) if len(peak) else None,
            "samples": n,
        }

    minimum = int(protocol["gate"]["minimum_completed_samples"])
    identity_limit = float(
        protocol["implementation_validation"]["maximum_identity_js_divergence_nats"]
    )
    identity_required = int(
        protocol["implementation_validation"]["identity_check_samples"]
    )
    max_identity = max(identity_js, default=float("inf"))
    identity_pass = (
        len(identity_js) >= identity_required
        and max_identity <= identity_limit
        and all(identity_top1)
    )
    verified = n >= minimum and identity_pass

    return {
        "schema_version": "arc-sceb/pm2-results-v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "experiment_id": "P-M2",
        "evidence_status": "verified" if verified else "placeholder",
        "measurement_mode": protocol["measurement_mode"],
        "completed_samples": n,
        "failed_samples": len(errors),
        "requested_samples": int(protocol["sample_selection"]["target_samples"]),
        "steady_state_samples": len(steady_rows),
        "warmup_samples_excluded": warmup,
        "arms": arm_summary,
        "implementation_gates": {
            "identity_samples": len(identity_js),
            "maximum_identity_js_divergence_nats": max_identity
            if identity_js
            else None,
            "identity_limit": identity_limit,
            "identity_top1_all_retained": all(identity_top1)
            if identity_top1
            else False,
            "passed": identity_pass,
        },
        "gates": {
            "minimum_completed_samples": minimum,
            "passed": verified,
        },
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
    print("P-M2 distance/bootstrap self-test passed.")


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
    n_records = len(records)
    if args.limit is not None:
        n_records = min(n_records, int(args.limit))
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
    atomic_json(
        output_dir / "environment.json",
        environment_record(protocol, dataset_path, model_path),
    )

    intervention = UniformBudgetIntervention(
        model, top_k=int(protocol["policy"]["original_top_k"])
    )
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
                        f"top6_e2e={row['arms']['uniform_top6']['end_to_end_latency_ms']:.1f}ms "
                        f"top4_e2e={row['arms']['uniform_top4']['end_to_end_latency_ms']:.1f}ms"
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
        final_summary = summarize(completed_rows, errors, protocol)
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

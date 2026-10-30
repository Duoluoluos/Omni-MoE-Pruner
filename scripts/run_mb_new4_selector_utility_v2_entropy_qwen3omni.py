#!/usr/bin/env python3
"""MB-NEW-4 selector-utility test v2 (entropy signal) on Qwen3-Omni.

Protocol:
research_paper/iclr2027/evidence/mb_new4_selector_utility_v2_entropy_qwen3omni_protocol.json

Cross-backbone confirmation of the entropy-quantile Axis C selector. On
Uni-MoE-2 the entropy signal was directionally correct but underpowered (small
4-dynamic-expert selection space). Qwen3-Omni has 128 experts and fixed top-8, a
far larger space where a useful router signal should be detectable if it exists.

Arms per sample:
  native                  - unmodified top-8 routing everywhere (reference)
  sceb_margin             - entropy-quantile budget in {4,6,8} on eligible video
                            tokens in layers 12..31 (Axis C v2)
  budget_matched_uniform  - budget-matched position split (signal-free)
  budget_matched_shuffled - permuted sceb budgets (signal-destroyed)

Leakage-free: reads only router logits and token position. No label of any kind.
Single-checkpoint, single-H20 deterministic inference; NOT a wall-clock claim.
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
    REPO
    / "research_paper/iclr2027/evidence/mb_new4_selector_utility_v2_entropy_qwen3omni_protocol.json"
)
sys.path.insert(0, str(REPO / "qwen-omni-utils" / "src"))

VIDEO_TOKEN_ID = 151656
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def log(*parts: Any) -> None:
    print("[MB-NEW-4-Qwen3Omni-v2]", *parts, file=sys.stderr, flush=True)


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


class SCEBController:
    """Per-token entropy-quantile SCEB budget injection for Qwen3-Omni MoE blocks.

    Wraps every thinker SparseMoeBlock.forward. For eligible video tokens in the
    middle band, computes normalized entropy over the top-k routing weights and
    maps it to a budget in the candidate set via calibrated quantiles, then runs
    only that many experts (masking the lowest routing weights, renormalizing),
    exactly mirroring the verified P-M4 TargetedBudgetIntervention math.

    Arms: native / sceb_margin (entropy) / budget_matched_uniform /
    budget_matched_shuffled. Non-native arms act only on eligible tokens/layers.
    """

    def __init__(
        self,
        model: Any,
        candidate_budgets: list[int],
        native_top_k: int,
        layer_band: tuple[int, int],
    ) -> None:
        self.candidate_budgets = sorted(int(b) for b in candidate_budgets)
        self.native_top_k = int(native_top_k)
        self.layer_lo, self.layer_hi = layer_band  # half-open [lo, hi)
        self._wrapped: list[tuple[Any, Callable[..., Any], int]] = []

        self._arm = "native"
        self._eligible_seq_mask: torch.Tensor | None = None
        self._shuffle_perm_by_layer: dict[int, torch.Tensor] | None = None
        self._record_entropy = False
        self._entropy_records: list[dict[str, Any]] = []
        self.q_low = 0.0
        self.q_high = 1.0
        self._budget_hist = {int(b): 0 for b in self.candidate_budgets}
        self._executed_budget_sum = 0.0
        self._executed_budget_count = 0
        self._sceb_budgets_by_layer: dict[int, torch.Tensor] = {}

        n = 0
        for name, module in model.thinker.named_modules():
            if "sparsemoeblock" not in module.__class__.__name__.lower():
                continue
            if not all(
                hasattr(module, attr)
                for attr in ("gate", "experts", "top_k", "num_experts")
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
            n += 1
        if not self._wrapped:
            raise RuntimeError("no SparseMoeBlock modules found")
        self.installed = n
        log(
            f"installed SCEB controller on {n} MoE layers; "
            f"band=[{self.layer_lo},{self.layer_hi}) candidates={self.candidate_budgets}"
        )

    @staticmethod
    def _layer_index(name: str) -> int | None:
        parts = name.split(".")
        for index, part in enumerate(parts[:-1]):
            if part == "layers" and parts[index + 1].isdigit():
                return int(parts[index + 1])
        return None

    def _install(self, module: Any, layer: int) -> None:
        original = module.forward
        controller = self

        def wrapped(hidden_states: torch.Tensor):
            eligible = (
                controller._arm != "native"
                and controller._in_band(layer)
                and controller._eligible_seq_mask is not None
            )
            if not eligible:
                return original(hidden_states)
            return controller._budgeted_forward(module, original, hidden_states, layer)

        module.forward = wrapped
        self._wrapped.append((module, original, layer))

    def _in_band(self, layer: int) -> bool:
        return self.layer_lo <= layer < self.layer_hi

    def set_quantiles(self, q_low: float, q_high: float) -> None:
        self.q_low = float(q_low)
        self.q_high = float(q_high)

    @staticmethod
    def _normalized_entropy(topk_weights: torch.Tensor) -> torch.Tensor:
        """Normalized Shannon entropy of the (renormalized) top-k routing weights.

        topk_weights: (n_tokens, top_k), softmax weights over the top_k experts.
        """
        k = topk_weights.shape[1]
        p = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        p = p.clamp_min(1e-12)
        ent = -(p * p.log()).sum(dim=-1)
        return ent / float(np.log(k))

    def _entropy_budget(self, topk_weights: torch.Tensor) -> torch.Tensor:
        cands = sorted(self.candidate_budgets)
        lo_b, mid_b, hi_b = cands[0], cands[len(cands) // 2], cands[-1]
        h = self._normalized_entropy(topk_weights)
        chosen = torch.full_like(h, float(mid_b))
        chosen = torch.where(h < self.q_low, torch.full_like(h, float(lo_b)), chosen)
        chosen = torch.where(h >= self.q_high, torch.full_like(h, float(hi_b)), chosen)
        return chosen.to(torch.int64).clamp_(min=lo_b, max=hi_b)

    def _budgeted_forward(self, module, original, hidden_states, layer):
        batch, seqlen, hidden_size = hidden_states.shape
        flat = hidden_states.view(-1, hidden_size)
        token_count = flat.shape[0]

        # Decode step or shape mismatch -> native passthrough.
        elig = self._eligible_seq_mask
        if elig is None or seqlen != elig.shape[0]:
            return original(hidden_states)

        router_logits = module.gate(flat)
        routing_weights = torch.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, module.top_k, dim=-1
        )

        elig_flat = elig.to(routing_weights.device).view(-1)
        budgets = torch.full(
            (token_count,),
            module.top_k,
            dtype=torch.long,
            device=routing_weights.device,
        )

        if self._arm == "sceb_margin":
            sceb_b = self._entropy_budget(routing_weights)
            budgets = torch.where(elig_flat, sceb_b, budgets)
            elig_b = sceb_b[elig_flat]
            self._sceb_budgets_by_layer[layer] = elig_b.detach().cpu()
            for b in self._budget_hist:
                self._budget_hist[b] += int((elig_b == b).sum().item())
        elif self._arm == "budget_matched_uniform":
            base = self._sceb_budgets_by_layer.get(layer)
            idx = torch.nonzero(elig_flat, as_tuple=True)[0]
            if base is not None and idx.shape[0] == base.shape[0] and idx.shape[0] > 0:
                target = float(base.float().mean().item())
                lo = max(
                    [c for c in self.candidate_budgets if c <= target],
                    default=min(self.candidate_budgets),
                )
                hi = min(
                    [c for c in self.candidate_budgets if c >= target],
                    default=max(self.candidate_budgets),
                )
                n_elig = idx.shape[0]
                if hi == lo:
                    assign = torch.full((n_elig,), lo, dtype=torch.long)
                else:
                    frac_hi = (target - lo) / (hi - lo)
                    n_hi = int(round(frac_hi * n_elig))
                    assign = torch.full((n_elig,), lo, dtype=torch.long)
                    assign[:n_hi] = hi
                budgets[idx] = assign.to(budgets.device)
        elif self._arm == "budget_matched_shuffled":
            base = self._sceb_budgets_by_layer.get(layer)
            perm = (
                self._shuffle_perm_by_layer.get(layer)
                if self._shuffle_perm_by_layer
                else None
            )
            if base is not None and perm is not None:
                shuffled = base[perm].to(budgets.device).long()
                idx = torch.nonzero(elig_flat, as_tuple=True)[0]
                if idx.shape[0] == shuffled.shape[0]:
                    budgets[idx] = shuffled

        if self._record_entropy and elig_flat.any():
            h = self._normalized_entropy(routing_weights[elig_flat])
            self._entropy_records.append(
                {
                    "layer": layer,
                    "n": int(elig_flat.sum().item()),
                    "entropy": h.detach().cpu().tolist(),
                }
            )

        if self._arm in (
            "sceb_margin",
            "budget_matched_uniform",
            "budget_matched_shuffled",
        ):
            if elig_flat.any():
                self._executed_budget_sum += float(budgets[elig_flat].sum().item())
                self._executed_budget_count += int(elig_flat.sum().item())

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
            expert_output = (
                expert_output * routing_weights[token_index, rank_index, None]
            )
            final.index_add_(0, token_index, expert_output.to(flat.dtype))
        return final.reshape(batch, seqlen, hidden_size), router_logits

    def prepare_sample(self, input_ids_1d: torch.Tensor) -> None:
        self._eligible_seq_mask = (input_ids_1d == VIDEO_TOKEN_ID).bool().cpu()
        self._sceb_budgets_by_layer = {}

    def reset_counters(self) -> None:
        self._executed_budget_sum = 0.0
        self._executed_budget_count = 0

    def executed_mean_budget(self) -> float | None:
        if self._executed_budget_count == 0:
            return None
        return self._executed_budget_sum / self._executed_budget_count

    @contextlib.contextmanager
    def arm(self, name: str, shuffle_seed: int | None = None):
        self._arm = name
        self._shuffle_perm_by_layer = None
        self.reset_counters()
        if name == "budget_matched_shuffled" and shuffle_seed is not None:
            gen = torch.Generator().manual_seed(int(shuffle_seed))
            self._shuffle_perm_by_layer = {
                layer: torch.randperm(vals.shape[0], generator=gen)
                for layer, vals in self._sceb_budgets_by_layer.items()
            }
        try:
            yield
        finally:
            self._arm = "native"
            self._shuffle_perm_by_layer = None

    def close(self) -> None:
        for module, original, _layer in self._wrapped:
            module.forward = original
        self._wrapped.clear()


@dataclass
class GenerationResult:
    first_logits: torch.Tensor
    first_top1: int
    output_text: str
    answer: str | None
    e2e_seconds: float


def generate_once(model, processor, inputs, max_new_tokens) -> GenerationResult:
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
    e2e = time.perf_counter() - start
    thinker = raw[0] if isinstance(raw, tuple) else raw
    first_logits = thinker.scores[0][0].detach().float().cpu()
    input_length = int(inputs["input_ids"].shape[1])
    generated = thinker.sequences[0, input_length:].detach().cpu()
    output_text = processor.tokenizer.decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    return GenerationResult(
        first_logits=first_logits,
        first_top1=int(first_logits.argmax().item()),
        output_text=output_text,
        answer=parse_answer(output_text),
        e2e_seconds=e2e,
    )


def distribution_js(native_logits, arm_logits) -> float:
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
    return max(0.0, float(js.item()))


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL, timeout=20
        ).strip()
    except Exception:
        return "unavailable"


def environment_record(
    protocol, dataset_path, model_path, calibration
) -> dict[str, Any]:
    config_path = model_path / "config.json"
    return {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "requested_worker": os.environ.get("MB4_WORKER_ID", "unknown"),
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
        "calibrated_entropy_quantiles": calibration,
        "protocol": protocol,
    }


def bootstrap_mean_ci(values, seed, draws: int = 10000) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 1000):
        stop = min(draws, start + 1000)
        idx = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def paired_diff_ci(a, b, seed, draws: int = 10000) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    if len(d) == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 1000):
        stop = min(draws, start + 1000)
        idx = rng.integers(0, len(d), size=(stop - start, len(d)))
        means[start:stop] = d[idx].mean(axis=1)
    return {
        "mean": float(d.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def build_inputs(processor, process_mm_info, conversation):
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
    return inputs


def calibrate_quantiles(
    *, records, model, processor, process_mm_info, controller, protocol
):
    n = int(protocol["entropy_calibration"]["calibrate_samples"])
    q_low_pct = float(protocol["entropy_calibration"]["q_low_percentile"])
    q_high_pct = float(protocol["entropy_calibration"]["q_high_percentile"])
    pooled: list[float] = []
    controller._record_entropy = True
    controller._entropy_records = []
    for i in range(min(n, len(records))):
        video_path = (
            REPO / "eval_data/OmniVideoBench" / str(records[i]["video"])
        ).resolve()
        if not video_path.exists():
            continue
        conv = build_conversation(video_path, records[i])
        inputs = build_inputs(processor, process_mm_info, conv)
        input_ids_1d = inputs["input_ids"].reshape(-1).clone()
        controller.prepare_sample(input_ids_1d)
        inputs = inputs.to(model.device).to(model.dtype)
        controller.set_quantiles(0.0, 1.0)
        with controller.arm("sceb_margin"):
            generate_once(model, processor, inputs, 1)
        for rec in controller._entropy_records:
            if "entropy" in rec:
                pooled.extend(rec["entropy"])
        controller._entropy_records = []
    controller._record_entropy = False
    if not pooled:
        raise RuntimeError("entropy calibration collected no values")
    arr = np.asarray(pooled, dtype=np.float64)
    q_low = float(np.percentile(arr, q_low_pct))
    q_high = float(np.percentile(arr, q_high_pct))
    log(
        f"calibrated entropy quantiles q_low={q_low:.6g} q_high={q_high:.6g} "
        f"from {len(pooled)} pooled entropies (min/med/max={arr.min():.4g}/{np.median(arr):.4g}/{arr.max():.4g})"
    )
    return q_low, q_high


def run_sample(
    *,
    sample_index,
    record,
    model,
    processor,
    process_mm_info,
    controller,
    protocol,
    max_new_tokens,
    passthrough_check,
) -> dict[str, Any]:
    video_path = (REPO / "eval_data/OmniVideoBench" / str(record["video"])).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    conv = build_conversation(video_path, record)
    inputs = build_inputs(processor, process_mm_info, conv)
    input_ids_1d = inputs["input_ids"].reshape(-1).clone()
    controller.prepare_sample(input_ids_1d)
    seq_len = int(inputs["input_ids"].shape[1])
    n_video = int((input_ids_1d == VIDEO_TOKEN_ID).sum().item())
    inputs = inputs.to(model.device).to(model.dtype)
    correct = str(record["correct_option"]).upper()

    native = generate_once(model, processor, inputs, max_new_tokens)

    passthrough = None
    if passthrough_check:
        with controller.arm("native"):
            pt = generate_once(model, processor, inputs, max_new_tokens)
        passthrough = {
            "js_divergence_nats": distribution_js(native.first_logits, pt.first_logits)
        }

    with controller.arm("sceb_margin"):
        sceb = generate_once(model, processor, inputs, max_new_tokens)
        sceb_mean_budget = controller.executed_mean_budget()

    with controller.arm("budget_matched_uniform"):
        umatch = generate_once(model, processor, inputs, max_new_tokens)
        uniform_mean_budget = controller.executed_mean_budget()

    with controller.arm(
        "budget_matched_shuffled",
        shuffle_seed=int(protocol["decoding"]["seed"]) + sample_index,
    ):
        shuf = generate_once(model, processor, inputs, max_new_tokens)
        shuffled_mean_budget = controller.executed_mean_budget()

    def arm_row(res, mean_budget):
        return {
            "js_divergence_nats": distribution_js(
                native.first_logits, res.first_logits
            ),
            "answer": res.answer,
            "output_text": res.output_text,
            "answer_flip_vs_native": res.answer != native.answer,
            "is_correct": res.answer == correct,
            "end_to_end_latency_ms": res.e2e_seconds * 1000.0,
            "mean_executed_budget": mean_budget,
        }

    return {
        "sample_index": sample_index,
        "video": record["video"],
        "correct_option": correct,
        "input_sequence_length": seq_len,
        "n_video_tokens": n_video,
        "passthrough_js_vs_native": passthrough,
        "native": {
            "answer": native.answer,
            "output_text": native.output_text,
            "is_correct": native.answer == correct,
            "end_to_end_latency_ms": native.e2e_seconds * 1000.0,
        },
        "arms": {
            "sceb_margin": arm_row(sceb, sceb_mean_budget),
            "budget_matched_uniform": arm_row(umatch, uniform_mean_budget),
            "budget_matched_shuffled": arm_row(shuf, shuffled_mean_budget),
        },
    }


def summarize(
    rows,
    errors,
    protocol,
    controller_blocks,
    controller_expected,
    calibration,
    passthrough_max,
    budget_hist,
) -> dict[str, Any]:
    n = len(rows)
    seed = int(protocol["decoding"]["seed"])
    tol = float(protocol["budget_match_tolerance"])
    arm_names = ["sceb_margin", "budget_matched_uniform", "budget_matched_shuffled"]

    native_acc = np.asarray(
        [int(bool(r["native"]["is_correct"])) for r in rows], dtype=np.float64
    )
    arm_summary: dict[str, Any] = {
        "native": {"accuracy": bootstrap_mean_ci(native_acc, seed + 1), "samples": n}
    }
    for i, arm in enumerate(arm_names):
        acc = np.asarray(
            [int(bool(r["arms"][arm]["is_correct"])) for r in rows], dtype=np.float64
        )
        flip = np.asarray(
            [int(bool(r["arms"][arm]["answer_flip_vs_native"])) for r in rows],
            dtype=np.float64,
        )
        js = np.asarray(
            [float(r["arms"][arm]["js_divergence_nats"]) for r in rows],
            dtype=np.float64,
        )
        budgets = np.asarray(
            [float(r["arms"][arm]["mean_executed_budget"] or np.nan) for r in rows],
            dtype=np.float64,
        )
        arm_summary[arm] = {
            "accuracy": bootstrap_mean_ci(acc, seed + 10 + i),
            "answer_flip_rate": float(flip.mean()),
            "first_token_js_divergence_nats": bootstrap_mean_ci(js, seed + 20 + i),
            "mean_executed_budget": float(np.nanmean(budgets)),
            "samples": n,
        }

    sceb_flip = np.asarray(
        [int(bool(r["arms"]["sceb_margin"]["answer_flip_vs_native"])) for r in rows],
        dtype=np.float64,
    )
    sceb_acc = np.asarray(
        [int(bool(r["arms"]["sceb_margin"]["is_correct"])) for r in rows],
        dtype=np.float64,
    )
    contrasts = {}
    for ctrl in ("budget_matched_uniform", "budget_matched_shuffled"):
        ctrl_flip = np.asarray(
            [int(bool(r["arms"][ctrl]["answer_flip_vs_native"])) for r in rows],
            dtype=np.float64,
        )
        ctrl_acc = np.asarray(
            [int(bool(r["arms"][ctrl]["is_correct"])) for r in rows], dtype=np.float64
        )
        contrasts[f"sceb_minus_{ctrl}"] = {
            "flip_rate_diff": paired_diff_ci(sceb_flip, ctrl_flip, seed + 100),
            "accuracy_diff": paired_diff_ci(sceb_acc, ctrl_acc, seed + 200),
            "interpretation": "flip_rate_diff<0 with CI excluding 0 => entropy signal reduces flips at matched budget; accuracy_diff>=~0 => no accuracy cost.",
        }

    match_ok = 0
    for r in rows:
        sb = r["arms"]["sceb_margin"]["mean_executed_budget"]
        ub = r["arms"]["budget_matched_uniform"]["mean_executed_budget"]
        hb = r["arms"]["budget_matched_shuffled"]["mean_executed_budget"]
        if sb is None:
            continue
        rel = lambda x: x is not None and abs(x - sb) <= tol * max(1.0, sb)
        if rel(ub) and rel(hb):
            match_ok += 1

    minimum = int(protocol["gate"]["minimum_completed_samples"])
    controller_ok = controller_blocks == controller_expected
    passthrough_ok = passthrough_max is not None and passthrough_max < 1e-6

    total_assign = sum(budget_hist.values())
    min_frac = float(protocol["dispersion_gate"]["min_fraction_per_bucket"])
    budget_fractions = {
        str(b): (budget_hist[b] / total_assign if total_assign else 0.0)
        for b in budget_hist
    }
    dispersion_ok = total_assign > 0 and all(
        f >= min_frac for f in budget_fractions.values()
    )

    verified = (
        n >= minimum
        and controller_ok
        and passthrough_ok
        and match_ok >= minimum
        and dispersion_ok
    )

    return {
        "schema_version": "sceb/mb-new-4-results-v2",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "experiment_id": "MB-NEW-4-Qwen3Omni-v2",
        "backbone": protocol["backbone"],
        "evidence_status": "verified" if verified else "placeholder",
        "calibration": calibration,
        "completed_samples": n,
        "failed_samples": len(errors),
        "requested_samples": int(protocol["sample_selection"]["target_samples"]),
        "arms": arm_summary,
        "selectivity_contrasts": contrasts,
        "sceb_budget_histogram": {str(b): budget_hist[b] for b in budget_hist},
        "sceb_budget_fractions": budget_fractions,
        "implementation_gates": {
            "controller_blocks": controller_blocks,
            "controller_blocks_expected": controller_expected,
            "controller_wiring_passed": controller_ok,
            "native_passthrough_js_max": passthrough_max,
            "native_passthrough_passed": passthrough_ok,
            "budget_match_ok_samples": match_ok,
            "budget_match_tolerance": tol,
            "dispersion_ok": dispersion_ok,
            "dispersion_min_fraction_required": min_frac,
            "passed": controller_ok
            and passthrough_ok
            and match_ok >= minimum
            and dispersion_ok,
        },
        "gates": {"minimum_completed_samples": minimum, "passed": verified},
        "scope": protocol["scope"],
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Debug-only override.")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> None:
    native = torch.tensor([2.0, 1.0, 0.5, 0.1])
    assert distribution_js(native, native.clone()) < 1e-9
    ci = bootstrap_mean_ci(np.array([1.0, 0.0, 1.0, 1.0]), 7)
    assert ci["ci95_low"] <= ci["mean"] <= ci["ci95_high"]
    d = paired_diff_ci(np.array([0.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0]), 3)
    assert d["mean"] < 0
    # entropy sanity: uniform top-k => H=1; peaked => H~0
    uni = torch.full((1, 8), 1.0 / 8)
    peak = torch.tensor([[0.99, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.004]])
    hu = SCEBController._normalized_entropy(uni).item()
    hp = SCEBController._normalized_entropy(peak).item()
    assert hu > 0.99 and hp < 0.3, (hu, hp)
    assert parse_answer("The answer is C.") == "C"
    print("MB-NEW-4-Qwen3Omni-v2 self-test passed.")


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
    n_records = (
        min(n_records, int(args.limit))
        if args.limit is not None
        else min(n_records, target)
    )
    selected = list(range(n_records))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_results.jsonl"
    error_path = output_dir / "errors.jsonl"
    summary_path = output_dir / "summary.json"

    import random as _random

    seed = int(protocol["decoding"]["seed"])
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    log("loading Qwen3-Omni model")
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info

    processor = Qwen3OmniMoeProcessor.from_pretrained(str(model_path))
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()

    total_layers = int(
        json.loads((model_path / "config.json").read_text())["thinker_config"][
            "text_config"
        ]["num_hidden_layers"]
    )
    band_lo = int(np.floor(0.25 * total_layers))
    band_hi = int(np.floor(0.67 * total_layers))
    candidates = list(protocol["moe_structure"]["candidate_budgets"])
    native_top_k = int(protocol["moe_structure"]["native_top_k"])
    controller = SCEBController(
        model,
        candidate_budgets=candidates,
        native_top_k=native_top_k,
        layer_band=(band_lo, band_hi),
    )

    q_low, q_high = calibrate_quantiles(
        records=records,
        model=model,
        processor=processor,
        process_mm_info=process_mm_info,
        controller=controller,
        protocol=protocol,
    )
    controller.set_quantiles(q_low, q_high)
    controller._budget_hist = {int(b): 0 for b in controller.candidate_budgets}
    atomic_json(
        output_dir / "environment.json",
        environment_record(
            protocol, dataset_path, model_path, {"q_low": q_low, "q_high": q_high}
        ),
    )

    completed_rows = load_jsonl(raw_path)
    completed = {int(r["sample_index"]) for r in completed_rows}
    errors = load_jsonl(error_path)
    passthrough_done = sum(
        1 for r in completed_rows if r.get("passthrough_js_vs_native") is not None
    )
    passthrough_required = 8
    passthrough_vals = [
        float(r["passthrough_js_vs_native"]["js_divergence_nats"])
        for r in completed_rows
        if r.get("passthrough_js_vs_native") is not None
    ]

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
                    controller=controller,
                    protocol=protocol,
                    max_new_tokens=args.max_new_tokens,
                    passthrough_check=passthrough_done < passthrough_required,
                )
                if row.get("passthrough_js_vs_native") is not None:
                    passthrough_done += 1
                    passthrough_vals.append(
                        float(row["passthrough_js_vs_native"]["js_divergence_nats"])
                    )
                append_jsonl(raw_path, row)
                completed_rows.append(row)
                completed.add(sample_index)
                if ordinal % 20 == 0 or ordinal <= 3:
                    s = row["arms"]["sceb_margin"]
                    u = row["arms"]["budget_matched_uniform"]
                    log(
                        f"[{ordinal}/{len(selected)}] sample {sample_index}: "
                        f"sceb_budget={s['mean_executed_budget']} uni_budget={u['mean_executed_budget']} "
                        f"sceb_flip={s['answer_flip_vs_native']} uni_flip={u['answer_flip_vs_native']}"
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
                err = {
                    "sample_index": sample_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(error_path, err)
                errors.append(err)
                log(f"sample {sample_index} OOM; recorded and continuing")
            except Exception as exc:
                err = {
                    "sample_index": sample_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(error_path, err)
                errors.append(err)
                log(f"sample {sample_index} failed: {exc!r}")
        passthrough_max = max(passthrough_vals) if passthrough_vals else None
        final = summarize(
            completed_rows,
            errors,
            protocol,
            controller_blocks=controller.installed,
            controller_expected=total_layers,
            calibration={"q_low": q_low, "q_high": q_high},
            passthrough_max=passthrough_max,
            budget_hist=controller._budget_hist,
        )
        atomic_json(summary_path, final)
        log(
            f"finished: status={final['evidence_status']} completed={final['completed_samples']} failed={final['failed_samples']}"
        )
        return 0 if final["evidence_status"] == "verified" else 3
    finally:
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""MB-NEW-2 speed-quality frontier on Uni-MoE-2.0-Omni.

Protocol:
research_paper/iclr2027/evidence/mb_new2_frontier_unimoe2_protocol.json

Question: does SCEB's label-free per-token boundary-margin ADAPTIVE dynamic-expert
budget produce a speed-quality operating point that strictly Pareto-dominates the
P-M2 equal-layer uniform FIXED-k points (top-1..4) under a matched wall-clock
protocol?

This is the FULL-SCOPE frontier variant of SCEB: the boundary-margin selector is
applied to EVERY token at EVERY MoE layer, exactly matching P-M2's uniform scope,
so the adaptive-vs-fixed-k comparison is scope-fair. It does NOT modify the frozen
video-only + middle-depth SCEB method definition.

Arms per sample:
  native          - unmodified GRIN top-p routing everywhere (reference)
  sceb_margin     - Axis C boundary-margin selection over {1,2,3,4}, applied to
                    every token at every MoE layer (adaptive operating point)
  uniform_top1    - fixed dynamic budget 1 on every token/every layer (P-M2)
  uniform_top2    - fixed dynamic budget 2 on every token/every layer (P-M2)
  uniform_top3    - fixed dynamic budget 3 on every token/every layer (P-M2)
  uniform_top4    - fixed dynamic budget 4 on every token/every layer (P-M2)

Leakage-free: the selector reads only the router's dynamic-expert probabilities
and token position. No task_type / audio_type / answer / label ever reaches the
runtime path.

Single-checkpoint, single-H20 deterministic inference. This is a matched
wall-clock frontier comparison; it is NOT a modality-utility claim and NOT a
kernel-level speedup claim.
"""

from __future__ import annotations

import argparse
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
    REPO / "research_paper/iclr2027/evidence/mb_new2_frontier_unimoe2_protocol.json"
)
sys.path.insert(0, "/home/tiger/Uni-MoE/Uni-MoE-2")
sys.path.insert(0, str(REPO / "scripts"))
from mb_new4_router_signals import boundary_budget, boundary_margins  # noqa: E402

VIDEO_TOKEN_ID = 151656

FIXED_ARMS = {
    "uniform_top1": 1,
    "uniform_top2": 2,
    "uniform_top3": 3,
    "uniform_top4": 4,
}


def log(*parts: Any) -> None:
    print("[MB-NEW-2-FRONTIER-UniMoE2]", *parts, file=sys.stderr, flush=True)


def _ensure_matplotlib_stub() -> None:
    """Uni-MoE's modeling file imports matplotlib.pyplot at module load but the
    inference path never plots. Inject a minimal stub if unavailable so the import
    succeeds without perturbing any numerics."""
    import types
    import importlib.machinery

    try:
        import matplotlib.pyplot  # noqa: F401

        return
    except Exception:
        pass
    mpl = types.ModuleType("matplotlib")
    mpl.__spec__ = importlib.machinery.ModuleSpec("matplotlib", loader=None)
    mpl.__path__ = []
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.__spec__ = importlib.machinery.ModuleSpec("matplotlib.pyplot", loader=None)

    def _noop(*_a: Any, **_k: Any) -> None:
        return None

    for name in (
        "figure",
        "plot",
        "savefig",
        "close",
        "clf",
        "title",
        "xlabel",
        "ylabel",
        "legend",
        "hist",
        "scatter",
        "imshow",
        "colorbar",
        "show",
    ):
        setattr(pyplot, name, _noop)
    mpl.pyplot = pyplot
    sys.modules.setdefault("matplotlib", mpl)
    sys.modules.setdefault("matplotlib.pyplot", pyplot)
    log("injected matplotlib.pyplot stub (plotting disabled, numerics unaffected)")


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


class FrontierController:
    """Full-scope per-token budget injection for GRIN MoE blocks.

    Wraps every GRINMoESparseMoeBlock.forward. For each arm it derives a per-token
    dynamic budget (adaptive boundary-margin for sceb_margin, or a constant for the
    fixed-k arms), applied to EVERY token at EVERY layer (prefill and decode). It
    then forces that budget through the native expert math by monkeypatching
    dynamic_expert_selection for the single call, keeping downstream math identical
    to native.

    Arms:
      native        -> passthrough (native top-p)
      sceb_margin   -> boundary-margin selection over candidate_budgets, all tokens
      uniform_topK  -> constant budget K on all tokens
    """

    def __init__(
        self,
        model: Any,
        expected_blocks: int,
        candidate_budgets: list[int],
        native_ceiling: int,
        layer_band: tuple[int, int] | None = None,
    ) -> None:
        self.model = model
        self.candidate_budgets = sorted(int(b) for b in candidate_budgets)
        self.native_ceiling = int(native_ceiling)
        # Half-open [lo, hi) MoE layer band the intervention acts on. Defaults to
        # every layer (full scope). When narrowed, BOTH the sceb and fixed-k arms
        # pass through native outside the band, keeping the comparison scope-fair.
        if layer_band is None:
            self.layer_lo, self.layer_hi = 0, expected_blocks
        else:
            self.layer_lo, self.layer_hi = int(layer_band[0]), int(layer_band[1])
        self._blocks: list[Any] = []
        self._orig_forward: list[Callable[..., Any]] = []
        self._orig_top_p: list[float] = []
        self._orig_top_k: list[int] = []
        self._layer_of_block: list[int] = []

        moe_blocks = []
        for name, module in model.named_modules():
            if module.__class__.__name__ == "GRINMoESparseMoeBlock":
                moe_blocks.append((name, module))
        if len(moe_blocks) != expected_blocks:
            raise RuntimeError(
                f"Found {len(moe_blocks)} GRIN MoE blocks, expected {expected_blocks}"
            )
        for name, module in moe_blocks:
            layer_idx = self._layer_idx(name)
            if layer_idx is None:
                raise RuntimeError(f"Could not parse layer index from {name!r}")
            self._blocks.append(module)
            self._layer_of_block.append(layer_idx)
            self._orig_top_p.append(float(module.mlp_dynamic_top_p))
            self._orig_top_k.append(int(module.mlp_dynamic_top_k))

        self._arm = "native"
        self._fixed_budget: int | None = None
        self._thresholds: dict[int, float] = {}
        # margin recording for calibration
        self._record_margins = False
        self._margin_records: list[dict[str, Any]] = []
        # executed budget tracking over ALL tokens (full scope)
        self._executed_budget_sum = 0.0
        self._executed_budget_count = 0
        # per-sample per-token sceb budget dispersion (for non-degeneracy gate)
        self._sceb_budget_values: list[int] = []

        self._install()
        log(
            f"installed frontier controller on {len(self._blocks)} blocks; "
            f"candidates={self.candidate_budgets} (full-scope: all tokens, all layers)"
        )

    @staticmethod
    def _layer_idx(name: str) -> int | None:
        parts = name.split(".")
        for i, part in enumerate(parts[:-1]):
            if part == "layers" and parts[i + 1].isdigit():
                return int(parts[i + 1])
        return None

    def _install(self) -> None:
        for block, layer_idx in zip(self._blocks, self._layer_of_block):
            self._orig_forward.append(block.forward)
            block.forward = self._make_forward(block, layer_idx)

    def close(self) -> None:
        for block, orig, top_p, top_k in zip(
            self._blocks, self._orig_forward, self._orig_top_p, self._orig_top_k
        ):
            block.forward = orig
            block.mlp_dynamic_top_p = top_p
            block.mlp_dynamic_top_k = top_k

    def _margin_budget(self, dynamic_probs: torch.Tensor) -> torch.Tensor:
        """boundary-margin selection with per-candidate quantile thresholds.

        dynamic_probs: (n_tokens, dynamic_expert_num) softmax probabilities.
        Returns int64 (n_tokens,) budgets in the candidate set.
        """
        k = dynamic_probs.shape[1]
        return boundary_budget(
            dynamic_probs, k, self.candidate_budgets, self._thresholds
        )

    def _make_forward(self, block: Any, layer_idx: int):
        controller = self

        def forward(hidden_states, attention_mask, aux_balance_weight):
            in_band = controller.layer_lo <= layer_idx < controller.layer_hi
            if controller._arm == "native" or not in_band:
                return controller._orig_forward[controller._blocks.index(block)](
                    hidden_states, attention_mask, aux_balance_weight
                )

            from uni_moe.model.modeling_qwen_grin_moe import (
                dynamic_expert_selection as _native_select,
            )

            _, _, hidden_dim = hidden_states.shape
            flat = hidden_states.view(-1, hidden_dim)
            full_router_logits = block.gate(flat)
            dyn_logits = full_router_logits[:, : block.mlp_dynamic_expert_num]
            dyn_probs = torch.softmax(dyn_logits.float(), dim=-1)
            n_tokens = dyn_probs.shape[0]

            if controller._arm == "sceb_margin":
                budget = controller._margin_budget(dyn_probs)
                if controller._record_margins:
                    ev = dyn_probs
                    if ev.shape[0] > 512:
                        ev = ev[torch.randperm(ev.shape[0], device=ev.device)[:512]]
                    k = ev.shape[1]
                    g = boundary_margins(ev, k, controller.candidate_budgets)
                    rec = {"layer": layer_idx}
                    for m in controller.candidate_budgets:
                        rec[f"margin_{m}"] = g[m].detach().cpu()
                    controller._margin_records.append(rec)
                else:
                    controller._sceb_budget_values.extend(
                        budget.detach().cpu().tolist()
                    )
            else:
                fb = int(controller._fixed_budget or controller.native_ceiling)
                budget = torch.full(
                    (n_tokens,), fb, dtype=torch.int64, device=dyn_probs.device
                )

            controller._executed_budget_sum += float(budget.sum().item())
            controller._executed_budget_count += int(budget.shape[0])

            return controller._forward_with_budget(
                block, hidden_states, attention_mask, aux_balance_weight, budget
            )

        return forward

    def _forward_with_budget(
        self, block, hidden_states, attention_mask, aux_balance_weight, budget
    ):
        """Run the native block math with an externally supplied per-token budget.

        Monkeypatch dynamic_expert_selection for this single call so it returns our
        per-token budget, keeping ALL downstream expert math identical to native.
        """
        import uni_moe.model.modeling_qwen_grin_moe as gm

        idx = self._blocks.index(block)
        orig_select = gm.dynamic_expert_selection
        orig_top_p = block.mlp_dynamic_top_p

        def _forced_select(logits, top_p):
            return budget.to(logits.device)

        try:
            gm.dynamic_expert_selection = _forced_select
            block.mlp_dynamic_top_p = self._orig_top_p[idx]  # non-zero => select path
            out = self._orig_forward[idx](
                hidden_states, attention_mask, aux_balance_weight
            )
        finally:
            gm.dynamic_expert_selection = orig_select
            block.mlp_dynamic_top_p = orig_top_p
        return out

    def set_thresholds(self, thresholds: dict[int, float]) -> None:
        self._thresholds = {int(m): float(t) for m, t in thresholds.items()}

    def reset_counters(self) -> None:
        self._executed_budget_sum = 0.0
        self._executed_budget_count = 0
        self._sceb_budget_values = []

    def executed_mean_budget(self) -> float | None:
        if self._executed_budget_count == 0:
            return None
        return self._executed_budget_sum / self._executed_budget_count

    def sceb_budget_dispersion(self) -> dict[str, Any]:
        vals = self._sceb_budget_values
        if not vals:
            return {"n": 0, "unique": 0, "hist": {}}
        arr = np.asarray(vals, dtype=np.int64)
        uniq, counts = np.unique(arr, return_counts=True)
        return {
            "n": int(arr.shape[0]),
            "unique": int(uniq.shape[0]),
            "hist": {int(u): int(c) for u, c in zip(uniq, counts)},
        }

    class _ArmCtx:
        def __init__(self, ctrl, name, fixed_budget):
            self.ctrl = ctrl
            self.name = name
            self.fixed_budget = fixed_budget

        def __enter__(self):
            self.ctrl._arm = self.name
            self.ctrl._fixed_budget = self.fixed_budget
            self.ctrl.reset_counters()
            return self.ctrl

        def __exit__(self, *exc):
            self.ctrl._arm = "native"
            self.ctrl._fixed_budget = None
            return False

    def arm(self, name: str, fixed_budget: int | None = None):
        return self._ArmCtx(self, name, fixed_budget)


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


def generate_once(model, processor, inputs, max_new_tokens) -> GenerationResult:
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


def distribution_distances(native_logits, arm_logits) -> dict[str, float]:
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
    return {"js_divergence_nats": max(0.0, float(js.item()))}


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL, timeout=20
        ).strip()
    except Exception:
        return "unavailable"


def environment_record(protocol, dataset_path, model_path, tau) -> dict[str, Any]:
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
        "calibrated_tau": tau,
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
    """Bootstrap CI for mean(a - b) on paired samples."""
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


def calibrate_tau(
    *, records, model, processor, process_mm_info, controller, protocol
) -> dict[int, float]:
    n = int(protocol["tau_calibration"]["calibrate_samples"])
    pct = float(protocol["tau_calibration"]["threshold_percentile"])
    pooled: dict[int, list[Any]] = {m: [] for m in controller.candidate_budgets}
    controller._record_margins = True
    controller._margin_records = []
    seen = 0
    for i in range(len(records)):
        if seen >= n:
            break
        video_path = (
            REPO / "eval_data/OmniVideoBench" / str(records[i]["video"])
        ).resolve()
        if not video_path.exists():
            continue
        conv = build_conversation(video_path, records[i])
        inputs = build_inputs(processor, process_mm_info, conv)
        inputs = move_inputs(inputs, model)
        controller.set_thresholds({m: 1e9 for m in controller.candidate_budgets})
        with controller.arm("sceb_margin"):
            generate_once(model, processor, inputs, 1)
        for rec in controller._margin_records:
            for m in controller.candidate_budgets:
                key = f"margin_{m}"
                if key in rec:
                    pooled[m].append(rec[key])
        controller._margin_records = []
        seen += 1
    controller._record_margins = False
    if not any(pooled[m] for m in controller.candidate_budgets):
        raise RuntimeError("boundary calibration collected no margins")
    thresholds = {
        m: float(np.percentile(torch.cat(pooled[m]).numpy().astype(np.float64), pct))
        for m in controller.candidate_budgets
        if pooled[m]
    }
    # Any candidate without collected margins (e.g. m == ceiling) gets +inf so it
    # is never chosen as a stopping point below the ceiling fallback.
    for m in controller.candidate_budgets:
        thresholds.setdefault(m, 1e9)
    log(f"calibrated boundary thresholds (pct={pct}) {thresholds} over {seen} samples")
    return thresholds


def run_sample(
    *,
    sample_index,
    record,
    model,
    processor,
    process_mm_info,
    controller,
    max_new_tokens,
    passthrough_check,
) -> dict[str, Any]:
    video_path = (REPO / "eval_data/OmniVideoBench" / str(record["video"])).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    conv = build_conversation(video_path, record)
    inputs = build_inputs(processor, process_mm_info, conv)
    input_ids_1d = inputs["input_ids"].reshape(-1)
    seq_len = int(inputs["input_ids"].shape[-1])
    n_video = int((input_ids_1d == VIDEO_TOKEN_ID).sum().item())
    inputs = move_inputs(inputs, model)
    correct = str(record["correct_option"]).upper()

    native = generate_once(model, processor, inputs, max_new_tokens)

    passthrough = None
    if passthrough_check:
        with controller.arm("native"):
            pt = generate_once(model, processor, inputs, max_new_tokens)
        passthrough = distribution_distances(native.first_logits, pt.first_logits)

    arms: dict[str, Any] = {}
    dispersion = None

    with controller.arm("sceb_margin"):
        res = generate_once(model, processor, inputs, max_new_tokens)
        mean_budget = controller.executed_mean_budget()
        dispersion = controller.sceb_budget_dispersion()
    arms["sceb_margin"] = _arm_row(res, native, correct, mean_budget)

    for arm_name, k in FIXED_ARMS.items():
        with controller.arm(arm_name, fixed_budget=k):
            res = generate_once(model, processor, inputs, max_new_tokens)
            mean_budget = controller.executed_mean_budget()
        arms[arm_name] = _arm_row(res, native, correct, mean_budget)

    return {
        "sample_index": sample_index,
        "video": record["video"],
        "correct_option": correct,
        "input_sequence_length": seq_len,
        "n_video_tokens": n_video,
        "passthrough_js_vs_native": passthrough,
        "sceb_budget_dispersion": dispersion,
        "native": {
            "answer": native.answer,
            "output_text": native.output_text,
            "is_correct": native.answer == correct,
            "end_to_end_latency_ms": native.e2e_seconds * 1000.0,
        },
        "arms": arms,
    }


def _arm_row(res, native, correct, mean_budget) -> dict[str, Any]:
    return {
        **distribution_distances(native.first_logits, res.first_logits),
        "answer": res.answer,
        "output_text": res.output_text,
        "answer_flip_vs_native": res.answer != native.answer,
        "is_correct": res.answer == correct,
        "end_to_end_latency_ms": res.e2e_seconds * 1000.0,
        "mean_executed_dynamic_budget": mean_budget,
    }


def summarize(
    rows, errors, protocol, controller_blocks, controller_expected, tau, passthrough_max
) -> dict[str, Any]:
    n = len(rows)
    seed = int(protocol["decoding"]["seed"])
    warmup = int(protocol["timing"]["warmup_samples"])
    arm_names = ["sceb_margin", *FIXED_ARMS.keys()]

    def latency_array(getter) -> np.ndarray:
        vals = [getter(r) for i, r in enumerate(rows) if i >= warmup]
        return np.asarray(vals, dtype=np.float64)

    native_acc = np.asarray(
        [int(bool(r["native"]["is_correct"])) for r in rows], dtype=np.float64
    )
    native_lat = latency_array(lambda r: float(r["native"]["end_to_end_latency_ms"]))

    arm_summary: dict[str, Any] = {
        "native": {
            "accuracy": bootstrap_mean_ci(native_acc, seed + 1),
            "end_to_end_latency_ms": bootstrap_mean_ci(native_lat, seed + 2),
            "samples": n,
        }
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
        lat = latency_array(
            lambda r, a=arm: float(r["arms"][a]["end_to_end_latency_ms"])
        )
        budgets = np.asarray(
            [
                float(r["arms"][arm]["mean_executed_dynamic_budget"] or np.nan)
                for r in rows
            ],
            dtype=np.float64,
        )
        arm_summary[arm] = {
            "accuracy": bootstrap_mean_ci(acc, seed + 10 + i),
            "answer_flip_rate": float(flip.mean()),
            "first_token_js_divergence_nats": bootstrap_mean_ci(js, seed + 30 + i),
            "end_to_end_latency_ms": bootstrap_mean_ci(lat, seed + 50 + i),
            "mean_executed_dynamic_budget": float(np.nanmean(budgets)),
            "samples": n,
        }

    # Pareto contrasts: SCEB vs each fixed-k arm (paired).
    sceb_acc = np.asarray(
        [int(bool(r["arms"]["sceb_margin"]["is_correct"])) for r in rows],
        dtype=np.float64,
    )
    sceb_lat = latency_array(
        lambda r: float(r["arms"]["sceb_margin"]["end_to_end_latency_ms"])
    )
    pareto = {}
    for i, arm in enumerate(FIXED_ARMS.keys()):
        arm_acc = np.asarray(
            [int(bool(r["arms"][arm]["is_correct"])) for r in rows], dtype=np.float64
        )
        arm_lat = latency_array(
            lambda r, a=arm: float(r["arms"][a]["end_to_end_latency_ms"])
        )
        acc_diff = paired_diff_ci(sceb_acc, arm_acc, seed + 100 + i)
        lat_diff = paired_diff_ci(sceb_lat, arm_lat, seed + 200 + i)
        # SCEB dominates if it is not worse on accuracy AND not slower, with at
        # least one direction significant.
        acc_not_worse = (
            acc_diff["ci95_low"] is not None and acc_diff["ci95_low"] >= -0.005
        )
        lat_not_worse = (
            lat_diff["ci95_high"] is not None and lat_diff["ci95_high"] <= 0.0
        )
        acc_better = acc_diff["ci95_low"] is not None and acc_diff["ci95_low"] > 0.0
        lat_better = lat_diff["ci95_high"] is not None and lat_diff["ci95_high"] < 0.0
        dominates = bool(acc_not_worse and lat_not_worse and (acc_better or lat_better))
        pareto[f"sceb_vs_{arm}"] = {
            "accuracy_diff": acc_diff,
            "latency_diff_ms": lat_diff,
            "strictly_dominates": dominates,
            "interpretation": "accuracy_diff = sceb - fixed_k; latency_diff = sceb - fixed_k. Dominance needs accuracy not worse AND latency not worse, with >=1 strict.",
        }

    minimum = int(protocol["gate"]["minimum_completed_samples"])
    controller_ok = controller_blocks == controller_expected
    passthrough_ok = passthrough_max is not None and passthrough_max < 1e-6
    sceb_mean = arm_summary["sceb_margin"]["mean_executed_dynamic_budget"]
    cands = sorted(
        int(b) for b in protocol["moe_structure"]["candidate_dynamic_budgets"]
    )
    total_unique = sum(int(r["sceb_budget_dispersion"]["unique"]) for r in rows)
    non_degenerate = bool(
        sceb_mean is not None
        and cands[0] < sceb_mean < cands[-1]
        and total_unique > n  # on average >1 distinct budget per sample
    )
    verified = n >= minimum and controller_ok and passthrough_ok and non_degenerate

    any_dominates = any(v["strictly_dominates"] for v in pareto.values())

    return {
        "schema_version": "sceb/mb-new-2-frontier-unimoe2-results",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "experiment_id": "MB-NEW-2-UniMoE2",
        "backbone": protocol["backbone"],
        "evidence_status": "verified" if verified else "placeholder",
        "calibrated_tau": tau,
        "threshold_percentile": float(
            protocol["tau_calibration"]["threshold_percentile"]
        ),
        "completed_samples": n,
        "failed_samples": len(errors),
        "requested_samples": int(protocol["sample_selection"]["target_samples"]),
        "arms": arm_summary,
        "pareto_contrasts": pareto,
        "sceb_strictly_dominates_any_fixed_k": any_dominates,
        "implementation_gates": {
            "controller_blocks": controller_blocks,
            "controller_blocks_expected": controller_expected,
            "controller_wiring_passed": controller_ok,
            "native_passthrough_js_max": passthrough_max,
            "native_passthrough_passed": passthrough_ok,
            "sceb_mean_budget": sceb_mean,
            "sceb_budget_non_degenerate": non_degenerate,
            "passed": controller_ok and passthrough_ok and non_degenerate,
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
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=None,
        help="Override protocol threshold percentile (frontier sweep knob).",
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Override SCEB candidate dynamic budgets, comma-separated (e.g. '2,3,4'). "
        "The fixed-k baseline arms always cover the full P-M2 set top1..4.",
    )
    parser.add_argument(
        "--layer-band-frac",
        type=str,
        default=None,
        help="Restrict the intervention to a middle-depth MoE layer band, given as "
        "'lo,hi' depth fractions (e.g. '0.25,0.67'). BOTH the sceb and fixed-k arms "
        "are gated to this band so the comparison stays scope-fair. Default: full scope.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> None:
    native = torch.tensor([2.0, 1.0, 0.5, 0.1])
    same = distribution_distances(native, native.clone())
    assert same["js_divergence_nats"] < 1e-9, same
    ci = bootstrap_mean_ci(np.array([1.0, 0.0, 1.0, 1.0]), 7)
    assert ci["ci95_low"] <= ci["mean"] <= ci["ci95_high"]
    d = paired_diff_ci(np.array([0.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0]), 3)
    assert d["mean"] < 0
    assert parse_answer("The answer is C.") == "C"
    # boundary_budget sanity: a sharply peaked distribution stops early.
    peaked = torch.tensor([[0.9, 0.06, 0.03, 0.01]])
    flat = torch.tensor([[0.30, 0.28, 0.22, 0.20]])
    thr = {1: 0.1, 2: 0.1, 3: 0.1, 4: 1e9}
    b_peaked = boundary_budget(peaked, 4, [1, 2, 3, 4], thr)
    b_flat = boundary_budget(flat, 4, [1, 2, 3, 4], thr)
    assert int(b_peaked[0]) <= int(b_flat[0]), (b_peaked, b_flat)
    print("MB-NEW-2-FRONTIER-UniMoE2 self-test passed.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --self-test is used")

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if args.threshold_percentile is not None:
        protocol["tau_calibration"]["threshold_percentile"] = float(
            args.threshold_percentile
        )
    if args.candidates is not None:
        cands = [int(x) for x in args.candidates.split(",") if x.strip()]
        if not cands:
            raise SystemExit("--candidates parsed to empty list")
        protocol["moe_structure"]["candidate_dynamic_budgets"] = sorted(set(cands))
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

    log("loading Uni-MoE-2 model")
    _ensure_matplotlib_stub()
    from uni_moe.model.processing_qwen2_vl import Qwen2VLProcessor
    from uni_moe.model.modeling_qwen_grin_moe import GrinQwen2VLForConditionalGeneration
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

    total_layers = int(
        json.loads((model_path / "config.json").read_text())["num_hidden_layers"]
    )
    candidates = list(protocol["moe_structure"]["candidate_dynamic_budgets"])
    native_ceiling = int(protocol["moe_structure"]["mlp_dynamic_real_expert_num"])
    layer_band = None
    if args.layer_band_frac is not None:
        lo_f, hi_f = (float(x) for x in args.layer_band_frac.split(","))
        layer_band = (
            int(np.floor(lo_f * total_layers)),
            int(np.floor(hi_f * total_layers)),
        )
        protocol["scope_layer_band"] = {
            "layer_band_frac": [lo_f, hi_f],
            "layer_band": list(layer_band),
            "note": "Both sceb and fixed-k arms are gated to this MoE layer band; scope-fair.",
        }
        log(
            f"layer band restricted to {layer_band} (frac {lo_f},{hi_f}) of {total_layers}"
        )
    controller = FrontierController(
        model,
        expected_blocks=total_layers,
        candidate_budgets=candidates,
        native_ceiling=native_ceiling,
        layer_band=layer_band,
    )

    tau = calibrate_tau(
        records=records,
        model=model,
        processor=processor,
        process_mm_info=process_mm_info,
        controller=controller,
        protocol=protocol,
    )
    controller.set_thresholds(tau)
    atomic_json(
        output_dir / "environment.json",
        environment_record(protocol, dataset_path, model_path, tau),
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
                    u2 = row["arms"]["uniform_top2"]
                    log(
                        f"[{ordinal}/{len(selected)}] sample {sample_index}: "
                        f"sceb_budget={s['mean_executed_dynamic_budget']:.3f} "
                        f"sceb_lat={s['end_to_end_latency_ms']:.0f}ms "
                        f"top2_lat={u2['end_to_end_latency_ms']:.0f}ms "
                        f"sceb_correct={s['is_correct']} disp={row['sceb_budget_dispersion']['unique']}"
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
            controller_blocks=len(controller._blocks),
            controller_expected=total_layers,
            tau=tau,
            passthrough_max=passthrough_max,
        )
        atomic_json(summary_path, final)
        log(
            f"finished: status={final['evidence_status']} "
            f"completed={final['completed_samples']} failed={final['failed_samples']} "
            f"dominates_any={final['sceb_strictly_dominates_any_fixed_k']}"
        )
        return 0 if final["evidence_status"] == "verified" else 3
    finally:
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())

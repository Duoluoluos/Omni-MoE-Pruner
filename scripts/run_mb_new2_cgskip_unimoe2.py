#!/usr/bin/env python3
"""MB-NEW-2 v2: Confidence-Gated expert Skip (CG-Skip) frontier on Uni-MoE-2.0-Omni.

Protocol: research_paper/iclr2027/evidence/mb_new2_frontier_unimoe2_protocol.json
(shared decoding / dataset / gate contract; this runner adds the CG-Skip policy
and a shared-only reference arm).

Motivation (from the MB-NEW-2 SCEB null analysis on 1,000 examples):
  - The frozen SCEB boundary-margin selector produced a near-constant ~2.54 mean
    dynamic budget and could not undercut uniform top-2 latency, because it never
    dropped below ~2 dynamic experts.
  - The GRIN block, however, cleanly supports dynamic_top_k = 0: such a token
    skips all 4 real dynamic experts and is served by only the 2 always-on shared
    experts (which are ~8x cheaper by intermediate size). This is a much lower
    latency floor than top-2 and is currently unused.

CG-Skip policy (leakage-free): for each token, read the router's own dynamic
softmax (5-way: 4 real + 1 null). Let p1 be the top-1 dynamic probability
(confidence) and let null_mass be the probability the router assigns to the null
slot. A token the router is very confident about (large p1) or already wants to
no-op on (large null_mass) is cheap to serve: give it budget 0 (shared-only).
An uncertain token gets the full/boosted budget. Concretely, per token:

    if p1 >= tau_skip  or  null_mass >= tau_null : budget = 0   (shared-only)
    elif p1 <= tau_hard                          : budget = boost_budget
    else                                          : budget = base_budget

All thresholds are calibrated from input router statistics only (percentiles of
the pooled p1 / null_mass), never from answers or labels. The mean executed
budget is a reported quantity; the frontier question is whether this adaptive
skip beats the uniform fixed-k points at matched wall-clock.

Arms per sample:
  native        - unmodified GRIN top-p routing (reference)
  cgskip        - the CG-Skip policy above (adaptive, can emit budget 0)
  shared_only   - budget 0 for every token (all-skip reference; latency floor)
  uniform_top1..4 - P-M2 fixed-k baselines

Leakage-free, single-checkpoint, single-H20 deterministic inference.
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

VIDEO_TOKEN_ID = 151656

FIXED_ARMS = {
    "uniform_top1": 1,
    "uniform_top2": 2,
    "uniform_top3": 3,
    "uniform_top4": 4,
}


def log(*parts: Any) -> None:
    print("[MB-NEW-2-CGSKIP-UniMoE2]", *parts, file=sys.stderr, flush=True)


def _ensure_matplotlib_stub() -> None:
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


def cgskip_budget(
    dyn_probs: torch.Tensor,
    real_expert_num: int,
    tau_skip: float,
    tau_null: float,
    tau_hard: float,
    base_budget: int,
    boost_budget: int,
) -> torch.Tensor:
    """Confidence-Gated Skip per-token budget.

    dyn_probs: (n_tokens, 5) softmax over [4 real + 1 null].
    Returns int64 (n_tokens,) in {0, base_budget, boost_budget}.
    """
    real = dyn_probs[:, :real_expert_num]  # (n, 4)
    p1 = real.max(dim=-1).values  # top-1 real confidence
    null_mass = dyn_probs[:, real_expert_num]  # mass on the null slot
    n = dyn_probs.shape[0]
    budget = torch.full(
        (n,), int(base_budget), dtype=torch.int64, device=dyn_probs.device
    )
    hard = p1 <= tau_hard
    budget = torch.where(hard, torch.full_like(budget, int(boost_budget)), budget)
    skip = (p1 >= tau_skip) | (null_mass >= tau_null)
    budget = torch.where(skip, torch.zeros_like(budget), budget)
    return budget


class CGSkipController:
    """Full-scope per-token budget injection supporting budget 0 (shared-only).

    Arms:
      native        -> passthrough
      cgskip        -> CG-Skip adaptive budget (may be 0)
      shared_only   -> budget 0 for all tokens
      uniform_topK  -> constant budget K
    """

    def __init__(
        self,
        model: Any,
        expected_blocks: int,
        real_expert_num: int,
        layer_band: tuple[int, int] | None,
        policy: dict[str, Any],
    ) -> None:
        self.model = model
        self.real_expert_num = int(real_expert_num)
        if layer_band is None:
            self.layer_lo, self.layer_hi = 0, expected_blocks
        else:
            self.layer_lo, self.layer_hi = int(layer_band[0]), int(layer_band[1])
        self.policy = policy
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
        self._force_budget: int | None = None
        self._thresholds: dict[str, float] = {}
        self._record_stats = False
        self._stat_records: list[dict[str, Any]] = []
        self._executed_budget_sum = 0.0
        self._executed_budget_count = 0
        self._budget_values: list[int] = []

        self._install()
        log(
            f"installed CG-Skip controller on {len(self._blocks)} blocks; "
            f"band=[{self.layer_lo},{self.layer_hi}) real_experts={self.real_expert_num}"
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

    def set_thresholds(self, thresholds: dict[str, float]) -> None:
        self._thresholds = {k: float(v) for k, v in thresholds.items()}

    def _make_forward(self, block: Any, layer_idx: int):
        controller = self

        def forward(hidden_states, attention_mask, aux_balance_weight):
            in_band = controller.layer_lo <= layer_idx < controller.layer_hi
            if controller._arm == "native" or not in_band:
                return controller._orig_forward[controller._blocks.index(block)](
                    hidden_states, attention_mask, aux_balance_weight
                )

            _, _, hidden_dim = hidden_states.shape
            flat = hidden_states.view(-1, hidden_dim)
            full_router_logits = block.gate(flat)
            dyn_logits = full_router_logits[:, : block.mlp_dynamic_expert_num]
            dyn_probs = torch.softmax(dyn_logits.float(), dim=-1)
            n_tokens = dyn_probs.shape[0]

            if controller._arm == "cgskip":
                budget = cgskip_budget(
                    dyn_probs,
                    controller.real_expert_num,
                    tau_skip=controller._thresholds["tau_skip"],
                    tau_null=controller._thresholds["tau_null"],
                    tau_hard=controller._thresholds["tau_hard"],
                    base_budget=int(controller.policy["base_budget"]),
                    boost_budget=int(controller.policy["boost_budget"]),
                )
                if not controller._record_stats:
                    controller._budget_values.extend(budget.detach().cpu().tolist())
            elif controller._arm == "shared_only":
                budget = torch.zeros(
                    n_tokens, dtype=torch.int64, device=dyn_probs.device
                )
            else:  # uniform_topK
                fb = int(controller._fixed_budget or controller.real_expert_num)
                budget = torch.full(
                    (n_tokens,), fb, dtype=torch.int64, device=dyn_probs.device
                )

            if controller._force_budget is not None:
                budget = torch.full(
                    (n_tokens,),
                    int(controller._force_budget),
                    dtype=torch.int64,
                    device=dyn_probs.device,
                )

            if controller._record_stats:
                real = dyn_probs[:, : controller.real_expert_num]
                p1 = real.max(dim=-1).values
                null_mass = dyn_probs[:, controller.real_expert_num]
                controller._stat_records.append(
                    {"p1": p1.detach().cpu(), "null_mass": null_mass.detach().cpu()}
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
        """Native block math with an externally supplied per-token budget.

        Budget may be 0: the grouped selection loop (range 1..5) simply never
        enters such a token, its dynamic expert_mask stays 0, routing_weights
        stay 0, and only the 2 always-on shared experts contribute. cal_global_weight
        softmaxes over the still-enabled shared slots, so the shared experts carry
        full weight for a 0-budget token. Verified NaN-free.
        """
        import uni_moe.model.modeling_qwen_grin_moe as gm

        idx = self._blocks.index(block)
        orig_select = gm.dynamic_expert_selection
        orig_top_p = block.mlp_dynamic_top_p

        def _forced_select(logits, top_p):
            return budget.to(logits.device)

        try:
            gm.dynamic_expert_selection = _forced_select
            block.mlp_dynamic_top_p = self._orig_top_p[idx]
            out = self._orig_forward[idx](
                hidden_states, attention_mask, aux_balance_weight
            )
        finally:
            gm.dynamic_expert_selection = orig_select
            block.mlp_dynamic_top_p = orig_top_p
        return out

    def reset_counters(self) -> None:
        self._executed_budget_sum = 0.0
        self._executed_budget_count = 0
        self._budget_values = []

    def executed_mean_budget(self) -> float | None:
        if self._executed_budget_count == 0:
            return None
        return self._executed_budget_sum / self._executed_budget_count

    def budget_dispersion(self) -> dict[str, Any]:
        vals = self._budget_values
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
        def __init__(self, ctrl, name, fixed_budget, force_budget):
            self.ctrl = ctrl
            self.name = name
            self.fixed_budget = fixed_budget
            self.force_budget = force_budget

        def __enter__(self):
            self.ctrl._arm = self.name
            self.ctrl._fixed_budget = self.fixed_budget
            self.ctrl._force_budget = self.force_budget
            self.ctrl.reset_counters()
            return self.ctrl

        def __exit__(self, *exc):
            self.ctrl._arm = "native"
            self.ctrl._fixed_budget = None
            self.ctrl._force_budget = None
            return False

    def arm(
        self,
        name: str,
        fixed_budget: int | None = None,
        force_budget: int | None = None,
    ):
        return self._ArmCtx(self, name, fixed_budget, force_budget)


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


def environment_record(
    protocol, dataset_path, model_path, tau, policy
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
        "calibrated_thresholds": tau,
        "cgskip_policy": policy,
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


def calibrate_thresholds(
    *, records, model, processor, process_mm_info, controller, protocol, policy
) -> dict[str, float]:
    """Calibrate tau_skip / tau_null / tau_hard from pooled input router stats only."""
    n = int(protocol["tau_calibration"]["calibrate_samples"])
    p1_pool: list[Any] = []
    null_pool: list[Any] = []
    controller._record_stats = True
    controller._stat_records = []
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
        with controller.arm("cgskip"):
            # thresholds not needed while recording; force a benign budget so the
            # forced_select path still runs and stats are captured.
            controller.set_thresholds(
                {"tau_skip": 2.0, "tau_null": 2.0, "tau_hard": -1.0}
            )
            generate_once(model, processor, inputs, 1)
        for rec in controller._stat_records:
            p1_pool.append(rec["p1"])
            null_pool.append(rec["null_mass"])
        controller._stat_records = []
        seen += 1
    controller._record_stats = False
    if not p1_pool:
        raise RuntimeError("threshold calibration collected no router stats")
    p1_all = torch.cat(p1_pool).numpy().astype(np.float64)
    null_all = torch.cat(null_pool).numpy().astype(np.float64)
    skip_pct = float(policy["skip_percentile"])
    hard_pct = float(policy["hard_percentile"])
    null_pct = float(policy["null_percentile"])
    thresholds = {
        # skip the easiest skip_pct fraction: p1 at (100 - skip_pct) percentile
        "tau_skip": float(np.percentile(p1_all, 100.0 - skip_pct)),
        "tau_null": float(np.percentile(null_all, 100.0 - null_pct)),
        "tau_hard": float(np.percentile(p1_all, hard_pct)),
    }
    log(
        f"calibrated thresholds {thresholds} over {seen} samples "
        f"(p1 mean {p1_all.mean():.3f}, null mean {null_all.mean():.3f})"
    )
    return thresholds


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
    with controller.arm("cgskip"):
        res = generate_once(model, processor, inputs, max_new_tokens)
        mean_budget = controller.executed_mean_budget()
        dispersion = controller.budget_dispersion()
    arms["cgskip"] = _arm_row(res, native, correct, mean_budget)

    with controller.arm("shared_only"):
        res = generate_once(model, processor, inputs, max_new_tokens)
        mean_budget = controller.executed_mean_budget()
    arms["shared_only"] = _arm_row(res, native, correct, mean_budget)

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
        "cgskip_budget_dispersion": dispersion,
        "native": {
            "answer": native.answer,
            "output_text": native.output_text,
            "is_correct": native.answer == correct,
            "end_to_end_latency_ms": native.e2e_seconds * 1000.0,
        },
        "arms": arms,
    }


def summarize(
    rows,
    errors,
    protocol,
    controller_blocks,
    controller_expected,
    tau,
    policy,
    passthrough_max,
) -> dict[str, Any]:
    n = len(rows)
    seed = int(protocol["decoding"]["seed"])
    warmup = int(protocol["timing"]["warmup_samples"])
    arm_names = ["cgskip", "shared_only", *FIXED_ARMS.keys()]

    def latency_array(getter) -> np.ndarray:
        return np.asarray(
            [getter(r) for i, r in enumerate(rows) if i >= warmup], dtype=np.float64
        )

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
            "end_to_end_latency_ms": bootstrap_mean_ci(lat, seed + 50 + i),
            "mean_executed_dynamic_budget": float(np.nanmean(budgets)),
            "samples": n,
        }

    cg_acc = np.asarray(
        [int(bool(r["arms"]["cgskip"]["is_correct"])) for r in rows], dtype=np.float64
    )
    cg_lat = latency_array(
        lambda r: float(r["arms"]["cgskip"]["end_to_end_latency_ms"])
    )
    pareto = {}
    for i, arm in enumerate(FIXED_ARMS.keys()):
        arm_acc = np.asarray(
            [int(bool(r["arms"][arm]["is_correct"])) for r in rows], dtype=np.float64
        )
        arm_lat = latency_array(
            lambda r, a=arm: float(r["arms"][a]["end_to_end_latency_ms"])
        )
        acc_diff = paired_diff_ci(cg_acc, arm_acc, seed + 100 + i)
        lat_diff = paired_diff_ci(cg_lat, arm_lat, seed + 200 + i)
        acc_not_worse = (
            acc_diff["ci95_low"] is not None and acc_diff["ci95_low"] >= -0.005
        )
        lat_not_worse = (
            lat_diff["ci95_high"] is not None and lat_diff["ci95_high"] <= 0.0
        )
        acc_better = acc_diff["ci95_low"] is not None and acc_diff["ci95_low"] > 0.0
        lat_better = lat_diff["ci95_high"] is not None and lat_diff["ci95_high"] < 0.0
        dominates = bool(acc_not_worse and lat_not_worse and (acc_better or lat_better))
        pareto[f"cgskip_vs_{arm}"] = {
            "accuracy_diff": acc_diff,
            "latency_diff_ms": lat_diff,
            "strictly_dominates": dominates,
        }

    minimum = int(protocol["gate"]["minimum_completed_samples"])
    controller_ok = controller_blocks == controller_expected
    passthrough_ok = passthrough_max is not None and passthrough_max < 1e-6
    cg_mean = arm_summary["cgskip"]["mean_executed_dynamic_budget"]
    total_unique = sum(int(r["cgskip_budget_dispersion"]["unique"]) for r in rows)
    non_degenerate = bool(cg_mean is not None and total_unique > n)
    verified = n >= minimum and controller_ok and passthrough_ok and non_degenerate
    any_dominates = any(v["strictly_dominates"] for v in pareto.values())

    return {
        "schema_version": "sceb/mb-new-2-cgskip-unimoe2-results",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "experiment_id": "MB-NEW-2-CGSKIP-UniMoE2",
        "backbone": protocol["backbone"],
        "evidence_status": "verified" if verified else "placeholder",
        "calibrated_thresholds": tau,
        "cgskip_policy": policy,
        "completed_samples": n,
        "failed_samples": len(errors),
        "requested_samples": int(protocol["sample_selection"]["target_samples"]),
        "arms": arm_summary,
        "pareto_contrasts": pareto,
        "cgskip_strictly_dominates_any_fixed_k": any_dominates,
        "implementation_gates": {
            "controller_blocks": controller_blocks,
            "controller_blocks_expected": controller_expected,
            "controller_wiring_passed": controller_ok,
            "native_passthrough_js_max": passthrough_max,
            "native_passthrough_passed": passthrough_ok,
            "cgskip_mean_budget": cg_mean,
            "cgskip_budget_non_degenerate": non_degenerate,
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--layer-band-frac", type=str, default=None)
    parser.add_argument("--base-budget", type=int, default=2)
    parser.add_argument("--boost-budget", type=int, default=3)
    parser.add_argument(
        "--skip-percentile",
        type=float,
        default=30.0,
        help="Fraction of tokens (by highest p1) sent to budget 0.",
    )
    parser.add_argument(
        "--hard-percentile",
        type=float,
        default=30.0,
        help="Fraction of tokens (by lowest p1) given the boost budget.",
    )
    parser.add_argument(
        "--null-percentile",
        type=float,
        default=10.0,
        help="Extra fraction sent to budget 0 by highest null-mass.",
    )
    parser.add_argument(
        "--probe-budget",
        type=int,
        default=None,
        help="Debug: force this constant budget on all tokens (e.g. 0) and skip other arms.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> None:
    native = torch.tensor([2.0, 1.0, 0.5, 0.1])
    same = distribution_distances(native, native.clone())
    assert same["js_divergence_nats"] < 1e-9
    ci = bootstrap_mean_ci(np.array([1.0, 0.0, 1.0, 1.0]), 7)
    assert ci["ci95_low"] <= ci["mean"] <= ci["ci95_high"]
    assert parse_answer("The answer is C.") == "C"
    # cgskip: confident token -> 0, uncertain token -> boost, mid -> base
    probs = torch.tensor(
        [
            [0.95, 0.02, 0.02, 0.005, 0.005],  # very confident -> skip (0)
            [0.30, 0.28, 0.22, 0.15, 0.05],  # uncertain -> boost
            [0.55, 0.25, 0.12, 0.06, 0.02],  # mid -> base
        ]
    )
    b = cgskip_budget(
        probs,
        4,
        tau_skip=0.9,
        tau_null=0.9,
        tau_hard=0.35,
        base_budget=2,
        boost_budget=3,
    )
    assert int(b[0]) == 0, b
    assert int(b[1]) == 3, b
    assert int(b[2]) == 2, b
    print("MB-NEW-2-CGSKIP-UniMoE2 self-test passed.")


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

    policy = {
        "base_budget": int(args.base_budget),
        "boost_budget": int(args.boost_budget),
        "skip_percentile": float(args.skip_percentile),
        "hard_percentile": float(args.hard_percentile),
        "null_percentile": float(args.null_percentile),
    }

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
    real_expert_num = int(protocol["moe_structure"]["mlp_dynamic_real_expert_num"])
    layer_band = None
    if args.layer_band_frac is not None:
        lo_f, hi_f = (float(x) for x in args.layer_band_frac.split(","))
        layer_band = (
            int(np.floor(lo_f * total_layers)),
            int(np.floor(hi_f * total_layers)),
        )
        log(f"layer band restricted to {layer_band} (frac {lo_f},{hi_f})")

    controller = CGSkipController(
        model,
        expected_blocks=total_layers,
        real_expert_num=real_expert_num,
        layer_band=layer_band,
        policy=policy,
    )

    # Debug probe: force a constant budget (e.g. 0) and measure latency/quality.
    if args.probe_budget is not None:
        log(f"PROBE MODE: forcing budget={args.probe_budget} on all tokens in band")
        for si in selected:
            rec = records[si]
            vp = (REPO / "eval_data/OmniVideoBench" / str(rec["video"])).resolve()
            if not vp.exists():
                continue
            conv = build_conversation(vp, rec)
            inputs = move_inputs(build_inputs(processor, process_mm_info, conv), model)
            nat = generate_once(model, processor, inputs, args.max_new_tokens)
            with controller.arm("cgskip", force_budget=args.probe_budget):
                controller.set_thresholds(
                    {"tau_skip": 2.0, "tau_null": 2.0, "tau_hard": -1.0}
                )
                pr = generate_once(model, processor, inputs, args.max_new_tokens)
            jsd = distribution_distances(nat.first_logits, pr.first_logits)
            log(
                f"probe sample {si}: nat_lat={nat.e2e_seconds * 1000:.0f}ms "
                f"probe_lat={pr.e2e_seconds * 1000:.0f}ms js={jsd['js_divergence_nats']:.4f} "
                f"nat_ans={nat.answer} probe_ans={pr.answer}"
            )
        controller.close()
        return 0

    tau = calibrate_thresholds(
        records=records,
        model=model,
        processor=processor,
        process_mm_info=process_mm_info,
        controller=controller,
        protocol=protocol,
        policy=policy,
    )
    controller.set_thresholds(tau)
    atomic_json(
        output_dir / "environment.json",
        environment_record(protocol, dataset_path, model_path, tau, policy),
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
                    c = row["arms"]["cgskip"]
                    s = row["arms"]["shared_only"]
                    u2 = row["arms"]["uniform_top2"]
                    log(
                        f"[{ordinal}/{len(selected)}] s{sample_index}: "
                        f"cg_bud={c['mean_executed_dynamic_budget']:.2f} "
                        f"cg_lat={c['end_to_end_latency_ms']:.0f} "
                        f"shared_lat={s['end_to_end_latency_ms']:.0f} "
                        f"top2_lat={u2['end_to_end_latency_ms']:.0f} "
                        f"cg_ok={c['is_correct']} disp={row['cgskip_budget_dispersion']['unique']}"
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
            policy=policy,
            passthrough_max=passthrough_max,
        )
        atomic_json(summary_path, final)
        log(
            f"finished: status={final['evidence_status']} completed={final['completed_samples']} "
            f"failed={final['failed_samples']} dominates_any={final['cgskip_strictly_dominates_any_fixed_k']}"
        )
        return 0 if final["evidence_status"] == "verified" else 3
    finally:
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())

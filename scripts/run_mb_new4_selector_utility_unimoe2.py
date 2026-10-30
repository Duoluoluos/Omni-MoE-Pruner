#!/usr/bin/env python3
"""MB-NEW-4 selector-utility test on Uni-MoE-2.0-Omni.

Protocol:
research_paper/iclr2027/evidence/mb_new4_selector_utility_unimoe2_protocol.json

Question: at a matched mean executed dynamic-expert budget on eligible video
tokens in the middle-depth MoE band, does SCEB Axis C (per-token boundary-margin
selection over the router's own probabilities) reduce answer flips and hold
accuracy relative to budget-matched signal-free controls?

Arms per sample:
  native                  - unmodified GRIN top-p routing everywhere (reference)
  sceb_margin             - Axis C: smallest budget m in {2,3,4} with
                            p_m - p_{m+1} >= tau, else grow to native ceiling,
                            applied only to eligible video tokens in layers 7..17
  budget_matched_uniform  - single fixed budget on the same scope, matched to the
                            realized sceb_margin mean budget (signal-free)
  budget_matched_shuffled - the exact sceb_margin per-token budgets permuted
                            within each (layer) eligible-video group (signal-
                            destroyed, budget-preserved)

Leakage-free: the selector reads only router logits and token position. No
task_type / audio_type / answer / label ever reaches the runtime path.

Single-checkpoint, single-H20 deterministic inference. This validates Axis C of
the frozen SCEB definition; it is NOT a wall-clock or modality-utility claim.
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
    / "research_paper/iclr2027/evidence/mb_new4_selector_utility_unimoe2_protocol.json"
)
sys.path.insert(0, "/home/tiger/Uni-MoE/Uni-MoE-2")

VIDEO_TOKEN_ID = 151656


def log(*parts: Any) -> None:
    print("[MB-NEW-4-UniMoE2]", *parts, file=sys.stderr, flush=True)


def _ensure_matplotlib_stub() -> None:
    """Uni-MoE's modeling file imports matplotlib.pyplot at module load but the
    inference path never plots. If matplotlib is unavailable (disk quota), inject
    a minimal stub so the import succeeds without perturbing any numerics."""
    import types
    import importlib.machinery

    try:
        import matplotlib.pyplot  # noqa: F401

        return
    except Exception:
        pass
    mpl = types.ModuleType("matplotlib")
    mpl.__spec__ = importlib.machinery.ModuleSpec("matplotlib", loader=None)
    mpl.__path__ = []  # mark as package
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


class SCEBController:
    """Per-token, position-conditioned SCEB budget injection for GRIN MoE blocks.

    Wraps every GRINMoESparseMoeBlock.forward. The wrapper recomputes the router
    logits, derives the per-token dynamic budget for the active arm, and forces
    that budget by writing mlp_dynamic_top_p=0 with a per-token budget override
    stored on the block. It relies on the fact that the native block already
    treats dynamic_top_k as a per-token quantity.

    Arms:
      native                  -> passthrough (native top-p)
      sceb_margin             -> boundary-margin selection on eligible video toks
      budget_matched_uniform  -> single fixed budget on eligible video toks
      budget_matched_shuffled -> permuted sceb_margin budgets on eligible video toks

    Eligibility = (token id == VIDEO_TOKEN_ID) AND (layer in middle band).
    Everything else keeps native top-p.
    """

    def __init__(
        self,
        model: Any,
        expected_blocks: int,
        candidate_budgets: list[int],
        native_ceiling: int,
        layer_band: tuple[int, int],
    ) -> None:
        self.model = model
        self.candidate_budgets = sorted(int(b) for b in candidate_budgets)
        self.native_ceiling = int(native_ceiling)
        self.layer_lo, self.layer_hi = layer_band  # half-open [lo, hi)
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
        # Recover the decoder layer index from the module name for the band gate.
        for name, module in moe_blocks:
            layer_idx = self._layer_idx(name)
            if layer_idx is None:
                raise RuntimeError(f"Could not parse layer index from {name!r}")
            self._blocks.append(module)
            self._layer_of_block.append(layer_idx)
            self._orig_top_p.append(float(module.mlp_dynamic_top_p))
            self._orig_top_k.append(int(module.mlp_dynamic_top_k))

        # Per-arm runtime state, set by prepare_sample / set_arm.
        self._arm = "native"
        self._eligible_seq_mask: torch.Tensor | None = None  # (seq,) bool
        self._uniform_budget: int | None = None
        self._shuffle_perm_by_layer: dict[int, torch.Tensor] | None = None
        # Recording buffers (per forward, per block).
        self._record_margins = False
        self._margin_records: list[dict[str, Any]] = []
        self._executed_budget_sum = 0.0
        self._executed_budget_count = 0
        # Captured native per-token budgets for the current sample, per layer,
        # restricted to eligible video positions (used by shuffled arm & matching).
        self._sceb_budgets_by_layer: dict[int, torch.Tensor] = {}

        self._install()
        log(
            f"installed SCEB controller on {len(self._blocks)} blocks; "
            f"band=[{self.layer_lo},{self.layer_hi}) candidates={self.candidate_budgets}"
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

    def _in_band(self, layer_idx: int) -> bool:
        return self.layer_lo <= layer_idx < self.layer_hi

    def _margin_budget(self, dynamic_probs: torch.Tensor) -> torch.Tensor:
        """SCEB Axis C: smallest candidate m with p_m - p_{m+1} >= tau, else grow.

        dynamic_probs: (n_tokens, dynamic_expert_num) softmax probabilities.
        Returns int64 (n_tokens,) budgets.
        """
        sorted_probs, _ = torch.sort(dynamic_probs, dim=-1, descending=True)
        n = sorted_probs.shape[0]
        chosen = torch.full(
            (n,), self.native_ceiling, dtype=torch.int64, device=sorted_probs.device
        )
        # Iterate from largest candidate down so smaller decisive budgets win.
        for m in sorted(self.candidate_budgets, reverse=True):
            if m >= sorted_probs.shape[1]:
                margin = sorted_probs[:, m - 1]  # no p_{m+1}: treat as fully decisive
            else:
                margin = sorted_probs[:, m - 1] - sorted_probs[:, m]
            decisive = margin >= self.tau
            chosen = torch.where(
                decisive,
                torch.tensor(m, device=chosen.device, dtype=torch.int64),
                chosen,
            )
        return torch.clamp(
            chosen, min=min(self.candidate_budgets), max=self.native_ceiling
        )

    def _make_forward(self, block: Any, layer_idx: int):
        from uni_moe.model.modeling_qwen_grin_moe import (
            dynamic_expert_selection as _native_select,
        )

        controller = self

        def forward(hidden_states, attention_mask, aux_balance_weight):
            eligible = (
                controller._arm != "native"
                and controller._in_band(layer_idx)
                and controller._eligible_seq_mask is not None
            )
            if not eligible:
                return controller._orig_forward[controller._blocks.index(block)](
                    hidden_states, attention_mask, aux_balance_weight
                )

            # Recompute router logits exactly as the native block does, to derive
            # the per-token budget, then force it via top_p=0 + per-token override.
            bsz, seqlen, hidden_dim = hidden_states.shape
            flat = hidden_states.view(-1, hidden_dim)
            full_router_logits = block.gate(flat)
            dyn_logits = full_router_logits[:, : block.mlp_dynamic_expert_num]
            dyn_probs = torch.softmax(dyn_logits.float(), dim=-1)

            native_topk = _native_select(
                dyn_logits, controller._orig_top_p[controller._blocks.index(block)]
            )

            # Build eligible mask aligned to flattened tokens. Prefill only:
            # seqlen matches the eligible_seq_mask; decode (seqlen==1) is never
            # eligible (asserted by band+video gate below).
            elig = controller._eligible_seq_mask
            if seqlen != elig.shape[0]:
                # decode step or mismatched shape -> passthrough native
                return controller._orig_forward[controller._blocks.index(block)](
                    hidden_states, attention_mask, aux_balance_weight
                )
            elig_flat = elig.to(dyn_probs.device).view(-1)

            budget = native_topk.clone().to(torch.int64)
            if controller._arm == "sceb_margin":
                sceb_b = controller._margin_budget(dyn_probs)
                budget = torch.where(elig_flat, sceb_b, budget)
                controller._sceb_budgets_by_layer[layer_idx] = (
                    sceb_b[elig_flat].detach().cpu()
                )
            elif controller._arm == "budget_matched_uniform":
                base = controller._sceb_budgets_by_layer.get(layer_idx)
                idx = torch.nonzero(elig_flat, as_tuple=True)[0]
                if (
                    base is not None
                    and idx.shape[0] == base.shape[0]
                    and idx.shape[0] > 0
                ):
                    # Signal-free budget-matched control: reproduce the SCEB
                    # per-layer mean budget with a fixed budget where possible,
                    # else a deterministic two-level split across the bracketing
                    # candidates so the per-layer mean matches exactly. The
                    # assignment order is independent of the router margin, so it
                    # carries no per-token selectivity information.
                    target = float(base.float().mean().item())
                    lo = max(
                        [c for c in controller.candidate_budgets if c <= target],
                        default=min(controller.candidate_budgets),
                    )
                    hi = min(
                        [c for c in controller.candidate_budgets if c >= target],
                        default=max(controller.candidate_budgets),
                    )
                    n_elig = idx.shape[0]
                    if hi == lo:
                        assign = torch.full((n_elig,), lo, dtype=torch.int64)
                    else:
                        frac_hi = (target - lo) / (hi - lo)
                        n_hi = int(round(frac_hi * n_elig))
                        assign = torch.full((n_elig,), lo, dtype=torch.int64)
                        # Assign the hi budget to the first n_hi eligible tokens in
                        # sequence order (position-based, margin-independent).
                        assign[:n_hi] = hi
                    budget[idx] = assign.to(budget.device)
                else:
                    ub = (
                        int(controller._uniform_budget)
                        if controller._uniform_budget
                        else controller.native_ceiling
                    )
                    budget = torch.where(
                        elig_flat,
                        torch.tensor(ub, device=budget.device, dtype=torch.int64),
                        budget,
                    )
            elif controller._arm == "budget_matched_shuffled":
                base = controller._sceb_budgets_by_layer.get(layer_idx)
                perm = (
                    controller._shuffle_perm_by_layer.get(layer_idx)
                    if controller._shuffle_perm_by_layer
                    else None
                )
                if base is not None and perm is not None:
                    shuffled = base[perm].to(budget.device)
                    idx = torch.nonzero(elig_flat, as_tuple=True)[0]
                    if idx.shape[0] == shuffled.shape[0]:
                        budget[idx] = shuffled

            if controller._record_margins:
                em = elig_flat
                if em.any():
                    sp, _ = torch.sort(dyn_probs[em], dim=-1, descending=True)
                    rec = {"layer": layer_idx, "n": int(em.sum().item())}
                    for m in controller.candidate_budgets:
                        if m < sp.shape[1]:
                            rec[f"margin_{m}"] = (
                                (sp[:, m - 1] - sp[:, m]).detach().cpu().tolist()
                            )
                    controller._margin_records.append(rec)

            # Track executed budget on eligible tokens for budget-match checks.
            if controller._arm in (
                "sceb_margin",
                "budget_matched_uniform",
                "budget_matched_shuffled",
            ):
                em = elig_flat
                if em.any():
                    controller._executed_budget_sum += float(budget[em].sum().item())
                    controller._executed_budget_count += int(em.sum().item())

            return controller._forward_with_budget(
                block, hidden_states, attention_mask, aux_balance_weight, budget
            )

        return forward

    def _forward_with_budget(
        self, block, hidden_states, attention_mask, aux_balance_weight, budget
    ):
        """Run the native block math but with an externally supplied per-token budget.

        We set the block to fixed-budget mode with a sentinel and monkeypatch the
        module's dynamic_expert_selection for this single call so it returns our
        per-token budget. This keeps ALL downstream expert math identical to native.
        """
        import uni_moe.model.modeling_qwen_grin_moe as gm

        idx = self._blocks.index(block)
        orig_select = gm.dynamic_expert_selection
        orig_top_p = block.mlp_dynamic_top_p

        def _forced_select(logits, top_p):
            return budget.to(logits.device)

        try:
            gm.dynamic_expert_selection = _forced_select
            block.mlp_dynamic_top_p = self._orig_top_p[
                idx
            ]  # non-zero => uses select path
            out = self._orig_forward[idx](
                hidden_states, attention_mask, aux_balance_weight
            )
        finally:
            gm.dynamic_expert_selection = orig_select
            block.mlp_dynamic_top_p = orig_top_p
        return out

    # ---- per-sample setup ----
    def set_tau(self, tau: float) -> None:
        self.tau = float(tau)

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
    def arm(
        self,
        name: str,
        uniform_budget: int | None = None,
        shuffle_seed: int | None = None,
    ):
        self._arm = name
        self._uniform_budget = uniform_budget
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
            self._uniform_budget = None
            self._shuffle_perm_by_layer = None


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
    chunk = 1000
    for start in range(0, draws, chunk):
        stop = min(draws, start + chunk)
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
) -> float:
    n = int(protocol["tau_calibration"]["calibrate_samples"])
    pct = float(protocol["tau_calibration"]["tau_percentile"])
    pooled: list[float] = []
    controller._record_margins = True
    controller._margin_records = []
    for i in range(min(n, len(records))):
        video_path = (
            REPO / "eval_data/OmniVideoBench" / str(records[i]["video"])
        ).resolve()
        if not video_path.exists():
            continue
        conv = build_conversation(video_path, records[i])
        inputs = build_inputs(processor, process_mm_info, conv)
        input_ids_1d = inputs["input_ids"].reshape(-1)
        controller.prepare_sample(input_ids_1d)
        inputs = move_inputs(inputs, model)
        controller.set_tau(
            1e9
        )  # never fires during calibration; margins still recorded
        with controller.arm("sceb_margin"):
            generate_once(model, processor, inputs, 1)
        for rec in controller._margin_records:
            for m in controller.candidate_budgets:
                key = f"margin_{m}"
                if key in rec:
                    pooled.extend(rec[key])
        controller._margin_records = []
    controller._record_margins = False
    if not pooled:
        raise RuntimeError("tau calibration collected no margins")
    tau = float(np.percentile(np.asarray(pooled, dtype=np.float64), pct))
    log(
        f"calibrated tau={tau:.6g} from {len(pooled)} pooled margins over {min(n, len(records))} samples"
    )
    return tau


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
    input_ids_1d = inputs["input_ids"].reshape(-1)
    controller.prepare_sample(input_ids_1d)
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

    # SCEB margin first (records per-layer budgets used by uniform/shuffled).
    with controller.arm("sceb_margin"):
        sceb = generate_once(model, processor, inputs, max_new_tokens)
        sceb_mean_budget = controller.executed_mean_budget()

    # Choose the matched uniform budget: nearest candidate to sceb mean.
    if sceb_mean_budget is None:
        uniform_budget = controller.native_ceiling
    else:
        uniform_budget = min(
            controller.candidate_budgets,
            key=lambda c: abs(c - sceb_mean_budget),
        )
    with controller.arm("budget_matched_uniform", uniform_budget=uniform_budget):
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
            **distribution_distances(native.first_logits, res.first_logits),
            "answer": res.answer,
            "output_text": res.output_text,
            "answer_flip_vs_native": res.answer != native.answer,
            "is_correct": res.answer == correct,
            "end_to_end_latency_ms": res.e2e_seconds * 1000.0,
            "mean_executed_dynamic_budget": mean_budget,
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
        "uniform_budget_used": int(uniform_budget),
        "arms": {
            "sceb_margin": arm_row(sceb, sceb_mean_budget),
            "budget_matched_uniform": arm_row(umatch, uniform_mean_budget),
            "budget_matched_shuffled": arm_row(shuf, shuffled_mean_budget),
        },
    }


def summarize(
    rows, errors, protocol, controller_blocks, controller_expected, tau, passthrough_max
) -> dict[str, Any]:
    n = len(rows)
    seed = int(protocol["decoding"]["seed"])
    warmup = int(protocol["timing"]["warmup_samples"])
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
            [
                float(r["arms"][arm]["mean_executed_dynamic_budget"] or np.nan)
                for r in rows
            ],
            dtype=np.float64,
        )
        arm_summary[arm] = {
            "accuracy": bootstrap_mean_ci(acc, seed + 10 + i),
            "answer_flip_rate": float(flip.mean()),
            "first_token_js_divergence_nats": bootstrap_mean_ci(js, seed + 20 + i),
            "mean_executed_dynamic_budget": float(np.nanmean(budgets)),
            "samples": n,
        }

    # Paired selectivity contrasts: sceb_margin vs each control.
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
            "interpretation": "flip_rate_diff<0 with CI excluding 0 => margin signal reduces flips at matched budget; accuracy_diff>=~0 => no accuracy cost.",
        }

    # Budget-match gate across samples.
    match_ok = 0
    for r in rows:
        sb = r["arms"]["sceb_margin"]["mean_executed_dynamic_budget"]
        ub = r["arms"]["budget_matched_uniform"]["mean_executed_dynamic_budget"]
        hb = r["arms"]["budget_matched_shuffled"]["mean_executed_dynamic_budget"]
        if sb is None:
            continue
        rel = lambda x: x is not None and abs(x - sb) <= tol * max(1.0, sb)
        if rel(ub) and rel(hb):
            match_ok += 1

    minimum = int(protocol["gate"]["minimum_completed_samples"])
    controller_ok = controller_blocks == controller_expected
    passthrough_ok = passthrough_max is not None and passthrough_max < 1e-6
    verified = n >= minimum and controller_ok and passthrough_ok and match_ok >= minimum

    return {
        "schema_version": "sceb/mb-new-4-results-v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "experiment_id": "MB-NEW-4-UniMoE2",
        "backbone": protocol["backbone"],
        "evidence_status": "verified" if verified else "placeholder",
        "calibrated_tau": tau,
        "completed_samples": n,
        "failed_samples": len(errors),
        "requested_samples": int(protocol["sample_selection"]["target_samples"]),
        "arms": arm_summary,
        "selectivity_contrasts": contrasts,
        "implementation_gates": {
            "controller_blocks": controller_blocks,
            "controller_blocks_expected": controller_expected,
            "controller_wiring_passed": controller_ok,
            "native_passthrough_js_max": passthrough_max,
            "native_passthrough_passed": passthrough_ok,
            "budget_match_ok_samples": match_ok,
            "budget_match_tolerance": tol,
            "passed": controller_ok and passthrough_ok and match_ok >= minimum,
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
    same = distribution_distances(native, native.clone())
    assert same["js_divergence_nats"] < 1e-9, same
    ci = bootstrap_mean_ci(np.array([1.0, 0.0, 1.0, 1.0]), 7)
    assert ci["ci95_low"] <= ci["mean"] <= ci["ci95_high"]
    d = paired_diff_ci(np.array([0.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0]), 3)
    assert d["mean"] < 0
    assert parse_answer("The answer is C.") == "C"
    print("MB-NEW-4-UniMoE2 self-test passed.")


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
    band_lo = int(np.floor(0.25 * total_layers))
    band_hi = int(np.floor(0.67 * total_layers))
    candidates = list(protocol["moe_structure"]["candidate_dynamic_budgets"])
    native_ceiling = int(protocol["moe_structure"]["mlp_dynamic_real_expert_num"])
    controller = SCEBController(
        model,
        expected_blocks=total_layers,
        candidate_budgets=candidates,
        native_ceiling=native_ceiling,
        layer_band=(band_lo, band_hi),
    )

    tau = calibrate_tau(
        records=records,
        model=model,
        processor=processor,
        process_mm_info=process_mm_info,
        controller=controller,
        protocol=protocol,
    )
    controller.set_tau(tau)
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
                        f"sceb_budget={s['mean_executed_dynamic_budget']} "
                        f"uni_budget={u['mean_executed_dynamic_budget']} "
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
            controller_blocks=len(controller._blocks),
            controller_expected=total_layers,
            tau=tau,
            passthrough_max=passthrough_max,
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

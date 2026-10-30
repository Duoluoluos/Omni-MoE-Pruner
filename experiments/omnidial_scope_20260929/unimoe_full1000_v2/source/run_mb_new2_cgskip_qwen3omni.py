#!/usr/bin/env python3
"""MB-NEW-2 CG-Skip frontier on Qwen3-Omni-30B-A3B (thinker MoE).

Ports the Uni-MoE CG-Skip idea (Confidence-Gated expert Skip) to Qwen3-Omni.

Backbone differences handled here:
  - Qwen thinker MoE is fixed top-8 over 128 experts, 48 layers, and has NO
    shared/always-on experts (shared_expert_intermediate_size == 0). Therefore a
    budget-0 token contributes ZERO MoE output that layer and is carried only by
    the residual stream. This is NaN-safe (the verified budgeted_forward clamps
    the renorm denominator), and is the Qwen analog of Uni-MoE's shared-only skip.

CG-Skip policy (leakage-free): for each prefill token, read the router's own
softmax over 128 experts, take the top-8, renormalize, and let p1 be the top-1
weight (confidence). A token the router is very confident about is cheap to serve
and its top experts already dominate, so it tolerates a small (or zero) budget:

    if p1 >= tau_skip : budget = 0            (skip: residual-only this layer)
    elif p1 <= tau_hard : budget = boost_budget (keep many experts)
    else : budget = base_budget

Thresholds are calibrated from pooled INPUT router statistics only (percentiles of
observed p1), never from answers/labels/task_type/audio_type.

Arms per sample:
  native_top8   - unmodified top-8 routing (reference)
  cgskip        - CG-Skip adaptive budget (may be 0)
  skip_all      - budget 0 everywhere (residual-only reference; latency floor)
  top1/2/4/6/8  - uniform fixed-k baselines (the Qwen frontier)

Reuses the VERIFIED per-token budgeted_forward from
scripts/run_pm4_modality_depth_sensitivity.py (softmax over 128 -> topk 8 ->
rank-prefix mask by per-token budget -> renorm clamp_min -> sparse expert loop).

Single-checkpoint, single-H20, deterministic greedy inference. Leakage-free.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "qwen-omni-utils/src"))

MODEL_PATH = REPO / "models/Qwen3-Omni-30B-A3B-Instruct"
VIDEO_TOKEN_ID = 151656
AUDIO_TOKEN_ID = 151675
NATIVE_TOP_K = 8
TOTAL_MOE_LAYERS = 48
OPTION_LETTERS = ("A", "B", "C", "D", "E", "F")
SEED = 20260909
SYSTEM_PROMPT = "You are a helpful assistant."

BENCHMARKS = {
    "omnivideobench": {
        "dataset": REPO / "eval_data/OmniVideoBench/data.json",
        "video_dir": REPO / "eval_data/OmniVideoBench",
    },
    "avut": {
        "dataset": REPO / "eval_data/AVUTBenchmark/AV_Human_filtered_data.json",
        "video_dir": REPO / "eval_data/AVUTBenchmark",
    },
    "worldsense": {
        "dataset": REPO / "eval_data/WorldSense/worldsense_qa.json",
        "video_dir": REPO / "eval_data/WorldSense/videos",
    },
}

FIXED_ARMS = {
    "top1": 1,
    "top2": 2,
    "top4": 4,
    "top6": 6,
    "top8": 8,
}


def log(*parts: Any) -> None:
    print("[MB-NEW-2-CGSKIP-Qwen]", *parts, file=sys.stderr, flush=True)


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
    match = re.search(r"(?:^|[\s.,:;()\[\]])([A-F])(?:[\s.,:;()\[\]]|$)", upper)
    if match:
        return match.group(1)
    for character in upper:
        if character in OPTION_LETTERS:
            return character
    return None


def load_benchmark(name: str) -> tuple[list[dict[str, Any]], Path]:
    """Load benchmark records into a unified format: list of dicts each with
    'question', 'options' (list of str), 'correct_option' (letter), 'video' (str).
    Returns (records, video_dir).
    """
    cfg = BENCHMARKS[name]
    raw = json.loads(cfg["dataset"].read_text(encoding="utf-8"))
    video_dir = cfg["video_dir"]
    records: list[dict[str, Any]] = []
    if name == "omnivideobench":
        for r in raw:
            records.append(
                {
                    "question": r["question"],
                    "options": r["options"],
                    "correct_option": str(r["correct_option"]).upper(),
                    "video": str(r["video"]),
                    "_raw": r,
                }
            )
    elif name == "avut":
        for r in raw:
            opts = []
            for letter in OPTION_LETTERS:
                val = r.get(f"option_{letter}")
                if val is not None:
                    opts.append(f"{letter}. {val}")
            correct = str(r.get("answer", "")).strip().upper()
            vp = Path(r["video_path"]).name
            records.append(
                {
                    "question": r["question"],
                    "options": opts,
                    "correct_option": correct,
                    "video": vp,
                    "_raw": r,
                }
            )
    elif name == "worldsense":
        for vid, entry in raw.items():
            video_file = f"{entry['video_id']}.mp4"
            for tk in sorted(k for k in entry if k.startswith("task")):
                t = entry[tk]
                cands = t.get("candidates", [])
                if not cands:
                    continue
                records.append(
                    {
                        "question": t["question"],
                        "options": list(cands),
                        "correct_option": str(t.get("answer", "")).strip().upper(),
                        "video": video_file,
                        "_raw": {
                            "video_id": entry["video_id"],
                            "task": tk,
                            "task_type": t.get("task_type"),
                            "task_domain": t.get("task_domain"),
                        },
                    }
                )
    else:
        raise ValueError(f"Unknown benchmark: {name}")
    return records, video_dir


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


def confidence_signal(
    routing_weights: torch.Tensor, reduce_budget: int
) -> torch.Tensor:
    """Leakage-free per-token confidence that the cheap (reduced) budget suffices.

    routing_weights: (n_tokens, top_k) renormalized top-k weights (sorted desc).
    Signal = cumulative routing mass captured by the top `reduce_budget` experts.
    A token whose top-`reduce_budget` experts already carry almost all the mass can
    safely drop the tail experts; a token whose mass is spread past `reduce_budget`
    needs the full budget. This separates tokens far better than the raw top-1
    weight on a 128-expert / near-uniform top-8 router.
    """
    rb = max(1, int(reduce_budget))
    rb = min(rb, routing_weights.shape[1])
    return routing_weights[:, :rb].sum(dim=-1)


def cg_reduce_budget(
    signal: torch.Tensor,
    tau_reduce: float,
    tau_boost: float,
    base_budget: int,
    reduce_budget: int,
    boost_budget: int,
) -> torch.Tensor:
    """Confidence-Gated Reduce per-token budget (Qwen: no budget-0).

    High signal (cheap budget already captures the mass) -> reduce_budget.
    Low signal (mass spread wide, token needs experts) -> boost_budget.
    Otherwise -> base_budget. All budgets are >= 1 (non-empty MoE).
    """
    n = signal.shape[0]
    budget = torch.full((n,), int(base_budget), dtype=torch.long, device=signal.device)
    budget = torch.where(
        signal <= tau_boost, torch.full_like(budget, int(boost_budget)), budget
    )
    budget = torch.where(
        signal >= tau_reduce, torch.full_like(budget, int(reduce_budget)), budget
    )
    return budget


class CGSkipIntervention:
    """Per-token expert-budget injection for Qwen3-Omni thinker MoE blocks.

    Reuses the verified PM4 budgeted_forward. Arms:
      native  -> passthrough
      cgskip  -> CG-Skip adaptive per-token budget (may be 0)
      fixed   -> constant budget k on every token
    Budget 0 => residual-only token that layer (NaN-safe via renorm clamp).
    """

    def __init__(
        self, model: Any, native_top_k: int, layer_band: tuple[int, int] | None
    ) -> None:
        self.native_top_k = int(native_top_k)
        self.layer_lo, self.layer_hi = (
            (0, TOTAL_MOE_LAYERS) if layer_band is None else layer_band
        )
        self._arm = "native"
        self._fixed_budget: int | None = None
        self._force_budget: int | None = None
        self._eligible_seq_mask: torch.Tensor | None = None
        self._thresholds: dict[str, float] = {}
        self.policy: dict[str, int] = {}
        self._record_stats = False
        self._p1_records: list[torch.Tensor] = []
        self._budget_values: list[int] = []
        self._executed_sum = 0.0
        self._executed_count = 0
        self._wrapped: list[tuple[Any, Callable[..., Any]]] = []

        for name, module in model.thinker.named_modules():
            if "sparsemoeblock" not in module.__class__.__name__.lower():
                continue
            if not all(
                hasattr(module, a) for a in ("gate", "experts", "top_k", "num_experts")
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
        log(
            f"installed CG-Skip on {len(self._wrapped)} MoE layers; band=[{self.layer_lo},{self.layer_hi})"
        )

    @property
    def installed(self) -> int:
        return len(self._wrapped)

    @staticmethod
    def _layer_index(name: str) -> int | None:
        parts = name.split(".")
        for i, part in enumerate(parts[:-1]):
            if part == "layers" and parts[i + 1].isdigit():
                return int(parts[i + 1])
        return None

    def _install(self, module: Any, layer: int) -> None:
        original = module.forward

        def wrapped(hidden_states: torch.Tensor):
            in_band = self.layer_lo <= layer < self.layer_hi
            if self._arm == "native" or not in_band:
                return original(hidden_states)
            return self._budgeted_forward(module, hidden_states)

        module.forward = wrapped
        self._wrapped.append((module, original))

    def _per_token_budget(
        self, routing_weights: torch.Tensor, token_count: int
    ) -> torch.Tensor:
        """routing_weights: (n_tokens, top_k) renormalized top-8 weights."""
        device = routing_weights.device
        if self._force_budget is not None:
            return torch.full(
                (token_count,), int(self._force_budget), dtype=torch.long, device=device
            )
        if self._arm == "skip_all":
            return torch.zeros(token_count, dtype=torch.long, device=device)
        if self._arm.startswith("fixed:"):
            k = int(self._arm.split(":")[1])
            return torch.full((token_count,), k, dtype=torch.long, device=device)
        # CG-Reduce. Token mode assigns each token independently. Layer mode
        # aggregates the same label-free router signal and assigns one budget to
        # the entire layer, avoiding irregular per-token dispatch overhead.
        sig = confidence_signal(routing_weights, int(self.policy["reduce_budget"]))
        layer_mode = self.policy.get("gate_mode", "token") == "layer"
        decision_sig = sig.mean().reshape(1) if layer_mode else sig
        if self._record_stats:
            self._p1_records.append(decision_sig.detach().cpu())
        decision_budget = cg_reduce_budget(
            decision_sig,
            tau_reduce=self._thresholds["tau_reduce"],
            tau_boost=self._thresholds["tau_boost"],
            base_budget=int(self.policy["base_budget"]),
            reduce_budget=int(self.policy["reduce_budget"]),
            boost_budget=int(self.policy["boost_budget"]),
        )
        budget = (
            decision_budget.expand(token_count).clone()
            if layer_mode
            else decision_budget
        )
        # Restrict skip to eligible prefill tokens; if a mask is set, non-eligible
        # tokens keep native top_k. (Default: all prefill tokens eligible.)
        if (
            self._eligible_seq_mask is not None
            and self._eligible_seq_mask.shape[0] == token_count
        ):
            elig = self._eligible_seq_mask.to(device)
            budget = torch.where(
                elig, budget, torch.full_like(budget, self.native_top_k)
            )
        if not self._record_stats:
            self._budget_values.extend(budget.detach().cpu().tolist())
        return budget

    def _budgeted_forward(self, module: Any, hidden_states: torch.Tensor):
        batch, sequence_length, hidden_size = hidden_states.shape
        flat = hidden_states.view(-1, hidden_size)
        router_logits = module.gate(flat)
        routing_weights = torch.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, module.top_k, dim=-1
        )
        token_count = flat.shape[0]

        if token_count != sequence_length * batch or (
            self._eligible_seq_mask is not None
            and token_count != self._eligible_seq_mask.shape[0]
        ):
            # decode step (seqlen 1) or shape mismatch: keep native budget
            budgets = torch.full(
                (token_count,),
                module.top_k,
                dtype=torch.long,
                device=routing_weights.device,
            )
        else:
            budgets = self._per_token_budget(routing_weights, token_count)

        self._executed_sum += float(budgets.sum().item())
        self._executed_count += int(budgets.shape[0])

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
        return final.reshape(batch, sequence_length, hidden_size), router_logits

    @contextlib.contextmanager
    def arm(
        self,
        name: str,
        fixed_budget: int | None = None,
        force_budget: int | None = None,
    ):
        self._arm = name
        self._fixed_budget = fixed_budget
        self._force_budget = force_budget
        self._executed_sum = 0.0
        self._executed_count = 0
        self._budget_values = []
        try:
            yield self
        finally:
            self._arm = "native"
            self._fixed_budget = None
            self._force_budget = None

    def set_thresholds(self, thresholds: dict[str, float]) -> None:
        self._thresholds = {k: float(v) for k, v in thresholds.items()}

    def prepare_sample(self, eligible_mask: torch.Tensor | None) -> None:
        self._eligible_seq_mask = eligible_mask

    def executed_mean_budget(self) -> float | None:
        if self._executed_count == 0:
            return None
        return self._executed_sum / self._executed_count

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

    def close(self) -> None:
        for module, original in self._wrapped:
            module.forward = original
        self._wrapped.clear()


def distribution_js(native_logits: torch.Tensor, arm_logits: torch.Tensor) -> float:
    native_logp = torch.log_softmax(native_logits.double(), dim=-1)
    arm_logp = torch.log_softmax(arm_logits.double(), dim=-1)
    native_p, arm_p = native_logp.exp(), arm_logp.exp()
    mixture_log = (
        (0.5 * (native_p + arm_p)).clamp_min(torch.finfo(torch.float64).tiny).log()
    )
    js = 0.5 * (
        torch.sum(native_p * (native_logp - mixture_log))
        + torch.sum(arm_p * (arm_logp - mixture_log))
    )
    return max(0.0, float(js.item()))


def generate_once(model, processor, inputs, max_new_tokens) -> dict[str, Any]:
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


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL, timeout=20
        ).strip()
    except Exception:
        return "unavailable"


def environment_record(tau, policy, band, benchmark_name) -> dict[str, Any]:
    config_path = MODEL_PATH / "config.json"
    dataset_path = BENCHMARKS[benchmark_name]["dataset"]
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
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "model_config_sha256": sha256_file(config_path)
        if config_path.exists()
        else None,
        "calibrated_thresholds": tau,
        "cgskip_policy": policy,
        "layer_band": list(band) if band else None,
        "benchmark": benchmark_name,
        "video_token_id": VIDEO_TOKEN_ID,
        "native_top_k": NATIVE_TOP_K,
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
    *,
    records,
    video_dir,
    model,
    processor,
    process_mm_info,
    intervention,
    policy,
    n_samples,
) -> dict[str, float]:
    p1_pool: list[Any] = []
    intervention._record_stats = True
    intervention._p1_records = []
    seen = 0
    for i in range(len(records)):
        if seen >= n_samples:
            break
        video_path = (video_dir / str(records[i]["video"])).resolve()
        if not video_path.exists():
            continue
        conv = build_conversation(video_path, records[i])
        text = processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conv, use_audio_in_video=True)
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
        intervention.prepare_sample(None)
        with intervention.arm("cgskip"):
            # Sentinel thresholds so calibration's forced pass keeps base budget
            # (reduce needs sig>=+inf, boost needs sig<=-inf: neither fires).
            intervention.set_thresholds({"tau_reduce": 2.0, "tau_boost": -1.0})
            generate_once(model, processor, inputs, 1)
        for rec in intervention._p1_records:
            p1_pool.append(rec)
        intervention._p1_records = []
        seen += 1
    intervention._record_stats = False
    if not p1_pool:
        raise RuntimeError("threshold calibration collected no router stats")
    sig_all = torch.cat(p1_pool).numpy().astype(np.float64)
    reduce_pct = float(policy["reduce_percentile"])
    boost_pct = float(policy["boost_percentile"])
    thresholds = {
        # reduce the highest-signal reduce_pct% of tokens (mass already captured)
        "tau_reduce": float(np.percentile(sig_all, 100.0 - reduce_pct)),
        # boost the lowest-signal boost_pct% of tokens (mass spread wide)
        "tau_boost": float(np.percentile(sig_all, boost_pct)),
    }
    log(
        f"calibrated thresholds {thresholds} over {seen} samples "
        f"(signal mean {sig_all.mean():.3f}, reduce_budget={policy['reduce_budget']})"
    )
    return thresholds


def _arm_row(res, native, correct, mean_budget) -> dict[str, Any]:
    return {
        "first_token_js_vs_native": distribution_js(
            native["first_logits"], res["first_logits"]
        ),
        "answer": res["answer"],
        "output_text": res["output_text"],
        "answer_flip_vs_native": res["answer"] != native["answer"],
        "is_correct": res["answer"] == correct,
        "end_to_end_latency_ms": res["end_to_end_latency_ms"],
        "mean_executed_budget": mean_budget,
    }


def run_sample(
    *,
    sample_index,
    record,
    video_dir,
    model,
    processor,
    process_mm_info,
    intervention,
    max_new_tokens,
    passthrough_check,
) -> dict[str, Any]:
    video_path = (video_dir / str(record["video"])).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    conv = build_conversation(video_path, record)
    text = processor.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(conv, use_audio_in_video=True)
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
    input_ids_1d = inputs["input_ids"].reshape(-1).detach().cpu()
    n_video = int((input_ids_1d == VIDEO_TOKEN_ID).sum().item())
    correct = str(record["correct_option"]).upper()
    intervention.prepare_sample(None)

    native = generate_once(model, processor, inputs, max_new_tokens)

    passthrough = None
    if passthrough_check:
        with intervention.arm("native"):
            pt = generate_once(model, processor, inputs, max_new_tokens)
        passthrough = distribution_js(native["first_logits"], pt["first_logits"])

    arms: dict[str, Any] = {}
    with intervention.arm("cgskip"):
        res = generate_once(model, processor, inputs, max_new_tokens)
        mean_budget = intervention.executed_mean_budget()
        dispersion = intervention.budget_dispersion()
    arms["cgskip"] = _arm_row(res, native, correct, mean_budget)

    with intervention.arm("skip_all"):
        res = generate_once(model, processor, inputs, max_new_tokens)
        mean_budget = intervention.executed_mean_budget()
    arms["skip_all"] = _arm_row(res, native, correct, mean_budget)

    for arm_name, k in FIXED_ARMS.items():
        with intervention.arm(f"fixed:{k}"):
            res = generate_once(model, processor, inputs, max_new_tokens)
            mean_budget = intervention.executed_mean_budget()
        arms[arm_name] = _arm_row(res, native, correct, mean_budget)

    native_out = {
        "answer": native["answer"],
        "output_text": native["output_text"],
        "is_correct": native["answer"] == correct,
        "end_to_end_latency_ms": native["end_to_end_latency_ms"],
    }
    return {
        "sample_index": sample_index,
        "video": record["video"],
        "correct_option": correct,
        "input_sequence_length": seq_len,
        "n_video_tokens": n_video,
        "passthrough_js_vs_native": passthrough,
        "cgskip_budget_dispersion": dispersion,
        "native": native_out,
        "arms": arms,
    }


def summarize(
    rows, errors, tau, policy, band, blocks, passthrough_max
) -> dict[str, Any]:
    n = len(rows)
    warmup = 5
    arm_names = ["cgskip", "skip_all", *FIXED_ARMS.keys()]

    def latency_array(getter):
        return np.asarray(
            [getter(r) for i, r in enumerate(rows) if i >= warmup], dtype=np.float64
        )

    native_acc = np.asarray(
        [int(bool(r["native"]["is_correct"])) for r in rows], dtype=np.float64
    )
    native_lat = latency_array(lambda r: float(r["native"]["end_to_end_latency_ms"]))
    arm_summary: dict[str, Any] = {
        "native": {
            "accuracy": bootstrap_mean_ci(native_acc, SEED + 1),
            "end_to_end_latency_ms": bootstrap_mean_ci(native_lat, SEED + 2),
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
            [float(r["arms"][arm]["mean_executed_budget"] or np.nan) for r in rows],
            dtype=np.float64,
        )
        arm_summary[arm] = {
            "accuracy": bootstrap_mean_ci(acc, SEED + 10 + i),
            "answer_flip_rate": float(flip.mean()),
            "end_to_end_latency_ms": bootstrap_mean_ci(lat, SEED + 50 + i),
            "mean_executed_budget": float(np.nanmean(budgets)),
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
        acc_diff = paired_diff_ci(cg_acc, arm_acc, SEED + 100 + i)
        lat_diff = paired_diff_ci(cg_lat, arm_lat, SEED + 200 + i)
        acc_not_worse = (
            acc_diff["ci95_low"] is not None and acc_diff["ci95_low"] >= -0.005
        )
        lat_not_worse = (
            lat_diff["ci95_high"] is not None and lat_diff["ci95_high"] <= 0.0
        )
        acc_better = acc_diff["ci95_low"] is not None and acc_diff["ci95_low"] > 0.0
        lat_better = lat_diff["ci95_high"] is not None and lat_diff["ci95_high"] < 0.0
        pareto[f"cgskip_vs_{arm}"] = {
            "accuracy_diff": acc_diff,
            "latency_diff_ms": lat_diff,
            "strictly_dominates": bool(
                acc_not_worse and lat_not_worse and (acc_better or lat_better)
            ),
        }

    controller_ok = blocks == TOTAL_MOE_LAYERS
    passthrough_ok = passthrough_max is not None and passthrough_max < 1e-6
    cg_mean = arm_summary["cgskip"]["mean_executed_budget"]
    total_unique = sum(int(r["cgskip_budget_dispersion"]["unique"]) for r in rows)
    non_degenerate = bool(cg_mean is not None and total_unique > n)
    any_dominates = any(v["strictly_dominates"] for v in pareto.values())

    return {
        "schema_version": "cgskip/mb-new-2-qwen3omni-results",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "experiment_id": "MB-NEW-2-CGSKIP-Qwen3Omni",
        "backbone": "Qwen3-Omni-30B-A3B-Instruct (thinker MoE, top-8/128, 48 layers, no shared experts)",
        "evidence_status": "verified"
        if (n >= 95 and controller_ok and passthrough_ok and non_degenerate)
        else "placeholder",
        "calibrated_thresholds": tau,
        "cgskip_policy": policy,
        "layer_band": list(band) if band else None,
        "completed_samples": n,
        "failed_samples": len(errors),
        "arms": arm_summary,
        "pareto_contrasts": pareto,
        "cgskip_strictly_dominates_any_fixed_k": any_dominates,
        "implementation_gates": {
            "controller_blocks": blocks,
            "controller_blocks_expected": TOTAL_MOE_LAYERS,
            "controller_wiring_passed": controller_ok,
            "native_passthrough_js_max": passthrough_max,
            "native_passthrough_passed": passthrough_ok,
            "cgskip_mean_budget": cg_mean,
            "cgskip_budget_non_degenerate": non_degenerate,
            "passed": controller_ok and passthrough_ok and non_degenerate,
        },
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path)
    p.add_argument(
        "--benchmark",
        type=str,
        default="omnivideobench",
        choices=list(BENCHMARKS.keys()),
        help="Which benchmark to evaluate on.",
    )
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument(
        "--skip-indices",
        type=str,
        default=None,
        help="Comma-separated sample indices to skip (undecodable videos).",
    )
    p.add_argument("--layer-band-frac", type=str, default=None)
    p.add_argument("--base-budget", type=int, default=6)
    p.add_argument("--reduce-budget", type=int, default=4)
    p.add_argument("--boost-budget", type=int, default=8)
    p.add_argument(
        "--reduce-percentile",
        type=float,
        default=30.0,
        help="Fraction of highest-confidence tokens reduced to reduce_budget.",
    )
    p.add_argument(
        "--boost-percentile",
        type=float,
        default=30.0,
        help="Fraction of lowest-confidence tokens boosted to boost_budget.",
    )
    p.add_argument("--calibrate-samples", type=int, default=16)
    p.add_argument("--probe-budget", type=int, default=None)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def self_test() -> None:
    x = torch.tensor([2.0, 1.0, 0.5])
    assert distribution_js(x, x.clone()) < 1e-9
    # routing weights sorted desc; reduce_budget=4 cumulative mass separates tokens
    rw = torch.tensor(
        [
            [0.5, 0.3, 0.1, 0.05, 0.03, 0.02, 0.0, 0.0],  # top4 mass 0.95 -> reduce
            [0.2, 0.18, 0.16, 0.14, 0.12, 0.10, 0.06, 0.04],  # spread -> boost
            [0.3, 0.25, 0.15, 0.1, 0.08, 0.07, 0.03, 0.02],  # mid
        ]
    )
    sig = confidence_signal(rw, reduce_budget=4)
    b = cg_reduce_budget(
        sig,
        tau_reduce=0.9,
        tau_boost=0.7,
        base_budget=6,
        reduce_budget=4,
        boost_budget=8,
    )
    assert int(b[0]) == 4 and int(b[1]) == 8 and int(b[2]) == 6, (sig, b)
    assert parse_answer("The answer is C.") == "C"
    print("MB-NEW-2-CGSKIP-Qwen self-test passed.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --self-test")

    records, video_dir = load_benchmark(args.benchmark)
    log(f"benchmark={args.benchmark} records={len(records)} video_dir={video_dir}")
    n_records = min(len(records), int(args.limit))
    skip = (
        set(int(x) for x in args.skip_indices.split(",") if x.strip())
        if args.skip_indices
        else set()
    )
    if skip:
        log(f"skipping {len(skip)} indices (undecodable videos): {sorted(skip)}")
    selected = [i for i in range(n_records) if i not in skip]

    import random

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_results.jsonl"
    error_path = output_dir / "errors.jsonl"
    summary_path = output_dir / "summary.json"

    policy = {
        "base_budget": int(args.base_budget),
        "reduce_budget": int(args.reduce_budget),
        "boost_budget": int(args.boost_budget),
        "reduce_percentile": float(args.reduce_percentile),
        "boost_percentile": float(args.boost_percentile),
    }

    log("loading Qwen3-Omni model")
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(str(MODEL_PATH), use_fast=False)

    band = None
    if args.layer_band_frac is not None:
        lo_f, hi_f = (float(x) for x in args.layer_band_frac.split(","))
        band = (
            int(np.floor(lo_f * TOTAL_MOE_LAYERS)),
            int(np.floor(hi_f * TOTAL_MOE_LAYERS)),
        )
        log(f"layer band {band} (frac {lo_f},{hi_f})")

    intervention = CGSkipIntervention(model, NATIVE_TOP_K, band)
    intervention.policy = policy

    if args.probe_budget is not None:
        log(f"PROBE MODE: forcing budget={args.probe_budget}")
        for si in selected:
            rec = records[si]
            vp = (video_dir / str(rec["video"])).resolve()
            if not vp.exists():
                continue
            conv = build_conversation(vp, rec)
            text = processor.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=False
            )
            audios, images, videos = process_mm_info(conv, use_audio_in_video=True)
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
            intervention.prepare_sample(None)
            nat = generate_once(model, processor, inputs, args.max_new_tokens)
            with intervention.arm("cgskip", force_budget=args.probe_budget):
                intervention.set_thresholds({"tau_reduce": 2.0, "tau_boost": -1.0})
                pr = generate_once(model, processor, inputs, args.max_new_tokens)
            js = distribution_js(nat["first_logits"], pr["first_logits"])
            log(
                f"probe s{si}: nat_lat={nat['end_to_end_latency_ms']:.0f} "
                f"probe_lat={pr['end_to_end_latency_ms']:.0f} js={js:.4f} "
                f"nat_ans={nat['answer']} probe_ans={pr['answer']}"
            )
        intervention.close()
        return 0

    tau = calibrate_thresholds(
        records=records,
        video_dir=video_dir,
        model=model,
        processor=processor,
        process_mm_info=process_mm_info,
        intervention=intervention,
        policy=policy,
        n_samples=int(args.calibrate_samples),
    )
    intervention.set_thresholds(tau)
    atomic_json(
        output_dir / "environment.json",
        environment_record(tau, policy, band, args.benchmark),
    )

    completed_rows = load_jsonl(raw_path)
    completed = {int(r["sample_index"]) for r in completed_rows}
    errors = load_jsonl(error_path)
    passthrough_done = sum(
        1 for r in completed_rows if r.get("passthrough_js_vs_native") is not None
    )
    passthrough_required = 8
    passthrough_vals = [
        float(r["passthrough_js_vs_native"])
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
                    video_dir=video_dir,
                    model=model,
                    processor=processor,
                    process_mm_info=process_mm_info,
                    intervention=intervention,
                    max_new_tokens=args.max_new_tokens,
                    passthrough_check=passthrough_done < passthrough_required,
                )
                if row.get("passthrough_js_vs_native") is not None:
                    passthrough_done += 1
                    passthrough_vals.append(float(row["passthrough_js_vs_native"]))
                append_jsonl(raw_path, row)
                completed_rows.append(row)
                completed.add(sample_index)
                if ordinal % 10 == 0 or ordinal <= 3:
                    c = row["arms"]["cgskip"]
                    sa = row["arms"]["skip_all"]
                    t4 = row["arms"]["top4"]
                    log(
                        f"[{ordinal}/{len(selected)}] s{sample_index}: "
                        f"cg_bud={c['mean_executed_budget']:.2f} cg_lat={c['end_to_end_latency_ms']:.0f} "
                        f"skipall_lat={sa['end_to_end_latency_ms']:.0f} top4_lat={t4['end_to_end_latency_ms']:.0f} "
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
                log(f"sample {sample_index} OOM; continuing")
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
            tau,
            policy,
            band,
            blocks=intervention.installed,
            passthrough_max=passthrough_max,
        )
        atomic_json(summary_path, final)
        log(
            f"finished: status={final['evidence_status']} completed={final['completed_samples']} "
            f"failed={final['failed_samples']} dominates_any={final['cgskip_strictly_dominates_any_fixed_k']}"
        )
        return 0 if final["evidence_status"] == "verified" else 3
    finally:
        intervention.close()


if __name__ == "__main__":
    raise SystemExit(main())

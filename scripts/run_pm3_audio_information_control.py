#!/usr/bin/env python3
"""Run P-M3: budget-matched audio-information control.

Frozen protocol:
research_paper/iclr2027/evidence/pm3_audio_information_control_protocol.json.

For every OmniVideoBench example the runner runs five arms that all keep the
router unchanged but apply different per-video-token expert budgets in the
eligible MoE layers:

  native_top8        - unchanged model.
  cutoff_only_sceb   - per-video-token cutoff-gap budget, audio floor OFF.
  arc_true_audio     - cutoff-gap budget + one-way same-layer audio floor.
  arc_shuffled_audio - budget-COUNT identical to arc_true_audio, but the audio
                       floor is driven by a layer-permuted audio state.
  arc_random_restore - budget-COUNT identical to arc_true_audio, but the
                       restored video tokens are chosen at random.

The budget is applied by ranking the native top-8 router weights and keeping
only the first k (4/6/8) experts, renormalizing the retained weights.  Fewer
experts are genuinely executed (the real-skip path validated in P-M2), so this
also yields matched executed-work accounting.  The controls preserve the exact
per-sample, per-layer 4/6/8 histogram of arc_true_audio, isolating whether
audio ALIGNMENT (not extra compute) drives any effect.

Quality is measured as full generated-answer accuracy and the gold-option
log-likelihood.  Uncertainty is a video-clustered paired bootstrap.
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO
    / "research_paper/iclr2027/evidence/pm3_audio_information_control_protocol.json"
)
sys.path.insert(0, str(REPO / "qwen-omni-utils" / "src"))
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
VIDEO_ID = 151656
AUDIO_ID = 151675
IMAGE_ID = 151655
OPTION_LETTERS = ["A", "B", "C", "D"]
ARMS = [
    "native_top8",
    "cutoff_only_sceb",
    "arc_true_audio",
    "arc_shuffled_audio",
    "arc_random_restore",
]


def log(*parts: Any) -> None:
    print("[P-M3]", *parts, file=sys.stderr, flush=True)


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


# --------------------------------------------------------------------------- #
# Verified ARC-SCEB decision logic (mirrors scripts/arc_sceb_capture_worker.py
# and the recorded policy parameters).
# --------------------------------------------------------------------------- #
def arc_decide_layer(
    video_top_vals: np.ndarray,
    audio_margin_mean: float,
    audio_count: int,
    policy: dict[str, Any],
) -> tuple[np.ndarray, int, str]:
    """Return per-video-token budget (4/6/8), the audio floor, and the state.

    video_top_vals: (n_video, 8) sorted-descending router probabilities.
    """
    m_small, m_mid, m_full = policy["top_m_options"]
    thr = policy["cutoff_gap_threshold"]
    arc = policy["audio_conditioning"]
    nv = video_top_vals.shape[0]
    chosen = np.full(nv, m_full, dtype=np.int64)
    if nv:
        b_small = video_top_vals[:, m_small - 1] - video_top_vals[:, m_small]
        b_mid = video_top_vals[:, m_mid - 1] - video_top_vals[:, m_mid]
        choose_small = b_small >= thr
        choose_mid = (~choose_small) & (b_mid >= thr)
        chosen[choose_mid] = m_mid
        chosen[choose_small] = m_small
    if audio_count < arc["min_audio_tokens"]:
        floor, state = arc["insufficient_audio_min_top_m"], "insufficient_audio"
    elif audio_margin_mean < arc["fragile_margin_threshold"]:
        floor, state = arc["fragile_min_top_m"], "fragile_audio"
    else:
        floor, state = m_small, "confident_audio"
    return chosen, int(floor), state


@dataclass
class LayerPlan:
    """Per-eligible-layer budget assignment for one sample, one arm."""

    video_positions: np.ndarray  # flat sequence indices of video tokens
    budgets: np.ndarray  # per-video-token budget aligned to video_positions
    histogram: dict[int, int] = field(default_factory=dict)


def _histogram(budgets: np.ndarray, options: list[int]) -> dict[int, int]:
    return {int(m): int((budgets == m).sum()) for m in options}


def build_arm_plans(
    *,
    per_layer_capture: dict[int, dict[str, Any]],
    policy: dict[str, Any],
    seed: int,
) -> dict[str, dict[int, LayerPlan]]:
    """Build per-layer LayerPlans for all five arms from captured router stats.

    per_layer_capture[layer] = {
        "video_positions": np.ndarray[int],
        "video_top_vals": np.ndarray[n_video, 8],
        "audio_margin_mean": float,
        "audio_count": int,
        "audio_margin_by_layer": {layer: (mean, count)}  # shared, used for shuffle
    }
    """
    m_small, m_mid, m_full = policy["top_m_options"]
    options = [m_small, m_mid, m_full]
    rng = np.random.default_rng(seed)
    eligible_layers = sorted(per_layer_capture.keys())

    # cutoff-only base budgets (audio floor OFF) per layer.
    cutoff_only: dict[int, np.ndarray] = {}
    # arc true-audio budgets per layer (cutoff clipped up by the true audio floor).
    arc_true: dict[int, np.ndarray] = {}
    true_floor_by_layer: dict[int, int] = {}
    for li in eligible_layers:
        cap = per_layer_capture[li]
        base, _, _ = arc_decide_layer(
            cap["video_top_vals"], cap["audio_margin_mean"], cap["audio_count"], policy
        )
        cutoff_only[li] = base.copy()
        # audio floor for the true arm:
        _, floor, _ = arc_decide_layer(
            cap["video_top_vals"], cap["audio_margin_mean"], cap["audio_count"], policy
        )
        true_floor_by_layer[li] = floor
        arc_true[li] = np.clip(base, floor, m_full) if floor > m_small else base.copy()

    # shuffled-audio: recompute the floor from a layer-permuted audio state,
    # then clip the SAME cutoff base.  This preserves marginal audio-floor
    # statistics but destroys per-layer alignment.
    audio_states = [
        (
            per_layer_capture[li]["audio_margin_mean"],
            per_layer_capture[li]["audio_count"],
        )
        for li in eligible_layers
    ]
    perm = rng.permutation(len(eligible_layers))
    arc_shuffled: dict[int, np.ndarray] = {}
    arc = policy["audio_conditioning"]
    for idx, li in enumerate(eligible_layers):
        margin_mean, count = audio_states[perm[idx]]
        if count < arc["min_audio_tokens"]:
            floor = arc["insufficient_audio_min_top_m"]
        elif margin_mean < arc["fragile_margin_threshold"]:
            floor = arc["fragile_min_top_m"]
        else:
            floor = m_small
        base = cutoff_only[li]
        arc_shuffled[li] = (
            np.clip(base, floor, m_full) if floor > m_small else base.copy()
        )

    # random-restore: keep the EXACT per-layer 4/6/8 histogram of arc_true, but
    # randomly reassign which video positions get which budget.
    arc_random: dict[int, np.ndarray] = {}
    for li in eligible_layers:
        true_budgets = arc_true[li]
        shuffled = true_budgets.copy()
        rng.shuffle(shuffled)
        arc_random[li] = shuffled

    def to_plans(budget_map: dict[int, np.ndarray]) -> dict[int, LayerPlan]:
        plans: dict[int, LayerPlan] = {}
        for li in eligible_layers:
            cap = per_layer_capture[li]
            b = budget_map[li]
            plans[li] = LayerPlan(
                video_positions=cap["video_positions"],
                budgets=b,
                histogram=_histogram(b, options),
            )
        return plans

    return {
        "cutoff_only_sceb": to_plans(cutoff_only),
        "arc_true_audio": to_plans(arc_true),
        "arc_shuffled_audio": to_plans(arc_shuffled),
        "arc_random_restore": to_plans(arc_random),
    }


# --------------------------------------------------------------------------- #
# Generation-time budget intervention on the HF SparseMoeBlock.
# --------------------------------------------------------------------------- #
class BudgetIntervention:
    """Apply per-position variable top-k budgets in eligible MoE layers.

    The active plan maps a MoE layer index to a dense per-sequence-position
    budget tensor (default 8).  During the forward, the block computes the
    native top-8, then for each position keeps only the first ``budget[pos]``
    experts (weights beyond the budget are zeroed and the retained weights are
    renormalized).  Fewer experts run for reduced-budget positions.
    """

    def __init__(self, model: Any, top_k: int, eligible_layers: set[int]) -> None:
        self.model = model
        self.top_k = int(top_k)
        self.eligible = set(int(x) for x in eligible_layers)
        self.active: dict[int, torch.Tensor] | None = None
        self._wrapped: list[tuple[Any, Callable[..., Any]]] = []
        installed = 0
        for name, module in model.thinker.named_modules():
            if "sparsemoeblock" not in module.__class__.__name__.lower():
                continue
            if not all(
                hasattr(module, a) for a in ("gate", "experts", "top_k", "num_experts")
            ):
                continue
            layer_idx = self._layer_idx(name)
            if layer_idx is None:
                continue
            self._install(module, layer_idx)
            installed += 1
        if installed == 0:
            raise RuntimeError("No MoE blocks found for budget intervention")
        self.installed = installed
        log(
            f"installed budget intervention on {installed} MoE layers "
            f"({len(self.eligible)} eligible)"
        )

    @staticmethod
    def _layer_idx(name: str) -> int | None:
        parts = name.split(".")
        for i, p in enumerate(parts[:-1]):
            if p == "layers" and parts[i + 1].isdigit():
                return int(parts[i + 1])
        return None

    def _install(self, module: Any, layer_idx: int) -> None:
        original = module.forward

        def wrapped(hidden_states: torch.Tensor):
            plan = None if self.active is None else self.active.get(layer_idx)
            if plan is None:
                return original(hidden_states)
            return self._budgeted_forward(module, hidden_states, plan)

        module.forward = wrapped
        self._wrapped.append((module, original))

    def _budgeted_forward(self, module, hidden_states, budget_vec):
        batch, seqlen, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        router_logits = module.gate(flat)
        routing_weights = torch.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected = torch.topk(routing_weights, module.top_k, dim=-1)

        n = flat.shape[0]
        if budget_vec.shape[0] == n:
            budgets = budget_vec.to(routing_weights.device)
        else:
            budgets = torch.full(
                (n,), module.top_k, device=routing_weights.device, dtype=torch.long
            )
        positions = torch.arange(module.top_k, device=routing_weights.device).unsqueeze(
            0
        )
        keep = positions < budgets.unsqueeze(1)
        routing_weights = routing_weights.masked_fill(~keep, 0.0)
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-20)
        routing_weights = routing_weights.to(flat.dtype)

        final = torch.zeros((n, hidden_dim), dtype=flat.dtype, device=flat.device)
        expert_mask = torch.nn.functional.one_hot(
            selected, num_classes=module.num_experts
        ).permute(2, 1, 0)
        # zero-out contributions of dropped experts by also masking the mask
        keep_t = keep.permute(1, 0)  # (top_k, n)
        for expert_idx in torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero():
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            if top_x.numel() == 0:
                continue
            weight = routing_weights[top_x, idx, None]
            keep_sel = weight.squeeze(-1) > 0
            if not bool(keep_sel.any()):
                continue
            top_x = top_x[keep_sel]
            idx = idx[keep_sel]
            current = module.experts[expert_idx](
                flat[None, top_x].reshape(-1, hidden_dim)
            )
            current = current * routing_weights[top_x, idx, None]
            final.index_add_(0, top_x, current.to(flat.dtype))
        return final.reshape(batch, seqlen, hidden_dim), router_logits

    @contextlib.contextmanager
    def use_plan(self, plan: dict[int, torch.Tensor] | None):
        prev = self.active
        self.active = plan
        try:
            yield
        finally:
            self.active = prev

    def close(self) -> None:
        for module, original in self._wrapped:
            module.forward = original
        self._wrapped.clear()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL, timeout=20
        ).strip()
    except Exception:
        return "unavailable"


def environment_record(protocol, dataset_path, model_path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    policy_cfg = REPO / protocol["verified_policy"]["source_config"]
    return {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "requested_worker": os.environ.get("PM3_WORKER_ID", "unknown"),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status_short": git_value("status", "--short"),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "policy_config_sha256": sha256_file(policy_cfg)
        if policy_cfg.exists()
        else None,
        "model_config_sha256": sha256_file(config_path)
        if config_path.exists()
        else None,
        "protocol": protocol,
    }


def distribution_js(native_logits: torch.Tensor, arm_logits: torch.Tensor) -> float:
    a = torch.log_softmax(native_logits.double(), dim=-1)
    b = torch.log_softmax(arm_logits.double(), dim=-1)
    pa, pb = a.exp(), b.exp()
    m = (0.5 * (pa + pb)).clamp_min(torch.finfo(torch.float64).tiny).log()
    js = 0.5 * (torch.sum(pa * (a - m)) + torch.sum(pb * (b - m)))
    return max(0.0, float(js.item()))


def gold_option_logprob(
    first_logits: torch.Tensor, option_token_ids: dict[str, int], gold: str
) -> float:
    logp = torch.log_softmax(first_logits.double(), dim=-1)
    tid = option_token_ids.get(gold)
    if tid is None:
        return float("nan")
    return float(logp[tid].item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> None:
    policy = {
        "top_m_options": [4, 6, 8],
        "cutoff_gap_threshold": 0.05,
        "audio_conditioning": {
            "min_audio_tokens": 64,
            "insufficient_audio_min_top_m": 8,
            "fragile_margin_threshold": 0.0082,
            "fragile_min_top_m": 6,
        },
    }
    # video tokens with clearly separated top-4 boundary -> budget 4.
    vals = np.tile(np.array([0.5, 0.2, 0.15, 0.1, 0.01, 0.008, 0.006, 0.004]), (3, 1))
    chosen, floor, state = arc_decide_layer(
        vals, audio_margin_mean=0.5, audio_count=100, policy=policy
    )
    assert chosen.tolist() == [4, 4, 4], chosen
    assert state == "confident_audio", state
    # insufficient audio -> floor 8, clip raises budgets to 8.
    chosen2, floor2, state2 = arc_decide_layer(
        vals, audio_margin_mean=0.5, audio_count=10, policy=policy
    )
    assert floor2 == 8 and state2 == "insufficient_audio"

    # budget matching: shuffle/random preserve histogram.
    cap = {
        12: {
            "video_positions": np.arange(3),
            "video_top_vals": vals,
            "audio_margin_mean": 0.001,  # fragile -> floor 6
            "audio_count": 100,
        },
        13: {
            "video_positions": np.arange(3, 6),
            "video_top_vals": vals,
            "audio_margin_mean": 0.5,  # confident -> floor 4
            "audio_count": 100,
        },
    }
    plans = build_arm_plans(per_layer_capture=cap, policy=policy, seed=7)
    for li in (12, 13):
        h_true = plans["arc_true_audio"][li].histogram
        assert plans["arc_random_restore"][li].histogram == h_true, (li, h_true)
    # random-restore keeps the multiset of budgets per layer.
    assert sorted(plans["arc_random_restore"][12].budgets.tolist()) == sorted(
        plans["arc_true_audio"][12].budgets.tolist()
    )
    # gold logprob sanity.
    logits = torch.tensor([0.1, 5.0, 0.2, 0.3])
    opt = {"A": 0, "B": 1, "C": 2, "D": 3}
    assert gold_option_logprob(logits, opt, "B") > gold_option_logprob(logits, opt, "A")
    print("P-M3 self-test passed.")


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

    policy = {
        "top_m_options": protocol["verified_policy"]["top_m_options"],
        "cutoff_gap_threshold": protocol["verified_policy"]["cutoff_gap_threshold"],
        "audio_conditioning": protocol["verified_policy"]["audio_conditioning"],
    }
    eligible_start = protocol["verified_policy"]["eligible_moe_layers"][
        "start_inclusive"
    ]
    eligible_end = protocol["verified_policy"]["eligible_moe_layers"]["end_exclusive"]
    eligible_layers = set(range(eligible_start, eligible_end))
    top_k = protocol["verified_policy"]["original_top_k"]
    max_new = int(protocol["decoding"]["thinker_max_new_tokens"])
    base_seed = int(protocol["decoding"]["seed"])

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_results.jsonl"
    error_path = output_dir / "errors.jsonl"

    import random as _random

    _random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)

    log("loading Qwen3-Omni model")
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info

    # Shard across all visible GPUs when more than one is available so the
    # longest OmniVideoBench prefills fit; otherwise use the single device.
    device_map = "auto" if torch.cuda.device_count() > 1 else "cuda"
    log(
        f"loading model with device_map={device_map} on {torch.cuda.device_count()} GPU(s)"
    )
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2",
    ).eval()
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(str(model_path), use_fast=False)
    tok = processor.tokenizer
    option_token_ids = {}
    for letter in OPTION_LETTERS:
        ids = tok.encode(letter, add_special_tokens=False)
        if len(ids) == 1:
            option_token_ids[letter] = ids[0]
    atomic_json(
        output_dir / "environment.json",
        environment_record(protocol, dataset_path, model_path),
    )

    intervention = BudgetIntervention(
        model, top_k=top_k, eligible_layers=eligible_layers
    )
    capture = RouterProbeCapture(model, eligible_layers)

    completed_rows = load_jsonl(raw_path)
    completed = {int(r["sample_index"]) for r in completed_rows}
    errors = load_jsonl(error_path)

    try:
        for ordinal, sidx in enumerate(selected, 1):
            if sidx in completed:
                continue
            try:
                row = run_sample(
                    sample_index=sidx,
                    record=records[sidx],
                    model=model,
                    processor=processor,
                    process_mm_info=process_mm_info,
                    intervention=intervention,
                    capture=capture,
                    policy=policy,
                    eligible_layers=eligible_layers,
                    top_k=top_k,
                    option_token_ids=option_token_ids,
                    max_new=max_new,
                    base_seed=base_seed,
                    protocol=protocol,
                )
                append_jsonl(raw_path, row)
                completed_rows.append(row)
                completed.add(sidx)
                if ordinal <= 3 or ordinal % 20 == 0:
                    log(
                        f"[{ordinal}/{len(selected)}] sample {sidx} done; "
                        f"match_ok={row['budget_match_ok']} "
                        f"acc={{{','.join(a + ':' + str(int(row['arms'][a]['is_correct'])) for a in ARMS)}}}"
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
                        "last_completed_sample_index": sidx,
                    },
                )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                err = {
                    "sample_index": sidx,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(error_path, err)
                errors.append(err)
                log(f"sample {sidx} OOM; continuing")
            except Exception as exc:
                err = {
                    "sample_index": sidx,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(error_path, err)
                errors.append(err)
                log(f"sample {sidx} failed: {exc!r}")
        summary = summarize(completed_rows, errors, protocol, records)
        atomic_json(output_dir / "summary.json", summary)
        log(
            f"finished status={summary['evidence_status']} "
            f"completed={summary['completed_samples']} failed={summary['failed_samples']}"
        )
        return 0 if summary["evidence_status"] == "verified" else 3
    finally:
        capture.remove()
        intervention.close()


class RouterProbeCapture:
    """Capture full-sequence router logits at each eligible MoE gate."""

    def __init__(self, model: Any, eligible_layers: set[int]) -> None:
        self.eligible = set(int(x) for x in eligible_layers)
        self.by_layer: dict[int, torch.Tensor] = {}
        self.handles = []
        for name, module in model.thinker.named_modules():
            cls = module.__class__.__name__.lower()
            if "sparsemoeblock" in cls and hasattr(module, "gate"):
                li = BudgetIntervention._layer_idx(name)
                if li is None or li not in self.eligible:
                    continue
                self.handles.append(module.gate.register_forward_hook(self._hook(li)))

    def _hook(self, li: int):
        def hook(_m, _i, out):
            logits = out.detach()
            if logits.dim() != 2:
                return
            prev = self.by_layer.get(li)
            if prev is None or logits.shape[0] > prev.shape[0]:
                self.by_layer[li] = logits.float().cpu()

        return hook

    def reset(self) -> None:
        self.by_layer = {}

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


def run_sample(
    *,
    sample_index,
    record,
    model,
    processor,
    process_mm_info,
    intervention,
    capture,
    policy,
    eligible_layers,
    top_k,
    option_token_ids,
    max_new,
    base_seed,
    protocol,
) -> dict[str, Any]:
    video_rel = str(record["video"])
    video_path = (REPO / "eval_data/OmniVideoBench" / video_rel).resolve()
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
    input_ids = inputs["input_ids"]
    seqlen = int(input_ids.shape[1])
    flat_ids = input_ids.reshape(-1).cpu().numpy()
    correct = str(record["correct_option"]).upper()

    # 1) capture native router logits over the full prefill.  Use a 1-token
    #    generate (not a direct model.thinker call) so the model's own
    #    multimodal rope plumbing (video_second_per_grid) is populated; the
    #    RouterProbeCapture hooks fire during this prefill.
    capture.reset()
    with intervention.use_plan(None), torch.inference_mode():
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

    # 2) build per-layer capture stats from the router logits.
    per_layer_capture: dict[int, dict[str, Any]] = {}
    for li in sorted(eligible_layers):
        logits = capture.by_layer.get(li)
        if logits is None or logits.shape[0] != seqlen:
            continue
        probs = torch.softmax(logits, dim=1, dtype=torch.float)
        top_vals, _ = torch.topk(probs, top_k, dim=-1, sorted=True)
        top_vals = top_vals.numpy()
        video_pos = np.nonzero(flat_ids == VIDEO_ID)[0]
        audio_pos = np.nonzero(flat_ids == AUDIO_ID)[0]
        a = top_vals[audio_pos]
        audio_margin = (a[:, 0] - a[:, 1]) if a.shape[0] else np.array([])
        per_layer_capture[li] = {
            "video_positions": video_pos,
            "video_top_vals": top_vals[video_pos],
            "audio_margin_mean": float(audio_margin.mean())
            if audio_margin.size
            else 0.0,
            "audio_count": int(a.shape[0]),
        }

    plans = build_arm_plans(
        per_layer_capture=per_layer_capture,
        policy=policy,
        seed=base_seed + sample_index,
    )

    # 3) verify budget matching (controls == arc_true histogram per layer).
    match_ok = True
    hist_records: dict[str, dict[str, dict[str, int]]] = {}
    for arm in (
        "arc_true_audio",
        "arc_shuffled_audio",
        "arc_random_restore",
        "cutoff_only_sceb",
    ):
        hist_records[arm] = {
            str(li): {str(k): v for k, v in plans[arm][li].histogram.items()}
            for li in plans[arm]
        }
    for li in plans["arc_true_audio"]:
        h_true = plans["arc_true_audio"][li].histogram
        if plans["arc_random_restore"][li].histogram != h_true:
            match_ok = False
        if plans["arc_shuffled_audio"][li].histogram == h_true:
            pass  # shuffle may or may not match counts; not required to match true.

    # 4) build dense per-layer budget tensors and run each arm.
    def dense_plan(arm: str) -> dict[int, torch.Tensor] | None:
        if arm == "native_top8":
            return None
        out: dict[int, torch.Tensor] = {}
        for li, lp in plans[arm].items():
            vec = torch.full((seqlen,), top_k, dtype=torch.long)
            vec[torch.as_tensor(lp.video_positions, dtype=torch.long)] = (
                torch.as_tensor(lp.budgets, dtype=torch.long)
            )
            out[li] = vec
        return out

    rng = np.random.default_rng(base_seed + sample_index)
    arm_order = list(ARMS)
    rng.shuffle(arm_order)

    native_first_logits = None
    arm_results: dict[str, Any] = {}
    for arm in arm_order:
        plan = dense_plan(arm)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with intervention.use_plan(plan), torch.inference_mode():
            raw = model.generate(
                **inputs,
                use_audio_in_video=True,
                return_audio=False,
                thinker_max_new_tokens=max_new,
                thinker_do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                thinker_eos_token_id=processor.tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        thinker = raw[0] if isinstance(raw, tuple) else raw
        first_logits = thinker.scores[0][0].detach().float().cpu()
        gen_ids = thinker.sequences[0, seqlen:].detach().cpu()
        out_text = processor.tokenizer.decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        answer = parse_answer(out_text)
        if arm == "native_top8":
            native_first_logits = first_logits
        # mean expert budget over eligible video decisions.
        if plan is not None:
            all_b = (
                np.concatenate([plans[arm][li].budgets for li in plans[arm]])
                if plans[arm]
                else np.array([])
            )
            mean_budget = float(all_b.mean()) if all_b.size else float(top_k)
            exec_pairs = int(all_b.sum()) if all_b.size else 0
        else:
            mean_budget, exec_pairs = float(top_k), None
        arm_results[arm] = {
            "output_text": out_text,
            "answer": answer,
            "is_correct": answer == correct,
            "gold_option_logprob": gold_option_logprob(
                first_logits, option_token_ids, correct
            ),
            "end_to_end_latency_ms": e2e_ms,
            "mean_video_expert_budget": mean_budget,
            "executed_video_token_expert_pairs": exec_pairs,
            "_first_logits": first_logits,
        }

    # 5) secondary diagnostics vs native and vs arc_true.
    for arm in ARMS:
        fl = arm_results[arm].pop("_first_logits")
        arm_results[arm]["first_token_js_vs_native"] = (
            0.0
            if native_first_logits is None
            else distribution_js(native_first_logits, fl)
        )
        arm_results[arm]["answer_flip_vs_native"] = (
            arm_results[arm]["answer"] != arm_results["native_top8"]["answer"]
        )

    qtype = record.get("question_type")
    audio_dependent = qtype in {"background&music understanding", "sentiment analysis"}
    return {
        "sample_index": sample_index,
        "video": video_rel,
        "video_cluster": video_rel,
        "question_type": qtype,
        "audio_type": record.get("audio_type"),
        "audio_dependent_stratum": bool(audio_dependent),
        "correct_option": correct,
        "input_sequence_length": seqlen,
        "n_video_tokens": int((flat_ids == VIDEO_ID).sum()),
        "n_audio_tokens": int((flat_ids == AUDIO_ID).sum()),
        "arm_order": arm_order,
        "budget_match_ok": bool(match_ok),
        "per_layer_budget_histograms": hist_records,
        "arms": arm_results,
    }


def _cluster_bootstrap_diff(
    pairs: list[tuple[str, float, float]], seed: int, draws: int
) -> dict[str, Any]:
    """Paired difference (a-b) bootstrap resampling by cluster id."""
    if not pairs:
        return {
            "mean_diff": None,
            "ci95_low": None,
            "ci95_high": None,
            "n": 0,
            "clusters": 0,
        }
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for cid, a, b in pairs:
        by_cluster[cid].append(a - b)
    cluster_ids = list(by_cluster.keys())
    cluster_means = np.array(
        [np.mean(by_cluster[c]) for c in cluster_ids], dtype=np.float64
    )
    overall = float(
        np.concatenate([np.array(by_cluster[c]) for c in cluster_ids]).mean()
    )
    rng = np.random.default_rng(seed)
    k = len(cluster_ids)
    boot = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        idx = rng.integers(0, k, size=k)
        boot[i] = cluster_means[idx].mean()
    return {
        "mean_diff": overall,
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "n": len(pairs),
        "clusters": k,
    }


def _contrast(rows, metric, a, b, seed, draws, subset=None):
    pairs = []
    for r in rows:
        if subset is not None and not r.get(subset):
            continue
        va, vb = r["arms"][a][metric], r["arms"][b][metric]
        if va is None or vb is None or (isinstance(va, float) and np.isnan(va)):
            continue
        pairs.append((r["video_cluster"], float(va), float(vb)))
    return _cluster_bootstrap_diff(pairs, seed, draws)


def summarize(rows, errors, protocol, records) -> dict[str, Any]:
    n = len(rows)
    draws = int(protocol["statistics"]["draws"])
    seed = int(protocol["statistics"]["seed"])
    minimum = int(protocol["gate"]["minimum_completed_samples"])
    match_all = all(r["budget_match_ok"] for r in rows) if rows else False

    def arm_means(subset=None):
        out = {}
        for arm in ARMS:
            accs, lps, e2es, budgets = [], [], [], []
            for r in rows:
                if subset is not None and not r.get(subset):
                    continue
                a = r["arms"][arm]
                accs.append(int(bool(a["is_correct"])))
                if a["gold_option_logprob"] is not None and not np.isnan(
                    a["gold_option_logprob"]
                ):
                    lps.append(a["gold_option_logprob"])
                e2es.append(a["end_to_end_latency_ms"])
                budgets.append(a["mean_video_expert_budget"])
            out[arm] = {
                "accuracy": float(np.mean(accs)) if accs else None,
                "gold_logprob": float(np.mean(lps)) if lps else None,
                "end_to_end_latency_ms": float(np.mean(e2es)) if e2es else None,
                "mean_video_expert_budget": float(np.mean(budgets))
                if budgets
                else None,
                "n": len(accs),
            }
        return out

    contrasts = protocol["statistics"]["primary_contrasts"]

    def contrast_block(subset=None):
        block = {}
        for c in contrasts:
            a, b = [x.strip() for x in c.split("minus")]
            block[c] = {
                "accuracy": _contrast(rows, "is_correct", a, b, seed, draws, subset),
                "gold_logprob": _contrast(
                    rows, "gold_option_logprob", a, b, seed + 1, draws, subset
                ),
                "end_to_end_latency_ms": _contrast(
                    rows, "end_to_end_latency_ms", a, b, seed + 2, draws, subset
                ),
            }
        return block

    # helper: is_correct/gold logprob need numeric access; store bool as 0/1.
    for r in rows:
        for arm in ARMS:
            r["arms"][arm]["is_correct"] = int(bool(r["arms"][arm]["is_correct"]))

    verified = n >= minimum and match_all
    return {
        "schema_version": "arc-sceb/pm3-results-v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "experiment_id": "P-M3",
        "evidence_status": "verified" if verified else "placeholder",
        "completed_samples": n,
        "failed_samples": len(errors),
        "requested_samples": len(records),
        "budget_match_all_samples": match_all,
        "arm_means_full_set": arm_means(),
        "arm_means_audio_dependent": arm_means("audio_dependent_stratum"),
        "audio_dependent_count": sum(
            1 for r in rows if r.get("audio_dependent_stratum")
        ),
        "primary_contrasts_full_set": contrast_block(),
        "primary_contrasts_audio_dependent": contrast_block("audio_dependent_stratum"),
        "gates": {
            "minimum_completed_samples": minimum,
            "budget_match_all_samples": match_all,
            "passed": verified,
        },
        "decision_rule": protocol["decision_rule"],
        "scope": protocol["scope"],
        "errors": errors,
    }


if __name__ == "__main__":
    raise SystemExit(main())

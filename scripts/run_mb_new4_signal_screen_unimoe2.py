#!/usr/bin/env python3
"""MB-NEW-4 multi-signal SCREEN on Uni-MoE-2.0-Omni (fast backbone).

Exhaustively screens all leakage-free router-geometry signals in
scripts/mb_new4_router_signals.py over a small sample budget. For each signal it
runs a SCEB arm (entropy/dominance/etc. mapped to {2,3,4} via calibrated
quantiles) and a budget-matched SHUFFLED control (exact per-token budget multiset
permuted within each layer group, so it is budget/histogram-matched but
signal-destroyed). A signal "works" if SCEB flip_rate <= shuffled flip_rate
(direction correct). Survivors graduate to the slow Qwen backbone.

Leakage-free: signals read only router probabilities and token position.
Single-checkpoint single-H20 deterministic inference. Screen only, not evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
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
sys.path.insert(0, "/home/tiger/Uni-MoE/Uni-MoE-2")
sys.path.insert(0, str(REPO / "scripts"))

from mb_new4_router_signals import ALL_SIGNALS, budget_from_quantiles, signal_values  # noqa: E402

MODEL_PATH = str(REPO / "models/Uni-MoE-2.0-Omni")
DATASET = REPO / "eval_data/OmniVideoBench/data.json"
VIDEO_TOKEN_ID = 151656
CANDIDATES = [2, 3, 4]
SEED = 20260906


def log(*parts: Any) -> None:
    print("[MB4-SCREEN-UniMoE]", *parts, file=sys.stderr, flush=True)


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
    for name in ("figure", "plot", "savefig", "close", "clf", "show"):
        setattr(pyplot, name, lambda *a, **k: None)
    mpl.pyplot = pyplot
    sys.modules.setdefault("matplotlib", mpl)
    sys.modules.setdefault("matplotlib.pyplot", pyplot)


def parse_answer(text: str) -> str | None:
    upper = str(text).strip().upper()
    m = re.search(r"(?:^|[^A-Z])([A-D])(?:[^A-Z]|$)", upper)
    if m:
        return m.group(1)
    for c in upper:
        if c in "ABCD":
            return c
    return None


def build_conversation(
    video_path: Path, record: dict[str, Any]
) -> list[dict[str, Any]]:
    options = "\n".join(str(o) for o in record["options"])
    prompt = f"{record['question']}\nOptions:\n{options}\nAnswer with only the option letter."
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


def expand_placeholders(t: str) -> str:
    return (
        t.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        .replace("<audio>", "<|audio_start|><|audio_pad|><|audio_end|>")
        .replace("<video>", "<|vision_start|><|video_pad|><|vision_end|>")
    )


class MultiSignalController:
    """Wraps GRIN MoE blocks; applies a chosen signal's per-token budget to
    eligible video tokens in the middle band, or a shuffled/native control."""

    def __init__(self, model, expected_blocks, layer_band):
        self.candidate_budgets = CANDIDATES
        self.layer_lo, self.layer_hi = layer_band
        self._blocks, self._orig_forward, self._orig_top_p, self._layer_of = (
            [],
            [],
            [],
            [],
        )
        moe = [
            (n, m)
            for n, m in model.named_modules()
            if m.__class__.__name__ == "GRINMoESparseMoeBlock"
        ]
        if len(moe) != expected_blocks:
            raise RuntimeError(
                f"found {len(moe)} MoE blocks, expected {expected_blocks}"
            )
        for name, module in moe:
            li = self._layer_idx(name)
            self._blocks.append(module)
            self._layer_of.append(li)
            self._orig_top_p.append(float(module.mlp_dynamic_top_p))
        self._arm = "native"  # "native" | "sceb" | "shuffled"
        self._signal = None
        self._elig_mask = None
        self._quantiles: dict[str, tuple[float, float]] = {}
        self._sceb_budgets_by_layer: dict[int, torch.Tensor] = {}
        self._shuffle_perm: dict[int, torch.Tensor] | None = None
        self._record = False
        self._sig_records: list[dict[str, Any]] = []
        self._exec_sum = 0.0
        self._exec_n = 0
        self._install()
        log(
            f"controller on {len(self._blocks)} blocks; band=[{self.layer_lo},{self.layer_hi})"
        )

    @staticmethod
    def _layer_idx(name):
        parts = name.split(".")
        for i, p in enumerate(parts[:-1]):
            if p == "layers" and parts[i + 1].isdigit():
                return int(parts[i + 1])
        return None

    def _install(self):
        for i, (block, layer) in enumerate(zip(self._blocks, self._layer_of)):
            self._orig_forward.append(block.forward)
            block.forward = self._make_forward(block, layer, i)

    def close(self):
        for block, orig, tp in zip(self._blocks, self._orig_forward, self._orig_top_p):
            block.forward = orig
            block.mlp_dynamic_top_p = tp

    def _in_band(self, layer):
        return self.layer_lo <= layer < self.layer_hi

    def _make_forward(self, block, layer, idx):
        controller = self

        def forward(hidden_states, attention_mask, aux_balance_weight):
            orig = controller._orig_forward[idx]
            active = (
                controller._arm != "native"
                and controller._in_band(layer)
                and controller._elig_mask is not None
            )
            if not active:
                return orig(hidden_states, attention_mask, aux_balance_weight)
            bsz, seqlen, hdim = hidden_states.shape
            if seqlen != controller._elig_mask.shape[0]:
                return orig(hidden_states, attention_mask, aux_balance_weight)
            flat = hidden_states.view(-1, hdim)
            full_logits = block.gate(flat)
            dyn_logits = full_logits[:, : block.mlp_dynamic_expert_num]
            dyn_probs = torch.softmax(dyn_logits.float(), dim=-1)
            k = dyn_probs.shape[1]
            elig = controller._elig_mask.to(dyn_probs.device).view(-1)

            from uni_moe.model.modeling_qwen_grin_moe import (
                dynamic_expert_selection as nsel,
            )

            budget = nsel(dyn_logits, controller._orig_top_p[idx]).to(torch.int64)

            if controller._record and elig.any():
                ev = dyn_probs[elig]
                # Subsample eligible tokens per layer to bound calibration cost.
                if ev.shape[0] > 512:
                    sel = torch.randperm(ev.shape[0], device=ev.device)[:512]
                    ev = ev[sel]
                sv = signal_values(ev, k)
                controller._sig_records.append(
                    {"layer": layer, **{s: sv[s].detach().cpu() for s in ALL_SIGNALS}}
                )

            if controller._arm == "sceb":
                sig = controller._signal
                q_low, q_high = controller._quantiles[sig]
                sv = signal_values(dyn_probs, k)[sig]
                sceb_b = budget_from_quantiles(
                    sv, q_low, q_high, controller.candidate_budgets
                )
                budget = torch.where(elig, sceb_b, budget)
                controller._sceb_budgets_by_layer[layer] = sceb_b[elig].detach().cpu()
            elif controller._arm == "shuffled":
                base = controller._sceb_budgets_by_layer.get(layer)
                perm = (
                    controller._shuffle_perm.get(layer)
                    if controller._shuffle_perm
                    else None
                )
                if base is not None and perm is not None:
                    sh = base[perm].to(budget.device)
                    ii = torch.nonzero(elig, as_tuple=True)[0]
                    if ii.shape[0] == sh.shape[0]:
                        budget[ii] = sh

            if elig.any():
                controller._exec_sum += float(budget[elig].sum().item())
                controller._exec_n += int(elig.sum().item())

            gm = sys.modules["uni_moe.model.modeling_qwen_grin_moe"]
            orig_sel = gm.dynamic_expert_selection
            try:
                gm.dynamic_expert_selection = lambda logits, top_p: budget.to(
                    logits.device
                )
                block.mlp_dynamic_top_p = controller._orig_top_p[idx]
                out = orig(hidden_states, attention_mask, aux_balance_weight)
            finally:
                gm.dynamic_expert_selection = orig_sel
                block.mlp_dynamic_top_p = controller._orig_top_p[idx]
            return out

        return forward

    def prepare_sample(self, input_ids_1d):
        self._elig_mask = (input_ids_1d == VIDEO_TOKEN_ID).bool().cpu()
        self._sceb_budgets_by_layer = {}

    def reset_exec(self):
        self._exec_sum = 0.0
        self._exec_n = 0

    def exec_mean(self):
        return None if self._exec_n == 0 else self._exec_sum / self._exec_n

    @contextlib.contextmanager
    def arm(self, name, signal=None, shuffle_seed=None):
        self._arm, self._signal = name, signal
        self._shuffle_perm = None
        self.reset_exec()
        if name == "shuffled" and shuffle_seed is not None:
            g = torch.Generator().manual_seed(int(shuffle_seed))
            self._shuffle_perm = {
                l: torch.randperm(v.shape[0], generator=g)
                for l, v in self._sceb_budgets_by_layer.items()
            }
        try:
            yield
        finally:
            self._arm, self._signal, self._shuffle_perm = "native", None, None


@dataclass
class Gen:
    answer: str | None
    text: str


def generate_once(model, processor, inputs):
    with torch.inference_mode():
        raw = model.generate(
            **inputs,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
            do_sample=False,
            num_beams=1,
            max_new_tokens=4,
            return_dict_in_generate=True,
            output_scores=True,
        )
    input_len = int(inputs["input_ids"].shape[-1])
    gen_ids = raw.sequences[0, input_len:].detach().cpu()
    text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return Gen(answer=parse_answer(text), text=text)


def build_inputs(processor, process_mm_info, conv):
    texts = expand_placeholders(
        processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    )
    imgs, vids, auds = process_mm_info(conv)
    inputs = processor(
        text=texts,
        images=imgs,
        videos=vids,
        audios=auds,
        padding=True,
        return_tensors="pt",
    )
    if "second_grid_ts" in inputs:
        inputs["second_per_grid_ts"] = inputs["second_grid_ts"]
        del inputs["second_grid_ts"]
    inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
    return inputs


def move(inputs, model):
    inputs = inputs.to(device=model.device)
    for k, v in list(inputs.items()):
        if k in ("pixel_values", "pixel_values_videos", "audio_features"):
            inputs[k] = v.to(dtype=torch.bfloat16)
    return inputs


def calibrate(records, model, processor, process_mm_info, controller, n_cal):
    pooled = {s: [] for s in ALL_SIGNALS}
    controller._record = True
    controller._sig_records = []
    for i in range(min(n_cal, len(records))):
        vp = (REPO / "eval_data/OmniVideoBench" / str(records[i]["video"])).resolve()
        if not vp.exists():
            continue
        conv = build_conversation(vp, records[i])
        inputs = build_inputs(processor, process_mm_info, conv)
        controller.prepare_sample(inputs["input_ids"].reshape(-1).clone())
        inputs = move(inputs, model)
        with controller.arm("sceb", signal="entropy_high"):
            # quantiles not yet set; recording happens regardless of budget mapping
            controller._quantiles["entropy_high"] = (0.0, 1.0)
            generate_once(model, processor, inputs)
        for rec in controller._sig_records:
            for s in ALL_SIGNALS:
                pooled[s].append(rec[s])
        controller._sig_records = []
    controller._record = False
    quant = {}
    for s in ALL_SIGNALS:
        arr = (
            torch.cat(pooled[s]).numpy().astype(np.float64)
            if pooled[s]
            else np.zeros(1)
        )
        quant[s] = (float(np.percentile(arr, 33.3)), float(np.percentile(arr, 66.7)))
    controller._quantiles = quant
    log(
        f"calibrated {len(ALL_SIGNALS)} signal quantiles over {min(n_cal, len(records))} samples"
    )
    return quant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--calibrate-samples", type=int, default=30)
    args = ap.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw_results.jsonl"

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    log("loading Uni-MoE-2")
    _ensure_matplotlib_stub()
    from uni_moe.model.processing_qwen2_vl import Qwen2VLProcessor
    from uni_moe.model.modeling_qwen_grin_moe import GrinQwen2VLForConditionalGeneration
    from uni_moe.qwen_vl_utils import process_mm_info
    from uni_moe.model import deepspeed_moe_inference_utils  # noqa: F401

    processor = Qwen2VLProcessor.from_pretrained(MODEL_PATH)
    model = (
        GrinQwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        .cuda()
        .eval()
    )
    processor.data_args = model.config

    records = json.loads(DATASET.read_text())
    total_layers = int(
        json.loads((Path(MODEL_PATH) / "config.json").read_text())["num_hidden_layers"]
    )
    band = (int(np.floor(0.25 * total_layers)), int(np.floor(0.67 * total_layers)))
    controller = MultiSignalController(
        model, expected_blocks=total_layers, layer_band=band
    )

    calibrate(
        records, model, processor, process_mm_info, controller, args.calibrate_samples
    )

    rows = []
    try:
        for si in range(min(args.limit, len(records))):
            rec = records[si]
            vp = (REPO / "eval_data/OmniVideoBench" / str(rec["video"])).resolve()
            if not vp.exists():
                continue
            try:
                conv = build_conversation(vp, rec)
                inputs = build_inputs(processor, process_mm_info, conv)
                controller.prepare_sample(inputs["input_ids"].reshape(-1).clone())
                inputs = move(inputs, model)
                correct = str(rec["correct_option"]).upper()
                native = generate_once(model, processor, inputs)
                row = {
                    "sample_index": si,
                    "native_answer": native.answer,
                    "native_correct": native.answer == correct,
                    "signals": {},
                }
                for sig in ALL_SIGNALS:
                    with controller.arm("sceb", signal=sig):
                        sceb = generate_once(model, processor, inputs)
                        sceb_budget = controller.exec_mean()
                    with controller.arm("shuffled", shuffle_seed=SEED + si):
                        shuf = generate_once(model, processor, inputs)
                        shuf_budget = controller.exec_mean()
                    row["signals"][sig] = {
                        "sceb_flip": sceb.answer != native.answer,
                        "shuf_flip": shuf.answer != native.answer,
                        "sceb_correct": sceb.answer == correct,
                        "shuf_correct": shuf.answer == correct,
                        "sceb_budget": sceb_budget,
                        "shuf_budget": shuf_budget,
                    }
                rows.append(row)
                with raw_path.open("a") as h:
                    h.write(json.dumps(row) + "\n")
                if (si + 1) % 5 == 0:
                    log(f"[{si + 1}/{args.limit}] done")
            except Exception as e:
                log(f"sample {si} failed: {e!r}")
                traceback.print_exc()
        # Aggregate direction per signal.
        n = len(rows)
        summary = {"backbone": "Uni-MoE-2.0-Omni", "n_samples": n, "signals": {}}
        for sig in ALL_SIGNALS:
            sf = sum(r["signals"][sig]["sceb_flip"] for r in rows)
            hf = sum(r["signals"][sig]["shuf_flip"] for r in rows)
            so = sum(
                1
                for r in rows
                if r["signals"][sig]["sceb_flip"] and not r["signals"][sig]["shuf_flip"]
            )
            ho = sum(
                1
                for r in rows
                if not r["signals"][sig]["sceb_flip"] and r["signals"][sig]["shuf_flip"]
            )
            summary["signals"][sig] = {
                "sceb_flip_rate": sf / n if n else None,
                "shuffled_flip_rate": hf / n if n else None,
                "direction_ok": sf <= hf,
                "mcnemar_sceb_only": so,
                "mcnemar_shuf_only": ho,
                "net": so - ho,
                "mean_sceb_budget": float(
                    np.nanmean(
                        [r["signals"][sig]["sceb_budget"] or np.nan for r in rows]
                    )
                ),
            }
        summary["survivors_direction_ok"] = [
            s for s in ALL_SIGNALS if summary["signals"][s]["direction_ok"]
        ]
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        log(f"SCREEN done: n={n} survivors={summary['survivors_direction_ok']}")
    finally:
        controller.close()


if __name__ == "__main__":
    main()

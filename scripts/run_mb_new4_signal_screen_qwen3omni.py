#!/usr/bin/env python3
"""MB-NEW-4 multi-signal SCREEN on Qwen3-Omni (slow, large-selection backbone).

Validates the Uni-MoE screen survivors on Qwen3-Omni (128 experts, top-8,
candidates {4,6,8}). For each screened signal it runs a SCEB arm and a
budget-matched SHUFFLED control. A signal "works" here if SCEB flip_rate <=
shuffled flip_rate (direction correct). Only signals whose direction is correct
on BOTH backbones survive to the full 1,000-example verification.

Leakage-free; reads only router probabilities and token position. Screen only.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "qwen-omni-utils" / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from mb_new4_router_signals import budget_from_quantiles, signal_values  # noqa: E402

MODEL_PATH = str(REPO / "models/Qwen3-Omni-30B-A3B-Instruct")
DATASET = REPO / "eval_data/OmniVideoBench/data.json"
VIDEO_TOKEN_ID = 151656
CANDIDATES = [4, 6, 8]
NATIVE_TOP_K = 8
SEED = 20260906
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
SCREEN_SIGNALS = [
    "participation_high",
    "gini",
    "top1_dominance",
    "cummass_half",
    "entropy_high",
]


def log(*parts: Any) -> None:
    print("[MB4-SCREEN-Qwen]", *parts, file=sys.stderr, flush=True)


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
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": prompt},
            ],
        },
    ]


class MultiSignalController:
    def __init__(self, model, layer_band):
        self.candidate_budgets = CANDIDATES
        self.top_k = NATIVE_TOP_K
        self.layer_lo, self.layer_hi = layer_band
        self._wrapped: list[tuple[Any, Callable[..., Any], int]] = []
        self._arm = "native"
        self._signal = None
        self._elig = None
        self._quantiles: dict[str, tuple[float, float]] = {}
        self._sceb_budgets_by_layer: dict[int, torch.Tensor] = {}
        self._shuffle_perm: dict[int, torch.Tensor] | None = None
        self._record = False
        self._sig_records: list[dict[str, Any]] = []
        self._exec_sum = 0.0
        self._exec_n = 0
        n = 0
        for name, module in model.thinker.named_modules():
            if "sparsemoeblock" not in module.__class__.__name__.lower():
                continue
            if not all(
                hasattr(module, a) for a in ("gate", "experts", "top_k", "num_experts")
            ):
                continue
            layer = self._layer_idx(name)
            if layer is None:
                continue
            self._install(module, layer)
            n += 1
        if not self._wrapped:
            raise RuntimeError("no SparseMoeBlock found")
        self.installed = n
        log(f"controller on {n} MoE layers; band=[{self.layer_lo},{self.layer_hi})")

    @staticmethod
    def _layer_idx(name):
        parts = name.split(".")
        for i, p in enumerate(parts[:-1]):
            if p == "layers" and parts[i + 1].isdigit():
                return int(parts[i + 1])
        return None

    def _install(self, module, layer):
        original = module.forward
        controller = self

        def wrapped(hidden_states):
            active = (
                controller._arm != "native"
                and controller.layer_lo <= layer < controller.layer_hi
                and controller._elig is not None
            )
            if not active:
                return original(hidden_states)
            return controller._budgeted(module, original, hidden_states, layer)

        module.forward = wrapped
        self._wrapped.append((module, original, layer))

    def _budgeted(self, module, original, hidden_states, layer):
        batch, seqlen, hdim = hidden_states.shape
        if self._elig is None or seqlen != self._elig.shape[0]:
            return original(hidden_states)
        flat = hidden_states.view(-1, hdim)
        token_count = flat.shape[0]
        router_logits = module.gate(flat)
        probs_full = torch.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(probs_full, module.top_k, dim=-1)
        elig = self._elig.to(routing_weights.device).view(-1)
        budgets = torch.full(
            (token_count,),
            module.top_k,
            dtype=torch.long,
            device=routing_weights.device,
        )

        if self._record and elig.any():
            ev = probs_full[elig]
            if ev.shape[0] > 512:
                ev = ev[torch.randperm(ev.shape[0], device=ev.device)[:512]]
            sv = signal_values(ev, module.top_k)
            self._sig_records.append(
                {"layer": layer, **{s: sv[s].detach().cpu() for s in SCREEN_SIGNALS}}
            )

        if self._arm == "sceb":
            q_low, q_high = self._quantiles[self._signal]
            sv = signal_values(probs_full, module.top_k)[self._signal]
            sceb_b = budget_from_quantiles(sv, q_low, q_high, self.candidate_budgets)
            budgets = torch.where(elig, sceb_b, budgets)
            self._sceb_budgets_by_layer[layer] = sceb_b[elig].detach().cpu()
        elif self._arm == "shuffled":
            base = self._sceb_budgets_by_layer.get(layer)
            perm = self._shuffle_perm.get(layer) if self._shuffle_perm else None
            if base is not None and perm is not None:
                sh = base[perm].to(budgets.device).long()
                ii = torch.nonzero(elig, as_tuple=True)[0]
                if ii.shape[0] == sh.shape[0]:
                    budgets[ii] = sh

        if elig.any():
            self._exec_sum += float(budgets[elig].sum().item())
            self._exec_n += int(elig.sum().item())

        ranks = torch.arange(module.top_k, device=routing_weights.device).unsqueeze(0)
        keep = ranks < budgets.unsqueeze(1)
        routing_weights = routing_weights.masked_fill(~keep, 0.0)
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-20)
        routing_weights = routing_weights.to(flat.dtype)

        final = torch.zeros((token_count, hdim), dtype=flat.dtype, device=flat.device)
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=module.num_experts
        ).permute(2, 1, 0)
        active = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for ei in active:
            rank_index, token_index = torch.where(expert_mask[ei].squeeze(0))
            if token_index.numel() == 0:
                continue
            nz = routing_weights[token_index, rank_index] > 0
            if not bool(nz.any()):
                continue
            token_index = token_index[nz]
            rank_index = rank_index[nz]
            eo = module.experts[ei](flat[None, token_index].reshape(-1, hdim))
            eo = eo * routing_weights[token_index, rank_index, None]
            final.index_add_(0, token_index, eo.to(flat.dtype))
        return final.reshape(batch, seqlen, hdim), router_logits

    def close(self):
        for module, original, _ in self._wrapped:
            module.forward = original

    def prepare_sample(self, input_ids_1d):
        self._elig = (input_ids_1d == VIDEO_TOKEN_ID).bool().cpu()
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


def generate_once(model, processor, inputs):
    with torch.inference_mode():
        raw = model.generate(
            **inputs,
            use_audio_in_video=True,
            return_audio=False,
            thinker_max_new_tokens=4,
            thinker_do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            thinker_eos_token_id=processor.tokenizer.eos_token_id,
        )
    thinker = raw[0] if isinstance(raw, tuple) else raw
    input_len = int(inputs["input_ids"].shape[1])
    gen_ids = thinker.sequences[0, input_len:].detach().cpu()
    text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return Gen(answer=parse_answer(text))


def build_inputs(processor, process_mm_info, conv):
    text = processor.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(conv, use_audio_in_video=True)
    return processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True,
    )


def calibrate(records, model, processor, process_mm_info, controller, n_cal):
    pooled = {s: [] for s in SCREEN_SIGNALS}
    controller._record = True
    controller._sig_records = []
    for i in range(min(n_cal, len(records))):
        vp = (REPO / "eval_data/OmniVideoBench" / str(records[i]["video"])).resolve()
        if not vp.exists():
            continue
        conv = build_conversation(vp, records[i])
        inputs = build_inputs(processor, process_mm_info, conv)
        controller.prepare_sample(inputs["input_ids"].reshape(-1).clone())
        inputs = inputs.to(model.device).to(model.dtype)
        controller._quantiles = {s: (0.0, 1.0) for s in SCREEN_SIGNALS}
        with controller.arm("sceb", signal=SCREEN_SIGNALS[0]):
            generate_once(model, processor, inputs)
        for rec in controller._sig_records:
            for s in SCREEN_SIGNALS:
                pooled[s].append(rec[s])
        controller._sig_records = []
    controller._record = False
    quant = {}
    for s in SCREEN_SIGNALS:
        arr = (
            torch.cat(pooled[s]).numpy().astype(np.float64)
            if pooled[s]
            else np.zeros(1)
        )
        quant[s] = (float(np.percentile(arr, 33.3)), float(np.percentile(arr, 66.7)))
    controller._quantiles = quant
    log(
        f"calibrated {len(SCREEN_SIGNALS)} signal quantiles over {min(n_cal, len(records))} samples"
    )
    return quant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--calibrate-samples", type=int, default=20)
    args = ap.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw_results.jsonl"

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    log("loading Qwen3-Omni")
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()

    records = json.loads(DATASET.read_text())
    total_layers = int(
        json.loads((Path(MODEL_PATH) / "config.json").read_text())["thinker_config"][
            "text_config"
        ]["num_hidden_layers"]
    )
    band = (int(np.floor(0.25 * total_layers)), int(np.floor(0.67 * total_layers)))
    controller = MultiSignalController(model, layer_band=band)

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
                inputs = inputs.to(model.device).to(model.dtype)
                correct = str(rec["correct_option"]).upper()
                native = generate_once(model, processor, inputs)
                row = {
                    "sample_index": si,
                    "native_answer": native.answer,
                    "native_correct": native.answer == correct,
                    "signals": {},
                }
                for sig in SCREEN_SIGNALS:
                    with controller.arm("sceb", signal=sig):
                        sceb = generate_once(model, processor, inputs)
                        sb = controller.exec_mean()
                    with controller.arm("shuffled", shuffle_seed=SEED + si):
                        shuf = generate_once(model, processor, inputs)
                        hb = controller.exec_mean()
                    row["signals"][sig] = {
                        "sceb_flip": sceb.answer != native.answer,
                        "shuf_flip": shuf.answer != native.answer,
                        "sceb_correct": sceb.answer == correct,
                        "shuf_correct": shuf.answer == correct,
                        "sceb_budget": sb,
                        "shuf_budget": hb,
                    }
                rows.append(row)
                with raw_path.open("a") as h:
                    h.write(json.dumps(row) + "\n")
                if (si + 1) % 5 == 0:
                    log(f"[{si + 1}/{args.limit}] done")
            except Exception as e:
                log(f"sample {si} failed: {e!r}")
                traceback.print_exc()
        n = len(rows)
        summary = {"backbone": "Qwen3-Omni", "n_samples": n, "signals": {}}
        for sig in SCREEN_SIGNALS:
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
            s for s in SCREEN_SIGNALS if summary["signals"][s]["direction_ok"]
        ]
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        log(f"SCREEN done: n={n} survivors={summary['survivors_direction_ok']}")
    finally:
        controller.close()


if __name__ == "__main__":
    main()

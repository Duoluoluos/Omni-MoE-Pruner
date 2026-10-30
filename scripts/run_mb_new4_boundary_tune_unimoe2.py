#!/usr/bin/env python3
"""MB-NEW-4 boundary-margin TUNING screen on Uni-MoE-2.0-Omni.

Goal: reproduce on Uni-MoE the boundary-margin selectivity that the original
SCEB showed on Qwen3-Omni. v1 failed on Uni-MoE because a single absolute
threshold on the tiny 5-way dynamic softmax collapsed the budget to a near
constant. Here we use the ORIGINAL boundary-margin mechanism (smallest candidate
m whose g_m = p_m - p_{m+1} exceeds tau_m) but with PER-CANDIDATE thresholds
calibrated as quantiles of each candidate's own margin distribution, so the
budget disperses. We sweep the threshold percentile to find a working operating
point, each compared against a budget-matched SHUFFLED control (direction test).

Leakage-free; router geometry + token position only. Screen only, not evidence.
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
sys.path.insert(0, "/home/tiger/Uni-MoE/Uni-MoE-2")
sys.path.insert(0, str(REPO / "scripts"))

from mb_new4_router_signals import boundary_budget, boundary_margins  # noqa: E402

MODEL_PATH = str(REPO / "models/Uni-MoE-2.0-Omni")
DATASET = REPO / "eval_data/OmniVideoBench/data.json"
VIDEO_TOKEN_ID = 151656
CANDIDATES = [2, 3, 4]
SEED = 20260906
# Threshold percentiles to sweep (per-candidate quantile of margin distribution).
# Lower percentile => easier to be "decisive" => smaller budgets (more cutting).
PERCENTILES = [30.0, 50.0, 70.0]


def log(*p: Any) -> None:
    print("[MB4-BND-UniMoE]", *p, file=sys.stderr, flush=True)


def _mpl_stub() -> None:
    import types, importlib.machinery

    try:
        import matplotlib.pyplot  # noqa

        return
    except Exception:
        pass
    m = types.ModuleType("matplotlib")
    m.__spec__ = importlib.machinery.ModuleSpec("matplotlib", loader=None)
    m.__path__ = []
    pp = types.ModuleType("matplotlib.pyplot")
    pp.__spec__ = importlib.machinery.ModuleSpec("matplotlib.pyplot", loader=None)
    for nm in ("figure", "plot", "savefig", "close", "clf", "show"):
        setattr(pp, nm, lambda *a, **k: None)
    m.pyplot = pp
    sys.modules.setdefault("matplotlib", m)
    sys.modules.setdefault("matplotlib.pyplot", pp)


def parse_answer(text: str) -> str | None:
    u = str(text).strip().upper()
    mt = re.search(r"(?:^|[^A-Z])([A-D])(?:[^A-Z]|$)", u)
    if mt:
        return mt.group(1)
    for c in u:
        if c in "ABCD":
            return c
    return None


def build_conversation(vp: Path, rec: dict) -> list[dict]:
    options = "\n".join(str(o) for o in rec["options"])
    prompt = (
        f"{rec['question']}\nOptions:\n{options}\nAnswer with only the option letter."
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(vp)},
                {"type": "video", "video": str(vp)},
                {"type": "text", "text": f"<video>\n<audio>\n{prompt}"},
            ],
        }
    ]


def expand(t: str) -> str:
    return (
        t.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        .replace("<audio>", "<|audio_start|><|audio_pad|><|audio_end|>")
        .replace("<video>", "<|vision_start|><|video_pad|><|vision_end|>")
    )


class BoundaryController:
    def __init__(self, model, expected_blocks, layer_band):
        self.cands = CANDIDATES
        self.layer_lo, self.layer_hi = layer_band
        self._blocks, self._orig, self._top_p, self._layer = [], [], [], []
        moe = [
            (n, m)
            for n, m in model.named_modules()
            if m.__class__.__name__ == "GRINMoESparseMoeBlock"
        ]
        if len(moe) != expected_blocks:
            raise RuntimeError(f"{len(moe)} blocks != {expected_blocks}")
        for name, module in moe:
            self._blocks.append(module)
            self._layer.append(self._li(name))
            self._top_p.append(float(module.mlp_dynamic_top_p))
        self._arm = "native"  # native | sceb | shuffled
        self._thresholds = None  # {m: tau}
        self._elig = None
        self._budgets_by_layer = {}
        self._perm = None
        self._record = False
        self._rec = []
        self._exec_sum = 0.0
        self._exec_n = 0
        for i, (b, l) in enumerate(zip(self._blocks, self._layer)):
            self._orig.append(b.forward)
            b.forward = self._mk(b, l, i)
        log(
            f"controller on {len(self._blocks)} blocks band=[{self.layer_lo},{self.layer_hi})"
        )

    @staticmethod
    def _li(name):
        parts = name.split(".")
        for i, p in enumerate(parts[:-1]):
            if p == "layers" and parts[i + 1].isdigit():
                return int(parts[i + 1])
        return None

    def close(self):
        for b, o, tp in zip(self._blocks, self._orig, self._top_p):
            b.forward = o
            b.mlp_dynamic_top_p = tp

    def _mk(self, block, layer, idx):
        controller = self

        def fwd(hidden_states, attention_mask, aux_balance_weight):
            orig = controller._orig[idx]
            active = (
                controller._arm != "native"
                and controller.layer_lo <= layer < controller.layer_hi
                and controller._elig is not None
            )
            if not active:
                return orig(hidden_states, attention_mask, aux_balance_weight)
            bsz, seqlen, hdim = hidden_states.shape
            if seqlen != controller._elig.shape[0]:
                return orig(hidden_states, attention_mask, aux_balance_weight)
            flat = hidden_states.view(-1, hdim)
            full = block.gate(flat)
            dyn = full[:, : block.mlp_dynamic_expert_num]
            probs = torch.softmax(dyn.float(), dim=-1)
            k = probs.shape[1]
            elig = controller._elig.to(probs.device).view(-1)
            from uni_moe.model.modeling_qwen_grin_moe import (
                dynamic_expert_selection as nsel,
            )

            budget = nsel(dyn, controller._top_p[idx]).to(torch.int64)

            if controller._record and elig.any():
                g = boundary_margins(probs[elig], k, controller.cands)
                controller._rec.append(
                    {
                        "layer": layer,
                        **{f"g{m}": g[m].detach().cpu() for m in controller.cands},
                    }
                )

            if controller._arm == "sceb":
                sceb_b = boundary_budget(
                    probs, k, controller.cands, controller._thresholds
                )
                budget = torch.where(elig, sceb_b, budget)
                controller._budgets_by_layer[layer] = sceb_b[elig].detach().cpu()
            elif controller._arm == "shuffled":
                base = controller._budgets_by_layer.get(layer)
                perm = controller._perm.get(layer) if controller._perm else None
                if base is not None and perm is not None:
                    sh = base[perm].to(budget.device)
                    ii = torch.nonzero(elig, as_tuple=True)[0]
                    if ii.shape[0] == sh.shape[0]:
                        budget[ii] = sh

            if elig.any():
                controller._exec_sum += float(budget[elig].sum().item())
                controller._exec_n += int(elig.sum().item())

            gm = sys.modules["uni_moe.model.modeling_qwen_grin_moe"]
            osel = gm.dynamic_expert_selection
            try:
                gm.dynamic_expert_selection = lambda logits, top_p: budget.to(
                    logits.device
                )
                block.mlp_dynamic_top_p = controller._top_p[idx]
                out = orig(hidden_states, attention_mask, aux_balance_weight)
            finally:
                gm.dynamic_expert_selection = osel
                block.mlp_dynamic_top_p = controller._top_p[idx]
            return out

        return fwd

    def prepare(self, ids):
        self._elig = (ids == VIDEO_TOKEN_ID).bool().cpu()
        self._budgets_by_layer = {}

    def reset(self):
        self._exec_sum = 0.0
        self._exec_n = 0

    def mean_budget(self):
        return None if self._exec_n == 0 else self._exec_sum / self._exec_n

    @contextlib.contextmanager
    def arm(self, name, thresholds=None, shuffle_seed=None):
        self._arm = name
        self._thresholds = thresholds
        self._perm = None
        self.reset()
        if name == "shuffled" and shuffle_seed is not None:
            g = torch.Generator().manual_seed(int(shuffle_seed))
            self._perm = {
                l: torch.randperm(v.shape[0], generator=g)
                for l, v in self._budgets_by_layer.items()
            }
        try:
            yield
        finally:
            self._arm, self._thresholds, self._perm = "native", None, None


@dataclass
class Gen:
    answer: str | None


def gen_once(model, processor, inputs):
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
    il = int(inputs["input_ids"].shape[-1])
    ids = raw.sequences[0, il:].detach().cpu()
    return Gen(
        parse_answer(processor.tokenizer.decode(ids, skip_special_tokens=True).strip())
    )


def build_inputs(processor, pmm, conv):
    texts = expand(
        processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    )
    imgs, vids, auds = pmm(conv)
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


def calibrate(records, model, processor, pmm, controller, n_cal):
    pooled = {m: [] for m in CANDIDATES}
    controller._record = True
    controller._rec = []
    for i in range(min(n_cal, len(records))):
        vp = (REPO / "eval_data/OmniVideoBench" / str(records[i]["video"])).resolve()
        if not vp.exists():
            continue
        conv = build_conversation(vp, records[i])
        inputs = build_inputs(processor, pmm, conv)
        controller.prepare(inputs["input_ids"].reshape(-1).clone())
        inputs = move(inputs, model)
        with controller.arm("sceb", thresholds={m: 1e9 for m in CANDIDATES}):
            gen_once(model, processor, inputs)
        for rec in controller._rec:
            for m in CANDIDATES:
                pooled[m].append(rec[f"g{m}"])
        controller._rec = []
    controller._record = False
    # thresholds per percentile setting
    th_by_pct = {}
    for pct in PERCENTILES:
        th_by_pct[pct] = {
            m: float(np.percentile(torch.cat(pooled[m]).numpy(), pct))
            for m in CANDIDATES
        }
    log(f"calibrated boundary thresholds for pct={PERCENTILES}")
    return th_by_pct


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
    log("loading Uni-MoE-2")
    _mpl_stub()
    from uni_moe.model.processing_qwen2_vl import Qwen2VLProcessor
    from uni_moe.model.modeling_qwen_grin_moe import GrinQwen2VLForConditionalGeneration
    from uni_moe.qwen_vl_utils import process_mm_info
    from uni_moe.model import deepspeed_moe_inference_utils  # noqa

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
    total = int(
        json.loads((Path(MODEL_PATH) / "config.json").read_text())["num_hidden_layers"]
    )
    band = (int(np.floor(0.25 * total)), int(np.floor(0.67 * total)))
    controller = BoundaryController(model, expected_blocks=total, layer_band=band)
    th_by_pct = calibrate(
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
                controller.prepare(inputs["input_ids"].reshape(-1).clone())
                inputs = move(inputs, model)
                correct = str(rec["correct_option"]).upper()
                native = gen_once(model, processor, inputs)
                row = {
                    "sample_index": si,
                    "native_correct": native.answer == correct,
                    "variants": {},
                }
                for pct in PERCENTILES:
                    with controller.arm("sceb", thresholds=th_by_pct[pct]):
                        sceb = gen_once(model, processor, inputs)
                        sb = controller.mean_budget()
                    with controller.arm("shuffled", shuffle_seed=SEED + si):
                        shuf = gen_once(model, processor, inputs)
                        hb = controller.mean_budget()
                    row["variants"][f"pct{int(pct)}"] = {
                        "sceb_flip": sceb.answer != native.answer,
                        "shuf_flip": shuf.answer != native.answer,
                        "sceb_correct": sceb.answer == correct,
                        "sceb_budget": sb,
                        "shuf_budget": hb,
                    }
                rows.append(row)
                with raw_path.open("a") as h:
                    h.write(json.dumps(row) + "\n")
                if (si + 1) % 5 == 0:
                    log(f"[{si + 1}/{args.limit}]")
            except Exception as e:
                log(f"sample {si} failed: {e!r}")
                traceback.print_exc()
        n = len(rows)
        summary = {
            "backbone": "Uni-MoE-2.0-Omni",
            "n_samples": n,
            "signal": "boundary_margin",
            "variants": {},
        }
        for pct in PERCENTILES:
            key = f"pct{int(pct)}"
            sf = sum(r["variants"][key]["sceb_flip"] for r in rows)
            hf = sum(r["variants"][key]["shuf_flip"] for r in rows)
            so = sum(
                1
                for r in rows
                if r["variants"][key]["sceb_flip"]
                and not r["variants"][key]["shuf_flip"]
            )
            ho = sum(
                1
                for r in rows
                if not r["variants"][key]["sceb_flip"]
                and r["variants"][key]["shuf_flip"]
            )
            summary["variants"][key] = {
                "threshold_percentile": pct,
                "sceb_flip_rate": sf / n if n else None,
                "shuffled_flip_rate": hf / n if n else None,
                "direction_ok": sf <= hf,
                "net": so - ho,
                "mean_sceb_budget": float(
                    np.nanmean(
                        [r["variants"][key]["sceb_budget"] or np.nan for r in rows]
                    )
                ),
                "mean_shuf_budget": float(
                    np.nanmean(
                        [r["variants"][key]["shuf_budget"] or np.nan for r in rows]
                    )
                ),
            }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        log(
            f"BND SCREEN done n={n}: "
            + " | ".join(
                f"{k} sceb={v['sceb_flip_rate']:.3f} shuf={v['shuffled_flip_rate']:.3f} net={v['net']} bud={v['mean_sceb_budget']:.2f}"
                for k, v in summary["variants"].items()
            )
        )
    finally:
        controller.close()


if __name__ == "__main__":
    main()

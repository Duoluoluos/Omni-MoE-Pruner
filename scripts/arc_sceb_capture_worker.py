#!/usr/bin/env python3
"""Instrumented ARC-SCEB router capture on real Qwen3-Omni audio-visual prefill.

Runs on the H20 worker (CUDA venv). For a handful of real AVUT clips it captures
the modality-resolved routing geometry that static profiles do not expose, and
replays the deployed mask-only ARC-SCEB policy token-by-token:
  * per-modality (video/audio/text) top1-top2 decisiveness margins;
  * video-token boundary margins at m=4 (p4-p5) and m=6 (p6-p7);
  * boundary-only budget decision -> effective top-4/6/8;
  * same-sample audio-router margin and resulting video floor (6 or 8).
Mirrors moe_expert_budget/configs/arc_video_boundary_audio_floor6_full_v1.json
and scripts/megatron_arc_sceb_router.py. Inference-only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner")
sys.path.insert(0, str(REPO / "qwen-omni-utils" / "src"))

MODEL_PATH = str(REPO / "models/Qwen3-Omni-30B-A3B-Instruct")
POLICY = json.loads(
    (REPO / "moe_expert_budget/configs/arc_video_boundary_audio_floor6_full_v1.json").read_text()
)
NUM_LAYERS = 48
L_START = int(NUM_LAYERS * POLICY["layer_start_ratio"])
L_END = int(NUM_LAYERS * POLICY["layer_end_ratio"])
TOP_K = int(POLICY["original_top_k"])
M_SMALL, M_MID, M_FULL = sorted(int(v) for v in POLICY["top_m_options"])
BOUNDARY_THR = float(POLICY["thresholds"]["boundary_threshold"])
ARC = POLICY["audio_router_conditioning"]
MIN_AUDIO_TOK = int(ARC["min_audio_tokens"])
INSUFF_FLOOR = int(ARC["insufficient_audio_min_top_m"])
FRAGILE_MARGIN = float(ARC["fragile_margin_threshold"])
FRAGILE_FLOOR = int(ARC["fragile_min_top_m"])

VIDEO_ID = 151656
AUDIO_ID = 151675
IMAGE_ID = 151655
MAX_RAW_PER_LAYER = 6000


def log(*a):
    print("[capture]", *a, file=sys.stderr, flush=True)


class RouterCapture:
    def __init__(self, model):
        self.by_layer = {}
        self.handles = []
        n = 0
        for name, module in model.thinker.named_modules():
            cls = module.__class__.__name__.lower()
            if "sparsemoeblock" in cls and hasattr(module, "gate") and hasattr(module, "experts"):
                layer_idx = self._layer_idx(name)
                if layer_idx is None:
                    continue
                self.handles.append(module.gate.register_forward_hook(self._make_hook(layer_idx)))
                n += 1
        log(f"hooked {n} MoE gates")

    @staticmethod
    def _layer_idx(name):
        parts = name.split(".")
        for i, p in enumerate(parts[:-1]):
            if p == "layers" and parts[i + 1].isdigit():
                return int(parts[i + 1])
        return None

    def _make_hook(self, layer_idx):
        def hook(_mod, _inp, out):
            logits = out.detach()
            if logits.dim() != 2:
                return
            prev = self.by_layer.get(layer_idx)
            if prev is None or logits.shape[0] > prev.shape[0]:
                self.by_layer[layer_idx] = logits.float().cpu()
        return hook

    def reset(self):
        self.by_layer = {}

    def remove(self):
        for h in self.handles:
            h.remove()


def arc_decide(video_top_vals, audio_margin_mean, audio_count):
    nv = video_top_vals.shape[0]
    b_small = video_top_vals[:, M_SMALL - 1] - video_top_vals[:, M_SMALL]
    b_mid = video_top_vals[:, M_MID - 1] - video_top_vals[:, M_MID]
    chosen = np.full(nv, M_FULL, dtype=np.int64)
    choose_small = b_small >= BOUNDARY_THR
    choose_mid = (~choose_small) & (b_mid >= BOUNDARY_THR)
    chosen[choose_mid] = M_MID
    chosen[choose_small] = M_SMALL
    if audio_count < MIN_AUDIO_TOK:
        floor, state = INSUFF_FLOOR, "insufficient_audio"
    elif audio_margin_mean < FRAGILE_MARGIN:
        floor, state = FRAGILE_FLOOR, "fragile_audio"
    else:
        floor, state = M_SMALL, "confident_audio"
    if floor > M_SMALL:
        chosen = np.clip(chosen, floor, M_FULL)
    return chosen, floor, state


def process_sample(cap, input_ids):
    ids = input_ids.reshape(-1)
    seq = ids.shape[0]
    video_mask = ids == VIDEO_ID
    audio_mask = ids == AUDIO_ID
    text_mask = ~(video_mask | audio_mask | (ids == IMAGE_ID))
    per_layer = []
    hist = {M_SMALL: 0, M_MID: 0, M_FULL: 0}
    floor_states = {}
    for li in range(L_START, L_END):
        logits = cap.by_layer.get(li)
        if logits is None or logits.shape[0] != seq:
            continue
        probs = torch.softmax(logits, dim=1, dtype=torch.float)
        top_vals, _ = torch.topk(probs, TOP_K, dim=-1, sorted=True)
        top_vals = top_vals.numpy()
        v = top_vals[video_mask]
        a = top_vals[audio_mask]
        t = top_vals[text_mask]
        audio_margin = (a[:, 0] - a[:, 1]) if a.shape[0] else np.array([])
        audio_margin_mean = float(audio_margin.mean()) if audio_margin.size else 0.0
        audio_count = int(a.shape[0])
        chosen, floor, state = arc_decide(v, audio_margin_mean, audio_count)
        for m in (M_SMALL, M_MID, M_FULL):
            hist[m] += int((chosen == m).sum())
        floor_states[state] = floor_states.get(state, 0) + 1
        vb_small = v[:, M_SMALL - 1] - v[:, M_SMALL]
        vb_mid = v[:, M_MID - 1] - v[:, M_MID]
        vm12 = v[:, 0] - v[:, 1]
        if v.shape[0] > MAX_RAW_PER_LAYER:
            sel = np.random.default_rng(0).choice(v.shape[0], MAX_RAW_PER_LAYER, replace=False)
            vb_small, vb_mid, vm12 = vb_small[sel], vb_mid[sel], vm12[sel]
        per_layer.append({
            "layer": li,
            "n_video": int(v.shape[0]), "n_audio": audio_count, "n_text": int(t.shape[0]),
            "video_margin12_mean": float(vm12.mean()) if vm12.size else 0.0,
            "audio_margin12_mean": audio_margin_mean,
            "text_margin12_mean": float((t[:, 0] - t[:, 1]).mean()) if t.shape[0] else 0.0,
            "video_boundary_small_mean": float(vb_small.mean()) if vb_small.size else 0.0,
            "video_boundary_mid_mean": float(vb_mid.mean()) if vb_mid.size else 0.0,
            "chosen_m_hist": {str(m): int((chosen == m).sum()) for m in (M_SMALL, M_MID, M_FULL)},
            "audio_floor": floor, "audio_state": state,
            "raw_video_boundary_small": [round(float(x), 6) for x in vb_small.tolist()],
            "raw_video_margin12": [round(float(x), 6) for x in vm12.tolist()],
            "raw_audio_margin12": [round(float(x), 6) for x in audio_margin.tolist()],
        })
    return {
        "seq_len": int(seq),
        "n_video_tokens": int(video_mask.sum()),
        "n_audio_tokens": int(audio_mask.sum()),
        "n_text_tokens": int(text_mask.sum()),
        "chosen_m_hist_total": {str(k): v for k, v in hist.items()},
        "audio_floor_states": floor_states,
        "per_layer": per_layer,
    }


def build_conversation(video_path, question):
    return [
        {"role": "system", "content": [{"type": "text", "text":
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
            "capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": question}]},
    ]


def pick_samples(limit):
    ann = json.loads((REPO / "eval_data/AVUTBenchmark/AV_Human_filtered_data.json").read_text())
    seen_type = {}
    picks = []
    for r in ann:
        vp = REPO / "eval_data/AVUTBenchmark" / (Path(r["video_path"]).name)
        if not vp.exists():
            continue
        vt = r.get("video_type", "?")
        if seen_type.get(vt, 0) >= 2:
            continue
        seen_type[vt] = seen_type.get(vt, 0) + 1
        q = r["question"]
        opts = " ".join(f"({k}) {r.get('option_'+k,'')}" for k in ["A", "B", "C", "D"])
        picks.append({
            "source": "avut", "video_id": r.get("video_id"), "task_type": r.get("task_type"),
            "video_type": vt, "video_path": str(vp),
            "question": f"{q}\nOptions: {opts}\nAnswer with the option letter.",
        })
        if len(picks) >= limit:
            break
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--out", type=str,
                    default=str(REPO / "logs/arc_sceb_interpretability/worker_capture.json"))
    args = ap.parse_args()

    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info

    samples = pick_samples(args.limit)
    log(f"selected {len(samples)} samples: "
        + ", ".join(f"{s['video_type']}/{s['task_type']}" for s in samples))

    log("loading model ...")
    t0 = time.time()
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    log(f"model loaded in {time.time()-t0:.1f}s")

    cap = RouterCapture(model)
    results = []
    for i, s in enumerate(samples):
        try:
            conv = build_conversation(s["video_path"], s["question"])
            text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conv, use_audio_in_video=True)
            inputs = processor(text=text, audio=audios, images=images, videos=videos,
                               return_tensors="pt", padding=True, use_audio_in_video=True)
            inputs = inputs.to(model.device).to(model.dtype)
            input_ids = inputs["input_ids"].detach().cpu().numpy()
            cap.reset()
            t1 = time.time()
            with torch.no_grad():
                model.generate(**inputs, use_audio_in_video=True, return_audio=False,
                               do_sample=False, max_new_tokens=1)
            rec = process_sample(cap, input_ids)
            rec.update({k: s[k] for k in ("source", "video_id", "task_type", "video_type")})
            rec["prefill_s"] = round(time.time() - t1, 2)
            results.append(rec)
            log(f"[{i+1}/{len(samples)}] {s['video_type']}/{s['task_type']} "
                f"seq={rec['seq_len']} video={rec['n_video_tokens']} audio={rec['n_audio_tokens']} "
                f"hist={rec['chosen_m_hist_total']} floors={rec['audio_floor_states']} "
                f"({rec['prefill_s']}s)")
        except Exception as e:
            log(f"[{i+1}] FAILED: {repr(e)[:300]}")
            import traceback
            traceback.print_exc()

    cap.remove()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "policy": {"L_START": L_START, "L_END": L_END, "TOP_K": TOP_K,
                   "top_m_options": [M_SMALL, M_MID, M_FULL],
                   "boundary_threshold": BOUNDARY_THR,
                   "min_audio_tokens": MIN_AUDIO_TOK, "fragile_margin": FRAGILE_MARGIN,
                   "insufficient_floor": INSUFF_FLOOR, "fragile_floor": FRAGILE_FLOOR},
        "n_samples": len(results),
        "samples": results,
    }, indent=1))
    log(f"wrote {out}  ({len(results)} samples)")


if __name__ == "__main__":
    main()

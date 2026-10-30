#!/usr/bin/env python3
"""Capture label-free audio routing signals for compressibility screening.

Frozen protocol:
research_paper/iclr2027/evidence/pm3d_audio_signal_capture_protocol.json.

For every OmniVideoBench example this runs a single native forward (via a
1-token generate so the multimodal rope plumbing is correct) and captures the
router logits at every MoE layer.  For each layer it computes four label-free,
deployable audio signals (no audio_type annotation needed):

  1. audio route uncertainty : audio-token top1-top2 margin (mean) and router
     entropy over all experts (mean).
  2. audio expert concentration: expert-usage entropy over the experts the
     audio tokens actually select (low = focused).
  3. audio-video route consistency: cosine similarity of the audio vs video
     top-k expert-usage histograms (high = coupled).
  4. audio vs text / video divergence: 1 - cosine similarity of audio vs text
     and audio vs video expert-usage histograms.

Sample-level aggregates (mean/median over eligible layers 12-31 and over all
layers) are the screening features, joined offline to P-M3 compressibility by
sample_index.  No generation quality is measured here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO / "research_paper/iclr2027/evidence/pm3d_audio_signal_capture_protocol.json"
)
sys.path.insert(0, str(REPO / "qwen-omni-utils" / "src"))
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
VIDEO_ID = 151656
AUDIO_ID = 151675
IMAGE_ID = 151655
NUM_EXPERTS = 128
TOP_K = 8
ELIGIBLE = set(range(12, 32))


def log(*p: Any) -> None:
    print("[P-M3d]", *p, file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(value, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"bad jsonl {path}:{ln}") from e
    return out


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


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


def _usage_hist(selected: np.ndarray) -> np.ndarray:
    """Expert-usage histogram (normalized) over NUM_EXPERTS from selected top-k ids."""
    h = np.bincount(selected.reshape(-1), minlength=NUM_EXPERTS).astype(np.float64)
    s = h.sum()
    return h / s if s > 0 else h


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a.dot(b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def layer_signals(
    probs: np.ndarray, sel: np.ndarray, masks: dict[str, np.ndarray]
) -> dict[str, float]:
    """probs: (seq,128) softmax; sel: (seq,top_k) selected expert ids."""
    vm, am, tm = masks["video"], masks["audio"], masks["text"]
    out: dict[str, float] = {}
    # audio route uncertainty
    if am.any():
        ap = probs[am]
        top = np.sort(ap, axis=1)[:, ::-1]
        out["audio_margin_top12"] = float((top[:, 0] - top[:, 1]).mean())
        out["audio_router_entropy"] = float(np.mean([_entropy(row) for row in ap]))
        # audio expert concentration: usage entropy over audio-selected experts
        out["audio_expert_usage_entropy"] = _entropy(_usage_hist(sel[am]))
    else:
        out["audio_margin_top12"] = float("nan")
        out["audio_router_entropy"] = float("nan")
        out["audio_expert_usage_entropy"] = float("nan")
    # cross-modal routing histograms
    ah = _usage_hist(sel[am]) if am.any() else np.zeros(NUM_EXPERTS)
    vh = _usage_hist(sel[vm]) if vm.any() else np.zeros(NUM_EXPERTS)
    th = _usage_hist(sel[tm]) if tm.any() else np.zeros(NUM_EXPERTS)
    out["audio_video_consistency"] = _cosine(ah, vh)
    out["audio_video_divergence"] = 1.0 - _cosine(ah, vh)
    out["audio_text_divergence"] = 1.0 - _cosine(ah, th)
    out["n_audio"] = int(am.sum())
    out["n_video"] = int(vm.sum())
    out["n_text"] = int(tm.sum())
    return out


class RouterCapture:
    def __init__(self, model: Any) -> None:
        self.by_layer: dict[int, torch.Tensor] = {}
        self.handles = []
        for name, module in model.thinker.named_modules():
            cls = module.__class__.__name__.lower()
            if "sparsemoeblock" in cls and hasattr(module, "gate"):
                li = self._layer_idx(name)
                if li is None:
                    continue
                self.handles.append(module.gate.register_forward_hook(self._hook(li)))

    @staticmethod
    def _layer_idx(name: str) -> int | None:
        parts = name.split(".")
        for i, p in enumerate(parts[:-1]):
            if p == "layers" and parts[i + 1].isdigit():
                return int(parts[i + 1])
        return None

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


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL, timeout=20
        ).strip()
    except Exception:
        return "unavailable"


def environment_record(protocol, dataset_path, model_path) -> dict[str, Any]:
    cfg = model_path / "config.json"
    return {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "requested_worker": os.environ.get("PM3D_WORKER_ID", "unknown"),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_devices": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
        "git_head": git_value("rev-parse", "HEAD"),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "model_config_sha256": sha256_file(cfg) if cfg.exists() else None,
        "protocol": protocol,
    }


def _agg(
    per_layer: list[dict[str, Any]], keys: list[str], layers: set[int]
) -> dict[str, float]:
    out = {}
    for k in keys:
        vals = [
            d[k]
            for d in per_layer
            if d["layer"] in layers and not (isinstance(d[k], float) and np.isnan(d[k]))
        ]
        out[k + "_mean"] = float(np.mean(vals)) if vals else float("nan")
        out[k + "_median"] = float(np.median(vals)) if vals else float("nan")
    return out


SIGNAL_KEYS = [
    "audio_margin_top12",
    "audio_router_entropy",
    "audio_expert_usage_entropy",
    "audio_video_consistency",
    "audio_video_divergence",
    "audio_text_divergence",
]


def run_sample(
    *, sample_index, record, model, processor, process_mm_info, capture
) -> dict[str, Any]:
    video_rel = str(record["video"])
    video_path = (REPO / "eval_data/OmniVideoBench" / video_rel).resolve()
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
    seqlen = int(inputs["input_ids"].shape[1])
    flat = inputs["input_ids"].reshape(-1).cpu().numpy()
    masks = {
        "video": flat == VIDEO_ID,
        "audio": flat == AUDIO_ID,
        "text": ~((flat == VIDEO_ID) | (flat == AUDIO_ID) | (flat == IMAGE_ID)),
    }

    capture.reset()
    with torch.inference_mode():
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

    per_layer = []
    for li in sorted(capture.by_layer.keys()):
        logits = capture.by_layer[li]
        if logits.shape[0] != seqlen:
            continue
        probs_t = torch.softmax(logits, dim=1, dtype=torch.float)
        sel_t = torch.topk(probs_t, TOP_K, dim=-1).indices
        sig = layer_signals(probs_t.numpy(), sel_t.numpy(), masks)
        sig["layer"] = li
        per_layer.append(sig)

    agg_elig = _agg(per_layer, SIGNAL_KEYS, ELIGIBLE)
    agg_all = _agg(per_layer, SIGNAL_KEYS, set(range(48)))
    return {
        "sample_index": sample_index,
        "video": video_rel,
        "video_cluster": video_rel,
        "question_type": record.get("question_type"),
        "audio_type": record.get("audio_type"),
        "input_sequence_length": seqlen,
        "n_audio_tokens": int(masks["audio"].sum()),
        "n_video_tokens": int(masks["video"].sum()),
        "per_layer_signals": per_layer,
        "agg_eligible_12_31": agg_elig,
        "agg_all_layers": agg_all,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--self-test", action="store_true")
    return ap.parse_args()


def self_test() -> None:
    seq = 10
    probs = np.full((seq, NUM_EXPERTS), 1e-6)
    # audio tokens (rows 0-2): concentrated on experts 0-7
    for r in range(3):
        probs[r, :TOP_K] = [0.5, 0.2, 0.1, 0.08, 0.05, 0.03, 0.02, 0.02]
    # video tokens (rows 3-6): concentrated on DIFFERENT experts 20-27
    for r in range(3, 7):
        probs[r, 20:28] = [0.5, 0.2, 0.1, 0.08, 0.05, 0.03, 0.02, 0.02]
    # text tokens (rows 7-9): experts 0-7 (like audio)
    for r in range(7, 10):
        probs[r, :TOP_K] = [0.5, 0.2, 0.1, 0.08, 0.05, 0.03, 0.02, 0.02]
    probs = probs / probs.sum(1, keepdims=True)
    sel = np.argsort(-probs, axis=1)[:, :TOP_K]
    masks = {
        "video": np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0], bool),
        "audio": np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], bool),
        "text": np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], bool),
    }
    s = layer_signals(probs, sel, masks)
    # audio routes like text (same experts) -> low audio-text divergence;
    # audio routes UNLIKE video (different experts) -> high audio-video divergence.
    assert s["audio_text_divergence"] < 0.2, s["audio_text_divergence"]
    assert s["audio_video_divergence"] > 0.8, s["audio_video_divergence"]
    assert s["audio_video_consistency"] < 0.2, s["audio_video_consistency"]
    assert s["audio_margin_top12"] > 0, s
    assert _entropy(np.array([0.5, 0.5])) > 0
    print("P-M3d self-test passed.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.output_dir is None:
        raise SystemExit("--output-dir required unless --self-test")

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    dataset_path = REPO / protocol["dataset"]
    model_path = REPO / protocol["model"]
    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    n = len(records) if args.limit is None else min(len(records), args.limit)
    selected = list(range(n))

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_signals.jsonl"
    err_path = out_dir / "errors.jsonl"

    log("loading Qwen3-Omni model")
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info

    device_map = "auto" if torch.cuda.device_count() > 1 else "cuda"
    log(f"device_map={device_map} gpus={torch.cuda.device_count()}")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2",
    ).eval()
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(str(model_path), use_fast=False)
    atomic_json(
        out_dir / "environment.json",
        environment_record(protocol, dataset_path, model_path),
    )

    capture = RouterCapture(model)
    done_rows = load_jsonl(raw_path)
    done = {int(r["sample_index"]) for r in done_rows}
    errors = load_jsonl(err_path)
    try:
        for ordinal, sidx in enumerate(selected, 1):
            if sidx in done:
                continue
            try:
                row = run_sample(
                    sample_index=sidx,
                    record=records[sidx],
                    model=model,
                    processor=processor,
                    process_mm_info=process_mm_info,
                    capture=capture,
                )
                append_jsonl(raw_path, row)
                done.add(sidx)
                if ordinal <= 3 or ordinal % 25 == 0:
                    a = row["agg_eligible_12_31"]
                    log(
                        f"[{ordinal}/{len(selected)}] s{sidx} "
                        f"a-v_div={a['audio_video_divergence_mean']:.3f} "
                        f"a_margin={a['audio_margin_top12_mean']:.4f}"
                    )
                atomic_json(
                    out_dir / "progress.json",
                    {
                        "updated_at": datetime.now(timezone.utc)
                        .astimezone()
                        .isoformat(),
                        "completed_samples": len(done),
                        "requested_samples": len(selected),
                        "error_records": len(errors),
                        "last_completed_sample_index": sidx,
                    },
                )
            except Exception as exc:
                err = {
                    "sample_index": sidx,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(err_path, err)
                errors.append(err)
                log(f"s{sidx} failed: {exc!r}")
        atomic_json(
            out_dir / "summary.json",
            {
                "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
                "experiment_id": "P-M3d-capture",
                "completed_samples": len(done),
                "failed_samples": len(errors),
                "requested_samples": len(selected),
                "signal_keys": SIGNAL_KEYS,
                "note": "Observational capture; join to P-M3 by sample_index for compressibility screening.",
                "scope": protocol["scope"],
            },
        )
        log(f"finished completed={len(done)} failed={len(errors)}")
        return 0
    finally:
        capture.remove()


if __name__ == "__main__":
    raise SystemExit(main())

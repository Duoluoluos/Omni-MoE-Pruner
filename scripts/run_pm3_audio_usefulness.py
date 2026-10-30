#!/usr/bin/env python3
"""Run P-M3: does cross-modal audio guidance help expert budgeting?

Frozen protocol:
research_paper/iclr2027/evidence/pm3_audio_usefulness_protocol.json.

Seven budget-matched arms test whether audio guidance transfers from token
compression (OmniZip/ContextGuard) to per-token expert budgeting:

  native_top8         - unchanged reference.
  structural_only     - video+depth cutoff budget, R candidates restored by the
                        video token's own least-decisive ranking; NO audio.
  audio_guided_u1     - U1: audio uncertainty picks WHICH R candidates restore.
  matched_random_u1   - U1 control: random R candidates (same per-layer counts).
  audio_guided_u2     - U2: per-sample audio-information score sets the sample
                        mean budget (more uncertain audio -> higher budget).
  matched_schedule_u2 - U2 control: audio-free schedule at the SAME per-sample
                        mean budget.
  audio_type_ceiling  - oracle audio_type guides restoration (non-deployable
                        information-ceiling probe).

Quality: full generated-answer accuracy and gold-option log-likelihood.
Statistics: video-clustered paired bootstrap + pre-registered TOST equivalence.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO / "research_paper/iclr2027/evidence/pm3_audio_usefulness_protocol.json"
)
sys.path.insert(0, str(REPO / "qwen-omni-utils" / "src"))
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
VIDEO_ID = 151656
AUDIO_ID = 151675
OPTION_LETTERS = ["A", "B", "C", "D"]
ARMS = [
    "native_top8",
    "structural_only",
    "audio_guided_u1",
    "matched_random_u1",
    "audio_guided_u2",
    "matched_schedule_u2",
    "audio_type_ceiling",
]
# audio_type -> restoration preference for the ceiling probe (Music/Sound get
# more restoration, i.e. the oracle "audio carries more here" hypothesis).
AUDIO_TYPE_RESTORE_PRIORITY = {"Music": 2, "Sound": 1, "Speech": 0}


def log(*p: Any) -> None:
    print("[P-M3]", *p, file=sys.stderr, flush=True)


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


def parse_answer(text: str) -> str | None:
    up = str(text).strip().upper()
    m = re.search(r"(?:^|[^A-Z])([A-D])(?:[^A-Z]|$)", up)
    if m:
        return m.group(1)
    for c in up:
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


def cutoff_budget(video_top_vals: np.ndarray, policy: dict[str, Any]) -> np.ndarray:
    m_small, m_mid, m_full = policy["top_m_options"]
    thr = policy["cutoff_gap_threshold"]
    nv = video_top_vals.shape[0]
    chosen = np.full(nv, m_full, dtype=np.int64)
    if nv:
        b_small = video_top_vals[:, m_small - 1] - video_top_vals[:, m_small]
        b_mid = video_top_vals[:, m_mid - 1] - video_top_vals[:, m_mid]
        cs = b_small >= thr
        cm = (~cs) & (b_mid >= thr)
        chosen[cm] = m_mid
        chosen[cs] = m_small
    return chosen


def audio_floor_for_layer(
    margin_mean: float, count: int, policy: dict[str, Any]
) -> int:
    m_small, m_mid, m_full = policy["top_m_options"]
    arc = policy["audio_conditioning"]
    if count < arc["min_audio_tokens"]:
        return m_full
    if margin_mean < arc["fragile_margin_threshold"]:
        return m_mid
    return m_small


@dataclass
class LayerPlan:
    video_positions: np.ndarray
    budgets: np.ndarray
    histogram: dict[int, int]


def _hist(b: np.ndarray, opts: list[int]) -> dict[int, int]:
    return {int(m): int((b == m).sum()) for m in opts}


def build_arm_plans(
    *, per_layer_capture, policy, seed, audio_type
) -> dict[str, dict[int, LayerPlan]]:
    """Build per-layer budget plans for the restoration arms.

    U1 arms (structural_only / audio_guided_u1 / matched_random_u1 /
    audio_type_ceiling) restore R = floor(n_candidates/2) below-floor candidates
    per layer, differing only in WHICH candidates. They share the identical
    per-layer 4/6/8 histogram.
    """
    m_small, m_mid, m_full = policy["top_m_options"]
    opts = [m_small, m_mid, m_full]
    rng = np.random.default_rng(seed)
    layers = sorted(per_layer_capture.keys())

    structural, guided, rand, ceiling = {}, {}, {}, {}
    for li in layers:
        cap = per_layer_capture[li]
        vals = cap["video_top_vals"]
        base = cutoff_budget(vals, policy)
        floor = audio_floor_for_layer(
            cap["audio_margin_mean"], cap["audio_count"], policy
        )
        nv = vals.shape[0]
        cand = (
            np.nonzero(base < floor)[0]
            if (nv and floor > m_small)
            else np.zeros(0, dtype=np.int64)
        )
        n_cand = int(cand.size)
        R = n_cand // 2

        def apply(order_in_cand: np.ndarray) -> np.ndarray:
            b = base.copy()
            if R > 0:
                sel = cand[order_in_cand[:R]]
                b[sel] = np.maximum(b[sel], floor)
            return b

        if R > 0:
            decis = vals[cand, 0] - vals[cand, 1]  # decisiveness among candidates
            # structural: restore least-decisive candidates (video-only signal).
            structural[li] = apply(np.argsort(decis))
            # audio_guided_u1: the audio router state at THIS layer genuinely
            # changes WHICH candidates are restored. When the layer's audio is
            # uncertain (margin below the fragile threshold), the audio signal
            # says "protect the tokens the video rule is least sure of" -> restore
            # the MOST-decisive candidates instead of the least-decisive ones.
            # When audio is confident, follow the video-only order. This makes the
            # arm a real per-layer audio-driven target choice, not a copy of
            # structural, while keeping the same count R (matched histogram).
            arc = policy["audio_conditioning"]
            audio_uncertain = (
                cap["audio_count"] >= arc["min_audio_tokens"]
                and cap["audio_margin_mean"] < arc["fragile_margin_threshold"]
            )
            if audio_uncertain:
                guided[li] = apply(np.argsort(-decis))  # audio flips the target set
            else:
                guided[li] = apply(np.argsort(decis))
            rand[li] = apply(rng.permutation(n_cand))
            # ceiling: ORACLE audio_type genuinely changes the restoration TARGET
            # set (same count R, same histogram, so budget stays matched). The
            # oracle hypothesis is that when audio carries independent information
            # (Music/Sound), the model should protect a DIFFERENT set of video
            # tokens than the video-only structural rule would. We encode that as:
            # Music/Sound restore the MOST-decisive candidates, Speech restores
            # the least-decisive (== structural). If even this oracle re-targeting
            # does not beat structural, the "audio signal too weak" attack fails.
            pri = AUDIO_TYPE_RESTORE_PRIORITY.get(audio_type, 0)
            if pri >= 1:
                ceiling[li] = apply(np.argsort(-decis))  # oracle: different targets
            else:
                ceiling[li] = apply(np.argsort(decis))  # Speech: == structural
        else:
            structural[li] = base.copy()
            guided[li] = base.copy()
            rand[li] = base.copy()
            ceiling[li] = base.copy()

    def to_plans(bmap):
        return {
            li: LayerPlan(
                per_layer_capture[li]["video_positions"],
                bmap[li],
                _hist(bmap[li], opts),
            )
            for li in layers
        }

    return {
        "structural_only": to_plans(structural),
        "audio_guided_u1": to_plans(guided),
        "matched_random_u1": to_plans(rand),
        "audio_type_ceiling": to_plans(ceiling),
    }


def build_u2_plans(
    *, per_layer_capture, policy, audio_info_score
) -> dict[str, dict[int, LayerPlan]]:
    """U2: per-sample audio-information score sets the sample mean budget.

    audio_guided_u2: more uncertain audio (higher score) -> restore MORE
    candidates per layer (fraction f_audio), targeting by decisiveness.
    matched_schedule_u2: restore the SAME per-layer count R without audio, using
    a fixed position-order schedule (audio-free). Same R per layer => matched
    per-sample mean budget; different targets => a genuine, non-degenerate
    control that isolates the audio-driven per-sample budget choice.
    """
    m_small, m_mid, m_full = policy["top_m_options"]
    opts = [m_small, m_mid, m_full]
    layers = sorted(per_layer_capture.keys())
    # map audio_info_score in [0,1] to a restoration fraction in [0.25, 0.75].
    f = 0.25 + 0.5 * float(np.clip(audio_info_score, 0.0, 1.0))
    guided, sched = {}, {}
    for li in layers:
        cap = per_layer_capture[li]
        vals = cap["video_top_vals"]
        base = cutoff_budget(vals, policy)
        floor = audio_floor_for_layer(
            cap["audio_margin_mean"], cap["audio_count"], policy
        )
        nv = vals.shape[0]
        cand = (
            np.nonzero(base < floor)[0]
            if (nv and floor > m_small)
            else np.zeros(0, dtype=np.int64)
        )
        n_cand = int(cand.size)
        R = int(round(f * n_cand))
        decis = vals[cand, 0] - vals[cand, 1] if n_cand else np.zeros(0)
        guided_order = np.argsort(decis)  # audio-set budget, decisiveness targets
        sched_order = np.argsort(cand)  # audio-free: fixed position-order schedule
        bg = base.copy()
        bs = base.copy()
        if R > 0:
            gsel = cand[guided_order[:R]]
            ssel = cand[sched_order[:R]]
            bg[gsel] = np.maximum(bg[gsel], floor)
            bs[ssel] = np.maximum(
                bs[ssel], floor
            )  # same R -> matched mean, different targets
        guided[li] = bg
        sched[li] = bs

    def to_plans(bmap):
        return {
            li: LayerPlan(
                per_layer_capture[li]["video_positions"],
                bmap[li],
                _hist(bmap[li], opts),
            )
            for li in layers
        }

    return {"audio_guided_u2": to_plans(guided), "matched_schedule_u2": to_plans(sched)}


class BudgetIntervention:
    def __init__(self, model, top_k, eligible_layers):
        self.model = model
        self.top_k = int(top_k)
        self.eligible = set(int(x) for x in eligible_layers)
        self.active = None
        self._wrapped = []
        installed = 0
        for name, module in model.thinker.named_modules():
            if "sparsemoeblock" not in module.__class__.__name__.lower():
                continue
            if not all(
                hasattr(module, a) for a in ("gate", "experts", "top_k", "num_experts")
            ):
                continue
            li = self._layer_idx(name)
            if li is None:
                continue
            self._install(module, li)
            installed += 1
        if installed == 0:
            raise RuntimeError("No MoE blocks found")
        self.installed = installed
        log(
            f"installed budget intervention on {installed} MoE layers ({len(self.eligible)} eligible)"
        )

    @staticmethod
    def _layer_idx(name):
        parts = name.split(".")
        for i, p in enumerate(parts[:-1]):
            if p == "layers" and parts[i + 1].isdigit():
                return int(parts[i + 1])
        return None

    def _install(self, module, li):
        original = module.forward

        def wrapped(hidden_states):
            plan = None if self.active is None else self.active.get(li)
            if plan is None:
                return original(hidden_states)
            return self._budgeted_forward(module, hidden_states, plan)

        module.forward = wrapped
        self._wrapped.append((module, original))

    def _budgeted_forward(self, module, hidden_states, budget_vec):
        batch, seqlen, hid = hidden_states.shape
        flat = hidden_states.view(-1, hid)
        router_logits = module.gate(flat)
        rw = torch.softmax(router_logits, dim=1, dtype=torch.float)
        rw, sel = torch.topk(rw, module.top_k, dim=-1)
        n = flat.shape[0]
        budgets = (
            budget_vec.to(rw.device)
            if budget_vec.shape[0] == n
            else torch.full((n,), module.top_k, device=rw.device, dtype=torch.long)
        )
        pos = torch.arange(module.top_k, device=rw.device).unsqueeze(0)
        keep = pos < budgets.unsqueeze(1)
        rw = rw.masked_fill(~keep, 0.0)
        rw = rw / rw.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        rw = rw.to(flat.dtype)
        final = torch.zeros((n, hid), dtype=flat.dtype, device=flat.device)
        emask = torch.nn.functional.one_hot(
            sel, num_classes=module.num_experts
        ).permute(2, 1, 0)
        for e in torch.greater(emask.sum(dim=(-1, -2)), 0).nonzero():
            idx, top_x = torch.where(emask[e].squeeze(0))
            if top_x.numel() == 0:
                continue
            w = rw[top_x, idx, None]
            ks = w.squeeze(-1) > 0
            if not bool(ks.any()):
                continue
            top_x = top_x[ks]
            idx = idx[ks]
            cur = (
                module.experts[e](flat[None, top_x].reshape(-1, hid))
                * rw[top_x, idx, None]
            )
            final.index_add_(0, top_x, cur.to(flat.dtype))
        return final.reshape(batch, seqlen, hid), router_logits

    @contextlib.contextmanager
    def use_plan(self, plan):
        prev = self.active
        self.active = plan
        try:
            yield
        finally:
            self.active = prev

    def close(self):
        for module, original in self._wrapped:
            module.forward = original
        self._wrapped.clear()


class RouterProbeCapture:
    def __init__(self, model, eligible_layers):
        self.eligible = set(int(x) for x in eligible_layers)
        self.by_layer = {}
        self.handles = []
        for name, module in model.thinker.named_modules():
            if "sparsemoeblock" in module.__class__.__name__.lower() and hasattr(
                module, "gate"
            ):
                li = BudgetIntervention._layer_idx(name)
                if li is None or li not in self.eligible:
                    continue
                self.handles.append(module.gate.register_forward_hook(self._hook(li)))

    def _hook(self, li):
        def hook(_m, _i, out):
            logits = out.detach()
            if logits.dim() != 2:
                return
            prev = self.by_layer.get(li)
            if prev is None or logits.shape[0] > prev.shape[0]:
                self.by_layer[li] = logits.float().cpu()

        return hook

    def reset(self):
        self.by_layer = {}

    def remove(self):
        for h in self.handles:
            h.remove()


def distribution_js(a_logits, b_logits):
    a = torch.log_softmax(a_logits.double(), dim=-1)
    b = torch.log_softmax(b_logits.double(), dim=-1)
    pa, pb = a.exp(), b.exp()
    m = (0.5 * (pa + pb)).clamp_min(torch.finfo(torch.float64).tiny).log()
    js = 0.5 * (torch.sum(pa * (a - m)) + torch.sum(pb * (b - m)))
    return max(0.0, float(js.item()))


def gold_logprob(first_logits, opt_ids, gold):
    lp = torch.log_softmax(first_logits.double(), dim=-1)
    tid = opt_ids.get(gold)
    return float("nan") if tid is None else float(lp[tid].item())


def git_value(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL, timeout=20
        ).strip()
    except Exception:
        return "unavailable"


def environment_record(protocol, dataset_path, model_path):
    cfg = model_path / "config.json"
    pol = REPO / protocol["verified_policy"]["source_config"]
    return {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "requested_worker": os.environ.get("PM3_WORKER_ID", "unknown"),
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
        "policy_config_sha256": sha256_file(pol) if pol.exists() else None,
        "model_config_sha256": sha256_file(cfg) if cfg.exists() else None,
        "protocol": protocol,
    }


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
    opt_ids,
    max_new,
    base_seed,
) -> dict[str, Any]:
    video_rel = str(record["video"])
    vp = (REPO / "eval_data/OmniVideoBench" / video_rel).resolve()
    if not vp.exists():
        raise FileNotFoundError(vp)
    conv = build_conversation(vp, record)
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
    correct = str(record["correct_option"]).upper()

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

    per_layer = {}
    audio_margins = []
    for li in sorted(eligible_layers):
        logits = capture.by_layer.get(li)
        if logits is None or logits.shape[0] != seqlen:
            continue
        probs = torch.softmax(logits, dim=1, dtype=torch.float)
        tv, _ = torch.topk(probs, top_k, dim=-1, sorted=True)
        tv = tv.numpy()
        vpos = np.nonzero(flat == VIDEO_ID)[0]
        apos = np.nonzero(flat == AUDIO_ID)[0]
        a = tv[apos]
        am = (a[:, 0] - a[:, 1]) if a.shape[0] else np.array([])
        amm = float(am.mean()) if am.size else 0.0
        per_layer[li] = {
            "video_positions": vpos,
            "video_top_vals": tv[vpos],
            "audio_margin_mean": amm,
            "audio_count": int(a.shape[0]),
        }
        if am.size:
            audio_margins.append(amm)
    # sample-level audio-information score: lower mean margin = more uncertain =
    # higher score in [0,1] (min-max within a fixed reasonable range).
    mm = float(np.mean(audio_margins)) if audio_margins else 0.05
    audio_info_score = float(np.clip((0.05 - mm) / 0.05, 0.0, 1.0))

    u1 = build_arm_plans(
        per_layer_capture=per_layer,
        policy=policy,
        seed=base_seed + sample_index,
        audio_type=record.get("audio_type"),
    )
    u2 = build_u2_plans(
        per_layer_capture=per_layer, policy=policy, audio_info_score=audio_info_score
    )
    plans = {**u1, **u2}

    # budget matching checks
    match_ok = True
    hist_records = {}
    for arm in [
        "structural_only",
        "audio_guided_u1",
        "matched_random_u1",
        "audio_type_ceiling",
    ]:
        hist_records[arm] = {
            str(li): {str(k): v for k, v in plans[arm][li].histogram.items()}
            for li in plans[arm]
        }
    for li in plans["structural_only"]:
        h = plans["structural_only"][li].histogram
        for arm in ["audio_guided_u1", "matched_random_u1", "audio_type_ceiling"]:
            if plans[arm][li].histogram != h:
                match_ok = False

    # U2 matched by construction (same targets/count); verify mean budget equal
    def mean_budget(arm):
        b = (
            np.concatenate([plans[arm][li].budgets for li in plans[arm]])
            if plans[arm]
            else np.array([])
        )
        return float(b.mean()) if b.size else float(top_k)

    if abs(mean_budget("audio_guided_u2") - mean_budget("matched_schedule_u2")) > 1e-9:
        match_ok = False

    def dense(arm):
        if arm == "native_top8":
            return None
        src = plans[arm]
        out = {}
        for li, lp in src.items():
            vec = torch.full((seqlen,), top_k, dtype=torch.long)
            vec[torch.as_tensor(lp.video_positions, dtype=torch.long)] = (
                torch.as_tensor(lp.budgets, dtype=torch.long)
            )
            out[li] = vec
        return out

    rng = np.random.default_rng(base_seed + sample_index)
    order = list(ARMS)
    rng.shuffle(order)

    native_fl = None
    res = {}
    for arm in order:
        plan = dense(arm)
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
        e2e = (time.perf_counter() - t0) * 1000.0
        th = raw[0] if isinstance(raw, tuple) else raw
        fl = th.scores[0][0].detach().float().cpu()
        gen = th.sequences[0, seqlen:].detach().cpu()
        out_text = processor.tokenizer.decode(
            gen, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        ans = parse_answer(out_text)
        if arm == "native_top8":
            native_fl = fl
        mb = float(top_k) if plan is None else mean_budget(arm)
        res[arm] = {
            "answer": ans,
            "is_correct": int(ans == correct),
            "gold_option_logprob": gold_logprob(fl, opt_ids, correct),
            "end_to_end_latency_ms": e2e,
            "mean_video_expert_budget": mb,
            "_fl": fl,
        }
    for arm in ARMS:
        fl = res[arm].pop("_fl")
        res[arm]["first_token_js_vs_native"] = (
            0.0 if native_fl is None else distribution_js(native_fl, fl)
        )

    qt = record.get("question_type")
    ad = qt in {"background&music understanding", "sentiment analysis"}
    return {
        "sample_index": sample_index,
        "video": video_rel,
        "video_cluster": video_rel,
        "question_type": qt,
        "audio_type": record.get("audio_type"),
        "audio_dependent_stratum": bool(ad),
        "correct_option": correct,
        "input_sequence_length": seqlen,
        "audio_info_score": audio_info_score,
        "arm_order": order,
        "budget_match_ok": bool(match_ok),
        "per_layer_budget_histograms": hist_records,
        "arms": res,
    }


def _cluster_boot(pairs, seed, draws):
    if not pairs:
        return {
            "mean_diff": None,
            "ci95_low": None,
            "ci95_high": None,
            "n": 0,
            "clusters": 0,
        }
    by = defaultdict(list)
    for c, a, b in pairs:
        by[c].append(a - b)
    ids = list(by)
    cm = np.array([np.mean(by[c]) for c in ids])
    ov = float(np.concatenate([np.array(by[c]) for c in ids]).mean())
    rng = np.random.default_rng(seed)
    k = len(ids)
    boot = np.array([cm[rng.integers(0, k, k)].mean() for _ in range(draws)])
    return {
        "mean_diff": ov,
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
    return _cluster_boot(pairs, seed, draws)


def _tost(ci, margin):
    """Equivalence: CI entirely within +/- margin."""
    if ci["ci95_low"] is None:
        return None
    return bool(ci["ci95_low"] > -margin and ci["ci95_high"] < margin)


def summarize(rows, errors, protocol, records):
    n = len(rows)
    draws = int(protocol["statistics"]["draws"])
    seed = int(protocol["statistics"]["seed"])
    minimum = int(protocol["gate"]["minimum_completed_samples"])
    match_all = all(r["budget_match_ok"] for r in rows) if rows else False
    eq = protocol["statistics"]["equivalence_test"]
    m_acc = float(eq["margin_accuracy_pp"]) / 100.0
    m_lp = float(eq["margin_gold_logprob_nats"])

    def arm_means(subset=None):
        out = {}
        for arm in ARMS:
            accs, lps, buds = [], [], []
            for r in rows:
                if subset is not None and not r.get(subset):
                    continue
                a = r["arms"][arm]
                accs.append(a["is_correct"])
                if a["gold_option_logprob"] is not None and not np.isnan(
                    a["gold_option_logprob"]
                ):
                    lps.append(a["gold_option_logprob"])
                buds.append(a["mean_video_expert_budget"])
            out[arm] = {
                "accuracy": float(np.mean(accs)) if accs else None,
                "gold_logprob": float(np.mean(lps)) if lps else None,
                "mean_video_expert_budget": float(np.mean(buds)) if buds else None,
                "n": len(accs),
            }
        return out

    def contrast_block(subset=None):
        block = {}
        for c in protocol["primary_contrasts"]:
            a, b = [x.strip() for x in c.split("minus")]
            acc = _contrast(rows, "is_correct", a, b, seed, draws, subset)
            lp = _contrast(rows, "gold_option_logprob", a, b, seed + 1, draws, subset)
            block[c] = {
                "accuracy": {**acc, "equivalent": _tost(acc, m_acc)},
                "gold_logprob": {**lp, "equivalent": _tost(lp, m_lp)},
            }
        return block

    verified = n >= minimum and match_all
    return {
        "schema_version": "arc-sceb/pm3-audio-usefulness-results-v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "experiment_id": "P-M3",
        "evidence_status": "verified" if verified else "placeholder",
        "completed_samples": n,
        "failed_samples": len(errors),
        "requested_samples": len(records),
        "budget_match_all_samples": match_all,
        "tost_margins": {
            "accuracy_pp": eq["margin_accuracy_pp"],
            "gold_logprob_nats": eq["margin_gold_logprob_nats"],
        },
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--self-test", action="store_true")
    return ap.parse_args()


def self_test():
    policy = {
        "top_m_options": [4, 6, 8],
        "cutoff_gap_threshold": 0.05,
        "audio_conditioning": {
            "min_audio_tokens": 64,
            "fragile_margin_threshold": 0.0082,
        },
    }
    vals = np.array(
        [
            [0.50, 0.30, 0.15, 0.10, 0.010, 0.008, 0.006, 0.004],
            [0.45, 0.30, 0.15, 0.10, 0.010, 0.008, 0.006, 0.004],
            [0.40, 0.30, 0.15, 0.10, 0.010, 0.008, 0.006, 0.004],
            [0.35, 0.30, 0.15, 0.10, 0.010, 0.008, 0.006, 0.004],
        ]
    )
    cap = {
        12: {
            "video_positions": np.arange(4),
            "video_top_vals": vals,
            "audio_margin_mean": 0.001,
            "audio_count": 100,
        }
    }
    u1 = build_arm_plans(
        per_layer_capture=cap, policy=policy, seed=1, audio_type="Music"
    )
    hs = u1["structural_only"][12].histogram
    for arm in ["audio_guided_u1", "matched_random_u1", "audio_type_ceiling"]:
        assert u1[arm][12].histogram == hs, (arm, u1[arm][12].histogram, hs)
    # non-degeneracy: with uncertain audio (margin 0.001 < 0.0082), audio_guided_u1
    # must target a DIFFERENT candidate set than the video-only structural rule.
    assert not np.array_equal(
        u1["audio_guided_u1"][12].budgets, u1["structural_only"][12].budgets
    ), "audio_guided_u1 degenerate: identical to structural"
    # non-degeneracy: with Music oracle, ceiling must target a DIFFERENT set than
    # structural (same count/histogram, different restored positions). The 4
    # candidates have distinct decisiveness, so least- vs most-decisive differ.
    assert not np.array_equal(
        u1["audio_type_ceiling"][12].budgets, u1["structural_only"][12].budgets
    ), "ceiling degenerate: identical to structural"
    # Speech oracle == structural (control that oracle only acts on Music/Sound).
    u1_speech = build_arm_plans(
        per_layer_capture=cap, policy=policy, seed=1, audio_type="Speech"
    )
    assert np.array_equal(
        u1_speech["audio_type_ceiling"][12].budgets,
        u1_speech["structural_only"][12].budgets,
    )
    u2 = build_u2_plans(per_layer_capture=cap, policy=policy, audio_info_score=0.8)
    bg = np.concatenate([u2["audio_guided_u2"][12].budgets])
    bs = np.concatenate([u2["matched_schedule_u2"][12].budgets])
    assert abs(bg.mean() - bs.mean()) < 1e-9, (bg.mean(), bs.mean())
    # non-degeneracy: U2 guided vs schedule must differ in targets (same mean).
    assert not np.array_equal(bg, bs), "U2 degenerate: guided == schedule"
    lg = torch.tensor([0.1, 5.0, 0.2, 0.3])
    opt = {"A": 0, "B": 1, "C": 2, "D": 3}
    assert gold_logprob(lg, opt, "B") > gold_logprob(lg, opt, "A")
    assert _tost({"ci95_low": -0.003, "ci95_high": 0.004}, 0.005) is True
    assert _tost({"ci95_low": -0.01, "ci95_high": 0.004}, 0.005) is False
    print("P-M3 self-test passed.")


def main():
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

    policy = {
        "top_m_options": protocol["verified_policy"]["top_m_options"],
        "cutoff_gap_threshold": protocol["verified_policy"]["cutoff_gap_threshold"],
        "audio_conditioning": protocol["verified_policy"]["audio_conditioning"],
    }
    es = protocol["verified_policy"]["eligible_moe_layers"]["start_inclusive"]
    ee = protocol["verified_policy"]["eligible_moe_layers"]["end_exclusive"]
    eligible = set(range(es, ee))
    top_k = protocol["verified_policy"]["original_top_k"]
    max_new = int(protocol["decoding"]["thinker_max_new_tokens"])
    base_seed = int(protocol["decoding"]["seed"])

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_results.jsonl"
    err_path = out_dir / "errors.jsonl"

    import random as _r

    _r.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)

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
    opt_ids = {}
    for letter in OPTION_LETTERS:
        ids = processor.tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) == 1:
            opt_ids[letter] = ids[0]
    atomic_json(
        out_dir / "environment.json",
        environment_record(protocol, dataset_path, model_path),
    )

    intervention = BudgetIntervention(model, top_k=top_k, eligible_layers=eligible)
    capture = RouterProbeCapture(model, eligible)
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
                    intervention=intervention,
                    capture=capture,
                    policy=policy,
                    eligible_layers=eligible,
                    top_k=top_k,
                    opt_ids=opt_ids,
                    max_new=max_new,
                    base_seed=base_seed,
                )
                append_jsonl(raw_path, row)
                done_rows.append(row)
                done.add(sidx)
                if ordinal <= 3 or ordinal % 20 == 0:
                    g = row["arms"]["audio_guided_u1"]
                    s = row["arms"]["structural_only"]
                    log(
                        f"[{ordinal}/{len(selected)}] s{sidx} match={row['budget_match_ok']} "
                        f"u1_guided_c={g['is_correct']} struct_c={s['is_correct']}"
                    )
                atomic_json(
                    out_dir / "progress.json",
                    {
                        "updated_at": datetime.now(timezone.utc)
                        .astimezone()
                        .isoformat(),
                        "completed_samples": len(done_rows),
                        "requested_samples": len(selected),
                        "error_records": len(errors),
                        "last_completed_sample_index": sidx,
                    },
                )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                e = {
                    "sample_index": sidx,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(err_path, e)
                errors.append(e)
                log(f"s{sidx} OOM; continuing")
            except Exception as exc:
                e = {
                    "sample_index": sidx,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(err_path, e)
                errors.append(e)
                log(f"s{sidx} failed: {exc!r}")
        summary = summarize(done_rows, errors, protocol, records)
        atomic_json(out_dir / "summary.json", summary)
        log(
            f"finished status={summary['evidence_status']} completed={summary['completed_samples']} "
            f"failed={summary['failed_samples']}"
        )
        return 0 if summary["evidence_status"] == "verified" else 3
    finally:
        capture.remove()
        intervention.close()


if __name__ == "__main__":
    raise SystemExit(main())

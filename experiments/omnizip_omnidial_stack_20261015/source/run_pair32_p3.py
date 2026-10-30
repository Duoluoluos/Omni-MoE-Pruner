#!/usr/bin/env python3
"""Fixed-32 OmniZip vs Stack paired eval for the FROZEN P3 zip-aware policy.

Hard constraints (audited):
- Fixed 32 indices from the frozen protocol; NEVER expanded (--limit still caps but
  the default is exactly 32 and the index set is hard-coded, not re-sampled).
- Same custom-Qwen + Megatron-Core GroupedMLP expert backend (P0 experts);
  HAVE_TE=False: no fused/TE dispatcher anywhere.
- Per-sample two-arm (omnizip, stack) order is fairly randomized with a fixed
  RNG seed (SEED + sample_index) and the realized order is recorded in arm_order.
- Real executed expert rows are measured at runtime via a GPU-side counter;
  nothing is copied from full32_v1.
- Reports paired Stack/OmniZip latency ratio + sample-level bootstrap CI.
- Saves answer, A-D option_logp, gold-logp delta (post-freeze quality only),
  failed samples, env/source hashes, and resumable per-sample progress.

This entry CONSUMES the already-frozen gates.json produced by calibrate_p3.py;
it does NOT recalibrate and does NOT read answers during routing.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, json, os, random, socket, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner")
HERE = Path(__file__).resolve().parent
# Vendored Megatron-Core 0.12.3 (from P0 run.log: .tmp/qwen_runtime_mcore_20260924/python).
# The shared venv does NOT pip-install megatron; this path MUST precede imports.
MCORE_PY = ROOT / ".tmp/qwen_runtime_mcore_20260924/python"
sys.path[:0] = [str(HERE), str(MCORE_PY), str(ROOT), str(ROOT / "qwen-omni-utils/src"), str(ROOT / "scripts")]
from modeling_qwen3_omni_moe_stack import Qwen3OmniMoeForConditionalGeneration  # noqa: E402
from transformers import Qwen3OmniMoeProcessor  # noqa: E402
from qwen_omni_utils import process_mm_info  # noqa: E402
from qwen_mcore_controller_p0 import MCoreQwenControllerP0  # noqa: E402
import run_mb_new2_cgskip_qwen3omni as old  # noqa: E402

MODEL = ROOT / "models/Qwen3-Omni-30B-A3B-Instruct"
DATA = ROOT / "eval_data/OmniVideoBench/data.json"
INPUTS = ROOT / "experiments/omnidial_layer_followup_20260930/qwen_diagnostic32_v1/inputs.json"
# Fixed, frozen 32 indices (identical to protocol.json / run_stack_p2.py). DO NOT EDIT.
INDICES = [32,41,71,88,94,121,200,208,211,288,295,297,315,321,333,344,389,424,515,539,546,585,641,672,678,699,769,859,870,874,882,916]
ARMS = ["omnizip", "stack"]          # two arms only for this paired P3 eval
SEED = 20261015                      # frozen RNG seed, shared with prior arms
SCOPE = (12, 32)
OMNIZIP_CONFIG = {"rho_audio": 0.3, "rho_video": 0.6, "g": 3, "contextual_ratio": 0.05}
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20261015


def atomic(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def arm_order_for(idx: int):
    """Fair, deterministic, machine-auditable within-sample two-arm order."""
    order = list(ARMS)
    random.Random(SEED + idx).shuffle(order)
    return order


def prefix_forward_p3(block, hidden, tau2, tau4, grouped):
    """Zip-aware tiered in-scope forward: per-token budget in {2,4,8}.

    s2 = top-2 softmax mass, s4 = top-4 mass. keep 2 experts if s2>=tau2,
    else 4 if s4>=tau4, else 8. Width stays 8; dropped pairs get weight 0 and
    the grouped experts dispatch only valid (nonzero) rows. Identical mechanism
    to the frozen cgcc forward, just {2,4,8} thresholds from tau2/tau4.
    """
    shape = hidden.shape
    flat = hidden.reshape(-1, shape[-1])
    logits = block.gate(flat)
    w8, s8 = torch.softmax(logits.float(), dim=-1).topk(8, dim=-1)
    s2 = w8[:, :2].sum(-1)
    s4 = w8[:, :4].sum(-1)
    budget = torch.where(s2 >= tau2, 2, torch.where(s4 >= tau4, 4, 8))
    keep = torch.arange(8, device=flat.device)[None, :] < budget[:, None]
    w = w8.masked_fill(~keep, 0.0)
    w = (w / w.sum(-1, keepdim=True).clamp_min(1e-20)).to(flat.dtype)
    return grouped(flat, s8, w).reshape(shape), logits


class MCoreQwenPair3(MCoreQwenControllerP0):
    """P0 backend + a single new audited policy 'p3' (tiered 2/4/8 budget)."""

    def __init__(self, model, scope, tau2, tau4, progress=None):
        self.tau2 = float(tau2)
        self.tau4 = float(tau4)
        super().__init__(model, scope=scope, tau_reduce=None, progress=progress)
        # GPU-side executed-row accumulator (zeroed per measured arm).
        self.row_counter = None

    def wrap(self, layer, block):
        super_wrap = super().wrap(layer, block)

        def forward(hidden):
            if self.policy == "p3" and self.scope[0] <= layer < self.scope[1]:
                return prefix_forward_p3(block, hidden, self.tau2, self.tau4, block.grouped_experts)
            return super_wrap(hidden)
        return forward

    @contextlib.contextmanager
    def arm(self, name):
        if name not in ("native", "top8", "p3"):
            raise ValueError(name)
        old = self.policy
        self.policy = name
        try:
            yield
        finally:
            self.policy = old


def install_row_counter(ctl):
    """Wrap each grouped_experts.forward to count dispatched (token,expert) rows.

    Counting happens on-device; a single .item() read happens per arm so the
    timed forward is not polluted by 48 GPU->CPU syncs.
    """
    ctl.row_counter = torch.zeros((), dtype=torch.long, device="cuda")
    for block in ctl.blocks:
        orig = block.grouped_experts.forward

        def make(orig=orig):
            def wrapped(flat, selected, weights):
                ctl.row_counter.add_((weights.reshape(-1) > 0).sum())
                return orig(flat, selected, weights)
            return wrapped
        block.grouped_experts.forward = make(orig)


def load_frozen_gates(gates_path: Path):
    g = json.loads(gates_path.read_text())
    assert g.get("answer_fields_read") is False, "gates.json unexpectedly read answers"
    sp = g["selected_policy"]
    tau2, tau4 = float(sp["tau2"]), float(sp["tau4"])
    assert g.get("scope") == [12, 32], "gates scope mismatch"
    return {"path": str(gates_path), "sha256": sha(gates_path), "selected": g["selected"],
            "tau2": tau2, "tau4": tau4, "captured_rows": g.get("captured_rows"),
            "selected_policy": sp}


def bootstrap_paired(s, z, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """Sample-level PAIRED bootstrap CI for the *mean* (not the individual spread).

    Each bootstrap replicate resamples the n sample indices as a matched pair
    (same idx for both arms). Within one replicate we compute the statistic on the
    resampled n, giving ONE value per replicate; the 95% CI is the 2.5/97.5
    percentiles over the B replicate-level statistics. This estimates uncertainty
    of the population mean, not dispersion across samples.
    Returns three paired estimators:
      - mean_of_ratios : mean_i( s_i/z_i )
      - ratio_of_means : mean(s)/mean(z)
      - paired_mean_diff: mean_i( s_i - z_i )
    """
    s = np.asarray(s, dtype=float); z = np.asarray(z, dtype=float)
    n = len(s)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))     # (B, n) paired draws
    sb = s[idx]; zb = z[idx]                        # (B, n)
    mor = (sb / zb).mean(axis=1)                    # mean-of-ratios per replicate
    rom = sb.mean(axis=1) / zb.mean(axis=1)         # ratio-of-means per replicate
    mdiff = (sb - zb).mean(axis=1)                  # paired mean diff per replicate
    def pct(v): return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {
        "mean_of_ratios": float((s / z).mean()),
        "ci95_mean_of_ratios": pct(mor),
        "ratio_of_means": float(s.mean() / z.mean()),
        "ci95_ratio_of_means": pct(rom),
        "paired_mean_diff_s": float((s - z).mean()),
        "ci95_paired_mean_diff_s": pct(mdiff),
        "bootstrap_method": ("per-replicate sample mean; paired index resampling "
                             "(B=%d, seed=%d); percentiles over %d replicate means, "
                             "NOT over flattened B*n individual values" % (n_boot, seed, n_boot)),
    }


# Back-compat name used by older callers.
def bootstrap_ratio_ci(stack_lat, zip_lat, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    return bootstrap_paired(stack_lat, zip_lat, n_boot=n_boot, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gates", type=Path, required=True, help="frozen gates.json from calibrate_p3.py")
    ap.add_argument("--limit", type=int, default=32, help="<=32; never expands the fixed set")
    args = ap.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True); (out / "predictions").mkdir(exist_ok=True)
    assert 1 <= args.limit <= 32
    indices = INDICES[:args.limit]                       # strict prefix of the frozen 32
    frozen = load_frozen_gates(args.gates)

    records = json.loads(INPUTS.read_text())
    gold = json.loads(DATA.read_text())

    # resumable: skip samples already written
    done = {}
    raw_path = out / "raw_results.jsonl"
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done[r["sample_index"]] = r
    todo = [i for i in indices if i not in done]
    # Resume guard: never mix results from a different frozen gates (e.g. smoke dir
    # accidentally reused as the formal dir). If a prior protocol recorded a different
    # gates sha, refuse to continue.
    prev_proto = out / "protocol.json"
    if done and prev_proto.exists():
        prev_gates_sha = json.loads(prev_proto.read_text()).get("frozen_gates", {}).get("sha256")
        if prev_gates_sha and prev_gates_sha != frozen["sha256"]:
            raise SystemExit(f"REFUSE: {out} has results from gates {prev_gates_sha[:12]}; "
                             f"current gates {frozen['sha256'][:12]}. Use a fresh --out.")

    protocol = {"schema": "omnizip-vs-stack-p3-paired-v1", "model": str(MODEL), "dataset": str(DATA),
                "dataset_sha256": sha(DATA), "indices": indices, "seed": SEED, "dtype": "bfloat16",
                "batch_size": 1, "max_new_tokens": 4, "arms": ARMS, "scope": list(SCOPE),
                "frozen_gates": frozen, "omnizip": OMNIZIP_CONFIG,
                "row_accounting": "runtime GPU counter; NOT copied from full32_v1",
                "ordering": "random.Random(SEED+sample_index).shuffle([omnizip,stack]); recorded per row",
                "metrics": ["latency_s", "expert_rows(real)", "option_logp", "gold_logp", "gold_delta_stack_minus_omnizip", "accuracy"]}
    atomic(out / "protocol.json", protocol)

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    progress = lambda stage, **kw: atomic(out / "progress.json", {"stage": stage, "time": time.time(),
                                                                  "pid": os.getpid(), "host": socket.gethostname(), **kw})
    progress("loading")
    processor = Qwen3OmniMoeProcessor.from_pretrained(str(MODEL), use_fast=False)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16,
                                                                 device_map="cuda", attn_implementation="flash_attention_2").eval()
    model.disable_talker()
    model.thinker.omnizip_config = dict(OMNIZIP_CONFIG); model.thinker.enable_omnizip = False
    progress("converting_mcore")
    ctl = MCoreQwenPair3(model, scope=SCOPE, tau2=frozen["tau2"], tau4=frozen["tau4"],
                         progress=lambda layer: progress("converting_mcore", layer=layer))
    install_row_counter(ctl)
    option_ids = {s: processor.tokenizer.encode(s, add_special_tokens=False)[0] for s in "ABCD"}
    import qwen_mcore_experts_p0, qwen_mcore_controller_p0
    env = {"python": sys.executable, "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(), "model_source": str(Path(sys.modules[Qwen3OmniMoeForConditionalGeneration.__module__].__file__)),
           "model_source_sha256": sha(Path(sys.modules[Qwen3OmniMoeForConditionalGeneration.__module__].__file__)),
           "runner_sha256": sha(Path(__file__)), "frozen_gates_sha256": frozen["sha256"],
           "mcore_experts_source": str(Path(qwen_mcore_experts_p0.__file__).resolve()),
           "mcore_controller_source": str(Path(qwen_mcore_controller_p0.__file__).resolve()),
           "backend": "custom-Qwen + MCore GroupedMLP P0; HAVE_TE=False; no fused/TE dispatcher",
           "mcore_python_path": str(MCORE_PY),
           "mcore": __import__("importlib.metadata", fromlist=["version"]).version("megatron-core")}
    atomic(out / "environment.json", env)

    def prepare(idx):
        rec = records[idx]
        conv = old.build_conversation(ROOT / "eval_data/OmniVideoBench" / rec["video"], rec)
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=True)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt",
                           padding=True, use_audio_in_video=True)
        return inputs.to(model.device).to(model.dtype)

    def run_arm(inputs, name):
        # both arms: OmniZip ON. Only in-scope expert policy differs.
        model.thinker.enable_omnizip = True
        model.thinker._stack_last_omnizip_mask = None
        policy = "native" if name == "omnizip" else "p3"
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        ctl.row_counter.zero_()
        with ctl.arm(policy):
            value = old.generate_once(model, processor, inputs, 4)
        torch.cuda.synchronize()
        expert_rows = int(ctl.row_counter.item())
        logits = value["first_logits"].reshape(-1).float(); lp = logits.log_softmax(-1)
        mask = getattr(model.thinker, "_stack_last_omnizip_mask", None)
        return {"answer": value["answer"], "output_text": value["output_text"],
                "latency_s": value["end_to_end_latency_ms"] / 1000.0,
                "option_logp": {s: float(lp[i]) for s, i in option_ids.items()},
                "retained_tokens": int(mask.sum()) if mask is not None else int(inputs["input_ids"].numel()),
                "input_tokens": int(inputs["input_ids"].numel()),
                "expert_rows": expert_rows, "expert_rows_source": "runtime_real_row_counter",
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}

    # warm both paths on first sample (not measured)
    if todo:
        warm = prepare(todo[0])
        for name in ARMS:
            progress("warmup", arm=name); run_arm(warm, name)
        del warm; torch.cuda.empty_cache()

    failures = out / "failures.jsonl"
    for pos, idx in enumerate(todo):
        progress("prepare", sample_index=idx, completed=len(done) + pos, total=len(indices))
        try:
            inputs = prepare(idx)
            order = arm_order_for(idx)
            arm_rows = {}
            for name in order:
                progress("generate", sample_index=idx, arm=name, completed=len(done) + pos, total=len(indices))
                arm_rows[name] = run_arm(inputs, name)
            correct = str(gold[idx]["correct_option"]).upper()
            for name in ARMS:
                arm_rows[name]["gold_logp"] = arm_rows[name]["option_logp"][correct]
                arm_rows[name]["correct"] = (arm_rows[name]["answer"] == correct)
            arm_rows["stack"]["gold_delta_minus_omnizip"] = arm_rows["stack"]["gold_logp"] - arm_rows["omnizip"]["gold_logp"]
            row = {"sample_index": idx, "video": records[idx]["video"], "correct_option": correct,
                   "arm_order": order, "arms": arm_rows, "complete": True}
            atomic(out / "predictions" / f"{idx:04d}.json", row); done[idx] = row
            with raw_path.open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n"); f.flush(); os.fsync(f.fileno())
            del inputs
        except Exception as e:  # record and continue; resumable later
            with failures.open("a") as f:
                f.write(json.dumps({"sample_index": idx, "time": time.time(), "error": repr(e),
                                    "traceback": traceback.format_exc()}) + "\n")
        torch.cuda.empty_cache()
        progress("sample_done", sample_index=idx, completed=len(done), total=len(indices))

    rows = [done[i] for i in indices if i in done]
    if len(rows) == len(indices):
        stack_lat = [r["arms"]["stack"]["latency_s"] for r in rows]
        zip_lat = [r["arms"]["omnizip"]["latency_s"] for r in rows]
        stack_rows = [r["arms"]["stack"]["expert_rows"] for r in rows]
        zip_rows = [r["arms"]["omnizip"]["expert_rows"] for r in rows]
        lat_bs = bootstrap_paired(stack_lat, zip_lat)
        row_bs = bootstrap_paired(stack_rows, zip_rows)
        atomic(out / "latency_summary.json", {
            "n": len(rows),
            "stack_mean_latency_s": float(np.mean(stack_lat)),
            "omnizip_mean_latency_s": float(np.mean(zip_lat)),
            "latency": lat_bs,
            "expert_row_ratio": {
                "mean_of_ratios": row_bs["mean_of_ratios"],
                "ci95_mean_of_ratios": row_bs["ci95_mean_of_ratios"],
                "ratio_of_means": row_bs["ratio_of_means"],
                "ci95_ratio_of_means": row_bs["ci95_ratio_of_means"],
            },
        })
        atomic(out / "inference_complete.json", {"n": len(rows), "time": time.time(), "raw_sha256": sha(raw_path)})
        progress("complete", completed=len(rows), total=len(indices))
    else:
        progress("incomplete", completed=len(rows), total=len(indices))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if "--out" in sys.argv:
            p = Path(sys.argv[sys.argv.index("--out") + 1]); p.mkdir(parents=True, exist_ok=True)
            atomic(p / "failure.json", {"time": time.time(), "traceback": traceback.format_exc()})
        raise

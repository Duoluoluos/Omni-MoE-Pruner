#!/usr/bin/env python3
"""Regenerate latency_summary.json from an existing raw_results.jsonl.
NO model is loaded: heavy modules are stubbed so we can import the FIXED
bootstrap_paired from run_pair32_p3. raw_results.jsonl and predictions/ are
READ-ONLY here. Idempotent derived product.
"""
import sys, types, json, hashlib
from pathlib import Path

for name in ["torch", "transformers", "modeling_qwen3_omni_moe_stack",
             "qwen_omni_utils", "qwen_mcore_controller_p0",
             "run_mb_new2_cgskip_qwen3omni"]:
    sys.modules[name] = types.ModuleType(name)
sys.modules["torch"].__version__ = "2.6.0+cu124"
sys.modules["qwen_mcore_controller_p0"].MCoreQwenControllerP0 = type("C", (), {"__init__": lambda *a, **k: None})
sys.modules["transformers"].Qwen3OmniMoeProcessor = object
sys.modules["modeling_qwen3_omni_moe_stack"].Qwen3OmniMoeForConditionalGeneration = object
sys.modules["qwen_omni_utils"].process_mm_info = lambda *a, **k: (None, None, None)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_pair32_p3 as R

OUT = Path(sys.argv[1]).resolve()
rows = [json.loads(l) for l in (OUT / "raw_results.jsonl").read_text().splitlines() if l.strip()]
stack_lat = [r["arms"]["stack"]["latency_s"] for r in rows]
zip_lat = [r["arms"]["omnizip"]["latency_s"] for r in rows]
stack_rows = [r["arms"]["stack"]["expert_rows"] for r in rows]
zip_rows = [r["arms"]["omnizip"]["expert_rows"] for r in rows]

import numpy as np
lat = R.bootstrap_paired(stack_lat, zip_lat)
row = R.bootstrap_paired(stack_rows, zip_rows)

summary = {
    "n": len(rows),
    "stack_mean_latency_s": float(np.mean(stack_lat)),
    "omnizip_mean_latency_s": float(np.mean(zip_lat)),
    "latency": lat,
    "expert_row_ratio": {
        "mean_of_ratios": row["mean_of_ratios"],
        "ci95_mean_of_ratios": row["ci95_mean_of_ratios"],
        "ratio_of_means": row["ratio_of_means"],
        "ci95_ratio_of_means": row["ci95_ratio_of_means"],
    },
    "regenerated_from": "raw_results.jsonl (read-only); bootstrap CI corrected to per-replicate mean",
}
(OUT / "latency_summary.json").write_text(json.dumps(summary, indent=2))

raw_sha = hashlib.sha256((OUT / "raw_results.jsonl").read_bytes()).hexdigest()
(Runner_sha := hashlib.sha256((HERE / "run_pair32_p3.py").read_bytes()).hexdigest())
corr = {
    "correction": "old flattened-B*n individual-value CI superseded; now per-replicate sample-mean CI",
    "runner_sha256": Runner_sha,
    "raw_sha256": raw_sha,
    "n": len(rows),
}
(OUT / "bootstrap_correction.json").write_text(json.dumps(corr, indent=2))
print("raw_sha256", raw_sha)
print("runner_sha256", Runner_sha)
print(json.dumps(summary, indent=2))

#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

ARMS=("native","omnizip","omnidial","stack")

def mean_ci(x,seed=20261015,draws=20000):
    x=np.asarray(x,float)
    if not len(x): return {"mean":None,"ci95":[None,None],"n":0}
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(x),(draws,len(x))); m=x[idx].mean(1)
    return {"mean":float(x.mean()),"ci95":[float(np.quantile(m,.025)),float(np.quantile(m,.975))],"n":len(x)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args()
    rows=[json.loads(x) for x in (a.out/"raw_results.jsonl").read_text().splitlines() if x.strip()]
    summary={"result_status":"verified" if (a.out/"inference_complete.json").exists() else "partial","n":len(rows),"arms":{},"interaction":{}}
    for arm in ARMS:
        vals=[r["arms"][arm] for r in rows]
        summary["arms"][arm]={
          "accuracy":mean_ci([v["answer"]==r["correct_option"] for v,r in zip(vals,rows)]),
          "gold_logp":mean_ci([v["option_logp"][r["correct_option"]] for v,r in zip(vals,rows)]),
          "latency_s":mean_ci([v["latency_s"] for v in vals]),
          "retained_tokens":mean_ci([v["retained_tokens"] for v in vals]),
          "retention_ratio":mean_ci([v["retained_tokens"]/v["input_tokens"] for v in vals]),
          "expert_rows":mean_ci([v["expert_rows"] for v in vals]),
          "expert_row_ratio":mean_ci([v["expert_rows"]/r["arms"]["native"]["expert_rows"] for v,r in zip(vals,rows)]),
          "peak_allocated_gib":mean_ci([v["peak_allocated_gib"] for v in vals]),
          "js_to_native":mean_ci([v["js_to_native"] for v in vals]),
          "flip_vs_native":mean_ci([v["answer"]!=r["arms"]["native"]["answer"] for v,r in zip(vals,rows)]),
        }
    def metric(name): return {arm:np.array([r["arms"][arm][name] for r in rows],float) for arm in ARMS}
    for name in ("latency_s","retained_tokens","expert_rows","peak_allocated_gib"):
        v=metric(name); interaction=v["stack"]-v["omnizip"]-v["omnidial"]+v["native"]
        ratios={arm:v[arm]/v["native"] for arm in ARMS}
        multiplicative=ratios["stack"]-(ratios["omnizip"]*ratios["omnidial"])
        summary["interaction"][name]={"difference_in_differences":mean_ci(interaction),"multiplicative_residual":mean_ci(multiplicative),"individual_product_ratio":mean_ci(ratios["omnizip"]*ratios["omnidial"]),"observed_stack_ratio":mean_ci(ratios["stack"])}
    for name,extract in {
      "accuracy":lambda r,a:float(r["arms"][a]["answer"]==r["correct_option"]),
      "gold_logp":lambda r,a:r["arms"][a]["option_logp"][r["correct_option"]],
    }.items():
        v={a:np.array([extract(r,a) for r in rows],float) for a in ARMS}
        summary["interaction"][name]={"difference_in_differences":mean_ci(v["stack"]-v["omnizip"]-v["omnidial"]+v["native"])}
    (a.out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__": main()

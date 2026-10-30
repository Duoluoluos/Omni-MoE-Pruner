#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
ARMS=("native","omnizip","omnidial","stack")
def ci(x,seed=20261015,draws=20000):
 x=np.asarray(x,float); rng=np.random.default_rng(seed); m=x[rng.integers(0,len(x),(draws,len(x)))].mean(1)
 return {"mean":float(x.mean()),"ci95":[float(np.quantile(m,.025)),float(np.quantile(m,.975))],"n":len(x)}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--baseline',type=Path,required=True); ap.add_argument('--p0',type=Path,required=True); a=ap.parse_args()
 old={r['sample_index']:r for r in map(json.loads,a.baseline.open())}; new={r['sample_index']:r for r in map(json.loads,a.p0.open())}; assert set(old)==set(new) and len(new)==32
 out={"n":32,"identity":{"passed":True,"max_abs_option_logp_delta":0.0,"text_mismatches":0},"arms":{},"incremental":{}}
 maxd=0.; mis=0
 for idx,r in new.items():
  for arm,v in r['arms'].items():
   q=old[idx]['arms'][arm]; maxd=max(maxd,max(abs(v['option_logp'][k]-q['option_logp'][k]) for k in 'ABCD')); mis+=v['output_text']!=q['output_text']
 out['identity']={"passed":maxd<=1e-6 and mis==0,"max_abs_option_logp_delta":maxd,"text_mismatches":mis}
 for arm in ARMS:
  p=np.array([new[i]['arms'][arm]['latency_s'] for i in sorted(new)]); b=np.array([old[i]['arms'][arm]['latency_s'] for i in sorted(old)])
  out['arms'][arm]={"baseline_latency_s":ci(b),"p0_latency_s":ci(p),"paired_delta_s":ci(p-b),"paired_delta_pct_of_baseline":ci((p-b)/b*100),"speedup_ratio_baseline_over_p0":float(b.mean()/p.mean())}
 for name,hi,lo in (("omnidial_vs_native","native","omnidial"),("stack_vs_omnizip","omnizip","stack")):
  old_gain=np.array([old[i]['arms'][hi]['latency_s']-old[i]['arms'][lo]['latency_s'] for i in sorted(old)])
  p0_gain=np.array([new[i]['arms'][hi]['latency_s']-new[i]['arms'][lo]['latency_s'] for i in sorted(new)])
  out['incremental'][name]={"baseline_saved_s":ci(old_gain),"p0_saved_s":ci(p0_gain),"p0_minus_baseline_saved_s":ci(p0_gain-old_gain)}
 (a.p0.parent/'p0_comparison.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
 assert out['identity']['passed']
if __name__=='__main__':main()

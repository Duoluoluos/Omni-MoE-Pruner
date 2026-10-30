#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
ARMS=("native","omnizip","omnidial","stack")
def ci(x,seed=20261015,draws=20000):
 x=np.asarray(x,float); rng=np.random.default_rng(seed); m=x[rng.integers(0,len(x),(draws,len(x)))].mean(1)
 return {"mean":float(x.mean()),"ci95":[float(np.quantile(m,.025)),float(np.quantile(m,.975))],"n":len(x)}
def load(p):return {r['sample_index']:r for r in map(json.loads,p.open())}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--p0',type=Path,required=True); ap.add_argument('--p1',type=Path,required=True); ap.add_argument('--p2',type=Path,required=True); a=ap.parse_args(); vs={'p0':load(a.p0),'p1':load(a.p1),'p2':load(a.p2)}; assert len(vs['p2'])==32 and all(set(v)==set(vs['p2']) for v in vs.values())
 out={'n':32,'identity_vs_p0':{},'arms':{},'incremental':{}}
 maxd=0.;mis=0
 for i,r in vs['p2'].items():
  for arm,v in r['arms'].items():
   q=vs['p0'][i]['arms'][arm]; maxd=max(maxd,max(abs(v['option_logp'][k]-q['option_logp'][k]) for k in 'ABCD')); mis+=v['output_text']!=q['output_text']
 out['identity_vs_p0']={'passed':maxd<=1e-6 and mis==0,'max_abs_option_logp_delta':maxd,'text_mismatches':mis}
 for arm in ARMS:
  out['arms'][arm]={}
  for right in ('p0','p1'):
   for metric in ('latency_s','peak_allocated_gib'):
    x=np.array([vs['p2'][i]['arms'][arm][metric] for i in sorted(vs['p2'])]); y=np.array([vs[right][i]['arms'][arm][metric] for i in sorted(vs[right])]); out['arms'][arm][f'p2_{metric}']=ci(x); out['arms'][arm][f'{right}_{metric}']=ci(y); out['arms'][arm][f'p2_minus_{right}_{metric}']=ci(x-y); out['arms'][arm][f'p2_minus_{right}_{metric}_pct']=ci((x-y)/y*100)
 for name,hi,lo in (('omnidial_vs_native','native','omnidial'),('stack_vs_omnizip','omnizip','stack')):
  gains={ver:np.array([rows[i]['arms'][hi]['latency_s']-rows[i]['arms'][lo]['latency_s'] for i in sorted(rows)]) for ver,rows in vs.items()}; out['incremental'][name]={ver+'_saved_s':ci(x) for ver,x in gains.items()}; out['incremental'][name]['p2_minus_p0_saved_s']=ci(gains['p2']-gains['p0']); out['incremental'][name]['p2_minus_p1_saved_s']=ci(gains['p2']-gains['p1'])
 (a.p2.parent/'p2_comparison.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2)); assert out['identity_vs_p0']['passed']
if __name__=='__main__':main()

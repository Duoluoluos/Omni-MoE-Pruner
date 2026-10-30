#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
ARMS=("native","omnizip","omnidial","stack")
def ci(x,seed=20261015,draws=20000):
 x=np.asarray(x,float); rng=np.random.default_rng(seed); m=x[rng.integers(0,len(x),(draws,len(x)))].mean(1)
 return {"mean":float(x.mean()),"ci95":[float(np.quantile(m,.025)),float(np.quantile(m,.975))],"n":len(x)}
def load(p): return {r['sample_index']:r for r in map(json.loads,p.open())}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--original',type=Path,required=True); ap.add_argument('--p0',type=Path,required=True); ap.add_argument('--p1',type=Path,required=True); a=ap.parse_args()
 versions={'original':load(a.original),'p0':load(a.p0),'p1':load(a.p1)}
 assert len(versions['p1'])==32 and all(set(v)==set(versions['p1']) for v in versions.values())
 out={'n':32,'identity_vs_p0':{},'arms':{},'incremental':{}}
 maxd=0.; mis=0
 for idx,r in versions['p1'].items():
  for arm,v in r['arms'].items():
   q=versions['p0'][idx]['arms'][arm]; maxd=max(maxd,max(abs(v['option_logp'][k]-q['option_logp'][k]) for k in 'ABCD')); mis+=v['output_text']!=q['output_text']
 out['identity_vs_p0']={'passed':maxd<=1e-6 and mis==0,'max_abs_option_logp_delta':maxd,'text_mismatches':mis}
 for arm in ARMS:
  out['arms'][arm]={}
  for left,right in (('p1','p0'),('p1','original')):
   x=np.array([versions[left][i]['arms'][arm]['latency_s'] for i in sorted(versions[left])]); y=np.array([versions[right][i]['arms'][arm]['latency_s'] for i in sorted(versions[right])])
   out['arms'][arm][f'{left}_latency_s']=ci(x); out['arms'][arm][f'{right}_latency_s']=ci(y); out['arms'][arm][f'{left}_minus_{right}_s']=ci(x-y); out['arms'][arm][f'{left}_minus_{right}_pct']=ci((x-y)/y*100)
 for name,hi,lo in (('omnidial_vs_native','native','omnidial'),('stack_vs_omnizip','omnizip','stack')):
  gains={}
  for ver,rows in versions.items(): gains[ver]=np.array([rows[i]['arms'][hi]['latency_s']-rows[i]['arms'][lo]['latency_s'] for i in sorted(rows)])
  out['incremental'][name]={ver+'_saved_s':ci(x) for ver,x in gains.items()}
  out['incremental'][name]['p1_minus_p0_saved_s']=ci(gains['p1']-gains['p0'])
  out['incremental'][name]['p1_minus_original_saved_s']=ci(gains['p1']-gains['original'])
 (a.p1.parent/'p1_comparison.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2)); assert out['identity_vs_p0']['passed']
if __name__=='__main__':main()

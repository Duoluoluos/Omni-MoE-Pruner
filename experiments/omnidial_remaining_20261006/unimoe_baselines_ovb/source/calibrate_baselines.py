"""Reuse the existing input-only Alloc/ASET implementation for Uni's partial band."""
import shutil,json
from common import *
def calibrate(backend,out,p,progress):
 import torch,numpy as np
 from collections import defaultdict
 import run_todo3_direct_comparison as existing
 from todo3_baselines import aset_budget,alloc_layer_dp
 existing.ROOT=ROOT
 local=Path('/tmp/omnidial_baseline_calibration_20261006');local.mkdir(exist_ok=True)
 records,_,_,_=dataset('omnivideobench')
 # These cached multimodal tensors live on local disk, away from the shared quota.
 existing.prepare_calibration(backend,local,records,p)
 chunks=defaultdict(list)
 for i in p['calibration_indices']:
  pool=torch.load(local/'calibration'/f'pool_{i}.pt',weights_only=False)
  for layer,rr in pool.items():chunks[layer].extend(rr)
 lo,hi=p['scope'];pp=[];nn=[];ww=[];numerator=denominator=0.
 for layer in range(lo,hi):
  for r in chunks[layer]:
   probs=r['probs'];native=r['native_budget'];weight=r['weight']
   with backend.controller.arm(p['policy']):counts=backend.controller.decide(probs,layer,native)
   numerator+=float(counts.sum())*weight;denominator+=len(counts)*weight
   pp.append(probs);nn.append(native);ww.append(torch.full((len(probs),),weight))
 target=numerator/denominator
 pp=torch.cat(pp).cuda();nn=torch.cat(nn).cuda();ww=torch.cat(ww).cuda();curve=[]
 for high in np.linspace(.2,1.,65):
  low=max(0.,float(high)-.2)
  mean=float((aset_budget(pp,nn,low,float(high),2).float()*ww).sum()/ww.sum())
  curve.append(dict(theta_min=low,theta_max=float(high),gamma=2.,assigned_mean=mean))
 best=min(curve,key=lambda c:abs(c['assigned_mean']-target));del pp,nn,ww
 progress('baseline_sensitivity',calibration_videos=len(p['calibration_indices']),scope=p['scope'])
 matrices=existing.sensitivity(backend,local,p)
 B=min((hi-lo)*4,max(hi-lo,round(target*(hi-lo))))
 schedule=[4]*len(backend.blocks);schedule[lo:hi]=alloc_layer_dp(matrices['scope'],B,exact=True)
 arms=dict(alloc=dict(kind='alloc',scope=p['scope'],schedule=schedule,global_assigned_budget=B,exact_budget_control=True),
  aset=dict(kind='aset',scope=p['scope'],**best,target=target))
 write(out/'baseline_calibration.json',dict(target_assigned_mean=target,alloc_integer_budget=B,aset_grid=curve,selected_aset=best,sensitivity=matrices,calibration_indices=p['calibration_indices'],gate_sha256=sha(out/'gates.json'),no_labels=True))
 dest=out/'calibration';dest.mkdir(exist_ok=True)
 for f in (local/'calibration').glob('*'):
  if f.name.startswith('pool_') or f.suffix=='.json':shutil.copy2(f,dest/f.name)
 shutil.copy2(local/'sensitivity_raw.jsonl',dest/'sensitivity_raw.jsonl')
 return arms

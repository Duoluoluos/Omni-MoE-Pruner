#!/usr/bin/env python3
"""Input-only Zip-aware router calibration for P3.

Uses 8 length-stratified inputs from the fixed 32-sample set. It never reads
answers or labels. Candidate selection minimizes distance to a frozen target
Stack/native expert-row ratio of 0.30 using router statistics only.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner'); HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'qwen-omni-utils/src'),str(ROOT/'scripts')]
from modeling_qwen3_omni_moe_stack import Qwen3OmniMoeForConditionalGeneration
from transformers import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
from qwen_mcore_controller_p0 import MCoreQwenControllerP0
import run_mb_new2_cgskip_qwen3omni as old
MODEL=ROOT/'models/Qwen3-Omni-30B-A3B-Instruct'
INPUTS=ROOT/'experiments/omnidial_layer_followup_20260930/qwen_diagnostic32_v1/inputs.json'
CALIBRATION_INDICES=[321,333,32,41,71,88,94,200]
SCOPE=(12,32); SEED=20261015; PRIOR_ZIP_RETENTION=0.4076261686632181
CANDIDATES={
 'p3_conservative':{'top2_fraction':0.50,'top4_fraction':0.35,'top8_fraction':0.15},
 'p3_balanced':{'top2_fraction':0.65,'top4_fraction':0.25,'top8_fraction':0.10},
 'p3_aggressive':{'top2_fraction':0.80,'top4_fraction':0.15,'top8_fraction':0.05},
}
TARGET_STACK_ROW_RATIO=0.30
OMNIZIP={'rho_audio':0.3,'rho_video':0.6,'g':3,'contextual_ratio':0.05}
def atomic(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def weighted_quantile(v,w,q):
 o=np.argsort(v,kind='stable');v,w=v[o],w[o];return float(np.interp(q*w.sum(),np.cumsum(w)-w/2,v))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
 records=json.loads(INPUTS.read_text());torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
 atomic(out/'protocol.json',{'schema':'p3-zip-aware-input-only-calibration-v1','indices':CALIBRATION_INDICES,'scope':list(SCOPE),'candidates':CANDIDATES,'selection':'closest predicted Stack/native expert-row ratio to 0.30; no labels or answers read','target_stack_row_ratio':TARGET_STACK_ROW_RATIO,'prior_zip_retention':PRIOR_ZIP_RETENTION,'omnizip':OMNIZIP})
 processor=Qwen3OmniMoeProcessor.from_pretrained(str(MODEL),use_fast=False)
 model=Qwen3OmniMoeForConditionalGeneration.from_pretrained(str(MODEL),torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='flash_attention_2').eval();model.disable_talker();model.thinker.omnizip_config=dict(OMNIZIP);model.thinker.enable_omnizip=True
 ctl=MCoreQwenControllerP0(model,scope=SCOPE,tau_reduce=0.08694529702904027)
 pools=[]
 def prepare(idx):
  r=records[idx];conv=old.build_conversation(ROOT/'eval_data/OmniVideoBench'/r['video'],r);text=processor.apply_chat_template(conv,add_generation_prompt=True,tokenize=False);audios,images,videos=process_mm_info(conv,use_audio_in_video=True);return processor(text=text,audio=audios,images=images,videos=videos,return_tensors='pt',padding=True,use_audio_in_video=True).to(model.device).to(model.dtype)
 for idx in CALIBRATION_INDICES:
  sample=[];handles=[]
  for layer,b in enumerate(ctl.blocks):
   if not SCOPE[0]<=layer<SCOPE[1]:continue
   def hook(module,args,value,layer=layer):
    if value.shape[0]<=1:return
    n=len(value);sel=torch.linspace(0,n-1,min(n,1024),device=value.device).long();probs=value[sel].float().softmax(-1);top=probs.topk(4,dim=-1).values
    sample.append({'layer':layer,'s2':top[:,:2].sum(-1).cpu(),'s4':top.sum(-1).cpu(),'weight':n/len(sel)})
   handles.append(b.gate.register_forward_hook(hook))
  inputs=prepare(idx)
  with ctl.arm('native'):old.generate_once(model,processor,inputs,1)
  for h in handles:h.remove()
  assert len(sample)==20
  pools.extend(sample);atomic(out/'progress.json',{'sample_index':idx,'completed_inputs':CALIBRATION_INDICES.index(idx)+1,'total_inputs':len(CALIBRATION_INDICES),'captured_layer_calls':len(pools),'time':time.time()})
  del inputs;torch.cuda.empty_cache()
 s2=np.concatenate([x['s2'].numpy() for x in pools]);s4=np.concatenate([x['s4'].numpy() for x in pools]);w=np.concatenate([np.full(len(x['s2']),x['weight']) for x in pools])
 results={}
 for name,c in CANDIDATES.items():
  f2,f4=c['top2_fraction'],c['top4_fraction'];tau2=weighted_quantile(s2,w,1-f2);remain=s2<tau2;conditional=f4/max(1e-12,1-f2);tau4=weighted_quantile(s4[remain],w[remain],1-conditional)
  b=np.where(s2>=tau2,2,np.where(s4>=tau4,4,8));mean_budget=float(np.average(b,weights=w));fractions={str(k):float(np.sum(w[b==k])/w.sum()) for k in (2,4,8)};conditional_ratio=(28*8+20*mean_budget)/(48*8);stack_ratio=PRIOR_ZIP_RETENTION*conditional_ratio
  results[name]={**c,'tau2':tau2,'tau4':tau4,'achieved_budget_fractions':fractions,'mean_budget_in_scope':mean_budget,'predicted_stack_expert_row_ratio':stack_ratio,'distance_to_target':abs(stack_ratio-TARGET_STACK_ROW_RATIO)}
 selected=min(results,key=lambda k:results[k]['distance_to_target'])
 gates={'schema':'p3-zip-aware-gates-v1','selected':selected,'selected_policy':results[selected],'candidates':results,'calibration_indices':CALIBRATION_INDICES,'captured_rows':len(s2),'weighted_rows':float(w.sum()),'scope':list(SCOPE),'answer_fields_read':False,'selection_rule':'closest predicted Stack/native expert-row ratio to frozen target 0.30'}
 atomic(out/'gates.json',gates);atomic(out/'complete.json',{'time':time.time(),'selected':selected});print(json.dumps(gates,indent=2))
 if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
if __name__=='__main__':
 try:main()
 except Exception:
  if '--out' in sys.argv:
   p=Path(sys.argv[sys.argv.index('--out')+1]);p.mkdir(parents=True,exist_ok=True);atomic(p/'failure.json',{'traceback':traceback.format_exc(),'time':time.time()})
  raise

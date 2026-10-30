#!/usr/bin/env python3
"""Resumable missing configurations using existing model adapters and policies."""
import argparse,inspect,json,os,random,socket,subprocess,sys,time,traceback
from pathlib import Path
from common import *
from scope_study_resume import run_lock,finite

def run(out):
 p=verify(out);records=read(out/'inputs.json');indices=p['evaluation_indices'];manifest_sha=sha(out/'manifest.json')
 sys.path.insert(0,str(ROOT/'qwen-omni-utils/src'));sys.path.insert(0,str(out/'source'))
 import torch,numpy as np,transformers
 import todo3_backend as adapter
 adapter.ROOT=ROOT
 if p['model']=='qwen':
  import run_mb_new2_cgskip_qwen3omni as old
  old.REPO=ROOT;old.MODEL_PATH=Path(p['checkpoint'])
 else:
  import run_mb_new2_cgskip_unimoe2 as old
  old.REPO=ROOT
 sys.path.insert(0,str(out/'source'))
 rows=load_rows(out,recover_tail=True)
 session=f'{time.strftime("%Y%m%dT%H%M%S")}_{socket.gethostname()}_{os.getpid()}'
 session_dir=out/'sessions'/session;session_dir.mkdir(parents=True)
 def progress(stage,**kw):
  obj=dict(stage=stage,completed=len(rows),expected=len(indices),pid=os.getpid(),host=socket.gethostname(),session=session,time=time.time(),**kw)
  write(out/'progress.json',obj);print(json.dumps(obj),flush=True)
 def analyze():
  verify(out)
  subprocess.run([sys.executable,'-B',str(out/'source/analyze.py'),'--out',str(out)],check=True)
  progress('complete',result_status='verified')
 if len(rows)==len(indices):analyze();return
 torch.manual_seed(p['model_seed']);np.random.seed(p['seed']);random.seed(p['seed'])
 progress('loading',backend=p['backend'])
 backend=adapter.Backend(p['model'])
 gates=read(out/'gates.json')['thresholds']
 if p['model']=='qwen':
  from scope_study_backend import ScopeController
  backend.controller=ScopeController(backend,gates,progress=lambda layer:progress('converting_mcore',layer=layer))
 else:backend.controller.thresholds=gates
 control=backend.controller
 env=dict(host=socket.gethostname(),pid=os.getpid(),session=session,torch=torch.__version__,transformers=transformers.__version__,
  cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
  python=sys.executable,backend=p['backend'],model_config_sha256=sha(backend.model_path/'config.json'),
  model_source_sha256=sha(inspect.getfile(type(backend.model))),processor_source_sha256=sha(inspect.getfile(type(backend.processor))),
  manifest_sha256=manifest_sha,ffmpeg=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
  weights=[dict(name=f.name,size=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns) for f in sorted(backend.model_path.glob('*.safetensors'))])
 stable=('torch','transformers','cuda','gpu','backend','model_config_sha256','model_source_sha256','processor_source_sha256','weights')
 if (out/'environment.json').exists():
  before=read(out/'environment.json')
  for key in stable:assert before[key]==env[key],f'Resume environment drift: {key}'
 else:write(out/'environment.json',env)
 write(session_dir/'environment.json',env)
 original_prepare=backend.prepare
 def prepare(record):
  # Absolute media path is passed through the existing adapter's path join;
  # Path join intentionally keeps an absolute right-hand argument.
  assert set(record)=={'sample_index','video','question','options'}
  return original_prepare(dict(record,video=str(Path(p['video_directory'])/record['video'])))
 backend.prepare=prepare
 if p['kind']=='baselines' and not (out/'arms.json').exists():
  from calibrate_baselines import calibrate
  arms=calibrate(backend,out,p,progress);write(out/'arms.json',arms)
 elif (out/'arms.json').exists():arms=read(out/'arms.json')
 else:
  arms={'omnidial':p['policy']};write(out/'arms.json',arms)
 arms_sha=sha(out/'arms.json');gates_sha=sha(out/'gates.json')
 for row in rows.values():
  assert row['manifest_sha256']==manifest_sha and row['arms_sha256']==arms_sha and row['gates_sha256']==gates_sha
  assert row['sample_index'] in indices and row['video']==records[row['sample_index']]['video']
  assert set(row['arms'])==set(arms);finite(row)
 option_ids={s:backend.processor.tokenizer.encode(s,add_special_tokens=False) for s in 'ABCD'}
 assert all(len(v)==1 for v in option_ids.values())
 def compact(value):
  logits=value['first_logits'].reshape(-1).float();assert torch.isfinite(logits).all()
  lp=logits.log_softmax(-1)
  return dict(answer=value['answer'],output_text=value['output_text'],generation_s=value['end_to_end_latency_ms']/1000,
   option_logp={s:float(lp[ids[0]]) for s,ids in option_ids.items()})
 # One Native pass proves that both modality embeddings reach the language model.
 pre_index=next(i for i in indices if i not in rows)
 progress('preflight_prepare',sample_index=pre_index)
 inputs,counts=backend.prepare(records[pre_index]);digest=adapter.input_digest(inputs)
 with control.arm({'kind':'native'}),backend.audit_embeddings(inputs) as embedding:
  native=compact(backend.generate(inputs,1))
 assert set(embedding)=={'audio','video'}
 warm={}
 for name,policy in arms.items():
  with control.arm(policy):warm[name]=compact(backend.generate(inputs,p['max_new_tokens']))
 write(session_dir/'preflight.json',dict(passed=True,sample_index=pre_index,counts=counts,input_sha256=digest,embedding=embedding,warmup_arms=list(arms)))
 progress('preflight_passed',backend=p['backend'],counts=counts)
 # Reuse terminal pilot predictions only after checking the same inputs, source,
 # package/weight identity and two archived policy outputs. No reuse of timing.
 if p['kind']=='terminal' and not (out/'reuse_check.json').exists():
  before=read(out/'reference_environment.json')
  for key in stable:assert before[key]==env[key],f'Pilot environment drift: {key}'
  refs=read(out/'reused_predictions.json');proofs=[]
  check_ids=[int(i) for i in list(refs)[:2]]
  for i in check_ids:
   r=refs[str(i)];assert sha(r['source'])==r['sha256']
   inp,c=backend.prepare(records[i]);dg=adapter.input_digest(inp);assert dg==r['input_sha256']
   with control.arm(p['policy']):value=compact(backend.generate(inp,p['max_new_tokens']))
   assert value['output_text']==r['arm']['output_text']
   assert max(abs(value['option_logp'][k]-r['arm']['option_logp'][k]) for k in option_ids)<1e-5
   proofs.append(dict(sample_index=i,input_sha256=dg,output_match=True));del inp
  write(out/'reuse_check.json',dict(passed=True,n=32,checks=proofs,session=session))
 if p['kind']=='terminal':
  assert read(out/'reuse_check.json')['passed']
  for key,r in read(out/'reused_predictions.json').items():
   i=int(key)
   if i in rows:continue
   assert sha(r['source'])==r['sha256']
   a={k:r['arm'][k] for k in ('answer','output_text','generation_s','option_logp')}
   oldrow=read(r['source'])
   row=dict(sample_index=i,video=r['video'],input_sha256=r['input_sha256'],counts=oldrow['counts'],arms={'omnidial':a},
    complete=True,timed=False,reused=True,reused_source=r['source'],reused_sha256=r['sha256'],session='reused_pilot',manifest_sha256=manifest_sha,arms_sha256=arms_sha,gates_sha256=gates_sha)
   append(out/'raw_results.jsonl',row);rows[i]=row
 del inputs,native,warm
 for ordinal,i in enumerate(indices):
  if i in rows:continue
  progress('prepare',sample_index=i)
  inputs,counts=backend.prepare(records[i]);digest=adapter.input_digest(inputs)
  order=list(arms);random.Random(p['seed']+i).shuffle(order);values={}
  progress('generate',sample_index=i,order=order)
  for name in order:
   with control.arm(arms[name]):values[name]=compact(backend.generate(inputs,p['max_new_tokens']))
  # Work instrumentation is confined to the existing diagnostic inputs.
  if p['kind']=='baselines' and i in p['diagnostic_indices']:
   for name in order:
    with control.arm(arms[name]),control.profile() as ledger:
     check=compact(backend.generate(inputs,p['max_new_tokens']))
    assert check['output_text']==values[name]['output_text']
    values[name]['work_ledger']={k:dict(v) for k,v in ledger.items()}
    values[name]['instrumentation_identity']=True
  row=dict(sample_index=i,ordinal=ordinal,video=records[i]['video'],input_sha256=digest,counts=counts,arms=values,
   complete=True,timed=ordinal>=5,reused=False,session=session,manifest_sha256=manifest_sha,arms_sha256=arms_sha,gates_sha256=gates_sha)
  finite(row);append(out/'raw_results.jsonl',row);rows[i]=row
  progress('sample_complete',sample_index=i);del inputs
 analyze()
 if torch.distributed.is_initialized():torch.distributed.destroy_process_group()

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--recover-lock',action='store_true');a=parser.parse_args()
 try:
  with run_lock(a.out,a.recover_lock):run(a.out)
 except Exception as exc:
  write(a.out/f'failure_{os.getpid()}.json',dict(error=str(exc),traceback=traceback.format_exc(),time=time.time()))
  raise

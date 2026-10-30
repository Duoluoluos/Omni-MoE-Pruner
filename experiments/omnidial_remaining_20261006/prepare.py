"""Freeze new configurations; never rerun archived Native or scope points."""
import argparse,shutil
from common import *
p=argparse.ArgumentParser();p.add_argument('--benchmark',choices=['avut','worldsense','omnivideobench'],required=True);a=p.parse_args()
records,directory,datafile,_=dataset(a.benchmark)
if a.benchmark=='omnivideobench':
 jobs=[('unimoe_baselines_ovb','unimoe','baselines',[7,19],['MT-U-ALLOC','MT-U-ASET']),('unimoe_terminal_ovb','unimoe','terminal',[16,27],['P0-B-full'])]
 indices=list(range(1000));selection='All 1000 OmniVideoBench questions.'
else:
 media=read(BASE/(a.benchmark+'_media_check.json'));indices=media['included_indices']
 assert len(indices)=={'avut':1733,'worldsense':3170}[a.benchmark], 'Resolve inclusion list before freezing'
 jobs=[(f'qwen_all_{a.benchmark}','qwen','transfer',[0,48],['MT-Q-'+('AVUT' if a.benchmark=='avut' else 'WS')]),(f'unimoe_partial_{a.benchmark}','unimoe','transfer',[7,19],['MT-U-'+('AVUT' if a.benchmark=='avut' else 'WS')])]
 selection='Input-only decodable subset after deletion of historical inclusion files. AVUT retains documented index 1495 exclusion (1733); WorldSense excludes corrupt-video questions 2293/2294 (3170 vs historical 3167). No arbitrary removal to match the old count. See media_check.json.'
for name,model,kind,scope,tasks in jobs:
 out=BASE/name
 if out.exists():verify(out);print('Frozen already:',name);continue
 out.mkdir();src=out/'source';src.mkdir()
 names=['todo3_backend.py','todo3_baselines.py','scope_study_backend.py','scope_study_resume.py','qwen_runtime_backend.py','qwen_mcore_controller.py','qwen_mcore_experts.py','run_mb_new2_cgskip_qwen3omni.py','run_mb_new2_cgskip_unimoe2.py','run_todo3_direct_comparison.py']
 for file in names:shutil.copy2(ROOT/'scripts'/file,src/file)
 # Reuse the original Alloc sensitivity implementation with an explicit scope override.
 f=src/'run_todo3_direct_comparison.py';s=f.read_text();needle='    logpath=out/"sensitivity_raw.jsonl"'
 assert s.count(needle)==1;s=s.replace(needle,'    scopes=protocol.get("baseline_scopes",scopes)\n'+needle);f.write_text(s)
 for file in ['common.py','run.py','calibrate_baselines.py','analyze.py']:shutil.copy2(BASE/file,src/file)
 ref=ROOT/f'experiments/omnidial_scope_20260929/{model}_full1000_v2'
 shutil.copy2(ref/'gates.json',out/'gates.json')
 write(out/'inputs.json',records)
 if kind=='transfer':shutil.copy2(BASE/(a.benchmark+'_media_check.json'),out/'media_check.json')
 protocol=dict(schema='remaining-v1',root=str(ROOT),model=model,kind=kind,benchmark=a.benchmark,scope=scope,task_ids=tasks,
  dataset=str(datafile),dataset_sha256=sha(datafile),video_directory=str(directory),evaluation_indices=indices,expected=len(indices),
  selection=selection,model_seed=20260909,seed=20261006,max_new_tokens=4,batch_size=1,dtype='bfloat16',greedy=True,
  backend='megatron_core_grouped_mlp' if model=='qwen' else 'unimoe_native_deepspeed',
  checkpoint=str(ROOT/('models/Qwen3-Omni-30B-A3B-Instruct' if model=='qwen' else 'models/Uni-MoE-2.0-Omni')),
  gate_source=str(ref/'gates.json'),gate_source_sha256=sha(ref/'gates.json'),no_gate_recalibration=True,
  timing='CUDA-synchronized model.generate including encoders, prefill and decode; excludes preparation, logging, scoring and work hooks; first five evaluated questions excluded from timing; each launch has one additional untimed warmup of every arm.',
  native='Reuse historical table means; small compatibility checks only, no full Native rerun.',
  resume='One fsynced JSONL line per complete question; reject duplicates and source/environment drift. Recovery may discard only an uncommitted final line.',
  comparisons='New operating points use the listed table Native only as descriptive cross-session ratios; no paired historical comparison.',
  calibration_indices=read(ref/'protocol.json')['calibration_indices'],
  diagnostic_indices=read(ref/'protocol.json')['diagnostic_indices'],
  policy=dict(kind='cgcc',scope=scope),
  baseline_calibration='Existing input-only Alloc Algorithm 1 sensitivity and exact-budget partial DP; ASET 65-point theta_max grid [0.2,1], theta_min=max(0,theta_max-0.2), gamma=2. Match scope-assigned OmniDial budget with frozen gates; no evaluation labels.',
  baseline_scopes={'scope':scope})
 if kind=='terminal':
  pilot=ROOT/'experiments/omnidial_layer_followup_20260930/unimoe_diagnostic32_v1'
  refs={}
  for f in (pilot/'predictions').glob('*.json'):
   if f.name.endswith('.partial.json'):continue
   r=read(f);refs[str(r['sample_index'])]=dict(video=r['video'],input_sha256=r['input_sha256'],arm=r['arms']['cg_late_protect_last'],source=str(f),sha256=sha(f))
  assert len(refs)==32;write(out/'reused_predictions.json',refs)
  shutil.copy2(pilot/'environment.json',out/'reference_environment.json')
  protocol['reused_n']=32;protocol['timing']+=' Reused pilot timings are excluded: terminal latency uses only fresh continuation questions after warmup.'
 write(out/'protocol.json',protocol)
 write(out/'manifest.json',{'files':{str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()}})
 print('Prepared',out,flush=True)

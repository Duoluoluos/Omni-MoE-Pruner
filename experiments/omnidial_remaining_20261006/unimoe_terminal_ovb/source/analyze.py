#!/usr/bin/env python3
"""Full-set scorer; evaluation labels are joined only after inference."""
import argparse,math,statistics,time
from common import *
def analyze(out):
 p=verify(out);rows=load_rows(out);records=read(out/'inputs.json');arms=read(out/'arms.json')
 _,_,_,gold=dataset(p['benchmark'],labels=True)
 assert len(rows)==p['expected'] and set(rows)==set(p['evaluation_indices'])
 for i,r in rows.items():
  assert r['video']==records[i]['video'] and r['input_sha256'] and r['counts']['audio']>0 and r['counts']['video']>0
  assert r['manifest_sha256']==sha(out/'manifest.json') and r['gates_sha256']==sha(out/'gates.json') and r['arms_sha256']==sha(out/'arms.json')
  assert set(r['arms'])==set(arms)
  if r.get('reused'):
   assert p['kind']=='terminal' and not r['timed'] and sha(r['reused_source'])==r['reused_sha256']
   continue
  session=out/'sessions'/r['session'];assert read(session/'preflight.json')['passed']
  assert r['timed']==(p['evaluation_indices'].index(i)>=5)
  for arm in r['arms'].values():
   assert math.isfinite(arm['generation_s']) and arm['generation_s']>0
   assert all(math.isfinite(v) for v in arm['option_logp'].values())
   if p['kind']=='baselines' and i in p['diagnostic_indices']:assert arm['instrumentation_identity'] and arm['work_ledger']
 timed=[r for r in rows.values() if r['timed']];assert timed
 result=dict(result_status='verified',n=len(rows),timed_n=len(timed),reused_n=sum(bool(r.get('reused')) for r in rows.values()),
  model=p['model'],benchmark=p['benchmark'],scope=p['scope'],backend=p['backend'],task_ids=p['task_ids'],
  raw_sha256=sha(out/'raw_results.jsonl'),manifest_sha256=sha(out/'manifest.json'),arms_sha256=sha(out/'arms.json'),arms={},checked_time=time.time(),
  verification='Full frozen question coverage, exact source/input/gate hashes, modality preflight, per-session warmup, finite metrics; historical controls are separate sessions.')
 for name,policy in arms.items():
  correct=sum(r['arms'][name]['answer']==gold[i] for i,r in rows.items())
  work={}
  for r in rows.values():
   for layer,metrics in r['arms'][name].get('work_ledger',{}).items():
    for k,v in metrics.items():work[k]=work.get(k,0)+v if layer.startswith('prefill:') else work.get(k,0)
  result['arms'][name]=dict(policy=policy,correct=correct,accuracy_percent=100*correct/len(rows),generation_seconds=statistics.mean(r['arms'][name]['generation_s'] for r in timed),diagnostic_prefill_work=work)
 write(out/'summary.json',result);print(json.dumps(result,indent=2))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();analyze(a.out)

#!/usr/bin/env python3
"""Two GPU lanes; adopt exact run processes, validate, backfill and continue."""
import fcntl,os,subprocess,sys,time
from common import *
WORKER='921134cf872ab4e8'
LANES={0:['qwen_all_avut','qwen_all_worldsense'],1:['unimoe_partial_avut','unimoe_partial_worldsense','unimoe_baselines_ovb','unimoe_terminal_ovb']}
MAX_BACKFILL_ATTEMPTS=3

def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]!='Z'
 except (OSError,IndexError):return False

def command_matches(pid,script,out):
 if not alive(pid):return False
 try:args=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
 except (OSError,UnicodeDecodeError):return False
 return str(script) in args and any(args[i:i+2]==['--out',str(out)] for i in range(len(args)-1))

def matching_pids(script,out):
 return [int(p.parent.name) for p in Path('/proc').glob('[0-9]*/cmdline') if command_matches(int(p.parent.name),script,out)]

def launch(name,gpu):
 out=BASE/name;p=verify(out)
 assert not matching_pids(out/'source/run.py',out),'Refusing duplicate inference launch'
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',
  OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES=str(gpu),
  NUMBA_CACHE_DIR=f'/tmp/omnidial_20261006_numba_{gpu}',TORCH_EXTENSIONS_DIR=f'/tmp/omnidial_20261006_torch_{gpu}',TRITON_CACHE_DIR=f'/tmp/omnidial_20261006_triton_{gpu}')
 env['PATH']=str(ROOT/'.runtime/scope_bin')+os.pathsep+env.get('PATH','')
 env['LD_LIBRARY_PATH']='/lib/x86_64-linux-gnu:'+env.get('LD_LIBRARY_PATH','')
 env['PYTHONPATH']=str(ROOT/'.tmp/qwen_runtime_mcore_20260924/python') if p['model']=='qwen' else '/home/tiger/Uni-MoE/Uni-MoE-2'
 interpreter=str(ROOT/'.venv/bin/python') if p['model']=='qwen' else '/home/tiger/uni_moe_venv/bin/python'
 with (out/'run.log').open('a') as log:
  child=subprocess.Popen([interpreter,'-B',str(out/'source/run.py'),'--out',str(out)],env=env,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (out/'launch.pid').write_text(str(child.pid)+'\n')
 print(f'Launched {name} on GPU{gpu}: PID {child.pid}',flush=True)
 return child

def verified_summary(out,p):
 s=optional_read(out/'summary.json')
 if not s or s.get('result_status')!='verified':return None
 assert s['n']==p['expected'] and s['scope']==p['scope'] and s['backend']==p['backend']
 assert s['raw_sha256']==sha(out/'raw_results.jsonl') and s['manifest_sha256']==sha(out/'manifest.json')
 return s

def backfill_done(out):
 record=optional_read(out/'backfill.json')
 return bool(record and record.get('build_exit_code')==0 and
  record.get('summary_sha256')==sha(out/'summary.json') and
  record.get('raw_sha256')==sha(out/'raw_results.jsonl') and
  record.get('backfill_script_sha256')==sha(BASE/'backfill.py'))

def main():
 with lock_path('queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  state=optional_read(BASE/'queue.json',dict(worker=WORKER,jobs={}))
  state.update(pid=os.getpid(),worker=WORKER,lock_directory=str(LOCK_DIR),supervisor_started=time.time())
  pending={};attempts={};protocols={};run_children={}
  print(f'Supervisor PID {os.getpid()}; worker-local locks: {LOCK_DIR}',flush=True)

  def schedule_backfill(name,job):
   out=BASE/name
   if name in pending:return True
   if backfill_done(out):
    job.update(backfill='complete',backfill_exit_code=0);return False
   if attempts.get(name,0)>=MAX_BACKFILL_ATTEMPTS:return False
   if time.time()<job.get('backfill_retry_after',0):return True
   existing=matching_pids(BASE/'backfill.py',out)
   assert len(existing)<=1,('Duplicate backfill processes',name,existing)
   if existing:
    pid=existing[0];child=None
    print(f'Adopted backfill {name}: PID {pid}',flush=True)
   else:
    if job.get('backfill'):
     job.setdefault('backfill_history',[]).append({k:job[k] for k in ('backfill','backfill_exit_code','backfill_pid','backfill_finished','backfill_error') if k in job})
    with (out/'backfill.log').open('a') as log:
     child=subprocess.Popen([sys.executable,'-B',str(BASE/'backfill.py'),'--out',str(out)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    pid=child.pid;attempts[name]=attempts.get(name,0)+1
    (out/'backfill.pid').write_text(str(pid)+'\n')
    print(f'Backfill {name}: PID {pid}; attempt {attempts[name]} this supervisor',flush=True)
   pending[name]=(pid,child);job.update(backfill='running',backfill_pid=pid,backfill_started=time.time())
   return True

  while True:
   finished=True
   for child in run_children.values():child.poll()
   for gpu,names in LANES.items():
    for name in names:
     out=BASE/name;job=state['jobs'].setdefault(name,dict(gpu=gpu,execution='queued',result_status='planned'))
     if not (out/'manifest.json').exists():
      job['execution']='waiting_for_input_protocol';finished=False;break
     try:
      if name not in protocols:protocols[name]=verify(out)
      p=protocols[name]
      # os.replace on the shared mount may briefly hide progress.json.
      job['progress']=optional_read(out/'progress.json',job.get('progress'))
      matches=matching_pids(out/'source/run.py',out)
      if len(matches)>1:
       job['failure']=f'Duplicate matching inference processes: {matches}';finished=False;break
      if matches:
       pid=matches[0]
       if job.get('pid')!=pid:
        job.update(pid=pid,adopted=True,started_check=time.time())
        print(f'Adopted {name}: PID {pid}',flush=True)
       job['execution']='running';job['pid_identity_verified']=True
      elif job.get('pid') and alive(job['pid']):
       job.setdefault('rejected_pid_history',[]).append(dict(pid=job['pid'],time=time.time(),reason='Command no longer matches this run'))
       job['pid_identity_verified']=False
      summary=verified_summary(out,p)
      # A verified summary after process exit is sufficient; progress is optional.
      if summary and not matches:
       job.update(execution='complete',result_status='verified',summary_sha256=sha(out/'summary.json'))
       if schedule_backfill(name,job):finished=False
       continue
      if matches:
       finished=False;break
      if job['execution'] in ('complete','failed'):continue
      if job.get('pid') or (out/'launch.pid').exists():
       job.update(execution='failed',failure='Process exited without a verified full result; see run.log. Other independent queued jobs may proceed.')
       continue
      child=launch(name,gpu);run_children[name]=child
      job.update(pid=child.pid,execution='running',started_check=time.time(),pid_identity_verified=True)
      finished=False;break
     except Exception as error:
      job.update(execution='failed',failure=f'{type(error).__name__}: {error}')
      print(f'Administrative check failed for {name}: {job["failure"]}',flush=True)
      if matching_pids(out/'source/run.py',out):finished=False;break
   for name,(pid,child) in list(pending.items()):
    rc=child.poll() if child is not None else (None if command_matches(pid,BASE/'backfill.py',BASE/name) else -1)
    if rc is not None:
     job=state['jobs'][name];success=backfill_done(BASE/name)
     job.update(backfill='complete' if success else 'failed',backfill_exit_code=0 if success else (rc or -1),backfill_finished=time.time())
     if not success:
      job.update(backfill_retry_after=time.time()+60,backfill_error='Backfill exited without a matching successful compile marker; see backfill.log.')
     print(f'Backfill {name}: {job["backfill"]} (exit {job["backfill_exit_code"]})',flush=True)
     pending.pop(name)
   state['checked_time']=time.time()
   failures=any(j.get('execution')=='failed' or j.get('backfill')=='failed' for j in state['jobs'].values())
   state['phase']=('complete_with_failures' if failures else 'complete') if finished and not pending else 'running'
   write(BASE/'queue.json',state)
   if finished and not pending:break
   time.sleep(20)
if __name__=='__main__':main()

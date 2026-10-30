#!/usr/bin/env python3
"""Launch author-requested MCore runs and backfill each completed benchmark."""
import json, os, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'research_paper/iclr2027/evidence/runs'
OUT=RUNS/'qwen_mcore_three_arm_queue_20260924'
def read(p):
    # Shared-filesystem rename visibility can briefly hide a progress file.
    # Missing/partial telemetry must not terminate supervision or backfill.
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
def write(p,obj):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');t.replace(p)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'queue.json').exists(), 'Queue already launched'
    state=dict(phase='launching',result_status='planned',pid=os.getpid(),backend='megatron_core_grouped_mlp',arms=['top4','top6','cgcc'],auto_paper_backfill=True,jobs={})
    env=dict(os.environ,OMP_NUM_THREADS='8',PYTHONPATH=str(ROOT/'.tmp/qwen_runtime_mcore_20260924/python'))
    env['PATH']=str(ROOT/'.tmp/qwen_runtime_mcore_20260924/bin')+os.pathsep+env.get('PATH','')
    children={};handles=[]
    for gpu,bench,n in [(0,'avut',1733),(1,'worldsense',3167)]:
        out=RUNS/f'qwen_mcore_three_arm_{bench}_full{n}_20260924';out.mkdir(parents=True,exist_ok=True)
        assert not (out/'launch.log').exists(), out
        cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/run_qwen_mcore_crossbench.py'),'--benchmark',bench,'--out',str(out)]
        handle=(out/'launch.log').open('w');handles.append(handle)
        child=subprocess.Popen(cmd,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        children[bench]=child;state['jobs'][bench]=dict(pid=child.pid,gpu=gpu,out=str(out),exit_code=None,backfill=None)
        write(OUT/'queue.json',state)
    state['phase']='running'
    while True:
        alive=False
        for bench,child in children.items():
            job=state['jobs'][bench];job['exit_code']=child.poll();alive |= job['exit_code'] is None
            job['progress']=read(Path(job['out'])/'progress.json')
            summary=read(Path(job['out'])/'summary.json')
            finalizer=ROOT/'research_paper/iclr2027/scripts/backfill_qwen_mcore.py'
            if child.returncode==0 and summary and summary.get('result_status')=='verified' and job['backfill'] is None and finalizer.exists():
                with (OUT/f'backfill_{bench}.log').open('w') as log:
                    completed=subprocess.run([str(ROOT/'.venv/bin/python'),str(finalizer),'--benchmark',bench],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                job['backfill']=dict(exit_code=completed.returncode,time=time.time())
        state['checked_unix']=time.time();write(OUT/'queue.json',state)
        if not alive and all(j['exit_code'] != 0 or j['backfill'] is not None for j in state['jobs'].values()):break
        time.sleep(30)
    valid=all(j['exit_code']==0 and j['backfill'] and j['backfill']['exit_code']==0 for j in state['jobs'].values())
    state.update(phase='complete' if valid else 'failed',result_status='verified' if valid else 'placeholder');write(OUT/'queue.json',state)
    for h in handles:h.close()
    if not valid:raise RuntimeError('Run or backfill failed; inspect queue records')
if __name__=='__main__':main()

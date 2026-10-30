#!/usr/bin/env python3
"""Run Native on available reserved GPUs, preserving the other agent's GPU."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'research_paper/iclr2027/evidence/runs'
OUT=RUNS/'qwen_mcore_native_queue_20260926'


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def gpu_idle(gpu):
    try:
        used=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=15)
        return int(used.strip())<200
    except (subprocess.SubprocessError,ValueError):return False


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'queue.json').exists(),'Queue already launched'
    write(OUT/'allocation.json',dict(gpus=[0],reason='Reserve GPU1 for the author CG-CC agent; optional clarification pending.'))
    state=dict(pid=os.getpid(),phase='running',result_status='planned',arms=['native'],
        backend='megatron_core_grouped_mlp',auto_paper_backfill=True,jobs={})
    for bench,n in [('avut',1733),('worldsense',3167)]:
        state['jobs'][bench]=dict(phase='pending',pid=None,gpu=None,out=str(RUNS/f'qwen_mcore_native_{bench}_full{n}_20260926'),
            exit_code=None,backfill=None)
    env=dict(os.environ,OMP_NUM_THREADS='8',PYTHONPATH=str(ROOT/'.tmp/qwen_runtime_mcore_20260924/python'))
    env['PATH']=str(ROOT/'.tmp/qwen_runtime_mcore_20260924/bin')+os.pathsep+env.get('PATH','')
    children={};handles=[]
    write(OUT/'queue.json',state)
    while True:
        for bench,child in children.items():
            job=state['jobs'][bench]
            job['exit_code']=child.poll()
            progress=read(Path(job['out'])/'progress.json')
            if progress is not None:job['progress']=progress
            if child.returncode is not None:
                job['phase']='complete' if child.returncode==0 else 'failed'
        allocation=read(OUT/'allocation.json') or {'gpus':[0]}
        occupied={j['gpu'] for j in state['jobs'].values() if j['phase']=='running'}
        for gpu in allocation['gpus']:
            pending=next((b for b,j in state['jobs'].items() if j['phase']=='pending'),None)
            if pending is None:break
            if gpu in occupied or not gpu_idle(gpu):continue
            job=state['jobs'][pending];out=Path(job['out']);out.mkdir(parents=True,exist_ok=True)
            assert not (out/'launch.log').exists(),out
            handle=(out/'launch.log').open('w');handles.append(handle)
            command=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/run_qwen_mcore_native.py'),
                     '--benchmark',pending,'--out',str(out)]
            child=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),
                stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            children[pending]=child;job.update(pid=child.pid,gpu=gpu,phase='running',command=command)
            write(OUT/'queue.json',state)
        for bench,job in state['jobs'].items():
            finalizer=ROOT/'research_paper/iclr2027/scripts/backfill_qwen_native.py'
            summary=read(Path(job['out'])/'summary.json')
            if job['phase']=='complete' and summary and summary.get('result_status')=='verified' and job['backfill'] is None and finalizer.exists():
                with (OUT/f'backfill_{bench}.log').open('w') as log:
                    result=subprocess.run([str(ROOT/'.venv/bin/python'),str(finalizer),'--benchmark',bench],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                job['backfill']=dict(exit_code=result.returncode,time=time.time())
        state['checked_unix']=time.time();write(OUT/'queue.json',state)
        if all(j['phase']=='failed' or (j['phase']=='complete' and j['backfill'] is not None) for j in state['jobs'].values()):break
        time.sleep(20)
    valid=all(j['phase']=='complete' and j['backfill']['exit_code']==0 for j in state['jobs'].values())
    state.update(phase='complete' if valid else 'failed',result_status='verified' if valid else 'placeholder')
    write(OUT/'queue.json',state)
    for handle in handles:handle.close()
    if not valid:raise RuntimeError('Native run or backfill failed; see queue records.')


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Run both complete benchmark sets after the preferred backend's input gate.

No manuscript writes. The chosen backend, gate decision and child exit codes
are persistent; a failed run cannot acquire verified status here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'research_paper/iclr2027/evidence/runs'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mcore-pid',type=int,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    clean=RUNS/'qwen_runtime_real_input_smoke_v2_20260924/summary.json'
    mcore=RUNS/'qwen_mcore_real_input_smoke_20260924/summary.json'
    state=dict(result_status='planned',phase='waiting_mcore_gate',pid=os.getpid(),mcore_validation_pid=a.mcore_pid,
               clean_gate=str(clean),mcore_gate=str(mcore),auto_paper_backfill=False,jobs={})
    write(a.out/'queue.json',state)
    # Wait for this exact validation process to release GPU0. A PID reused by
    # an unrelated process never counts as a running validation.
    while True:
        try:cmdline=Path(f'/proc/{a.mcore_pid}/cmdline').read_bytes()
        except FileNotFoundError:break
        if b'validate_qwen_mcore_real.py' not in cmdline:break
        time.sleep(20)
    candidate=read(mcore)
    core_ok=bool(candidate and candidate.get('result_status')=='verified' and candidate.get('identity_gate') and candidate.get('n')==6)
    fallback=read(clean)
    assert fallback and fallback['result_status']=='verified' and fallback['identity_gate'] and fallback['n']==6
    backend='mcore' if core_ok else 'clean_hf';gate=mcore if core_ok else clean
    state.update(phase='launching',backend=backend,selected_gate=str(gate),
        gate_sha256=hashlib.sha256(gate.read_bytes()).hexdigest(),
        reason='Preferred MCore full-model gate passed.' if core_ok else 'MCore full-model gate did not pass; use the clean HF dispatch that passed all six-input identity checks.')
    write(a.out/'queue.json',state)
    env_base=dict(os.environ)
    env_base['PATH']=str(ROOT/'.tmp/qwen_runtime_mcore_20260924/bin')+os.pathsep+env_base.get('PATH','')
    env_base['OMP_NUM_THREADS']='8'
    if core_ok:env_base['PYTHONPATH']=str(ROOT/'.tmp/qwen_runtime_mcore_20260924/python')
    children={};handles=[]
    for gpu,bench,n in [(0,'avut',1733),(1,'worldsense',3167)]:
        out=RUNS/f'qwen_runtime_{backend}_{bench}_full{n}_20260924'
        out.mkdir(parents=True,exist_ok=True)
        assert not (out/'raw_results.jsonl').exists() and not (out/'launch.log').exists(), out
        cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/run_qwen_crossbench_clean.py'),
             '--benchmark',bench,'--backend',backend,'--identity-gate',str(gate),'--out',str(out)]
        env=dict(env_base,CUDA_VISIBLE_DEVICES=str(gpu))
        handle=(out/'launch.log').open('w');handles.append(handle)
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        children[bench]=child
        state['jobs'][bench]=dict(pid=child.pid,gpu=gpu,out=str(out),command=cmd,exit_code=None)
        write(a.out/'queue.json',state)
    state['phase']='running';write(a.out/'queue.json',state)
    while True:
        alive=False
        for bench,child in children.items():
            job=state['jobs'][bench];job['exit_code']=child.poll();alive |= job['exit_code'] is None
            job['progress']=read(Path(job['out'])/'progress.json')
        state['checked_unix']=time.time();write(a.out/'queue.json',state)
        if not alive:break
        time.sleep(30)
    valid=True
    for bench,child in children.items():
        job=state['jobs'][bench];summary_path=Path(job['out'])/'summary.json';summary=read(summary_path)
        ok=child.returncode==0 and summary is not None and summary.get('result_status')=='verified'
        job['verified']=ok;valid &= ok
        if summary is not None:job['summary_sha256']=hashlib.sha256(summary_path.read_bytes()).hexdigest()
    state.update(phase='complete' if valid else 'failed',result_status='verified' if valid else 'placeholder')
    write(a.out/'queue.json',state)
    for f in handles:f.close()
    if not valid:raise RuntimeError('At least one full run failed; no paper backfill.')


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Validate the isolated MCore expert port before any model/benchmark run."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import socket
import torch
from diagnose_qwen_moe_runtime import load_block,write_json
from qwen_runtime_backend import prefix_forward
from qwen_mcore_experts import MCoreGroupedExperts


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    write_json(a.out/'environment.json',dict(hostname=socket.gethostname(),gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda,
        sources={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                 [Path(__file__),root/'scripts/qwen_mcore_experts.py',root/'scripts/qwen_runtime_backend.py']}))
    torch.manual_seed(20260924)
    block,config=load_block(20);grouped=MCoreGroupedExperts(block);rows=[]
    with torch.inference_mode():
        for n in [1,512,4096]:
            x=torch.randn(1,n,config.hidden_size,device='cuda',dtype=torch.bfloat16)
            variants=[('prefix',k) for k in (4,6,8)]+[('mcore',k) for k in (4,6,8)]
            values={f'{b}_{k}':[] for b,k in variants};refs={};parity={}
            for k in (4,6,8):refs[k]=prefix_forward(block,x,budget=k)[0]
            for repeat in range(10):
                order=variants.copy();random.Random(20260924+n+repeat).shuffle(order)
                for backend,k in order:
                    torch.cuda.synchronize();start=time.perf_counter()
                    y=prefix_forward(block,x,budget=k,grouped=grouped if backend=='mcore' else None)[0]
                    torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
                    if repeat>=3:values[f'{backend}_{k}'].append(ms)
                    if repeat==0 and backend=='mcore':
                        diff=y.float()-refs[k].float()
                        parity[str(k)]=dict(max_abs=float(diff.abs().max()),relative_rms=float(diff.square().mean().sqrt()/refs[k].float().square().mean().sqrt().clamp_min(1e-12)),cosine=float(torch.nn.functional.cosine_similarity(y.float().reshape(1,-1),refs[k].float().reshape(1,-1))))
            row=dict(tokens=n,parity=parity,timing_ms={k:dict(mean=statistics.mean(v),median=statistics.median(v),values=v) for k,v in values.items()})
            rows.append(row);print(json.dumps(row),flush=True)
            write_json(a.out/'progress.json',dict(completed=len(rows)))
    passed=all(v['relative_rms']<0.01 and v['cosine']>0.9999 for r in rows for v in r['parity'].values())
    write_json(a.out/'summary.json',dict(result_status='verified' if passed else 'placeholder',verification_scope='Synthetic-activation single checkpoint block; no full-model quality or end-to-end speed claim.',relative_rms_gate=.01,cosine_gate=.9999,parity_gate=passed,megatron_core='0.12.3',rows=rows))
    torch.distributed.destroy_process_group()
    if not passed:raise RuntimeError('MCore block numerical gate failed')


if __name__=='__main__':main()

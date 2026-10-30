#!/usr/bin/env python3
"""Small scoped probes using the frozen September 29 HF/DeepSpeed backends."""
import argparse
import contextlib
import inspect
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
import traceback
from scope_study_resume import sha, write, finite, run_lock, rebuild_ledger


def run(out):
    p=json.loads((out/'protocol.json').read_text())
    manifest=json.loads((out/'manifest.json').read_text())
    def verify():
        for file,digest in manifest['files'].items():
            assert sha(out/file)==digest, file
    verify()
    root=Path(p['root'])
    sys.path.insert(0,str(root/'qwen-omni-utils/src'))
    sys.path.insert(0,str(out/'source'))
    import numpy as np
    import torch
    import transformers
    import todo3_backend as adapter
    adapter.ROOT=root
    if p['model']=='qwen':
        import run_mb_new2_cgskip_qwen3omni as old
        old.REPO=root;old.MODEL_PATH=root/'models/Qwen3-Omni-30B-A3B-Instruct'
    else:
        import run_mb_new2_cgskip_unimoe2 as old
        old.REPO=root
    sys.path.insert(0,str(out/'source'))
    from scope_study_backend import ScopeController
    from analyze_scope_study import validate_work
    class HFScopedController(ScopeController):
        def __init__(self,backend):
            for b,f in zip(backend.blocks,backend.controller.originals): b.forward=f
            adapter.Controller.__init__(self,backend)
        def _wrap(self,layer,block): return adapter.Controller._wrap(self,layer,block)
        @contextlib.contextmanager
        def profile(self):
            with adapter.Controller.profile(self) as ledger:
                handles=[]
                if self.backend.name=='qwen':
                    for layer,b in enumerate(self.blocks):
                        def tokens(module,args,layer=layer):
                            phase='prefill' if args[0].shape[-2]>1 else 'decode'
                            ledger[f'{phase}:{layer}']['tokens']+=args[0].numel()//args[0].shape[-1]
                        handles.append(b.register_forward_pre_hook(tokens))
                try: yield ledger
                finally:
                    for h in handles: h.remove()
    records=json.loads((out/'inputs.json').read_text())
    refs=json.loads((out/'references.json').read_text())
    assert all(set(r)=={'sample_index','video','question','options'} for r in records)
    indices=p['evaluation_indices']; gates_sha=sha(out/'gates.json')
    pred=out/'predictions';pred.mkdir(exist_ok=True)
    session_id=f'{time.strftime("%Y%m%dT%H%M%S")}_{socket.gethostname()}_{os.getpid()}'
    session=out/'sessions'/session_id;session.mkdir(parents=True)
    rows={}
    def validate(row,complete):
        i=row['sample_index'];assert i in indices
        assert row['video']==records[i]['video'] and row['input_sha256']==refs[str(i)]['input_sha256']
        assert row['manifest_sha256']==sha(out/'manifest.json') and row['gates_sha256']==gates_sha
        assert set(row['arms'])<=set(p['arms']) and set(row['probes'])<=set(p['probes'])
        if complete:
            assert row['complete'] and set(row['arms'])==set(p['arms']) and set(row['probes'])==set(p['probes'])
            for name,policy in p['arms'].items():
                assert row['arms'][name]['instrumentation_identity']
                validate_work(p['model'],policy,row['arms'][name]['work_ledger'])
        finite(row)
    for f in pred.glob('*.json'):
        if f.name.endswith('.partial.json'):continue
        row=json.loads(f.read_text());validate(row,True);rows[row['sample_index']]=row
    rebuild_ledger(out,rows,session)
    def progress(stage,**kw):
        value=dict(stage=stage,time=time.time(),pid=os.getpid(),host=socket.gethostname(),
                   model=p['model'],completed=len(rows),expected=len(indices),session=session_id,**kw)
        write(out/'progress.json',value);print(json.dumps(value),flush=True)
    def analyze():
        verify();assert len(rows)==len(indices)
        rebuild_ledger(out,rows,session)
        write(out/'inference_complete.json',dict(n=len(rows),raw_sha256=sha(out/'raw_results.jsonl'),time=time.time()))
        progress('analyzing')
        subprocess.run([sys.executable,str(out/'source/analyze_layer_followup.py'),'--out',str(out)],check=True)
        progress('complete')
    if len(rows)==len(indices):analyze();return
    torch.manual_seed(p['model_seed']);np.random.seed(p['seed']);random.seed(p['seed'])
    progress('loading')
    backend=adapter.Backend(p['model']);control=HFScopedController(backend);backend.controller=control
    control.thresholds=json.loads((out/'gates.json').read_text())['thresholds']
    env=dict(host=socket.gethostname(),pid=os.getpid(),session=session_id,torch=torch.__version__,
             transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
             visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),python=sys.executable,
             backend='qwen_hf_expert_loop' if p['model']=='qwen' else 'unimoe_native_deepspeed',
             model_config_sha256=sha(backend.model_path/'config.json'),
             model_source_sha256=sha(inspect.getfile(type(backend.model))),
             processor_source_sha256=sha(inspect.getfile(type(backend.processor))),
             ffmpeg=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
             weights=[dict(name=f.name,size=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns)
                      for f in sorted(backend.model_path.glob('*.safetensors'))])
    before=json.loads((out/'reference_environment.json').read_text())
    for k in ('torch','transformers','cuda','gpu','backend','model_config_sha256',
              'model_source_sha256','processor_source_sha256','ffmpeg','weights'):
        assert before[k]==env[k],f'Prior-run environment drift: {k}'
    write(session/'environment.json',env);write(out/'environment.json',env)
    option_ids={s:backend.processor.tokenizer.encode(s,add_special_tokens=False) for s in 'ABCD'}
    assert all(len(v)==1 for v in option_ids.values())
    def compact(value):
        logits=value['first_logits'].reshape(-1).float();assert torch.isfinite(logits).all()
        lp=logits.log_softmax(-1)
        return dict(answer=value['answer'],output_text=value['output_text'],
                    option_logp={s:float(lp[ix[0]]) for s,ix in option_ids.items()})
    def same_options(a,b):
        diff=max(abs(a['option_logp'][s]-b['option_logp'][s]) for s in 'ABCD')
        assert a['output_text']==b['output_text'] and a['answer']==b['answer'] and diff<=p['native_logp_tolerance'],diff
        return diff
    checks=[]
    for idx in p['identity_indices']:
        progress('identity_prepare',sample_index=idx)
        inputs,counts=backend.prepare(records[idx])
        assert adapter.input_digest(inputs)==refs[str(idx)]['input_sha256']
        progress('identity_forward',sample_index=idx)
        with control.arm({'kind':'native'}),backend.audit_embeddings(inputs) as audit:
            native=compact(backend.generate(inputs))
        diff=same_options(native,refs[str(idx)]['native'])
        with control.arm({'kind':'native'}),control.zero_branch('none',p['all_scope']):
            noop=compact(backend.generate(inputs))
        same_options(native,noop)
        checks.append(dict(sample_index=idx,max_abs_option_logp_delta=diff,embedding_audit=audit,noop_identity=True))
        del inputs;torch.cuda.empty_cache()
    write(session/'preflight.json',dict(passed=True,checks=checks,gates_sha256=gates_sha))
    write(out/'preflight.json',dict(passed=True,checks=checks,gates_sha256=gates_sha))
    for idx in indices:
        if idx in rows:continue
        progress('prepare',sample_index=idx)
        inputs,counts=backend.prepare(records[idx]);digest=adapter.input_digest(inputs)
        assert digest==refs[str(idx)]['input_sha256'] and counts==refs[str(idx)]['counts']
        partial=pred/f'{idx:04d}.partial.json'
        row=json.loads(partial.read_text()) if partial.exists() else dict(sample_index=idx,
            video=records[idx]['video'],input_sha256=digest,counts=counts,arms={},probes={},
            execution_session=session_id,gates_sha256=gates_sha,manifest_sha256=sha(out/'manifest.json'))
        validate(row,False)
        order=list(p['probes']);random.Random(p['seed']+idx).shuffle(order)
        for name in order:
            if name in row['probes']:continue
            progress('generate_probe',sample_index=idx,arm=name)
            with control.arm({'kind':'native'}),control.zero_branch(**p['probes'][name]):
                value=compact(backend.generate(inputs))
            row['probes'][name]=dict(value,execution_session=session_id);write(partial,row)
        for name,policy in p['arms'].items():
            if row['arms'].get(name,{}).get('instrumentation_identity'):continue
            progress('generate_policy',sample_index=idx,arm=name)
            with control.arm(policy):value=backend.generate(inputs)
            item=compact(value)
            # Save practical output before independently counting actual expert work.
            row['arms'][name]=dict(item,generation_s=value['end_to_end_latency_ms']/1000,execution_session=session_id)
            write(partial,row)
            with control.arm(policy),control.profile() as ledger:instrumented=backend.generate(inputs)
            same_options(item,compact(instrumented))
            # Stronger within-session instrumentation check uses all first-token logits.
            a=value['first_logits'].reshape(-1).double().log_softmax(-1)
            b=instrumented['first_logits'].reshape(-1).double().log_softmax(-1)
            m=torch.logaddexp(a,b)-np.log(2.)
            js=float(.5*((a.exp()*(a-m)).sum()+(b.exp()*(b-m)).sum()))
            assert js<1e-6;validate_work(p['model'],policy,ledger)
            row['arms'][name].update(work_ledger=dict(ledger),instrumentation_identity=True,profile_js=js)
            write(partial,row);del value,instrumented
        assert adapter.input_digest(inputs)==digest
        row['complete']=True;validate(row,True);write(pred/f'{idx:04d}.json',row)
        partial.unlink(missing_ok=True);rows[idx]=row
        with (out/'raw_results.jsonl').open('a') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        progress('running');del inputs;torch.cuda.empty_cache()
    analyze()
    if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--recover-lock',action='store_true');args=parser.parse_args()
    with run_lock(args.out,args.recover_lock):run(args.out)

if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc()
        if '--out' in sys.argv:
            write(Path(sys.argv[sys.argv.index('--out')+1])/'failure.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise

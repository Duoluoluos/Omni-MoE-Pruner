#!/usr/bin/env python3
"""Reduced missing-scope suite; reuses historical HF/DeepSpeed policy and probes."""
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
from scope_study_resume import sha, write, completed_rows, rebuild_ledger, run_lock


def run(out):
    p=json.loads((out/'protocol.json').read_text())
    manifest=json.loads((out/'manifest.json').read_text())
    for key,file in [('protocol_sha256','protocol.json'),('inputs_sha256','inputs.json'),
                     ('reused_main_table_sha256','reused_main_table.json')]:
        assert sha(out/file)==manifest[key]
    for name,digest in manifest['sources'].items():assert sha(out/'source'/name)==digest
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
    # Keep the proven HF policy; reuse diagnostic methods without MCore conversion.
    class HFScopedController(ScopeController):
        def __init__(self,backend):
            for b,f in zip(backend.blocks,backend.controller.originals):b.forward=f
            adapter.Controller.__init__(self,backend)
        def _wrap(self,layer,block):return adapter.Controller._wrap(self,layer,block)
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
                try:yield ledger
                finally:
                    for h in handles:h.remove()
    records=json.loads((out/'inputs.json').read_text())
    assert all(set(r)=={'sample_index','video','question','options'} for r in records)
    pred=out/'predictions';pred.mkdir(exist_ok=True)
    session_id=f'{time.strftime("%Y%m%dT%H%M%S")}_{socket.gethostname()}_{os.getpid()}'
    session=out/'sessions'/session_id;session.mkdir(parents=True)
    rows=completed_rows(out,p,records,sha(out/'gates.json') if (out/'gates.json').exists() else None)
    assert not rows or (out/'gates.json').exists()
    rebuild_ledger(out,rows,session)
    def progress(stage,**kw):
        value=dict(stage=stage,pid=os.getpid(),host=socket.gethostname(),time=time.time(),
                   model=p['model'],completed=len(rows),session=session_id,**kw)
        write(out/'progress.json',value);print(json.dumps(value),flush=True)
    def analyze():
        assert len(rows)==p['expected']
        rebuild_ledger(out,rows,session)
        write(out/'inference_complete.json',dict(completed=len(rows),raw_sha256=sha(out/'raw_results.jsonl'),time=time.time()))
        progress('analyzing')
        subprocess.run([sys.executable,str(out/'source/analyze_scope_study_v2.py'),'--out',str(out)],check=True)
        progress('complete')
    if len(rows)==p['expected']:
        analyze();return
    torch.manual_seed(p['model_seed']);np.random.seed(p['seed']);random.seed(p['seed'])
    progress('loading')
    backend=adapter.Backend(p['model']);control=HFScopedController(backend);backend.controller=control
    env=dict(host=socket.gethostname(),pid=os.getpid(),session=session_id,torch=torch.__version__,
        transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),python=sys.executable,
        backend='qwen_hf_expert_loop' if p['model']=='qwen' else 'unimoe_native_deepspeed',
        model_path=str(backend.model_path),model_config_sha256=sha(backend.model_path/'config.json'),
        manifest_sha256=sha(out/'manifest.json'),
        model_source_sha256=sha(inspect.getfile(type(backend.model))),
        processor_source_sha256=sha(inspect.getfile(type(backend.processor))),
        ffmpeg=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
        weights=[dict(name=f.name,size=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns)
                 for f in sorted(backend.model_path.glob('*.safetensors'))])
    if (out/'environment.json').exists():
        before=json.loads((out/'environment.json').read_text())
        for k in ('torch','transformers','cuda','gpu','backend','model_config_sha256',
                  'manifest_sha256','model_source_sha256','processor_source_sha256','weights'):
            assert before[k]==env[k],f'Resume environment drift: {k}'
    else:write(out/'environment.json',env)
    write(session/'environment.json',env)
    option_ids={s:backend.processor.tokenizer.encode(s,add_special_tokens=False) for s in 'ABCD'}
    assert all(len(v)==1 for v in option_ids.values())
    def compact(value):
        logits=value['first_logits'].reshape(-1).float();assert torch.isfinite(logits).all()
        lp=logits.log_softmax(-1)
        return dict(answer=value['answer'],output_text=value['output_text'],
                    generation_s=value['end_to_end_latency_ms']/1000,
                    option_logp={s:float(lp[ix[0]]) for s,ix in option_ids.items()})
    def js(a,b):
        la,lb=a.reshape(-1).double().log_softmax(-1),b.reshape(-1).double().log_softmax(-1)
        lm=torch.logaddexp(la,lb)-np.log(2.)
        return float(.5*((la.exp()*(la-lm)).sum()+(lb.exp()*(lb-lm)).sum()))
    def same(a,b):
        assert a['output_text']==b['output_text'] and js(a['first_logits'],b['first_logits'])<1e-6
    # Same weighted-quantile rule as run_todo3_direct_comparison.freeze_gates.
    def quantile(values,weights,q):
        order=np.argsort(values,kind='stable');values,weights=values[order],weights[order]
        return float(np.interp(q*np.sum(weights),np.cumsum(weights)-weights/2,values))
    gates_path=out/'gates.json'
    if not gates_path.exists():
        calibration=out/'calibration';calibration.mkdir(exist_ok=True)
        pools=[];proofs=[]
        for idx in p['calibration_indices']:
            path=calibration/f'pool_{idx}.pt';meta=path.with_suffix('.json')
            if path.exists() and meta.exists():
                proof=json.loads(meta.read_text());assert sha(path)==proof['pool_sha256']
                assert proof['manifest_sha256']==env['manifest_sha256']
                pool=torch.load(path,weights_only=False)
            else:
                progress('calibration_prepare',sample_index=idx)
                inputs,counts=backend.prepare(records[idx]);digest=adapter.input_digest(inputs)
                progress('calibration_forward',sample_index=idx)
                with control.arm({'kind':'native'}),control.capture_pool() as pool:
                    value=backend.generate(inputs,1)
                assert len(pool)==len(backend.blocks)
                assert all(torch.isfinite(r['probs']).all() for rs in pool.values() for r in rs)
                assert adapter.input_digest(inputs)==digest
                tmp=path.with_suffix('.tmp');torch.save(dict(pool),tmp);os.replace(tmp,path)
                proof=dict(sample_index=idx,video=records[idx]['video'],input_sha256=digest,
                           pool_sha256=sha(path),manifest_sha256=env['manifest_sha256'],counts=counts)
                write(meta,proof);del inputs,value
            pools.append(pool);proofs.append(proof)
            progress('calibration_saved',sample_index=idx)
        lo,hi=p['calibration_scope']
        chunks=[r for pool in pools for l,rs in pool.items() if lo<=l<hi for r in rs]
        probs=torch.cat([r['probs'] for r in chunks])
        weights=torch.cat([torch.full((len(r['probs']),),r['weight']) for r in chunks]).numpy()
        if p['model']=='qwen':
            thresholds=dict(tau_reduce=quantile(probs.topk(4,dim=-1).values.sum(-1).numpy(),weights,.4))
        else:
            real=probs[:,:4].max(-1).values.numpy()
            thresholds=dict(tau_skip=quantile(real,weights,.85),tau_hard=quantile(real,weights,.15),
                            tau_null=quantile(probs[:,4].numpy(),weights,.92))
        write(gates_path,dict(thresholds=thresholds,calibration_scope=p['calibration_scope'],
            manifest_sha256=env['manifest_sha256'],input_proofs=proofs,rule=p['gate_policy']))
        del pools,pool,chunks,probs,weights
    gates=json.loads(gates_path.read_text());assert gates['manifest_sha256']==env['manifest_sha256']
    control.thresholds=gates['thresholds'];gates_sha=sha(gates_path)
    progress('gates_frozen',thresholds=control.thresholds)
    # Reuse the existing short/long identity, embedding and diagnostic gates.
    checks=[]
    for idx in p['preflight_indices']:
        progress('preflight',sample_index=idx)
        inputs,counts=backend.prepare(records[idx])
        with control.arm({'kind':'native'}),backend.audit_embeddings(inputs) as audit:
            native=backend.generate(inputs)
        with control.arm({'kind':'identity'}):identity=backend.generate(inputs)
        same(native,identity)
        with control.arm({'kind':'native'}),control.zero_branch('none',p['scopes']['all']):
            noop=backend.generate(inputs)
        same(native,noop)
        for name,policy in p['arms'].items():
            with control.arm(policy):value=backend.generate(inputs)
            with control.arm(policy),control.profile() as ledger:instrumented=backend.generate(inputs)
            same(value,instrumented);compact(value);validate_work(p['model'],policy,ledger)
        with control.arm({'kind':'native'}),control.profile() as ledger,control.decomposition(inputs) as norms:
            instrumented=backend.generate(inputs)
        same(native,instrumented);validate_work(p['model'],{'kind':'native'},ledger)
        assert len(norms)==len(backend.blocks)*3
        for probe in p['probes'].values():
            with control.arm({'kind':'native'}),control.zero_branch(**probe):value=backend.generate(inputs)
            compact(value)
        checks.append(dict(sample_index=idx,counts=counts,embedding_audit=audit,
                           identity=True,noop=True,instrumentation_identity=True))
        del inputs,native,identity,noop,value,instrumented
        torch.cuda.empty_cache()
    flight=dict(passed=True,session=session_id,gates_sha256=gates_sha,checks=checks)
    write(session/'preflight.json',flight);write(out/'preflight.json',flight)
    diagnostic=set(p['diagnostic_indices']);start=time.time();initial=len(rows)
    for idx in p['evaluation_indices']:
        if idx in rows:continue
        progress('prepare',sample_index=idx)
        inputs,counts=backend.prepare(records[idx]);digest=adapter.input_digest(inputs)
        if len(rows)==initial:
            progress('warming',sample_index=idx)
            for policy in p['arms'].values():
                with control.arm(policy):backend.generate(inputs)
        row=dict(sample_index=idx,video=records[idx]['video'],counts=counts,input_sha256=digest,
                 timed=idx>=5,execution_session=session_id,gates_sha256=gates_sha,arms={},probes={})
        order=list(p['arms']);random.Random(p['seed']+idx).shuffle(order);row['arm_order']=order
        outputs={}
        for name in order:
            progress('generate',sample_index=idx,arm=name)
            with control.arm(p['arms'][name]):value=backend.generate(inputs)
            outputs[name]=value;row['arms'][name]=compact(value)
            write(pred/f'{idx:04d}.partial.json',row)
        if idx in diagnostic:
            progress('diagnostic',sample_index=idx)
            with control.arm({'kind':'native'}):native=backend.generate(inputs)
            row['native']=compact(native);row['native'].pop('generation_s')
            for name in order:
                with control.arm(p['arms'][name]),control.profile() as ledger:value=backend.generate(inputs)
                same(outputs[name],value);validate_work(p['model'],p['arms'][name],ledger)
                row['arms'][name].update(work_ledger=dict(ledger),js_to_native=js(native['first_logits'],outputs[name]['first_logits']))
            with control.arm({'kind':'native'}),control.profile() as ledger,control.decomposition(inputs) as norms:
                value=backend.generate(inputs)
            same(native,value);validate_work(p['model'],{'kind':'native'},ledger)
            row['native']['work_ledger']=dict(ledger);row['native_decomposition']=norms
            assert len(norms)==len(backend.blocks)*3
            for name,probe in p['probes'].items():
                with control.arm({'kind':'native'}),control.zero_branch(**probe):value=backend.generate(inputs)
                item=compact(value);item.pop('generation_s')
                item['js_to_native']=js(native['first_logits'],value['first_logits']);row['probes'][name]=item
            del native
        assert adapter.input_digest(inputs)==digest
        row['complete']=True;write(pred/f'{idx:04d}.json',row)
        (pred/f'{idx:04d}.partial.json').unlink(missing_ok=True)
        with (out/'raw_results.jsonl').open('a') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        rows[idx]=row
        progress('running',elapsed_s=time.time()-start,mean_wall_s_per_question=(time.time()-start)/(len(rows)-initial))
        del inputs,outputs,value
        torch.cuda.empty_cache()
    for name,digest in manifest['sources'].items():assert sha(out/'source'/name)==digest
    assert sha(gates_path)==gates_sha
    assert sha(inspect.getfile(type(backend.model)))==env['model_source_sha256']
    analyze()
    if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--recover-lock',action='store_true');args=parser.parse_args()
    out=args.out.resolve()
    with run_lock(out,args.recover_lock):run(out)


if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc()
        if '--out' in sys.argv:
            write(Path(sys.argv[sys.argv.index('--out')+1])/'failure.json',
                  dict(time=time.time(),pid=os.getpid(),traceback=traceback.format_exc()))
        raise

#!/usr/bin/env python3
"""Run one frozen backbone suite. Gold labels are joined only by final analysis."""
import argparse
import contextlib
import hashlib
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    p = json.loads((out/'protocol.json').read_text())
    manifest = json.loads((out/'manifest.json').read_text())
    assert sha(out/'protocol.json') == manifest['protocol_sha256']
    assert sha(out/'inputs.json') == manifest['inputs_sha256']
    for name, digest in manifest['sources'].items():
        assert sha(out/'source'/name) == digest
    root = Path(p['root'])
    sys.path.insert(0,str(root/'qwen-omni-utils/src'))
    sys.path.insert(0,str(out/'source'))
    import torch
    import transformers
    import numpy as np
    import todo3_backend as adapter
    adapter.ROOT = root
    if p['model'] == 'qwen':
        import run_mb_new2_cgskip_qwen3omni as old
        old.REPO = root
        old.MODEL_PATH = root/'models/Qwen3-Omni-30B-A3B-Instruct'
    else:
        import run_mb_new2_cgskip_unimoe2 as old
        old.REPO = root
    # Helpers may add paths. Restore the frozen module precedence.
    sys.path.insert(0,str(out/'source'))
    from scope_study_backend import ScopeController
    from analyze_scope_study import validate_work
    torch.manual_seed(p['model_seed']); np.random.seed(p['seed']); random.seed(p['seed'])
    records = json.loads((out/'inputs.json').read_text())
    assert all(set(r) == {'sample_index','video','question','options'} for r in records)
    pred = out/'predictions'; pred.mkdir(exist_ok=True)
    assert not list(pred.glob('*.json')), 'Use a new version for restart after measured output.'
    def progress(stage, **kw):
        state = dict(stage=stage,pid=os.getpid(),time=time.time(),model=p['model'],**kw)
        write(out/'progress.json', state)
        print(json.dumps(state),flush=True)
    progress('loading',completed=0)
    backend = adapter.Backend(p['model'])
    control = ScopeController(backend,p['thresholds'],
                             progress=lambda layer:progress('converting_mcore',layer=layer,completed=0))
    backend.controller = control
    env = dict(host=socket.gethostname(),pid=os.getpid(),torch=torch.__version__,
        transformers=transformers.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        omp_threads=os.environ.get('OMP_NUM_THREADS'),python=sys.executable,
        model_path=str(backend.model_path),model_config_sha256=sha(backend.model_path/'config.json'),
        manifest_sha256=sha(out/'manifest.json'),
        ffmpeg=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
        backend='mcore_grouped_mlp' if p['model']=='qwen' else 'unimoe_native_deepspeed',
        model_source=inspect.getfile(type(backend.model)),
        model_source_sha256=sha(inspect.getfile(type(backend.model))),
        processor_source_sha256=sha(inspect.getfile(type(backend.processor))))
    if p['model'] == 'qwen':
        from megatron.core.transformer.moe.experts import GroupedMLP
        assert all(isinstance(b.grouped_experts.mlp,GroupedMLP) and b.experts is None for b in backend.blocks)
        env.update(megatron_core='0.12.3',grouped_gemm='1.1.4',mcore_layers=48)
    else:
        assert len(backend.blocks)==28 and all(len(b.fixed_real_moe)==2 for b in backend.blocks)
        env.update(shared_experts_per_layer=2)
    write(out/'environment.json',env)
    option_ids = {s:backend.processor.tokenizer.encode(s,add_special_tokens=False) for s in 'ABCD'}
    assert all(len(v)==1 for v in option_ids.values())
    def compact(value):
        logits = value['first_logits'].reshape(-1).float()
        assert torch.isfinite(logits).all()
        logp = logits.log_softmax(-1)
        return dict(answer=value['answer'],output_text=value['output_text'],
            generation_s=value['end_to_end_latency_ms']/1000,
            option_logp={s:float(logp[ix[0]]) for s,ix in option_ids.items()})
    def js(a,b):
        la,lb = a.reshape(-1).double().log_softmax(-1),b.reshape(-1).double().log_softmax(-1)
        lm = torch.logaddexp(la,lb)-np.log(2.)
        return float(.5*((la.exp()*(la-lm)).sum()+(lb.exp()*(lb-lm)).sum()))
    def same(a,b):
        assert a['output_text']==b['output_text'], 'Instrumented output changed'
        assert js(a['first_logits'],b['first_logits']) < 1e-6, 'Instrumented logits changed'
    # Two length extremes; actual audio/video embeddings, native identity and no-op probes.
    preflight=[]
    for idx in p['preflight_indices']:
        progress('preflight',completed=0,sample_index=idx)
        inputs,counts=backend.prepare(records[idx])
        with control.arm({'kind':'native'}), backend.audit_embeddings(inputs) as audit:
            native=backend.generate(inputs)
        with control.arm({'kind':'identity'}):
            identity=backend.generate(inputs)
        same(native,identity)
        with control.arm({'kind':'native'}), control.zero_branch('none',p['scopes']['all']):
            noop=backend.generate(inputs)
        same(native,noop)
        checks={}
        for name in ('low_early','cg_middle','cg_all'):
            with control.arm(p['arms'][name]),control.profile() as ledger:
                val=backend.generate(inputs)
            compact(val)
            assert ledger and sum(v.get('routed_rows',0) for v in ledger.values()) > 0
            validate_work(p['model'],p['arms'][name],ledger)
            checks[name]=dict(js_to_native=js(native['first_logits'],val['first_logits']),
                              profiled_layers=len(ledger))
        with control.arm({'kind':'native'}),control.decomposition(inputs) as norms:
            instrumented=backend.generate(inputs)
        same(native,instrumented)
        assert len(norms)==len(backend.blocks)*3
        for name in ('routed_zero_middle',) + (('shared_zero_middle','both_zero_middle') if p['model']=='unimoe' else ()):
            with control.arm({'kind':'native'}),control.zero_branch(**p['probes'][name]):
                val=backend.generate(inputs)
            checks[name]=dict(js_to_native=js(native['first_logits'],val['first_logits']),finite=True)
            compact(val)
        preflight.append(dict(sample_index=idx,counts=counts,embedding_audit=audit,
                              identity=True,noop=True,decomposition_identity=True,checks=checks))
        del inputs,native,identity,noop,instrumented,val
        torch.cuda.empty_cache()
    write(out/'preflight.json',dict(passed=True,checks=preflight))
    progress('preflight_passed',completed=0)
    diagnostic=set(p['diagnostic_indices'])
    started=time.time()
    for idx in p['evaluation_indices']:
        progress('prepare',completed=idx,sample_index=idx)
        inputs,counts=backend.prepare(records[idx])
        row=dict(sample_index=idx,video=records[idx]['video'],counts=counts,
                 input_sha256=adapter.input_digest(inputs),timed=idx>=5,arms={},probes={})
        order=list(p['arms']);random.Random(p['seed']+idx).shuffle(order)
        row['arm_order']=order
        outputs={}
        for name in order:
            progress('generate',completed=idx,sample_index=idx,arm=name)
            with control.arm(p['arms'][name]):
                val=backend.generate(inputs)
            outputs[name]=val
            row['arms'][name]=compact(val)
            write(pred/f'{idx:04d}.partial.json',row)
        native=outputs['native']
        for name,val in outputs.items():
            row['arms'][name]['js_to_native']=js(native['first_logits'],val['first_logits'])
        if idx in diagnostic:
            progress('diagnostic',completed=idx,sample_index=idx)
            for name in order:
                with control.arm(p['arms'][name]),control.profile() as ledger:
                    val=backend.generate(inputs)
                same(outputs[name],val)
                validate_work(p['model'],p['arms'][name],ledger)
                row['arms'][name]['work_ledger']=dict(ledger)
            with control.arm({'kind':'native'}),control.decomposition(inputs) as norms:
                val=backend.generate(inputs)
            same(native,val)
            row['native_decomposition']=norms
            for name,probe in p['probes'].items():
                with control.arm({'kind':'native'}),control.zero_branch(**probe):
                    val=backend.generate(inputs)
                item=compact(val)
                item.pop('generation_s')  # Output zeroing is explicitly not timed evidence.
                item['js_to_native']=js(native['first_logits'],val['first_logits'])
                row['probes'][name]=item
        row['complete']=True
        write(pred/f'{idx:04d}.json',row)
        (pred/f'{idx:04d}.partial.json').unlink()
        with (out/'raw_results.jsonl').open('a') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        progress('running',completed=idx+1,elapsed_s=time.time()-started,
                 mean_wall_s_per_question=(time.time()-started)/(idx+1))
        del inputs,outputs,native,val
        torch.cuda.empty_cache()
    for name,digest in manifest['sources'].items():
        assert sha(out/'source'/name)==digest
    assert sha(inspect.getfile(type(backend.model)))==env['model_source_sha256']
    write(out/'inference_complete.json',dict(completed=1000,raw_sha256=sha(out/'raw_results.jsonl'),time=time.time()))
    progress('analyzing',completed=1000)
    code=subprocess.call([sys.executable,str(out/'source/analyze_scope_study.py'),'--out',str(out)])
    assert code==0, f'Analysis exited {code}'
    progress('complete',completed=1000)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__=='__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        if '--out' in sys.argv:
            out=Path(sys.argv[sys.argv.index('--out')+1])
            write(out/'failure.json',dict(time=time.time(),pid=os.getpid(),traceback=traceback.format_exc()))
        raise

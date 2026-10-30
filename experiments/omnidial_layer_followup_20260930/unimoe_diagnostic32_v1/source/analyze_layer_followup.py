#!/usr/bin/env python3
"""Post-inference gold join for descriptive, mean-only diagnostics."""
import argparse
import json
from pathlib import Path
from statistics import mean
from scope_study_resume import sha, write, finite
from analyze_scope_study import validate_work


def analyze(out):
    p=json.loads((out/'protocol.json').read_text());m=json.loads((out/'manifest.json').read_text())
    for file,digest in m['files'].items():assert sha(out/file)==digest,file
    raw=out/'raw_results.jsonl'
    assert sha(raw)==json.loads((out/'inference_complete.json').read_text())['raw_sha256']
    refs=json.loads((out/'references.json').read_text())
    rows=[json.loads(line) for line in raw.read_text().splitlines()]
    assert [r['sample_index'] for r in rows]==sorted(p['evaluation_indices']) and len(rows)==32
    for r in rows:
        i=r['sample_index'];assert r==json.loads((out/f'predictions/{i:04d}.json').read_text())
        assert r['complete'] and r['input_sha256']==refs[str(i)]['input_sha256']
        assert r['gates_sha256']==sha(out/'gates.json') and r['manifest_sha256']==sha(out/'manifest.json')
        assert set(r['probes'])==set(p['probes']) and set(r['arms'])==set(p['arms'])
        for a in [*r['arms'].values(),*r['probes'].values()]:
            assert json.loads((out/'sessions'/a['execution_session']/'preflight.json').read_text())['passed']
        for name,policy in p['arms'].items():
            assert r['arms'][name]['instrumentation_identity'];validate_work(p['model'],policy,r['arms'][name]['work_ledger'])
        finite(r)
    # Labels are opened only after all 32 inputs have completed all new arms.
    assert sha(p['dataset'])==p['dataset_sha256']
    gold=json.loads(Path(p['dataset']).read_text())
    assert all(gold[r['sample_index']]['video']==r['video'] for r in rows)
    def measure(name,kind,reference):
        a=[r[kind][name] for r in rows];b=[refs[str(r['sample_index'])][reference] for r in rows]
        g=[gold[r['sample_index']]['correct_option'] for r in rows]
        return dict(n=len(rows),accuracy_pct=100*mean(x['answer']==y for x,y in zip(a,g)),
            accuracy_delta_pp=100*mean(int(x['answer']==z)-int(y['answer']==z) for x,y,z in zip(a,b,g)),
            gold_logp_delta=mean(x['option_logp'][z]-y['option_logp'][z] for x,y,z in zip(a,b,g)),
            flip_pct=100*mean(x['answer']!=y['answer'] for x,y in zip(a,b)))
    probes={name:dict(policy=policy,vs_native=measure(name,'probes','native')) for name,policy in p['probes'].items()}
    arms={}
    for name,policy in p['arms'].items():
        work={}
        for reference in ('new','late','native'):
            ledger=[r['arms'][name]['work_ledger'] if reference=='new' else refs[str(r['sample_index'])][reference]['work_ledger'] for r in rows]
            work[reference]={f'{phase}_{field}':sum(v.get(field,0) for l in ledger for k,v in l.items() if k.startswith(phase+':'))
                for phase in ('prefill','decode') for field in ('routed_rows','shared_rows','tokens')}
        arms[name]=dict(policy=policy,vs_native=measure(name,'arms','native'),vs_late=measure(name,'arms','late'),work=work)
    result=dict(result_status='verified',n=32,model=p['model'],probes=probes,arms=arms,
        raw_sha256=sha(raw),manifest_sha256=sha(out/'manifest.json'),main_table_eligible=False,
        limitations=p['limitations'],timing='Pilot times retained in raw records only; no full-set latency claim.')
    write(out/'summary.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    analyze(parser.parse_args().out)

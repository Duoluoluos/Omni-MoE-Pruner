#!/usr/bin/env python3
"""Complete-coverage validation and gold-label join; no Main Table overwrite."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values, videos):
    values=np.asarray(values,dtype=float)
    assert np.isfinite(values).all() and len(values)>0
    groups={}
    for value,video in zip(values,videos):
        groups.setdefault(video,[]).append(value)
    sums=np.array([sum(v) for v in groups.values()])
    counts=np.array([len(v) for v in groups.values()])
    rng=np.random.default_rng(20260928)
    means=[]
    for _ in range(2000):
        ix=rng.integers(0,len(groups),len(groups))
        means.append(sums[ix].sum()/counts[ix].sum())
    return dict(mean=float(values.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
                n=len(values),videos=len(groups))


def validate_work(model,policy,ledger):
    depth=48 if model=='qwen' else 28
    assert all(f'prefill:{layer}' in ledger for layer in range(depth))
    scope=policy.get('scope',[0,depth])
    for key,v in ledger.items():
        layer=int(key.split(':')[1]); n=v['tokens']
        assert n>0 and all(x>=0 for x in v.values())
        inside=scope[0]<=layer<scope[1]
        if model=='qwen':
            rows=v['routed_rows']
            if policy['kind']=='native' or not inside:
                assert rows==8*n, (key,rows,n,policy)
            elif policy['kind']=='fixed':
                assert rows==policy['budget']*n
            else:
                assert 4*n<=rows<=8*n
        else:
            assert v['shared_rows']==2*n
            assert v.get('routed_rows',0)>=v['selected_real']
            assert v['selected_real']+v['selected_null']<=v['selected_slots']
            if inside and policy['kind']=='cap':
                assert v['selected_slots']<=policy['budget']*n


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    out=parser.parse_args().out
    p=json.loads((out/'protocol.json').read_text())
    manifest=json.loads((out/'manifest.json').read_text())
    assert sha(out/'protocol.json')==manifest['protocol_sha256']
    assert sha(out/'inputs.json')==manifest['inputs_sha256']
    assert json.loads((out/'preflight.json').read_text())['passed']
    for name,digest in manifest['sources'].items():
        assert sha(out/'source'/name)==digest
    raw=out/'raw_results.jsonl'
    assert sha(raw)==json.loads((out/'inference_complete.json').read_text())['raw_sha256']
    rows=[json.loads(s) for s in raw.read_text().splitlines()]
    assert len(rows)==p['expected']==1000
    assert [r['sample_index'] for r in rows]==p['evaluation_indices']
    assert all(r['complete'] and set(r['arms'])==set(p['arms']) for r in rows)
    diagnostic=set(p['diagnostic_indices'])
    for r in rows:
        assert r['counts']['audio']>0 and r['counts']['video']>0
        if r['sample_index'] in diagnostic:
            assert set(r['probes'])==set(p['probes'])
            assert len(r['native_decomposition'])==(48 if p['model']=='qwen' else 28)*3
            for name,policy in p['arms'].items():
                validate_work(p['model'],policy,r['arms'][name]['work_ledger'])
    gold_path=Path(p['dataset'])
    assert sha(gold_path)==p['dataset_sha256']
    gold=json.loads(gold_path.read_text())
    assert all(r['video']==gold[r['sample_index']]['video'] for r in rows)
    arms,contrasts={},{}
    videos=[r['video'] for r in rows]
    scores={}
    for name in p['arms']:
        correct=np.array([r['arms'][name]['answer']==gold[r['sample_index']]['correct_option'] for r in rows],float)
        logp=np.array([r['arms'][name]['option_logp'][gold[r['sample_index']]['correct_option']] for r in rows])
        seconds=np.array([r['arms'][name]['generation_s'] for r in rows])
        scores[name]=dict(correct=correct,logp=logp,seconds=seconds)
        arms[name]=dict(policy=p['arms'][name],accuracy=interval(correct,videos),
            gold_logp=interval(logp,videos),generation_s=interval(seconds[5:],videos[5:]),
            flip=interval([r['arms'][name]['answer']!=r['arms']['native']['answer'] for r in rows],videos),
            js=interval([r['arms'][name]['js_to_native'] for r in rows],videos))
        subset=[r for r in rows if r['sample_index'] in diagnostic]
        totals={}
        for phase in ('prefill','decode'):
            for field in ('routed_rows','shared_rows','selected_real','selected_null','selected_slots','tokens'):
                totals[phase+'_'+field]=sum(v.get(field,0) for r in subset
                    for key,v in r['arms'][name]['work_ledger'].items() if key.startswith(phase+':'))
        arms[name]['work_subset_totals']=totals
    pairs=[(name,'native') for name in p['arms'] if name!='native']
    pairs += [('cg_middle','cg_all'),('cg_early','cg_late'),('low_early','low_late'),
              ('cg_quarter','cg_half'),('cg_half','cg_threequarter'),('cg_threequarter','cg_all')]
    for a,b in pairs:
        contrasts[a+'__minus__'+b]=dict(
            accuracy_pp=interval(100*(scores[a]['correct']-scores[b]['correct']),videos),
            gold_logp=interval(scores[a]['logp']-scores[b]['logp'],videos),
            generation_s=interval((scores[a]['seconds']-scores[b]['seconds'])[5:],videos[5:]))
    probes={}
    subset=[r for r in rows if r['sample_index'] in diagnostic]
    sv=[r['video'] for r in subset]
    for name,policy in p['probes'].items():
        gl=[];acc=[];flip=[]
        for r in subset:
            answer=gold[r['sample_index']]['correct_option']
            a,b=r['probes'][name],r['arms']['native']
            gl.append(a['option_logp'][answer]-b['option_logp'][answer])
            acc.append(100*(int(a['answer']==answer)-int(b['answer']==answer)))
            flip.append(a['answer']!=b['answer'])
        probes[name]=dict(policy=policy,gold_logp_delta=interval(gl,sv),
                          accuracy_delta_pp=interval(acc,sv),flip=interval(flip,sv))
    interactions={}
    if p['model']=='unimoe':
        for scope in ('middle','all'):
            values=[]
            for r in subset:
                g=gold[r['sample_index']]['correct_option']
                value=lambda branch:r['probes'][branch+'_zero_'+scope]['option_logp'][g]
                values.append(value('both')-value('routed')-value('shared')+r['arms']['native']['option_logp'][g])
            interactions[scope]=interval(values,sv)
    summary=dict(result_status='verified',model=p['model'],n=1000,videos=len(set(videos)),
        diagnostic_n=len(subset),arms=arms,paired_contrasts=contrasts,branch_probes=probes,
        branch_interaction_gold_logp=interactions,raw_sha256=sha(raw),
        protocol_sha256=sha(out/'protocol.json'),limitations=p['limitations'],
        interval_note='Pointwise descriptive CIs; no multiplicity-adjusted scope selection or equivalence claim.',
        main_table_replacement=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with (out/'scope_results.csv').open('w') as f:
        fields=['arm','scope','accuracy_pct','accuracy_low_pct','accuracy_high_pct',
                'generation_s','generation_low_s','generation_high_s','gold_logp_delta',
                'gold_logp_low','gold_logp_high','flip_pct','diagnostic_routed_rows_ratio',
                'diagnostic_shared_rows_ratio']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        native_work=arms['native']['work_subset_totals']
        for name,a in arms.items():
            delta=contrasts.get(name+'__minus__native',{}).get('gold_logp',dict(mean=0,ci95=[0,0]))
            work=a['work_subset_totals']
            ratio=lambda field:sum(work[k] for k in work if k.endswith('_'+field))/max(1,sum(native_work[k] for k in native_work if k.endswith('_'+field)))
            writer.writerow(dict(arm=name,scope=json.dumps(p['arms'][name].get('scope')),
                accuracy_pct=100*a['accuracy']['mean'],accuracy_low_pct=100*a['accuracy']['ci95'][0],
                accuracy_high_pct=100*a['accuracy']['ci95'][1],generation_s=a['generation_s']['mean'],
                generation_low_s=a['generation_s']['ci95'][0],generation_high_s=a['generation_s']['ci95'][1],
                gold_logp_delta=delta['mean'],gold_logp_low=delta['ci95'][0],gold_logp_high=delta['ci95'][1],
                flip_pct=100*a['flip']['mean'],diagnostic_routed_rows_ratio=ratio('routed_rows'),
                diagnostic_shared_rows_ratio=ratio('shared_rows') if p['model']=='unimoe' else 'NA'))
    report=['# Partial-layer scope study — '+p['model'],'','Status: verified (bounded by protocol).',
            '', '| Arm | Accuracy (%) | Generation (s) | Δ gold log p | Flip (%) |',
            '|---|---:|---:|---:|---:|']
    for name,a in arms.items():
        delta=contrasts.get(name+'__minus__native',{}).get('gold_logp',{}).get('mean',0)
        report.append(f"| {name} | {100*a['accuracy']['mean']:.2f} | {a['generation_s']['mean']:.3f} | {delta:+.4f} | {100*a['flip']['mean']:.2f} |")
    report += ['', 'Work/decomposition and branch probes use the predeclared 128-video subset only.',
               'All CIs and scope contrasts: summary.json. Main Table points remain unchanged.',
               '', *('- '+x for x in p['limitations'])]
    (out/'report.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(result_status='verified',out=str(out),n=1000)),flush=True)


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Validate complete new scopes; retain reused manuscript means as unpaired anchors."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from analyze_scope_study import interval, validate_work
from scope_study_resume import sha, write, completed_rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    out=parser.parse_args().out
    p=json.loads((out/'protocol.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
    for k,f in [('protocol_sha256','protocol.json'),('inputs_sha256','inputs.json'),
                ('reused_main_table_sha256','reused_main_table.json')]:assert sha(out/f)==manifest[k]
    for name,digest in manifest['sources'].items():assert sha(out/'source'/name)==digest
    gates_sha=sha(out/'gates.json')
    assert json.loads((out/'gates.json').read_text())['manifest_sha256']==sha(out/'manifest.json')
    raw=out/'raw_results.jsonl'
    assert sha(raw)==json.loads((out/'inference_complete.json').read_text())['raw_sha256']
    clean=json.loads((out/'inputs.json').read_text())
    by_idx=completed_rows(out,p,clean,gates_sha)
    assert sorted(by_idx)==p['evaluation_indices'] and len(by_idx)==p['expected']==1000
    rows=[by_idx[i] for i in p['evaluation_indices']]
    assert rows==[json.loads(line) for line in raw.read_text().splitlines()]
    for session in {r['execution_session'] for r in rows}:
        flight=json.loads((out/'sessions'/session/'preflight.json').read_text())
        assert flight['passed'] and flight['gates_sha256']==gates_sha
    subset=[r for r in rows if r['sample_index'] in p['diagnostic_indices']]
    assert len(subset)==len(p['diagnostic_indices'])==32
    for r in subset:
        for name,policy in p['arms'].items():validate_work(p['model'],policy,r['arms'][name]['work_ledger'])
        validate_work(p['model'],{'kind':'native'},r['native']['work_ledger'])
    # No gold file is opened until all inference, identity and work checks pass.
    assert sha(Path(p['dataset']))==p['dataset_sha256']
    gold=json.loads(Path(p['dataset']).read_text())
    assert all(gold[r['sample_index']]['video']==r['video'] for r in rows)
    videos=[r['video'] for r in rows];sv=[r['video'] for r in subset]
    arms,scores,contrasts,diagnostics={},{},{},{}
    for name,policy in p['arms'].items():
        scores[name]=dict(correct=np.array([r['arms'][name]['answer']==gold[r['sample_index']]['correct_option'] for r in rows],float),
            logp=np.array([r['arms'][name]['option_logp'][gold[r['sample_index']]['correct_option']] for r in rows]),
            seconds=np.array([r['arms'][name]['generation_s'] for r in rows]))
        s=scores[name]
        arms[name]=dict(policy=policy,accuracy=interval(s['correct'],videos),gold_logp=interval(s['logp'],videos),
                       generation_s=interval(s['seconds'][5:],videos[5:]))
    for a,b in itertools.combinations(p['arms'],2):
        contrasts[a+'__minus__'+b]=dict(
            accuracy_pp=interval(100*(scores[a]['correct']-scores[b]['correct']),videos),
            gold_logp=interval(scores[a]['logp']-scores[b]['logp'],videos),
            generation_s=interval((scores[a]['seconds']-scores[b]['seconds'])[5:],videos[5:]))
    for name in [*p['arms'],*p['probes']]:
        items=[r['arms'][name] if name in p['arms'] else r['probes'][name] for r in subset]
        diagnostics[name]=dict(
            gold_logp_delta=interval([a['option_logp'][gold[r['sample_index']]['correct_option']]-r['native']['option_logp'][gold[r['sample_index']]['correct_option']] for a,r in zip(items,subset)],sv),
            accuracy_delta_pp=interval([100*(int(a['answer']==gold[r['sample_index']]['correct_option'])-int(r['native']['answer']==gold[r['sample_index']]['correct_option'])) for a,r in zip(items,subset)],sv),
            flip=interval([a['answer']!=r['native']['answer'] for a,r in zip(items,subset)],sv),
            js=interval([a['js_to_native'] for a in items],sv))
    work={}
    for name in ['native',*p['arms']]:
        ledgers=[r['native']['work_ledger'] if name=='native' else r['arms'][name]['work_ledger'] for r in subset]
        work[name]={phase+'_'+field:sum(v.get(field,0) for ledger in ledgers for k,v in ledger.items() if k.startswith(phase+':'))
                    for phase in ('prefill','decode') for field in ('tokens','routed_rows','shared_rows','selected_real','selected_null','selected_slots')}
    norms={}
    for layer in range(48 if p['model']=='qwen' else 28):
        for modality in ('audio','video','other'):
            values=[v for r in subset for v in r['native_decomposition'] if v['layer']==layer and v['modality']==modality]
            assert len(values)==len(subset)
            norms[f'{layer}:{modality}']={field:interval([v[field] for v in values],sv)
                for field in ('input_norm','routed_norm','shared_norm','routed_over_input','shared_over_input')}
    interactions={}
    if p['model']=='unimoe':
        values=[]
        for r in subset:
            g=gold[r['sample_index']]['correct_option']
            values.append(r['probes']['both_zero_all']['option_logp'][g]-r['probes']['routed_zero_all']['option_logp'][g]
                          -r['probes']['shared_zero_all']['option_logp'][g]+r['native']['option_logp'][g])
        interactions['all']=interval(values,sv)
    summary=dict(result_status='verified',model=p['model'],n=1000,diagnostic_n=32,arms=arms,
        paired_contrasts=contrasts,diagnostics_vs_native=diagnostics,diagnostic_work_totals=work,
        native_branch_norms=norms,branch_interaction_gold_logp=interactions,
        reused_aggregate_references=json.loads((out/'reused_main_table.json').read_text()),
        main_table_replacement=False,raw_sha256=sha(raw),protocol_sha256=sha(out/'protocol.json'),
        gates_sha256=gates_sha,limitations=p['limitations'],
        interval_note='Pointwise source-video-cluster bootstrap. Paired comparisons only among new scopes or within the 32-video diagnostic subset. No CIs or paired deltas are inferred for reused aggregates.')
    write(out/'summary.json',summary)
    with (out/'scope_results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['arm','scope','accuracy_pct','generation_s','n','timed_n'])
        writer.writeheader()
        for name,a in arms.items():writer.writerow(dict(arm=name,scope=json.dumps(p['arms'][name]['scope']),
            accuracy_pct=100*a['accuracy']['mean'],generation_s=a['generation_s']['mean'],n=1000,timed_n=995))
    print(json.dumps(dict(result_status='verified',model=p['model'],n=1000,diagnostic_n=32)),flush=True)


if __name__=='__main__':main()

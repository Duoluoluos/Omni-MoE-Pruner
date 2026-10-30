#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
for model in ('qwen','unimoe'):
    prior=ROOT/'experiments/omnidial_scope_20260929'/f'{model}_full1000_v2'
    old=json.loads((prior/'protocol.json').read_text())
    out=BASE/f'{model}_diagnostic32_v1';out.mkdir(exist_ok=False)
    src=out/'source';src.mkdir()
    for f in (prior/'source').glob('*.py'):shutil.copy2(f,src/f.name)
    for file in ('run_layer_followup.py','analyze_layer_followup.py'):shutil.copy2(BASE/file,src/file)
    for file in ('inputs.json','gates.json'):shutil.copy2(prior/file,out/file)
    shutil.copy2(prior/'environment.json',out/'reference_environment.json')
    refs={};raw_hashes={}
    for i in old['diagnostic_indices']:
        f=prior/f'predictions/{i:04d}.json';r=json.loads(f.read_text());raw_hashes[str(i)]=sha(f)
        assert r['complete']
        refs[str(i)]={k:r[k] for k in ('video','counts','input_sha256','native')}
        refs[str(i)]['late']=r['arms']['cg_late']
    write(out/'references.json',refs)
    bands={k:old['scopes'][k] for k in ('early','middle','late')}
    probes={f'{branch}_zero_{band}':dict(branch=branch,scope=scope) for band,scope in bands.items()
            for branch in (('routed',) if model=='qwen' else ('routed','shared'))}
    identity=sorted(old['diagnostic_indices'],key=lambda i:refs[str(i)]['counts']['total'])
    p={k:old[k] for k in ('model','root','dataset','dataset_sha256','seed','model_seed','batch_size','dtype','greedy','max_new_tokens')}
    p.update(schema='layer-followup-20260930-v1',result_status='planned',expected=32,
             evaluation_indices=old['diagnostic_indices'],identity_indices=[identity[0],identity[-1]],
             diagnostic_strata=old['diagnostic_strata'],native_logp_tolerance=1e-5,
             all_scope=old['scopes']['all'],probes=probes,
             arms={} if model=='qwen' else {'cg_late_protect_last':dict(kind='cgcc',scope=[16,27])},
             parent_run=str(prior),parent_manifest_sha256=sha(prior/'manifest.json'),
             parent_raw_prediction_sha256=raw_hashes,parent_summary_sha256=sha(prior/'summary.json'),
             native_reuse='Two length-extreme Native output and option-logp identity checks; all input digests and model/runtime metadata must match.',
             gates='Exact prior gates.json, no calibration or tuning.',
             phase='Probes zero prefill branch outputs at Native weights; policy applies prefill+decode.',
             timing='Diagnostics only: raw policy wall times not a full benchmark operating point.',
             continuation='Uni full-1000 extension remains planned; pilot policy outputs can be reused without rerunning 32.',
             limitations=['32 preselected unique videos; descriptive means, not a full-benchmark accuracy estimate.',
                          'Native-weight branch removal is not the renormalized OmniDial fallback.',
                          'Protection changes both layer coverage and realized work.',
                          'Archived references are from an earlier session; timing is not a paired same-session comparison.',
                          'One checkpoint per model, OmniVideoBench only.'])
    write(out/'protocol.json',p)
    files=[f for f in out.rglob('*') if f.is_file()]
    write(out/'manifest.json',dict(files={str(f.relative_to(out)):sha(f) for f in sorted(files)}))
    print(model,out,'probes',len(probes),'arms',len(p['arms']),'identity',p['identity_indices'])

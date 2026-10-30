#!/usr/bin/env python3
"""Fresh, outcome-blind protocol after author-confirmed removal of v1 outputs."""
import csv
import json
from pathlib import Path
import random
import shutil
from prepare_scope_study import design
from scope_study_resume import sha, write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'experiments/omnidial_scope_20260929'
SOURCES = ['prepare_scope_study_v2.py','prepare_scope_study.py','run_scope_study_v2.py',
           'scope_study_resume.py','analyze_scope_study_v2.py','analyze_scope_study.py','scope_study_backend.py',
           'todo3_backend.py','todo3_baselines.py','qwen_mcore_controller.py',
           'qwen_mcore_experts.py','qwen_runtime_backend.py',
           'run_mb_new2_cgskip_qwen3omni.py','run_mb_new2_cgskip_unimoe2.py']


def main():
    dataset = ROOT/'eval_data/OmniVideoBench/data.json'
    raw = json.loads(dataset.read_text())
    clean = [dict(sample_index=i, **{k:r[k] for k in ('video','question','options')})
             for i,r in enumerate(raw)]
    assert len(clean) == 1000
    # Whitelist only input length/identity from the surviving verified ledger.
    source = ROOT/'models/pm1_unimoe2_full_4222222/raw_results.jsonl'
    videos = {}
    for line in source.read_text().splitlines():
        r = json.loads(line)
        idx,video,length = r['sample_index'],r['video'],r['input_sequence_length']
        assert clean[idx]['video'] == video
        videos.setdefault(video, (length,idx))
    ordered = sorted(videos.items(), key=lambda v:(v[1][0],v[0]))
    assert len(ordered)==628
    drng,crng = random.Random(20260928),random.Random(20260929)
    chosen,calibration,strata = [],[],[]
    for q in range(4):
        group=ordered[q*628//4:(q+1)*628//4]
        selected=[v[1][1] for v in drng.sample(group,8)]
        cal=[v[1][1] for v in crng.sample(group,2)]
        chosen += selected; calibration += cal
        strata.append(dict(quartile=q+1,indices=sorted(selected),calibration_indices=cal,
                           min_length=group[0][1][0],max_length=group[-1][1][0]))
    for model in ('qwen','unimoe'):
        out=BASE/f'{model}_full1000_v2';out.mkdir(parents=True,exist_ok=False)
        src=out/'source';src.mkdir()
        for name in SOURCES:shutil.copy2(ROOT/'scripts'/name,src/name)
        scopes,all_arms,all_probes=design(model)
        names=('cg_early','cg_late','cg_all') if model=='qwen' else ('cg_early','cg_middle','cg_late')
        arms={name:all_arms[name] for name in names}
        probes={name:policy for name,policy in all_probes.items() if name.endswith('_all')}
        table=ROOT/'research_paper/acl_arr_oct2027/results/main_inference_display.csv'
        refs=[r for r in csv.DictReader(table.open()) if r['benchmark']=='OmniVideoBench'
              and r['backbone'].startswith('Qwen' if model=='qwen' else 'Uni')]
        write(out/'reused_main_table.json',dict(source=str(table.relative_to(ROOT)),source_sha256=sha(table),
            omnidial_scope=scopes['middle'] if model=='qwen' else scopes['all'],rows=refs,
            use='Historical aggregate references only. No synthetic paired deltas, transferred CIs or reruns.'))
        p=dict(schema='partial-scope-v2-resumable',result_status='planned',model=model,
            root=str(ROOT),dataset=str(dataset),dataset_sha256=sha(dataset),expected=1000,
            evaluation_indices=list(range(1000)),seed=20260928,model_seed=20260909,
            arms=arms,scopes=scopes,probes=probes,
            calibration_indices=calibration,calibration_seed=20260929,
            calibration_scope=scopes['middle'] if model=='qwen' else scopes['all'],
            calibration='Eight shared input-only videos, two per archived Uni input-length quartile; Native prefill router statistics; overlap with evaluated inputs is transductive, not held-out.',
            gate_quantiles=dict(reduce=.4) if model=='qwen' else dict(skip=.85,hard=.15,null=.92),
            gate_policy='Fresh deterministic input-only calibration before measured output; never tune on answers or across scopes. gates.json is immutable after calibration.',
            preflight_indices=[ordered[0][1][1],ordered[-1][1][1]],
            diagnostic_indices=sorted(chosen),diagnostic_strata=strata,
            selection_source=str(source.relative_to(ROOT)),selection_source_sha256=sha(source),
            selection='32 common unique videos, 8 per archived Uni input-length quartile; only video/index/input_sequence_length read.',
            restart_difference='Old Qwen-ledger strata and exact gates were deleted; v2 uses the retained Uni input ledger and fresh input-only gates. It is a new experiment, not a v1 resume.',
            practical_phase='prefill and decode; outside scope Native',
            mechanism_phase='prefill only; zero branch OUTPUT after computation at Native weights; not timed',
            batch_size=1,dtype='bfloat16',greedy=True,max_new_tokens=4,
            timing='CUDA-synchronized model.generate including encoders, prefill and decode; excludes preparation and instrumentation',
            warmup='First five evaluation questions excluded from latency only; additionally, each launch warms all arms on its first pending input before timed generation.',
            arm_order='random.Random(seed + sample_index), shuffled identically for both models',
            primary='Full-1000 accuracy, generation time and paired contrasts among the three NEW scopes; gold joined after inference. Historical Main Table references are unpaired.',
            reused_main_table_sha256=sha(out/'reused_main_table.json'),
            native='Native is evaluated only on 32 diagnostic videos and calibration/preflight/warmup inputs, not rerun over 1000.',
            deferred='Width sweep, fixed-budget reruns and repeated Main Table scopes are excluded by author priority.',
            secondary=['generated accuracy','answer flips','full-vocabulary JS','generation time','actual expert rows','branch norms'],
            inference='2000 pointwise source-video-cluster bootstrap resamples; seed 20260928',
            limitations=['One checkpoint per family and one benchmark; no architecture-family causal claim.',
                'Uni cap/top-p/null and confidence scopes can differ in achieved work; record work rather than claim isolated depth effects.',
                'Native-weight branch removal differs from renormalized confidence shared-only fallback and is not speed evidence.',
                'Input-only calibration overlaps evaluated inputs; no test-answer-based tuning or best-scope selection.',
                'Two frameworks; compare within-model paired interventions, not cross-model speed ratios.',
                'New gates/subset after loss of v1 files; not a reproduction of the confidential Main Table points.',
                'Resumed questions are complete paired blocks; worker/session identities are recorded and timing is session-dependent.'])
        write(out/'protocol.json',p);write(out/'inputs.json',clean)
        write(out/'manifest.json',dict(protocol_sha256=sha(out/'protocol.json'),inputs_sha256=sha(out/'inputs.json'),
            reused_main_table_sha256=sha(out/'reused_main_table.json'),
            sources={x.name:sha(x) for x in src.glob('*.py')}))
        print(out,flush=True)


if __name__=='__main__':main()

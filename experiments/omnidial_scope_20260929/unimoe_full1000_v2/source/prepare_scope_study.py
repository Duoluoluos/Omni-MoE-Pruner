#!/usr/bin/env python3
"""Freeze the two author-authorized scope studies before any new outcomes."""
import hashlib
import json
from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research_paper/iclr2027/evidence'
SOURCES = ['scope_study_backend.py', 'run_scope_study.py', 'analyze_scope_study.py',
           'todo3_backend.py', 'todo3_baselines.py', 'qwen_mcore_controller.py',
           'qwen_mcore_experts.py', 'qwen_runtime_backend.py',
           'run_mb_new2_cgskip_qwen3omni.py', 'run_mb_new2_cgskip_unimoe2.py',
           'prepare_scope_study.py']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def design(model):
    if model == 'qwen':
        scopes = dict(early=[0,20], middle=[12,32], late=[28,48], all=[0,48],
                      quarter=[18,30], half=[12,36], threequarter=[6,42])
        low = dict(kind='fixed', budget=4)
    else:
        scopes = dict(early=[0,12], middle=[7,19], late=[16,28], all=[0,28],
                      quarter=[10,17], half=[7,21], threequarter=[3,24])
        low = dict(kind='cap', budget=1)
    arms = {'native': {'kind':'native'}}
    for name in ('early','middle','late','all'):
        arms['low_'+name] = dict(**low, scope=scopes[name])
    for name, scope in scopes.items():
        arms['cg_'+name] = dict(kind='cgcc', scope=scope)
    probes = {'routed_zero_'+name: dict(branch='routed',scope=scopes[name])
              for name in ('early','middle','late','all')}
    if model == 'unimoe':
        probes.update({branch+'_zero_'+name:dict(branch=branch,scope=scopes[name])
                       for branch in ('shared','both') for name in ('middle','all')})
    assert len(arms) == 12
    return scopes, arms, probes


def main():
    dataset = ROOT/'eval_data/OmniVideoBench/data.json'
    records = json.loads(dataset.read_text())
    assert len(records) == 1000
    clean = [dict(sample_index=i, **{k:r[k] for k in ('video','question','options')})
             for i,r in enumerate(records)]
    pm1 = BASE/'runs/pm1_workload_decomposition_worker4193170_20260828/raw_results.jsonl'
    # Select only input length and video identity. No outcome or task labels used.
    videos = {}
    for line in pm1.read_text().splitlines():
        r = json.loads(line)
        videos.setdefault(r['video'], (r['input_sequence_length'],r['sample_index']))
    ordered = sorted(videos.items(), key=lambda v:(v[1][0],v[0]))
    assert len(ordered) == 628
    rng = random.Random(20260928)
    strata, chosen = [], []
    for q in range(4):
        group = ordered[q*len(ordered)//4:(q+1)*len(ordered)//4]
        selected = rng.sample(group,32)
        strata.append(dict(quartile=q+1, indices=sorted(v[1][1] for v in selected),
                           min_length=group[0][1][0], max_length=group[-1][1][0]))
        chosen += [v[1][1] for v in selected]
    for model in ('qwen','unimoe'):
        out = BASE/f'runs/scope_study_{model}_full1000_20260928_v1'
        out.mkdir(parents=True, exist_ok=False)
        source = out/'source'; source.mkdir()
        for name in SOURCES:
            shutil.copy2(ROOT/'scripts'/name, source/name)
        gates = BASE/f'runs/todo3_{model}_full1000_20260921_v1/gates.json'
        thresholds = json.loads(gates.read_text())['thresholds']
        scopes,arms,probes = design(model)
        protocol = dict(schema='partial-scope-v1', result_status='planned', model=model,
            root=str(ROOT), dataset=str(dataset), dataset_sha256=sha(dataset), expected=1000,
            evaluation_indices=list(range(1000)), seed=20260928, model_seed=20260909,
            arms=arms, scopes=scopes, probes=probes, thresholds=thresholds,
            gate_source=str(gates.relative_to(ROOT)), gate_source_sha256=sha(gates),
            gate_policy='Frozen TODO3 reference; no new outcome-driven calibration.',
            preflight_indices=[ordered[0][1][1], ordered[-1][1][1]],
            diagnostic_indices=sorted(chosen), diagnostic_strata=strata,
            selection_source=str(pm1.relative_to(ROOT)), selection_source_sha256=sha(pm1),
            selection='128 unique videos, 32 per archived Qwen input-length quartile; first question per video',
            practical_phase='prefill and decode; outside scope native',
            mechanism_phase='prefill-only; zero branch output AFTER computation at native router weights',
            batch_size=1, dtype='bfloat16', greedy=True, max_new_tokens=4,
            timing='CUDA synchronized generation incl. encoders/prefill/decode; excludes preparation and hooks',
            warmup='First five questions excluded from timing only; all 1000 scored.',
            arm_order='Deterministic per-question shuffle, random.Random(seed + sample_index)',
            work='Independent output-checked passes on diagnostic subset only; phase/layer actual MLP rows, selected real/null',
            primary='Paired correct-option first-token log-probability relative to within-run Native',
            secondary=['generated accuracy','answer flips','full-vocabulary JS','generation seconds',
                       'actual expert rows','native routed/shared norm decomposition'],
            inference='2000 source-video-cluster bootstrap draws; seed 20260928; all predeclared contrasts reported',
            limitations=['one checkpoint per family; one benchmark',
                'Uni top-p/null/cap1 scopes are not exactly work matched',
                'CG scope contrasts mix depth, width and achieved work; no pure component attribution',
                'branch zeroing retains native weights, unlike renormalized CG shared-only fallback',
                'historical gate calibration used inputs from this benchmark; not a held-out calibration test',
                'separate frameworks per backbone; compare interventions within each model',
                'not a reproduction or replacement of author-confidential optimized CG points'])
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        (out/'inputs.json').write_text(json.dumps(clean,indent=2)+'\n')
        manifest = dict(protocol_sha256=sha(out/'protocol.json'), inputs_sha256=sha(out/'inputs.json'),
                        sources={p.name:sha(p) for p in source.glob('*.py')})
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(out)


if __name__ == '__main__':
    main()

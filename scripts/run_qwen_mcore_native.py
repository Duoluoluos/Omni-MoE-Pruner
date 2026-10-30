#!/usr/bin/env python3
"""Native top-8 using the frozen backend of the completed MCore comparisons."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(rows, values):
    groups = {}
    for row, value in zip(rows, values):
        groups.setdefault(row['video'], []).append(value)
    sums = np.array([sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(20260924)
    means = []
    for _ in range(2000):
        ids = rng.integers(0, len(groups), len(groups))
        means.append(sums[ids].sum()/counts[ids].sum())
    return dict(mean=float(np.mean(values)), ci95=np.quantile(means, [.025, .975]).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=['avut', 'worldsense'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    expected = {'avut':1733, 'worldsense':3167}[args.benchmark]
    baseline = ROOT/f'research_paper/iclr2027/evidence/runs/qwen_mcore_three_arm_{args.benchmark}_full{expected}_20260924'
    prior_protocol = json.loads((baseline/'protocol.json').read_text())
    prior_environment = json.loads((baseline/'environment.json').read_text())
    prior_summary = json.loads((baseline/'summary.json').read_text())
    assert prior_summary['result_status'] == 'verified'
    assert sha(baseline/'raw_results.jsonl') == prior_summary['raw_sha256']
    archived = [json.loads(x) for x in (baseline/'raw_results.jsonl').read_text().splitlines()]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'raw_results.jsonl').exists(), 'Use a fresh output directory.'
    frozen = out/'source';frozen.mkdir()
    files = ['qwen_mcore_experts.py', 'qwen_mcore_controller.py', 'qwen_runtime_backend.py',
             'run_mb_new2_cgskip_qwen3omni.py']
    for name in files:
        original = baseline/'source'/name
        assert sha(original) == prior_environment['sources']['scripts/'+name]
        shutil.copy2(original, frozen/name)
    shutil.copy2(Path(__file__), frozen/Path(__file__).name)
    sys.path.insert(0, str(ROOT/'qwen-omni-utils/src'))
    sys.path.insert(0, str(frozen))
    import torch
    import transformers
    import run_mb_new2_cgskip_qwen3omni as old
    # Frozen helpers derive paths from __file__; redirect only asset roots.
    archived_root = old.REPO
    old.REPO = ROOT
    old.MODEL_PATH = ROOT/old.MODEL_PATH.relative_to(archived_root)
    for cfg in old.BENCHMARKS.values():
        for key in ['dataset', 'video_dir']:
            cfg[key] = ROOT/cfg[key].relative_to(archived_root)
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import _get_feat_extract_output_lengths
    from qwen_omni_utils import process_mm_info
    from qwen_mcore_controller import MCoreQwenController
    from megatron.core.transformer.moe.experts import GroupedMLP
    assert shutil.which('ffmpeg')
    assert sha(old.BENCHMARKS[args.benchmark]['dataset']) == prior_protocol['dataset_sha256']
    assert sha(old.MODEL_PATH/'config.json') == prior_environment['model_config_sha256']
    assert torch.__version__ == prior_environment['torch']
    assert transformers.__version__ == prior_environment['transformers']
    records, video_dir = old.load_benchmark(args.benchmark)
    protocol = dict(result_status='planned', benchmark=args.benchmark, expected=expected,
        arms=['native'], intervention_scope=None, routing='Native top-8 at every MoE layer; no budget reduction',
        backend='megatron_core_grouped_mlp', native_rerun=True,
        checkpoint=str(old.MODEL_PATH), model_config_sha256=sha(old.MODEL_PATH/'config.json'),
        dataset_sha256=prior_protocol['dataset_sha256'], indices=prior_protocol['indices'],
        baseline_run=str(baseline.relative_to(ROOT)), baseline_raw_sha256=prior_summary['raw_sha256'],
        batch_size=1, dtype='bfloat16', max_new_tokens=4, greedy=True, model_seed=20260909,
        generation_timing=prior_protocol['generation_timing'],
        warmup='First five Native questions excluded from timing only; all questions scored.',
        bootstrap='2000 source-video-cluster resamples, seed 20260924',
        comparison='Same backend and question set, separately measured from the three-arm run.',
        historical_length_policy='Record input changes; do not exclude questions for cross-run differences.',
        cross_backend_text_equality_required=False, main_table_backfill_authorized=True)
    old.atomic_json(out/'protocol.json', protocol)
    def progress(stage, **details):
        state=dict(stage=stage,pid=os.getpid(),time=time.time(),expected=expected,**details)
        old.atomic_json(out/'progress.json',state);print(json.dumps(state),flush=True)
    torch.manual_seed(20260909)
    progress('loading',completed=0)
    model=Qwen3OmniMoeForConditionalGeneration.from_pretrained(str(old.MODEL_PATH),
        dtype=torch.bfloat16,device_map='cuda',attn_implementation='flash_attention_2').eval()
    model.disable_talker()
    processor=Qwen3OmniMoeProcessor.from_pretrained(str(old.MODEL_PATH),use_fast=False)
    control=MCoreQwenController(model,scope=(12,32),tau_reduce=None,
        progress=lambda layer:progress('converting',layer=layer,completed=0))
    assert len(control.blocks)==48
    assert all(isinstance(b.grouped_experts.mlp,GroupedMLP) and b.experts is None for b in control.blocks)
    old.atomic_json(out/'environment.json',dict(hostname=socket.gethostname(),torch=torch.__version__,
        transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),omp_threads=os.environ.get('OMP_NUM_THREADS'),
        ffmpeg=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
        backend='megatron_core_grouped_mlp',moe_layers=48,original_hf_expert_loops_present=False,
        megatron_core='0.12.3',grouped_gemm='1.1.4',
        frozen_sources={p.name:sha(p) for p in frozen.glob('*.py')},
        model_config_sha256=sha(old.MODEL_PATH/'config.json')))
    progress('megatron_ready',completed=0,moe_layers=48)
    rows=[]
    try:
        for ordinal, previous in enumerate(archived):
            index=previous['sample_index']
            record={k:records[index][k] for k in ('video','question','options')}
            assert record['video']==previous['video']
            progress('prepare',completed=len(rows),sample_index=index)
            conv=old.build_conversation(video_dir/record['video'],record)
            prompt=processor.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
            audios,images,videos=process_mm_info(conv,use_audio_in_video=True)
            inputs=processor(text=prompt,audio=audios,images=images,videos=videos,
                return_tensors='pt',padding=True,use_audio_in_video=True)
            audio_expected=int(_get_feat_extract_output_lengths(inputs['feature_attention_mask'].sum(-1)).sum())
            counts={m:int((inputs['input_ids']==token).sum()) for m,token in [('audio',151675),('video',151656)]}
            n=int(inputs['input_ids'].numel())
            assert counts['audio']==audio_expected>0 and counts['video']>0
            inputs=inputs.to(model.device).to(model.dtype)
            h=hashlib.sha256()
            for key,value in sorted(inputs.items()):
                if isinstance(value,torch.Tensor):
                    h.update(key.encode());h.update(str((tuple(value.shape),value.dtype)).encode())
                    h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            progress('generate',completed=len(rows),sample_index=index)
            with control.arm('native'):
                value=old.generate_once(model,processor,inputs,4)
            value.pop('first_logits')
            row=dict(ordinal=ordinal,sample_index=index,video=record['video'],
                input_sequence_length=n,input_sha256=h.hexdigest(),modality_counts=counts,input_gate=True,
                comparison_input_sequence_length=previous['input_sequence_length'],
                comparison_input_sha256=previous['input_sha256'],arms={'native':value})
            old.append_jsonl(out/'raw_results.jsonl',row);rows.append(row)
            progress('sample_complete',completed=len(rows),sample_index=index)
            del inputs,value
    except Exception as exc:
        old.append_jsonl(out/'errors.jsonl',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
        progress('failed',completed=len(rows),error=str(exc));raise
    # Labels are consulted only after generation for scoring.
    scores=[int(r['arms']['native']['answer']==records[r['sample_index']]['correct_option']) for r in rows]
    timed=rows[5:]
    summary=dict(result_status='verified',backend='megatron_core_grouped_mlp',
        n=len(rows),expected=expected,timed_samples=len(timed),errors=0,
        gates=dict(complete=len(rows)==expected,positive_audio_video=all(r['input_gate'] for r in rows)),
        historical_input_length_differences=sum(r['input_sequence_length']!=r['comparison_input_sequence_length'] for r in rows),
        comparison_input_digest_differences=sum(r['input_sha256']!=r['comparison_input_sha256'] for r in rows),
        arms={'native':dict(accuracy=bootstrap(rows,scores),
            generation_seconds=bootstrap(timed,[r['arms']['native']['end_to_end_latency_ms']/1000 for r in timed]))},
        contrasts={},bootstrap=protocol['bootstrap'],
        comparison='Same backend, separate measurement session; no within-run Native contrasts.')
    assert all(summary['gates'].values())
    for name,filename in [('raw','raw_results.jsonl'),('protocol','protocol.json'),('environment','environment.json')]:
        summary[name+'_sha256']=sha(out/filename)
    old.atomic_json(out/'summary.json',summary)
    torch.distributed.destroy_process_group()
    progress('complete',completed=len(rows),result_status='verified')


if __name__=='__main__':
    main()

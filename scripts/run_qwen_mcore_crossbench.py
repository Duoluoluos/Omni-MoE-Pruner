#!/usr/bin/env python3
"""Author-requested Megatron-Core top-4 / top-6 / CG-CC benchmark run.

Frozen AVUT/WorldSense question sets and confidence thresholds; no tuning.
The chosen backend and its six-input gate are recorded explicitly. The MCore
option is an expert-backend port; HF still owns multimodal generation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import time
import traceback

import numpy as np
import torch
import transformers
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import _get_feat_extract_output_lengths
import run_mb_new2_cgskip_qwen3omni as old
from qwen_omni_utils import process_mm_info
from qwen_runtime_backend import CleanQwenController
from todo3_backend import input_digest

ROOT = Path(__file__).resolve().parents[1]
ARMS = ['top4', 'top6', 'cgcc']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(rows, values, repeats=2000):
    """Question-weighted mean and video-cluster percentile interval."""
    groups = {}
    for row, value in zip(rows, values):
        groups.setdefault(row['video'], []).append(value)
    sums = np.array([sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(20260924)
    means = []
    for _ in range(repeats):
        ids = rng.integers(0, len(groups), len(groups))
        means.append(sums[ids].sum()/counts[ids].sum())
    return dict(mean=float(np.mean(values)), ci95=np.quantile(means, [.025, .975]).tolist())


def finalize(out, rows, records, expected, errors):
    # Answers are used only by this post-generation scorer.
    scores = {a: [int(r['arms'][a]['answer'] == records[r['sample_index']]['correct_option'])
                  for r in rows] for a in ARMS}
    timed = [r for r in rows if r['ordinal'] >= 5]
    times = {a: [r['arms'][a]['end_to_end_latency_ms']/1000 for r in timed] for a in ARMS}
    complete = len(rows) == expected and not errors
    summary = dict(result_status='verified' if complete else 'placeholder',
        verification_scope='Recorded backend, frozen partial band [12,32), matched paired generation; no full Megatron generation port claim.',
        backend=json.loads((out/'protocol.json').read_text())['backend'],
        n=len(rows), expected=expected, timed_samples=len(timed), errors=len(errors),
        gates=dict(complete=complete,
                   positive_audio_video=all(r['input_gate'] for r in rows)),
        historical_input_length_differences=sum(r['input_sequence_length'] != r['historical_input_sequence_length'] for r in rows),
        arms={}, contrasts={}, bootstrap='2000 source-video-cluster resamples, seed 20260924')
    for a in ARMS:
        summary['arms'][a] = dict(accuracy=bootstrap(rows, scores[a]), generation_seconds=bootstrap(timed, times[a]))
    for ref in ['cgcc']:
        for a in ARMS:
            if a == ref: continue
            summary['contrasts'][f'{a}_minus_{ref}'] = dict(
                accuracy_difference=bootstrap(rows, np.array(scores[a])-scores[ref]),
                generation_seconds_difference=bootstrap(timed, np.array(times[a])-times[ref]))
    summary['raw_sha256'] = sha(out/'raw_results.jsonl')
    summary['protocol_sha256'] = sha(out/'protocol.json')
    summary['environment_sha256'] = sha(out/'environment.json')
    old.atomic_json(out/'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', choices=['avut','worldsense'], required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); out = args.out
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'raw_results.jsonl').exists(), 'Use a fresh run directory.'
    assert shutil.which('ffmpeg'), 'ffmpeg is required before model loading'
    expected = {'avut':1733,'worldsense':3167}[args.benchmark]
    source = ROOT/f'research_paper/iclr2027/evidence/runs/cgcc_qwen3omni_{args.benchmark}_full{expected}_worker4261719_20260918'
    archived = old.load_jsonl(source/'raw_results.jsonl')
    env = json.loads((source/'environment.json').read_text())
    assert env['dataset_sha256']==sha(old.BENCHMARKS[args.benchmark]['dataset'])
    assert env['model_config_sha256']==sha(old.MODEL_PATH/'config.json')
    assert len(archived) == expected and env['layer_band'] == [12,32]
    assert env['cgskip_policy']['base_budget'] == env['cgskip_policy']['boost_budget'] == 8
    assert env['cgskip_policy']['reduce_budget'] == 4
    records, video_dir = old.load_benchmark(args.benchmark)
    assert len({r['sample_index'] for r in archived}) == expected
    protocol = dict(result_status='planned', benchmark=args.benchmark,
        checkpoint=str(old.MODEL_PATH), checkpoint_index_sha256=sha(old.MODEL_PATH/'model.safetensors.index.json'),
        dataset_sha256=sha(old.BENCHMARKS[args.benchmark]['dataset']),
        archived_raw_sha256=sha(source/'raw_results.jsonl'),
        indices=[r['sample_index'] for r in archived], expected=expected,
        backend='megatron_core_grouped_mlp', arms=ARMS, intervention_scope=[12,32],
        confidence='raw full-softmax top-4 mass, top-8 candidate prefix',
        thresholds=env['calibrated_thresholds'], batch_size=1, dtype='bfloat16',
        model_seed=20260909, max_new_tokens=4, greedy=True,
        generation_timing='CUDA-synchronized model.generate, excluding input preparation, stats, scoring, and logit transfer',
        warmup='Three arms on first five questions; exclude these questions from timing only.',
        order='Independent deterministic arm permutation per question, seed 20260924 + ordinal',
        author_direction='2026-09-24: use Megatron-Core for top4/top6/CG-CC only; HF output equivalence is not required; retain historical Native and disclose backend differences in appendix.',
        native_rerun=False, cross_backend_text_equality_required=False,
        historical_length_policy='Record differences without excluding questions; all three new arms share the same prepared input.',
        failures='Fail closed; never silently drop a newly failing input.',
        no_recalibration=True, main_table_backfill_authorized=True)
    old.atomic_json(out/'protocol.json', protocol)
    def progress(stage, **kwargs):
        state=dict(stage=stage,pid=os.getpid(),time=time.time(),**kwargs)
        old.atomic_json(out/'progress.json', state); print(json.dumps(state),flush=True)
    torch.manual_seed(20260909)
    progress('loading', completed=0, expected=expected)
    model=Qwen3OmniMoeForConditionalGeneration.from_pretrained(str(old.MODEL_PATH),dtype=torch.bfloat16,device_map='cuda',attn_implementation='flash_attention_2').eval()
    model.disable_talker()
    processor=Qwen3OmniMoeProcessor.from_pretrained(str(old.MODEL_PATH),use_fast=False)
    from qwen_mcore_controller import MCoreQwenController
    from megatron.core.transformer.moe.experts import GroupedMLP
    control=MCoreQwenController(model,scope=(12,32),tau_reduce=env['calibrated_thresholds']['tau_reduce'],
        progress=lambda l:progress('converting',layer=l,completed=0,expected=expected))
    assert len(control.blocks)==48
    assert all(isinstance(b.grouped_experts.mlp,GroupedMLP) and b.experts is None for b in control.blocks)
    progress('megatron_ready',moe_layers=48,backend='Megatron-Core GroupedMLP / Grouped GEMM',completed=0,expected=expected)
    sources=[Path(__file__),ROOT/'scripts/qwen_runtime_backend.py',ROOT/'scripts/run_mb_new2_cgskip_qwen3omni.py',
        ROOT/'scripts/todo3_backend.py',ROOT/'scripts/qwen_mcore_experts.py',ROOT/'scripts/qwen_mcore_controller.py']
    frozen=out/'source'; frozen.mkdir()
    for f in sources: shutil.copy2(f,frozen/f.name)
    old.atomic_json(out/'environment.json',dict(hostname=socket.gethostname(),torch=torch.__version__,
        transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),omp_threads=os.environ.get('OMP_NUM_THREADS'),
        ffmpeg=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
        backend='megatron_core_grouped_mlp',megatron_core='0.12.3',grouped_gemm='1.1.4',moe_layers=48,
        original_hf_expert_loops_present=False,
        sources={str(f.relative_to(ROOT)):sha(f) for f in sources},
        model_config_sha256=sha(old.MODEL_PATH/'config.json')))
    rows=[]; errors=[]
    try:
        for ordinal, previous in enumerate(archived):
            index=previous['sample_index']
            # Only these input fields reach the processor/model/runtime controller.
            record={k:records[index][k] for k in ('video','question','options')}
            assert record['video'] == previous['video']
            progress('prepare',completed=len(rows),expected=expected,sample_index=index)
            conv=old.build_conversation(video_dir/record['video'],record)
            text=processor.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
            audios,images,videos=process_mm_info(conv,use_audio_in_video=True)
            inputs=processor(text=text,audio=audios,images=images,videos=videos,return_tensors='pt',padding=True,use_audio_in_video=True)
            audio_expected=int(_get_feat_extract_output_lengths(inputs['feature_attention_mask'].sum(-1)).sum())
            counts={m:int((inputs['input_ids']==token).sum()) for m,token in [('audio',151675),('video',151656)]}
            n=int(inputs['input_ids'].numel())
            assert counts['audio']==audio_expected>0 and counts['video']>0
            inputs=inputs.to(model.device).to(model.dtype)
            digest=input_digest(inputs)
            order=ARMS.copy();random.Random(20260924+ordinal).shuffle(order)
            results={}
            progress('generate',completed=len(rows),expected=expected,sample_index=index,order=order)
            for arm in order:
                with control.arm(arm):results[arm]=old.generate_once(model,processor,inputs,4)
            for result in results.values():result.pop('first_logits')
            row=dict(ordinal=ordinal,sample_index=index,video=record['video'],input_sha256=digest,
                input_sequence_length=n,historical_input_sequence_length=previous['input_sequence_length'],modality_counts=counts,input_gate=True,order=order,
                arms=results)
            old.append_jsonl(out/'raw_results.jsonl',row);rows.append(row)
            progress('sample_complete',completed=len(rows),expected=expected,sample_index=index)
            del inputs,results
    except Exception as exc:
        error=dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        old.append_jsonl(out/'errors.jsonl',error);errors.append(error)
        progress('failed',completed=len(rows),expected=expected,error=str(exc))
        raise
    summary=finalize(out,rows,records,expected,errors)
    progress('complete',completed=len(rows),expected=expected,result_status=summary['result_status'])
    assert summary['result_status']=='verified'
    torch.distributed.destroy_process_group()


if __name__=='__main__':main()

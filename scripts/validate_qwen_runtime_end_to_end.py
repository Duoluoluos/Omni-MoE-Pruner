#!/usr/bin/env python3
"""Matched AVUT/WorldSense runtime smoke, preserving frozen policy and input path.

Samples are selected only by archived input length, not scores. No task labels
enter preparation, routing, timing or gates. Results are diagnostic and cannot
replace a full benchmark result.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import shutil
import statistics
import time

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration,Qwen3OmniMoeProcessor
import run_mb_new2_cgskip_qwen3omni as old
from qwen_runtime_backend import CleanQwenController
from todo3_backend import input_digest
from qwen_omni_utils import process_mm_info

ROOT=Path(__file__).resolve().parents[1]


def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'raw_results.jsonl').exists():
        raise RuntimeError('Use a fresh run directory; this diagnostic does not resume.')
    assert shutil.which('ffmpeg'), 'ffmpeg must be on PATH before loading the model'
    def progress(stage,**extra):
        d=dict(stage=stage,pid=os.getpid(),time=time.time(),**extra)
        write(args.out/'progress.json',d);print(json.dumps(d),flush=True)
    torch.manual_seed(20260909)
    manifests={}
    for bench,total in [('avut',1733),('worldsense',3167)]:
        source=ROOT/f'research_paper/iclr2027/evidence/runs/cgcc_qwen3omni_{bench}_full{total}_worker4261719_20260918'
        raw=[json.loads(s) for s in (source/'raw_results.jsonl').open()]
        unique={}
        for row in raw:unique.setdefault(row['video'],row)
        ordered=sorted(unique.values(),key=lambda r:(r['input_sequence_length'],r['sample_index']))
        selected=[ordered[int(q*(len(ordered)-1))]['sample_index'] for q in (.25,.5,.75)]
        env=json.loads((source/'environment.json').read_text())
        assert env['cgskip_policy']['base_budget']==env['cgskip_policy']['boost_budget']==8
        assert env['cgskip_policy']['reduce_budget']==4 and env['layer_band']==[12,32]
        manifests[bench]=dict(indices=selected,thresholds=env['calibrated_thresholds'],policy=env['cgskip_policy'],source=str(source))
    write(args.out/'protocol.json',dict(result_status='planned',selection='one input-length quartile representative at 25/50/75% per benchmark, unique videos; no outcome selection',benchmarks=manifests,repeats=args.repeats,scope=[12,32],max_new_tokens=4,batch_size=1,seed=20260909,measurement='CUDA-synchronized generate; prepared tensors shared; randomized order; stats outside timed passes; one warmup for each arm/input'))
    progress('loading')
    model=Qwen3OmniMoeForConditionalGeneration.from_pretrained(str(old.MODEL_PATH),dtype=torch.bfloat16,device_map='cuda',attn_implementation='flash_attention_2').eval()
    model.disable_talker();processor=Qwen3OmniMoeProcessor.from_pretrained(str(old.MODEL_PATH),use_fast=False)
    legacy=old.CGSkipIntervention(model,8,(12,32))
    clean=CleanQwenController(model)
    write(args.out/'environment.json',dict(hostname=socket.gethostname(),torch=torch.__version__,gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'scripts/qwen_runtime_backend.py',ROOT/'scripts/run_mb_new2_cgskip_qwen3omni.py']}))
    rows=[]
    for bench,manifest in manifests.items():
        records,video_dir=old.load_benchmark(bench)
        legacy.policy=manifest['policy'];legacy.set_thresholds(manifest['thresholds'])
        clean.tau_reduce=manifest['thresholds']['tau_reduce']
        for index in manifest['indices']:
            progress('prepare',benchmark=bench,sample_index=index)
            # Strip answers/categories before the input-only boundary.
            record={k:records[index][k] for k in ('video','question','options')}
            conv=old.build_conversation(video_dir/record['video'],record)
            text=processor.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
            audios,images,videos=process_mm_info(conv,use_audio_in_video=True)
            inputs=processor(text=text,audio=audios,images=images,videos=videos,return_tensors='pt',padding=True,use_audio_in_video=True)
            from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import _get_feat_extract_output_lengths
            expected=int(_get_feat_extract_output_lengths(inputs['feature_attention_mask'].sum(-1)).sum())
            counts={m:int((inputs['input_ids']==token).sum()) for m,token in [('audio',151675),('video',151656)]}
            assert counts['audio']==expected>0 and counts['video']>0
            inputs=inputs.to(model.device).to(model.dtype);identity=input_digest(inputs)
            legacy.prepare_sample(None)
            reference={}
            # Original wrapper correctness passes are separate from timing comparisons.
            for arm in ['native','top4','top6','top8','cgcc']:
                name='cgskip' if arm=='cgcc' else f'fixed:{arm[3:]}' if arm.startswith('top') else 'native'
                with legacy.arm(name),clean.arm('native'):
                    reference[arm]=old.generate_once(model,processor,inputs,4)
            times={a:[] for a in reference};parity={}
            for repeat in range(args.repeats+1):
                order=list(reference);random.Random(20260924+index*10+repeat).shuffle(order)
                progress('generate',benchmark=bench,sample_index=index,repeat=repeat)
                for arm in order:
                    with clean.arm(arm):res=old.generate_once(model,processor,inputs,4)
                    if repeat:
                        times[arm].append(res['end_to_end_latency_ms'])
                    if repeat==0:
                        ref=reference[arm];diff=res['first_logits']-ref['first_logits']
                        parity[arm]=dict(answer_equal=res['answer']==ref['answer'],output_equal=res['output_text']==ref['output_text'],first_logits_max_abs=float(diff.abs().max()),js=old.distribution_js(ref['first_logits'],res['first_logits']))
            row=dict(benchmark=bench,sample_index=index,video=record['video'],input_sha256=identity,tokens=int(inputs['input_ids'].numel()),modality_counts=counts,parity_vs_original_wrapper=parity,
                clean_ms={a:dict(mean=statistics.mean(v),median=statistics.median(v),values=v) for a,v in times.items()},
                original_wrapper_ms={a:v['end_to_end_latency_ms'] for a,v in reference.items()})
            with (args.out/'raw_results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
            rows.append(row);progress('sample_complete',benchmark=bench,sample_index=index,completed=len(rows))
            del inputs,reference
    identity_gate=all(p['output_equal'] and p['js']<1e-5 for r in rows for p in r['parity_vs_original_wrapper'].values())
    summary=dict(result_status='verified' if identity_gate else 'placeholder',verification_scope='Six input-only-selected real-input runtime diagnostics, not full benchmark results.',identity_gate=identity_gate,n=len(rows),rows=rows)
    write(args.out/'summary.json',summary);progress('complete',identity_gate=identity_gate)
    if not identity_gate:raise RuntimeError('Clean dispatch failed the output/first-logit gate; no full rerun authorized by this diagnostic.')


if __name__=='__main__':main()

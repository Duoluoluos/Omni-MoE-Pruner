#!/usr/bin/env python3
"""Gate full-model hybrid MCore expert execution against clean HF on six inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import socket
import statistics
import time

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration,Qwen3OmniMoeProcessor
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import _get_feat_extract_output_lengths
import run_mb_new2_cgskip_qwen3omni as old
from qwen_omni_utils import process_mm_info
from qwen_runtime_backend import CleanQwenController
from qwen_mcore_controller import MCoreQwenController
from todo3_backend import input_digest

ROOT=Path(__file__).resolve().parents[1]
ARMS=['native','top4','top6','top8','cgcc']


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--block-gate',type=Path,required=True);a=p.parse_args()
    gate=json.loads(a.block_gate.read_text());assert gate['parity_gate']
    a.out.mkdir(parents=True,exist_ok=True)
    assert not (a.out/'raw_results.jsonl').exists()
    assert shutil.which('ffmpeg')
    def progress(stage,**kwargs):
        d=dict(stage=stage,pid=os.getpid(),time=time.time(),**kwargs)
        old.atomic_json(a.out/'progress.json',d);print(json.dumps(d),flush=True)
    protocol=dict(result_status='planned',backend='HF generation and multimodal encoders + MCore 0.12.3 GroupedMLP experts',
        selection='unique-video input-length quartiles .25/.5/.75 from each archived benchmark; no outcomes',
        scope=[12,32],arms=ARMS,seed=20260909,max_new_tokens=4,
        correctness_gate='All outputs equal clean HF, first-logit JS < 1e-5; native/top8 output identity.',
        timing='HF references before conversion are not paired speed evidence. MCore timing: one per-arm per-input warmup, three randomized measured repetitions.',
        block_gate=str(a.block_gate),block_gate_sha256=hashlib.sha256(a.block_gate.read_bytes()).hexdigest())
    old.atomic_json(a.out/'protocol.json',protocol)
    torch.manual_seed(20260909);progress('loading')
    model=Qwen3OmniMoeForConditionalGeneration.from_pretrained(str(old.MODEL_PATH),dtype=torch.bfloat16,device_map='cuda',attn_implementation='flash_attention_2').eval()
    model.disable_talker();processor=Qwen3OmniMoeProcessor.from_pretrained(str(old.MODEL_PATH),use_fast=False)
    clean=CleanQwenController(model);inputs_all=[];refs_all=[];meta=[]
    for bench,total in [('avut',1733),('worldsense',3167)]:
        source=ROOT/f'research_paper/iclr2027/evidence/runs/cgcc_qwen3omni_{bench}_full{total}_worker4261719_20260918'
        archived=old.load_jsonl(source/'raw_results.jsonl');unique={}
        for r in archived:unique.setdefault(r['video'],r)
        ordered=sorted(unique.values(),key=lambda r:(r['input_sequence_length'],r['sample_index']))
        chosen=[ordered[int(q*(len(ordered)-1))] for q in (.25,.5,.75)]
        env=json.loads((source/'environment.json').read_text())
        tau=env['calibrated_thresholds']['tau_reduce'];clean.tau_reduce=tau
        records,video_dir=old.load_benchmark(bench)
        for prev in chosen:
            index=prev['sample_index'];progress('hf_reference',benchmark=bench,sample_index=index)
            record={k:records[index][k] for k in ('video','question','options')}
            conv=old.build_conversation(video_dir/record['video'],record)
            text=processor.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
            audios,images,videos=process_mm_info(conv,use_audio_in_video=True)
            inputs=processor(text=text,audio=audios,images=images,videos=videos,return_tensors='pt',padding=True,use_audio_in_video=True)
            n=int(inputs['input_ids'].numel());assert n==prev['input_sequence_length']
            audio_n=int((inputs['input_ids']==151675).sum())
            assert audio_n==int(_get_feat_extract_output_lengths(inputs['feature_attention_mask'].sum(-1)).sum())>0
            assert int((inputs['input_ids']==151656).sum())>0
            inputs=inputs.to(model.device).to(model.dtype);refs={}
            for arm in ARMS:
                with clean.arm(arm):refs[arm]=old.generate_once(model,processor,inputs,4)
            meta.append(dict(benchmark=bench,sample_index=index,video=record['video'],tokens=n,tau_reduce=tau,input_sha256=input_digest(inputs)))
            refs_all.append(refs);inputs_all.append(inputs.to('cpu'))
    clean.remove();progress('converting')
    control=MCoreQwenController(model,progress=lambda l:progress('converting',layer=l))
    files=[Path(__file__),ROOT/'scripts/qwen_mcore_experts.py',ROOT/'scripts/qwen_mcore_controller.py',ROOT/'scripts/qwen_runtime_backend.py']
    old.atomic_json(a.out/'environment.json',dict(hostname=socket.gethostname(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        cuda=torch.version.cuda,megatron_core='0.12.3',
        sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}))
    frozen=a.out/'source';frozen.mkdir()
    for f in files:shutil.copy2(f,frozen/f.name)
    rows=[]
    for i,(m,inputs,refs) in enumerate(zip(meta,inputs_all,refs_all)):
        inputs=inputs.to(model.device);assert input_digest(inputs)==m['input_sha256']
        control.tau_reduce=m['tau_reduce'];times={a:[] for a in ARMS};parity={};warm={}
        for repeat in range(4):
            order=ARMS.copy();random.Random(20260924+i*10+repeat).shuffle(order)
            progress('mcore_generate',benchmark=m['benchmark'],sample_index=m['sample_index'],repeat=repeat)
            for arm in order:
                with control.arm(arm):res=old.generate_once(model,processor,inputs,4)
                if repeat:times[arm].append(res['end_to_end_latency_ms'])
                else:
                    warm[arm]=res
                    parity[arm]=dict(output_equal=res['output_text']==refs[arm]['output_text'],
                        first_logits_max_abs=float((res['first_logits']-refs[arm]['first_logits']).abs().max()),
                        js=old.distribution_js(refs[arm]['first_logits'],res['first_logits']))
        identity=dict(output_equal=warm['native']['output_text']==warm['top8']['output_text'],js=old.distribution_js(warm['native']['first_logits'],warm['top8']['first_logits']))
        row=dict(**m,parity_vs_clean_hf=parity,native_top8_identity=identity,
                 mcore_ms={k:dict(mean=statistics.mean(v),median=statistics.median(v),values=v) for k,v in times.items()})
        rows.append(row);old.append_jsonl(a.out/'raw_results.jsonl',row);progress('sample_complete',completed=len(rows))
        del inputs
    passed=all(p['output_equal'] and p['js']<1e-5 for r in rows for p in r['parity_vs_clean_hf'].values())
    identity=all(r['native_top8_identity']['output_equal'] and r['native_top8_identity']['js']<1e-10 for r in rows)
    old.atomic_json(a.out/'summary.json',dict(result_status='verified' if passed and identity else 'placeholder',
        verification_scope='Six-input full-model numerical/runtime diagnostic only, not full benchmark quality or speed.',
        identity_gate=passed and identity,n=len(rows),rows=rows))
    progress('complete',identity_gate=passed and identity)
    torch.distributed.destroy_process_group()
    if not passed or not identity:raise RuntimeError('MCore full-model numerical gate failed; no full rerun with this backend.')


if __name__=='__main__':main()

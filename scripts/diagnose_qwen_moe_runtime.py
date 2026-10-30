#!/usr/bin/env python3
"""Isolated checkpoint-block runtime diagnosis, not a benchmark accuracy run.

Compare the current instrumented budget wrapper with the original HF MoE
forward at the same fixed k. Uses one real checkpoint layer and synthetic
activations; never changes an archived experiment or the shared model code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import socket
import statistics
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextSparseMoeBlock

from run_mb_new2_cgskip_qwen3omni import CGSkipIntervention
from qwen_runtime_backend import prefix_forward

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def load_block(layer):
    model = ROOT / 'models/Qwen3-Omni-30B-A3B-Instruct'
    config = Qwen3OmniMoeTextConfig(**json.loads((model / 'config.json').read_text())['thinker_config']['text_config'])
    with torch.device('meta'):
        block = Qwen3OmniMoeThinkerTextSparseMoeBlock(config)
    prefix = f'thinker.model.layers.{layer}.mlp.'
    mapping = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    weights = {}
    for shard in sorted({v for k, v in mapping.items() if k.startswith(prefix)}):
        with safe_open(model / shard, framework='pt', device='cpu') as f:
            for key in f.keys():
                if key.startswith(prefix):
                    weights[key[len(prefix):]] = f.get_tensor(key).to('cuda', dtype=torch.bfloat16)
    block.load_state_dict(weights, assign=True)
    return block.eval(), config


def legacy_controller():
    # Only the fixed-budget branch is exercised; no confidence policy/labels.
    obj = CGSkipIntervention.__new__(CGSkipIntervention)
    obj._arm = 'native'
    obj._force_budget = None
    obj._eligible_seq_mask = None
    obj._record_stats = False
    obj._executed_sum = 0.0
    obj._executed_count = 0
    obj._budget_values = []
    return obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--layer', type=int, default=20)
    p.add_argument('--lengths', default='1,512,4096')
    p.add_argument('--repeats', type=int, default=7)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260924)
    block, config = load_block(args.layer)
    control = legacy_controller()
    arms = [(backend, k) for backend in ('native_path', 'budget_wrapper', 'prefix_path') for k in (4, 6, 8)]
    rows = []
    for n in map(int, args.lengths.split(',')):
        x = torch.randn(1, n, config.hidden_size, device='cuda', dtype=torch.bfloat16)
        refs = {}
        with torch.inference_mode():
            for k in (4, 6, 8):
                block.top_k = 8
                with control.arm(f'fixed:{k}'):
                    refs[k] = control._budgeted_forward(block,x)[0].clone()
            probs = block.gate(x.reshape(-1,config.hidden_size)).float().softmax(-1)
            candidates = probs.topk(8,dim=-1).indices
            changed_candidates = {str(k):int((candidates[:,:k].sort(-1).values != probs.topk(k,dim=-1).indices.sort(-1).values).any(-1).sum()) for k in (4,6)}
            timings = {f'{a}_top{k}': [] for a, k in arms}
            parity = {}
            for rep in range(args.repeats + 3):
                order = arms.copy()
                random.Random(20260924 + n * 100 + rep).shuffle(order)
                for backend, k in order:
                    name = f'{backend}_top{k}'
                    block.top_k = k if backend == 'native_path' else 8
                    with control.arm(f'fixed:{k}'):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        if backend == 'native_path':
                            y = block(x)[0]
                        elif backend == 'prefix_path':
                            y = block(x)[0] if k == 8 else prefix_forward(block,x,budget=k)[0]
                        else:
                            y = control._budgeted_forward(block,x)[0]
                        torch.cuda.synchronize()
                        ms = (time.perf_counter() - start) * 1000
                    if rep >= 3:
                        timings[name].append(ms)
                    if rep == 0:
                        diff = y.float() - refs[k].float()
                        parity[name] = dict(max_abs=float(diff.abs().max()),
                            relative_rms=float(diff.square().mean().sqrt() / refs[k].float().square().mean().sqrt().clamp_min(1e-12)))
            row = dict(tokens=n, direct_topk_changed_candidates=changed_candidates,
                timings_ms={k:dict(mean=statistics.mean(v),median=statistics.median(v),values=v) for k,v in timings.items()}, parity_vs_original_prefix=parity)
            if n == 512:
                profiles = {}
                for backend in ('native_path', 'budget_wrapper'):
                    block.top_k = 8
                    with control.arm('fixed:8'), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                        y = block(x) if backend == 'native_path' else control._budgeted_forward(block, x)
                        torch.cuda.synchronize()
                    profiles[backend] = {e.key:dict(count=e.count,self_cpu_time_us=e.self_cpu_time_total) for e in prof.key_averages() if any(s in e.key for s in ('nonzero','item','_local_scalar_dense','cudaDeviceSynchronize','cudaStreamSynchronize','DtoH'))}
                    prof.export_chrome_trace(str(args.out / f'profile_{backend}.json'))
                row['profiles'] = profiles
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='profiles'}), flush=True)
        write_json(args.out / 'progress.json', dict(completed_lengths=len(rows), planned_lengths=args.lengths, pid=__import__('os').getpid()))
    result = dict(result_status='verified', verification_scope='Isolated checkpoint-layer diagnostic only; synthetic hidden states; no model quality or end-to-end speed claim.',
        hostname=socket.gethostname(), torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), layer=args.layer, dtype='bfloat16',
        legacy_source_sha256=hashlib.sha256((ROOT/'scripts/run_mb_new2_cgskip_qwen3omni.py').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        warmups=3,repeats=args.repeats,randomized_order=True,rows=rows)
    write_json(args.out / 'summary.json', result)
    print('isolated block diagnosis complete', flush=True)


if __name__ == '__main__':
    main()

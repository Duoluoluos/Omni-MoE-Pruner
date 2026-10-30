"""Scope interventions; timing, work instrumentation and branch probes are separate."""
from collections import defaultdict
import contextlib
import torch
from todo3_backend import Controller


class ScopeController(Controller):
    def __init__(self, backend, thresholds, progress=None):
        # Undo the adapter's historical expert-loop wrapper before installing MCore.
        for block, original in zip(backend.blocks, backend.controller.originals):
            block.forward = original
        if backend.name == 'qwen':
            from qwen_mcore_controller import MCoreQwenController
            self.mcore = MCoreQwenController(backend.model, progress=progress)
        super().__init__(backend)
        self.thresholds = thresholds

    def decide(self, probs, layer, native_budget):
        if self.policy['kind'] == 'cap':
            return native_budget.clamp_max(self.policy['budget'])
        return super().decide(probs, layer, native_budget)

    def _wrap(self, layer, block):
        if self.backend.name != 'qwen':
            return super()._wrap(layer, block)
        from qwen_runtime_backend import prefix_forward
        def forward(hidden_states):
            p = self.policy
            scope = p.get('scope', [0, 48])
            if p['kind'] in ('native', 'identity') or not scope[0] <= layer < scope[1]:
                return prefix_forward(block, hidden_states, budget=8, grouped=block.grouped_experts)
            if p['kind'] == 'cgcc':
                return prefix_forward(block, hidden_states, tau_reduce=self.thresholds['tau_reduce'],
                                      grouped=block.grouped_experts)
            assert p['kind'] == 'fixed'
            return prefix_forward(block, hidden_states, budget=p['budget'], grouped=block.grouped_experts)
        return forward

    @contextlib.contextmanager
    def profile(self):
        if self.backend.name == 'unimoe':
            with super().profile() as ledger:
                yield ledger
            return
        ledger, handles = defaultdict(lambda: defaultdict(int)), []
        for layer, block in enumerate(self.blocks):
            phase = ['prefill']
            def pre(module, args, layer=layer, phase=phase):
                phase[0] = 'prefill' if args[0].shape[-2] > 1 else 'decode'
                ledger[f'{phase[0]}:{layer}']['tokens'] += args[0].numel() // args[0].shape[-1]
            def count(module, args, layer=layer, phase=phase):
                key = f'{phase[0]}:{layer}'
                rows = args[0].shape[0]
                assert rows == int(args[1].sum()), 'GroupedMLP dispatch count mismatch'
                ledger[key]['routed_rows'] += rows
                ledger[key]['selected_real'] += rows
            handles += [block.register_forward_pre_hook(pre),
                        block.grouped_experts.mlp.register_forward_pre_hook(count)]
        try:
            yield ledger
        finally:
            for h in handles:
                h.remove()

    @contextlib.contextmanager
    def zero_branch(self, branch, scope):
        """Prefill-only output ablation at native weights; never a speed experiment."""
        handles = []
        assert branch in ('none', 'routed', 'shared', 'both')
        assert self.policy['kind'] == 'native'
        for layer in range(*scope):
            block = self.blocks[layer]
            if self.backend.name == 'qwen':
                assert branch in ('none', 'routed')
                def zero(module, args, out):
                    if branch != 'none' and args[0].shape[-2] > 1:
                        return (torch.zeros_like(out[0]), *out[1:])
                handles.append(block.register_forward_hook(zero))
            else:
                def zero(module, args, out):
                    if branch != 'none' and args[0].numel() // args[0].shape[-1] > 1:
                        return torch.zeros_like(out)
                if branch in ('none', 'routed', 'both'):
                    handles.append(block.dynamic_real_moe.register_forward_hook(zero))
                if branch in ('shared', 'both'):
                    handles += [e.register_forward_hook(zero) for e in block.fixed_real_moe]
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @contextlib.contextmanager
    def decomposition(self, inputs):
        """Native prefill branch norms on <=128 deterministic positions / modality.

        CPU copies and hooks only occur in independent untimed instrumentation.
        Non-audio/video positions include text and structural sentinels.
        """
        ids = inputs['input_ids'][0]
        audio = 151675 if self.backend.name == 'qwen' else self.backend.model.config.audio_token_id
        video = 151656 if self.backend.name == 'qwen' else self.backend.model.config.video_token_id
        masks = {'audio': ids == audio, 'video': ids == video,
                 'other': (ids != audio) & (ids != video)}
        indices, groups = [], []
        for name, mask in masks.items():
            pos = torch.where(mask)[0]
            if len(pos):
                pos = pos[torch.linspace(0, len(pos)-1, min(len(pos),128),device=ids.device).long()]
                indices.append(pos)
                groups.append((name, len(pos), int(mask.sum())))
        index = torch.cat(indices)
        records, handles = [], []
        def sample(x):
            return x.reshape(-1, x.shape[-1])[index].detach().float().cpu()
        for layer, block in enumerate(self.blocks):
            cache = {}
            def pre(module, args, cache=cache):
                cache.clear()
                if args[0].shape[-2] > 1:
                    cache['input'] = sample(args[0])
            handles.append(block.register_forward_pre_hook(pre))
            if self.backend.name == 'unimoe':
                def routed(module, args, out, cache=cache):
                    if 'input' in cache:
                        cache['routed'] = sample(out)
                handles.append(block.dynamic_real_moe.register_forward_hook(routed))
                for j, expert in enumerate(block.fixed_real_moe):
                    def shared(module, args, out, j=j, cache=cache):
                        if 'input' in cache:
                            cache[f'shared{j}'] = sample(out)
                    handles.append(expert.register_forward_hook(shared))
            def finish(module, args, out, layer=layer, cache=cache):
                if 'input' not in cache:
                    return
                h = cache['input']
                if self.backend.name == 'qwen':
                    r, s = sample(out[0]), torch.zeros_like(h)
                else:
                    r = cache['routed']
                    weights = out[4][index, 5:].detach().float().cpu()
                    s = sum(cache[f'shared{j}'] * weights[:,j,None] for j in range(2))
                offset = 0
                for name, n, population in groups:
                    hh, rr, ss = [v[offset:offset+n] for v in (h,r,s)]
                    hn, rn, sn = [v.norm(dim=-1) for v in (hh,rr,ss)]
                    records.append(dict(layer=layer, modality=name, sampled_tokens=n,
                        population_tokens=population, input_norm=float(hn.mean()),
                        routed_norm=float(rn.mean()), shared_norm=float(sn.mean()),
                        routed_over_input=float((rn/hn.clamp_min(1e-12)).mean()),
                        shared_over_input=float((sn/hn.clamp_min(1e-12)).mean()),
                        routed_shared_cosine=(float(torch.nn.functional.cosine_similarity(rr,ss).mean())
                                               if self.backend.name == 'unimoe' else None)))
                    offset += n
                cache.clear()
            handles.append(block.register_forward_hook(finish))
        try:
            yield records
        finally:
            for h in handles:
                h.remove()

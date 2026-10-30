"""Uninstrumented Qwen expert dispatch preserving the original top-8 prefix.

Keep accounting outside model.generate timing. The fixed-budget identity arm
calls exactly the native forward. Lower fixed budgets retain the same top-8
candidate selection before dropping tail ranks (important for tied router scores).
"""
from __future__ import annotations
import contextlib
import torch
import torch.nn.functional as F


def prefix_forward(block, hidden_states, budget=None, tau_reduce=None, grouped=None):
    shape = hidden_states.shape
    flat = hidden_states.reshape(-1, shape[-1])
    logits = block.gate(flat)
    weights, selected = F.softmax(logits, dim=-1, dtype=torch.float32).topk(8, dim=-1)
    if budget is None:
        # Historical Qwen policy uses raw full-router mass, before top-8 renorm.
        budgets = torch.where(weights[:, :4].sum(-1) >= tau_reduce, 4, 8)
        keep = torch.arange(8, device=flat.device)[None, :] < budgets[:, None]
    else:
        keep = torch.arange(8, device=flat.device)[None, :] < budget
    weights = weights.masked_fill(~keep, 0.0)
    weights = (weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)).to(flat.dtype)
    if budget is not None:
        weights, selected = weights[:, :budget], selected[:, :budget]
    if grouped is not None:
        return grouped(flat, selected, weights).reshape(shape), logits
    mask = F.one_hot(selected, num_classes=block.num_experts).permute(2, 1, 0)
    if budget is None:
        mask = mask * keep.T.unsqueeze(0)
    active = (mask.sum(dim=(-1,-2)) > 0).nonzero()
    out = torch.zeros_like(flat)
    for expert in active:
        ranks, tokens = torch.where(mask[expert].squeeze(0))
        value = block.experts[expert](flat[None,tokens].reshape(-1,shape[-1]))
        out.index_add_(0,tokens,(value * weights[tokens,ranks,None]).to(flat.dtype))
    return out.reshape(shape), logits


class CleanQwenController:
    def __init__(self, model, scope=(12,32), tau_reduce=None):
        self.scope = scope
        self.tau_reduce = tau_reduce
        self.policy = 'native'
        self.blocks = [m for m in model.thinker.model.modules()
                       if m.__class__.__name__ == 'Qwen3OmniMoeThinkerTextSparseMoeBlock']
        assert len(self.blocks) == 48
        self.originals = [m.forward for m in self.blocks]
        for layer, block in enumerate(self.blocks):
            block.forward = self.wrap(layer,block)

    def wrap(self,layer,block):
        original = self.originals[layer]
        def forward(hidden_states):
            if self.policy in ('native','top8') or not self.scope[0] <= layer < self.scope[1]:
                return original(hidden_states)
            if self.policy == 'cgcc':
                return prefix_forward(block,hidden_states,tau_reduce=self.tau_reduce)
            return prefix_forward(block,hidden_states,budget=int(self.policy[3:]))
        return forward

    @contextlib.contextmanager
    def arm(self,name):
        if name not in ('native','top8','top4','top6','cgcc'):
            raise ValueError(name)
        old = self.policy
        self.policy = name
        try:
            yield
        finally:
            self.policy = old

    def remove(self):
        for block,original in zip(self.blocks,self.originals):
            block.forward = original

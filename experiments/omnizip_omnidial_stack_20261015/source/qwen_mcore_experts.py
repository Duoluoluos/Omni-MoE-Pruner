"""Megatron-Core GroupedMLP adapter for a Qwen expert block.

This is an expert-backend port, not Megatron's full multimodal generation
runtime. Qwen attention, encoders and KV cache remain in their original backend.
Router probabilities are applied after the down projection, matching Qwen;
MCore receives unit probabilities to avoid moving BF16 rounding across GEMMs.
"""
from __future__ import annotations
import os
import tempfile
import torch
import torch.nn.functional as F
from torch import nn


def initialize_single_rank():
    from megatron.core import parallel_state
    if not torch.distributed.is_initialized():
        fd,path=tempfile.mkstemp(prefix='qwen_mcore_rank_');os.close(fd);os.unlink(path)
        torch.distributed.init_process_group('nccl',init_method='file://'+path,world_size=1,rank=0)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,pipeline_model_parallel_size=1,expert_model_parallel_size=1)


class MCoreGroupedExperts(nn.Module):
    def __init__(self,block):
        super().__init__()
        from megatron.core.transformer.moe.experts import GroupedMLP
        from megatron.core.transformer.transformer_config import TransformerConfig
        initialize_single_rank()
        e=block.num_experts;h=block.gate.in_features
        intermediate=block.experts[0].up_proj.out_features
        config=TransformerConfig(num_layers=1,hidden_size=h,num_attention_heads=1,
            ffn_hidden_size=intermediate,moe_ffn_hidden_size=intermediate,
            num_moe_experts=e,moe_router_topk=8,moe_grouped_gemm=True,
            gated_linear_unit=True,activation_func=F.silu,add_bias_linear=False,
            params_dtype=torch.bfloat16,bf16=True,perform_initialization=False,
            use_cpu_initialization=False,hidden_dropout=0.,attention_dropout=0.)
        self.mlp=GroupedMLP(e,config).eval();self.num_experts=e
        self.reset_accounting()
        # Preserve HF's separate BF16 SiLU/multiply rounding boundaries and
        # avoid counting shape-triggered torch.compile work as generation.
        def hf_activation(x,probs):
            gate,up=x.chunk(2,dim=-1)
            return ((F.silu(gate)*up)*probs).to(x.dtype)
        self.mlp.activation_func_with_probs=hf_activation
        with torch.no_grad():
            w1=self.mlp.weight1.view(e,h,2*intermediate)
            w2=self.mlp.weight2.view(e,intermediate,h)
            for i,expert in enumerate(block.experts):
                w1[i,:,:intermediate].copy_(expert.gate_proj.weight.T)
                w1[i,:,intermediate:].copy_(expert.up_proj.weight.T)
                w2[i].copy_(expert.down_proj.weight.T)

    def reset_accounting(self):
        self.executed_rows = 0
        self.calls = 0

    def forward(self,flat,selected,weights):
        width=selected.shape[1]
        valid=weights.reshape(-1)>0
        self.executed_rows += int(valid.sum().item())
        self.calls += 1
        pair_indices=torch.arange(selected.numel(),device=flat.device)[valid]
        experts=selected.reshape(-1)[valid]
        token_indices=torch.arange(len(flat),device=flat.device).repeat_interleave(width)[valid]
        probabilities=weights.reshape(-1)[valid]
        order=experts.argsort(stable=True)
        tokens=token_indices[order]
        counts=torch.bincount(experts,minlength=self.num_experts).to('cpu')
        ones=torch.ones((len(tokens),),device=flat.device,dtype=flat.dtype)
        values,_=self.mlp(flat[tokens],counts,ones)
        values=values*probabilities[order,None]
        # Match HF's ascending-expert BF16 additions, avoiding nondeterministic
        # atomic accumulation over multiple experts assigned to the same token.
        contributions=torch.zeros((selected.numel(),flat.shape[-1]),device=flat.device,dtype=flat.dtype)
        contributions[pair_indices[order]]=values.to(flat.dtype)
        contributions=contributions.view(len(flat),width,-1)
        ranks=selected.argsort(dim=-1)
        result=torch.zeros_like(flat)
        row=torch.arange(len(flat),device=flat.device)
        for k in range(width):
            result=result+contributions[row,ranks[:,k]]
        return result

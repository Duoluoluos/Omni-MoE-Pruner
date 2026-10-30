"""P2 compact deterministic aggregation on the proven P0 dispatch path.

Keeps P0 routing, stable expert-major order, GroupedMLP and ascending-expert
BF16 accumulation. Replaces the [tokens, topk, hidden] contribution tensor with
a compact [tokens, topk] int64 location table plus one [tokens, hidden] gathered
contribution per accumulation step. No telemetry scalar synchronization.
"""
from __future__ import annotations
import os,tempfile
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


class MCoreGroupedExpertsP2(nn.Module):
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

    def forward(self,flat,selected,weights):
        width=selected.shape[1]
        valid=weights.reshape(-1)>0
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

        # Map every original top-k pair to its compact expert-major output row.
        # Invalid pairs remain -1 and contribute exact BF16 zero.
        locations=torch.full((selected.numel(),),-1,device=flat.device,dtype=torch.long)
        locations[pair_indices[order]]=torch.arange(len(values),device=flat.device)
        locations=locations.view(len(flat),width)
        ranks=selected.argsort(dim=-1)
        result=torch.zeros_like(flat)
        for k in range(width):
            pos=locations.gather(1,ranks[:,k:k+1]).squeeze(1)
            keep=pos>=0
            contribution=values[pos.clamp_min(0)].masked_fill(~keep.unsqueeze(1),0)
            result=result+contribution
        return result

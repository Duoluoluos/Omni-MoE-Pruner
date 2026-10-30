"""P1 MCore-native single-rank token dispatcher adapter.

Uses MoEAllGatherTokenDispatcher at TP=EP=1 for compact expert-major
permutation. Transformer Engine is unavailable in this environment, so MCore's
unfused permute implementation is used. Native scatter-add unpermutation changed
BF16 accumulation order and failed the identity gate; this exact variant keeps
only a compact [tokens, topk] integer reverse map and accumulates expert outputs
in ascending expert-id order, matching the established Qwen adapter without the
old [tokens, topk, hidden] contribution buffer. Router probabilities remain
applied after the expert down projection.
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


class MCoreGroupedExpertsP1(nn.Module):
    def __init__(self,block):
        super().__init__()
        from megatron.core.transformer.moe.experts import GroupedMLP
        from megatron.core.transformer.moe.token_dispatcher import MoEAllGatherTokenDispatcher
        from megatron.core.transformer.transformer_config import TransformerConfig
        initialize_single_rank()
        e=block.num_experts;h=block.gate.in_features
        intermediate=block.experts[0].up_proj.out_features
        config=TransformerConfig(num_layers=1,hidden_size=h,num_attention_heads=1,
            ffn_hidden_size=intermediate,moe_ffn_hidden_size=intermediate,
            num_moe_experts=e,moe_router_topk=8,moe_grouped_gemm=True,
            moe_token_dispatcher_type='allgather',moe_permute_fusion=False,
            gated_linear_unit=True,activation_func=F.silu,add_bias_linear=False,
            params_dtype=torch.bfloat16,bf16=True,perform_initialization=False,
            use_cpu_initialization=False,hidden_dropout=0.,attention_dropout=0.)
        self.mlp=GroupedMLP(e,config).eval();self.num_experts=e
        self.dispatcher=MoEAllGatherTokenDispatcher(e,list(range(e)),config)
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
        valid=weights>0
        routing_map=torch.zeros((len(flat),self.num_experts),device=flat.device,dtype=torch.bool)
        routing_map.scatter_(1,selected,valid)
        dense_probs=torch.zeros((len(flat),self.num_experts),device=flat.device,dtype=weights.dtype)
        dense_probs.scatter_(1,selected,weights)
        permuted,tokens_per_expert,permuted_probs=self.dispatcher.token_permutation(
            flat,dense_probs,routing_map
        )
        reverse=self.dispatcher.reversed_local_input_permutation_mapping
        ones=torch.ones_like(permuted_probs,dtype=flat.dtype)
        values,_=self.mlp(permuted,tokens_per_expert,ones)
        values=values*permuted_probs.to(values.dtype).unsqueeze(-1)

        # MCore permutation orders rows by ascending expert id, then token id.
        # Build only compact integer locations [tokens, topk]. Accumulate each
        # expert slice in ascending expert order to preserve the established
        # BF16 addition order, without allocating [tokens, topk, hidden].
        width=selected.shape[1]
        ranks=selected.argsort(dim=-1)
        sorted_valid=valid.gather(1,ranks)
        # reverse contains the original token index for every expert-major row.
        # Recover its original top-k slot without a [tokens, experts] location table.
        expert_ids=torch.arange(self.num_experts,device=flat.device).repeat_interleave(
            routing_map.sum(dim=0)
        )
        permuted_ranks=(selected[reverse]==expert_ids.unsqueeze(1)).to(torch.int64).argmax(dim=1)
        locations=torch.full((len(flat),width),-1,device=flat.device,dtype=torch.long)
        locations[reverse,permuted_ranks]=torch.arange(len(reverse),device=flat.device)
        output=torch.zeros_like(flat)
        for rank in range(width):
            pos=locations.gather(1,ranks[:,rank:rank+1]).squeeze(1)
            keep=sorted_valid[:,rank]
            safe_pos=pos.clamp_min(0)
            contribution=values[safe_pos].masked_fill(~keep.unsqueeze(1),0)
            output=output+contribution
        return output

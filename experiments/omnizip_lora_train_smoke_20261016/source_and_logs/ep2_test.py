import os, torch, torch.nn.functional as F
from megatron.core.transformer.moe.experts import GroupedMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core import parallel_state
torch.distributed.init_process_group("nccl")
rank=int(os.environ["RANK"]); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,pipeline_model_parallel_size=1,expert_model_parallel_size=2)
E=8; ep=2; nloc=E//ep
cfg=TransformerConfig(num_layers=1,hidden_size=16,num_attention_heads=1,ffn_hidden_size=32,moe_ffn_hidden_size=32,
    num_moe_experts=E,moe_router_topk=2,moe_grouped_gemm=True,gated_linear_unit=True,activation_func=F.silu,
    add_bias_linear=False,params_dtype=torch.bfloat16,bf16=True,perform_initialization=False,use_cpu_initialization=False,
    expert_model_parallel_size=ep)
mlp=GroupedMLP(nloc,cfg).train()
epr=parallel_state.get_expert_model_parallel_rank(); epw=parallel_state.get_expert_model_parallel_world_size()
off=epr*nloc
print(f"[rank{rank}] ep_rank={epr} ep_world={epw} num_local={mlp.num_local_experts} global_range=[{off},{off+nloc}) w1={tuple(mlp.weight1.shape)}", flush=True)
assert epw==2 and mlp.num_local_experts==nloc==4
T=4
x=torch.randn(T,16,device="cuda",dtype=torch.bfloat16,requires_grad=True)
cnt=torch.tensor([1,1,1,1],device="cpu",dtype=torch.long)
with torch.no_grad(): mlp.weight1.copy_(torch.randn_like(mlp.weight1)*0.1); mlp.weight2.copy_(torch.randn_like(mlp.weight2)*0.1)
probs=torch.ones(T,device="cuda",dtype=torch.bfloat16)
out,_=mlp(x,cnt,probs); loss=out.float().sum(); loss.backward()
print(f"[rank{rank}] loss={float(loss):.3f} finite={torch.isfinite(loss).item()} w1grad={float(mlp.weight1.grad.abs().sum())>0}", flush=True)
assert torch.isfinite(loss) and float(mlp.weight1.grad.abs().sum())>0
torch.distributed.barrier()
if rank==0: print("EP2_OK", flush=True)

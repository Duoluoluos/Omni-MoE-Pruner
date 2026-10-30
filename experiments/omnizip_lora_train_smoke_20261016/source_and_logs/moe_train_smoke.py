import torch, torch.nn.functional as F, os, tempfile
from torch import nn
from megatron.core.transformer.moe.experts import GroupedMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core import parallel_state
if not torch.distributed.is_initialized():
    fd,p=tempfile.mkstemp();os.close(fd);os.unlink(p); torch.distributed.init_process_group("nccl",init_method="file://"+p,world_size=1,rank=0)
if not parallel_state.model_parallel_is_initialized():
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,pipeline_model_parallel_size=1,expert_model_parallel_size=1)
torch.manual_seed(0)
dev="cuda"; E,H,I=8,16,32
cfg=TransformerConfig(num_layers=1,hidden_size=H,num_attention_heads=1,ffn_hidden_size=I,moe_ffn_hidden_size=I,num_moe_experts=E,moe_router_topk=2,moe_grouped_gemm=True,gated_linear_unit=True,activation_func=F.silu,add_bias_linear=False,params_dtype=torch.bfloat16,bf16=True,perform_initialization=False,use_cpu_initialization=False,hidden_dropout=0.,attention_dropout=0.)
mlp=GroupedMLP(E,cfg).train()
with torch.no_grad():  # deterministic sane weights (P3 copies real weights; here synth)
    mlp.weight1.copy_(torch.randn_like(mlp.weight1)*0.1)
    mlp.weight2.copy_(torch.randn_like(mlp.weight2)*0.1)
gate=nn.Linear(H,E,bias=False).to(dev).to(torch.bfloat16)
down=nn.Linear(H,5,bias=False).to(dev).float()
opt=torch.optim.SGD(list(mlp.parameters())+list(gate.parameters())+list(down.parameters()),lr=0.1)

def route_and_run(x, topk):
    T,Hd=x.shape
    logits=gate(x)
    w,sidx=torch.softmax(logits.float(),dim=-1).topk(topk,dim=-1)
    w=w/(w.sum(-1,keepdim=True).clamp_min(1e-20)).to(x.dtype)
    valid=w.reshape(-1)>0
    pairs=torch.arange(sidx.numel(),device=dev)[valid]
    experts=sidx.reshape(-1)[valid]
    tok=torch.arange(T,device=dev).repeat_interleave(topk)[valid]
    order=experts.argsort(stable=True)
    counts=torch.bincount(experts[order],minlength=E).cpu().to(torch.long)
    perm=x[tok[order]]; probs=w.reshape(-1)[valid][order]
    out,_=mlp(perm,counts,probs)
    contrib=torch.zeros((sidx.numel(),Hd),device=dev,dtype=x.dtype)
    contrib[pairs[order]]=out
    contrib=contrib.view(T,topk,Hd)
    ranks=sidx.argsort(dim=-1); res=torch.zeros_like(x); rows=torch.arange(T,device=dev)
    for k in range(topk): res=res+contrib[rows,ranks[:,k]]
    return res, int(valid.sum())

x=torch.randn(4,H,device=dev,dtype=torch.bfloat16,requires_grad=True)
for topk,tag in [(8,"native_top8"),(2,"omnidial_budget2")]:
    opt.zero_grad()
    res,nrows=route_and_run(x,topk)
    loss=down(res.float()).log_softmax(-1).sum()
    loss.backward()
    gg=float(gate.weight.grad.abs().sum()); gw=float(mlp.weight1.grad.abs().sum()); gx=float(x.grad.abs().sum())
    opt.step()
    print(f"[{tag}] rows={nrows} loss={float(loss):.3f} finite={torch.isfinite(loss).item()} grad_gate={gg>0} grad_w1={gw>0} grad_x={gx>0}", flush=True)
    assert torch.isfinite(loss) and gg>0 and gw>0 and gx>0
print("SMOKE_OK", flush=True)

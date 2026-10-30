#!/usr/bin/env python3
"""Paired 2x2 OmniZip x OmniDial experiment on Qwen3-Omni with MCore GroupedMLP."""
from __future__ import annotations
import argparse, hashlib, json, os, random, socket, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "qwen-omni-utils/src"), str(ROOT / "scripts")]
from modeling_qwen3_omni_moe_stack import Qwen3OmniMoeForConditionalGeneration
from transformers import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
from qwen_mcore_controller import MCoreQwenController
import run_mb_new2_cgskip_qwen3omni as old

MODEL = ROOT / "models/Qwen3-Omni-30B-A3B-Instruct"
DATA = ROOT / "eval_data/OmniVideoBench/data.json"
INPUTS = ROOT / "experiments/omnidial_layer_followup_20260930/qwen_diagnostic32_v1/inputs.json"
INDICES = [32,41,71,88,94,121,200,208,211,288,295,297,315,321,333,344,389,424,515,539,546,585,641,672,678,699,769,859,870,874,882,916]
ARMS = {
    "native": (False, "native"),
    "omnizip": (True, "native"),
    "omnidial": (False, "cgcc"),
    "stack": (True, "cgcc"),
}
SEED = 20261015
TAU_REDUCE = 0.08694529702904027
OMNIZIP_CONFIG = {"rho_audio": 0.3, "rho_video": 0.6, "g": 3, "contextual_ratio": 0.05}


def atomic(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def sha(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def js(a,b):
    la=a.double().log_softmax(-1); lb=b.double().log_softmax(-1)
    lm=torch.logaddexp(la,lb)-np.log(2.)
    return float(.5*((la.exp()*(la-lm)).sum()+(lb.exp()*(lb-lm)).sum()))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--limit",type=int,default=2); args=ap.parse_args()
    out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True); (out/"predictions").mkdir(exist_ok=True)
    records=json.loads(INPUTS.read_text()); gold=json.loads(DATA.read_text())
    indices=INDICES[:args.limit]
    protocol={"schema":"omnizip-omnidial-2x2-v1","model":str(MODEL),"dataset":str(DATA),"dataset_sha256":sha(DATA),"indices":indices,"seed":SEED,"dtype":"bfloat16","batch_size":1,"max_new_tokens":4,"backend":"megatron-core GroupedMLP expert backend; HF multimodal/attention/KV cache","omnizip":OMNIZIP_CONFIG,"omnidial":{"scope":[12,32],"kind":"cgcc","tau_reduce":TAU_REDUCE,"source":"frozen 2026-09-29 input-only calibration"},"arms":{k:{"omnizip":v[0],"omnidial":v[1]=="cgcc"} for k,v in ARMS.items()},"timing":"CUDA-synchronized generate; prepared inputs excluded; randomized within-sample arm order","primary":["paired accuracy","gold option logp","latency","retained tokens","MCore expert rows","peak allocated memory"],"interaction":{"additive":"y11-y10-y01+y00","multiplicative_efficiency":"observed stack work/latency ratio versus product of individual ratios"}}
    atomic(out/"protocol.json",protocol)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    progress=lambda stage,**kw: atomic(out/"progress.json",{"stage":stage,"time":time.time(),"pid":os.getpid(),"host":socket.gethostname(),**kw})
    progress("loading")
    processor=Qwen3OmniMoeProcessor.from_pretrained(str(MODEL),use_fast=False)
    model=Qwen3OmniMoeForConditionalGeneration.from_pretrained(str(MODEL),torch_dtype=torch.bfloat16,device_map="cuda",attn_implementation="flash_attention_2").eval(); model.disable_talker()
    model.thinker.omnizip_config=dict(OMNIZIP_CONFIG); model.thinker.enable_omnizip=False
    progress("converting_mcore")
    ctl=MCoreQwenController(model,scope=(12,32),tau_reduce=TAU_REDUCE,progress=lambda layer: progress("converting_mcore",layer=layer))
    option_ids={s:processor.tokenizer.encode(s,add_special_tokens=False)[0] for s in "ABCD"}
    import qwen_mcore_experts, qwen_mcore_controller
    env={"python":sys.executable,"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),"model_source":str(Path(sys.modules[Qwen3OmniMoeForConditionalGeneration.__module__].__file__)),"model_source_sha256":sha(Path(sys.modules[Qwen3OmniMoeForConditionalGeneration.__module__].__file__)),"runner_sha256":sha(Path(__file__)),"mcore_experts_source":str(Path(qwen_mcore_experts.__file__).resolve()),"mcore_controller_source":str(Path(qwen_mcore_controller.__file__).resolve()),"mcore":__import__("megatron.core").core.__version__ if hasattr(__import__("megatron.core").core,"__version__") else "installed"}
    atomic(out/"environment.json",env)

    def prepare(idx):
        rec=records[idx]; conv=old.build_conversation(ROOT/"eval_data/OmniVideoBench"/rec["video"],rec)
        text=processor.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
        audios,images,videos=process_mm_info(conv,use_audio_in_video=True)
        inputs=processor(text=text,audio=audios,images=images,videos=videos,return_tensors="pt",padding=True,use_audio_in_video=True)
        return inputs.to(model.device).to(model.dtype)

    def run_arm(inputs,name):
        use_zip,policy=ARMS[name]; model.thinker.enable_omnizip=use_zip; model.thinker._stack_last_omnizip_mask=None
        for b in ctl.blocks: b.grouped_experts.reset_accounting()
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        with ctl.arm(policy): value=old.generate_once(model,processor,inputs,4)
        torch.cuda.synchronize()
        logits=value["first_logits"].reshape(-1).float(); lp=logits.log_softmax(-1)
        mask=getattr(model.thinker,"_stack_last_omnizip_mask",None)
        rows=sum(b.grouped_experts.executed_rows for b in ctl.blocks)
        return {"answer":value["answer"],"output_text":value["output_text"],"latency_s":value["end_to_end_latency_ms"]/1000,"option_logp":{s:float(lp[i]) for s,i in option_ids.items()},"retained_tokens":int(mask.sum()) if mask is not None else int(inputs["input_ids"].numel()),"input_tokens":int(inputs["input_ids"].numel()),"expert_rows":int(rows),"peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30},logits

    # Warm all paths once on the first sample; not measured.
    warm=prepare(indices[0])
    for name in ARMS: progress("warmup",arm=name); run_arm(warm,name)
    del warm; torch.cuda.empty_cache()
    rows=[]
    for pos,idx in enumerate(indices):
        progress("prepare",sample_index=idx,completed=pos,total=len(indices)); inputs=prepare(idx)
        order=list(ARMS); random.Random(SEED+idx).shuffle(order); arm_rows={}; logits={}
        for name in order:
            progress("generate",sample_index=idx,arm=name,completed=pos,total=len(indices))
            arm_rows[name],logits[name]=run_arm(inputs,name)
        for name in ARMS: arm_rows[name]["js_to_native"]=js(logits["native"],logits[name])
        correct=str(gold[idx]["correct_option"]).upper()
        row={"sample_index":idx,"video":records[idx]["video"],"correct_option":correct,"arm_order":order,"arms":arm_rows,"complete":True}
        atomic(out/"predictions"/f"{idx:04d}.json",row); rows.append(row)
        with (out/"raw_results.jsonl").open("a") as f: f.write(json.dumps(row,allow_nan=False)+"\n"); f.flush(); os.fsync(f.fileno())
        del inputs,logits; torch.cuda.empty_cache()
    atomic(out/"inference_complete.json",{"n":len(rows),"time":time.time(),"raw_sha256":sha(out/"raw_results.jsonl")})
    progress("complete",completed=len(rows),total=len(indices))

if __name__=="__main__":
    try: main()
    except Exception:
        if "--out" in sys.argv:
            out=Path(sys.argv[sys.argv.index("--out")+1]); out.mkdir(parents=True,exist_ok=True); atomic(out/"failure.json",{"time":time.time(),"traceback":traceback.format_exc()})
        raise

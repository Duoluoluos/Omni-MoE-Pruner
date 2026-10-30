#!/usr/bin/env python3
"""Fresh paired comparison; outcomes never reach calibration or scheduling.

Stages are resumable. Shared run directories retain source snapshots, input
identities, sensitivity observations, frozen arms, predictions, and work ledgers.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import time
import traceback

import numpy as np
import torch
from todo3_baselines import alloc_layer_dp, aset_budget, self_test
from todo3_backend import Backend, ROOT, input_digest, tree_to

SEED = 20260921
EVIDENCE = ROOT / "research_paper/iclr2027/evidence"
PROTOCOL = EVIDENCE / "todo3_alloc_aset_protocol.json"


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")
    os.replace(tmp,path)


def append(path,value):
    with path.open("a") as f:
        f.write(json.dumps(value,ensure_ascii=False)+"\n")
        f.flush()
        os.fsync(f.fileno())


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def progress(out,stage,**kwargs):
    row = dict(stage=stage,time=time.time(),pid=os.getpid(),host=socket.gethostname(),result_status="planned",**kwargs)
    write_json(out/"progress.json",row)
    write_json(out/f"progress_{os.getpid()}.json",row)
    print(json.dumps(row),flush=True)


def load_records():
    # Evaluation answers/categories do not survive this boundary. Analysis joins
    # gold only after prediction files have been closed.
    raw = json.loads((ROOT/"eval_data/OmniVideoBench/data.json").read_text())
    return [{"sample_index":i,**{k:r[k] for k in ("video","question","options")}} for i,r in enumerate(raw)]


def freeze_protocol():
    if PROTOCOL.exists():
        return json.loads(PROTOCOL.read_text())
    records = load_records()
    pm1 = read_rows(EVIDENCE/"runs/pm1_workload_decomposition_worker4193170_20260828/raw_results.jsonl")
    unique = {}
    for row in pm1:
        unique.setdefault(row["video"],(row["sample_index"],row["input_sequence_length"]))
    ordered = sorted(unique,key=lambda v:(unique[v][1],v))
    rng = np.random.default_rng(SEED)
    selected = []
    for band in np.array_split(np.asarray(ordered),4):
        selected.extend(unique[str(v)][0] for v in rng.choice(band,2,replace=False))
    protocol = dict(schema_version="todo3-direct-v1",result_status="planned",seed=SEED,
        benchmark="OmniVideoBench",evaluation_indices=list(range(len(records))),
        calibration_indices=selected,calibration_videos=[records[i]["video"] for i in selected],
        calibration_access="8 input-only unique videos, 2 per archived input-length quartile; shared by all policies; transductive overlap with full evaluation is disclosed; no answers/categories or quality-based parameter search",
        generation=dict(greedy=True,max_new_tokens=4,batch_size=1),
        measurement="CUDA-synchronized model.generate including modality encoders, routing and decode; prepared tensors reused; seeded arm-order randomization; 5 separate warm-up inputs excluded entirely; accounting uses independent output-matched passes",
        calibration_objective="Alloc-L: next-token perplexity of final input question/options/instruction text, conditioned on multimodal context; no reference answer. Candidate expert-count schedules follow paper Algorithm 1 with deeper eligible layers at 1. Native outside scope.",
        qwen=dict(scope=[12,32],cgcc=dict(base=8,reduce=4,boost=8,reduce_percentile=60),
                  alloc_k_original=8,full_budget_points="6 per layer and input-calibrated CG global assigned budget",matched_scope="same [12,32) scope; exact-budget DP control and ASET calibrated to CG assigned count"),
        unimoe=dict(scope=[0,28],cgcc=dict(base=2,boost=3,skip_percentile=15,hard_percentile=15,null_percentile=8),
                   historical_config_caveat="new explicit reference; historical exact Uni-MoE settings are unconfirmed",
                   alloc_k_original=4,alloc_adaptation="fixed top-4 slot reference (including possible null selection), original GRIN expert weighting/shared path retained; top-4 is not native top-p",
                   aset_adaptation="replace fixed native n_a with each token's native top-p slot count, preserve its native cap; null and padded expert rows reported separately",
                   full_budget_points="2 per layer and input-calibrated CG assigned count"),
        aset_parameter_selection="gamma=2; theta_min=max(0,theta_max-0.2); theta_cand=theta_max; deterministic 65-point theta_max grid [0.2,1], closest input-calibration assigned count, tie prefers smaller theta_max; no task scores",
        sensitivity_normalization="raw perplexities from printed Algorithm 1; subtract row minimum (argmin invariant); no unspecified scaling",
        alloc_t="K_base=1; use K_orig candidates as Sec 3.3 operational description; globally retain (K_l-1)*T extra probabilities; deterministic prefix-preserving ties",
        uncertainty="10000 paired source-video-cluster bootstrap draws; report accuracy, native flips, latency, real routed rows including padding, selected real/null slots, and achieved matching error",
        sources=dict(alloc="https://aclanthology.org/2026.acl-long.437.pdf",aset="https://aclanthology.org/2026.findings-acl.2140.pdf",aset_repository_commit="22fe5db77cb148072914179fb95d1ab28b4e3554",implementation="independent equation ports; ASET public repository contains no implementation at recorded commit"))
    write_json(PROTOCOL,protocol)
    return protocol


def source_record(out,backend):
    import transformers
    import importlib.metadata
    paths = [Path(__file__),ROOT/"scripts/todo3_backend.py",ROOT/"scripts/todo3_baselines.py",
             ROOT/"scripts/run_mb_new2_cgskip_qwen3omni.py",ROOT/"scripts/run_mb_new2_cgskip_unimoe2.py",PROTOCOL,
             backend.model_path/"config.json",ROOT/"research_paper/iclr2027/scripts/analyze_todo3_direct_comparison.py"]
    import inspect
    paths.extend([Path(inspect.getfile(backend.language.__class__)),Path(inspect.getfile(backend.processor.__class__))])
    paths.extend(sorted(backend.model_path.glob("*.index.json")))
    sources = {str(p):sha(p) for p in paths}
    existing = out/"environment.json"
    if existing.exists():
        old = json.loads(existing.read_text())
        if old["sources"] != sources:
            raise RuntimeError("Source files changed since run started; use a new run directory")
        assert old["torch"]==torch.__version__ and old["transformers"]==transformers.__version__
    else:
        write_json(existing,dict(backbone=backend.name,host=socket.gethostname(),torch=torch.__version__,
                                  transformers=transformers.__version__,cuda=torch.version.cuda,
                                  visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                                  gpu=torch.cuda.get_device_name(),sources=sources,protocol_sha=sha(PROTOCOL)))
        for i,p in enumerate(paths):
            (out/"sources").mkdir(exist_ok=True)
            (out/"sources"/f"{i:02d}_{p.name}").write_bytes(p.read_bytes())
    packages={}
    for name in ["torch","transformers","flash-attn","deepspeed","decord","librosa","moviepy","soundfile"]:
        try:
            packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name]=None
    write_json(out/f"worker_environment_{os.getpid()}.json",dict(pid=os.getpid(),host=socket.gethostname(),
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),packages=packages,sources=sources,
        gpu_inventory=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,name,driver_version,memory.total","--format=csv,noheader"],text=True),
        model_weights=[dict(name=p.name,size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(backend.model_path.glob("*.safetensors"))]))


def js(a,b):
    a,b=a.double().log_softmax(-1),b.double().log_softmax(-1)
    m=((a.exp()+b.exp())/2).clamp_min(1e-300).log()
    return max(0.,float(((a.exp()*(a-m)).sum()+(b.exp()*(b-m)).sum())/2))


def smoke(backend,out,records,protocol):
    reports=[]
    for index in [protocol["calibration_indices"][0],protocol["calibration_indices"][-1]]:
        progress(out,"smoke_prepare",sample_index=index)
        inputs,counts=backend.prepare(records[index])
        digest=input_digest(inputs)
        with backend.controller.arm({"kind":"native"}),backend.audit_embeddings(inputs) as embedding:
            native=backend.generate(inputs,1)
        with backend.controller.arm({"kind":"identity"}):
            identity=backend.generate(inputs,1)
        divergence=js(native["first_logits"],identity["first_logits"])
        assert divergence < 1e-6 and native["output_text"]==identity["output_text"],divergence
        # Validate all method decision paths and true dispatch, not only native bypass.
        maximum=8 if backend.name=="qwen" else 4
        tests=[dict(kind="fixed",budget=2),dict(kind="alloc",schedule=[2]*len(backend.blocks)),
               dict(kind="aset",theta_min=.5,theta_max=.7,gamma=2)]
        work={}
        times={"native":native["end_to_end_latency_ms"],"identity":identity["end_to_end_latency_ms"]}
        for policy in tests:
            with backend.controller.arm(policy):
                res=backend.generate(inputs,1)
                with backend.controller.profile() as ledger:
                    prof=backend.generate(inputs,1)
            assert res["output_text"]==prof["output_text"] and js(res["first_logits"],prof["first_logits"]) < 1e-6
            work[policy["kind"]]={k:dict(v) for k,v in ledger.items()}
            times[policy["kind"]]=res["end_to_end_latency_ms"]
        cache=backend.cache_language(inputs)
        with backend.controller.capture_pool() as pool:
            with backend.controller.arm(dict(kind="native")):
                nll,n=backend.input_nll(cache)
        assert len(pool)==len(backend.blocks) and math.isfinite(nll)
        assert input_digest(inputs)==digest
        reports.append(dict(sample_index=index,counts=counts,input_sha=digest,identity_js=divergence,
                            input_ppl=math.exp(nll/n),work=work,embedding_rows=embedding,probe_latency_ms=times))
        del inputs,cache,pool
    write_json(out/"preflight.json",dict(passed=True,probes=reports))
    progress(out,"smoke_passed",probes=len(reports))


def prepare_calibration(backend,out,records,protocol):
    for index in protocol["calibration_indices"]:
        cache_path=out/"calibration"/f"input_{index}.pt"
        pool_path=out/"calibration"/f"pool_{index}.pt"
        if cache_path.exists() and pool_path.exists():
            continue
        progress(out,"calibration_prepare",sample_index=index)
        inputs,counts=backend.prepare(records[index])
        digest=input_digest(inputs)
        cache=backend.cache_language(inputs)
        with backend.controller.capture_pool() as pool:
            with backend.controller.arm(dict(kind="native")):
                nll,n=backend.input_nll(cache)
        assert len(pool)==len(backend.blocks)
        cache_path.parent.mkdir(exist_ok=True)
        torch.save(cache,cache_path)
        torch.save(dict(pool),pool_path)
        write_json(cache_path.with_suffix(".json"),dict(sample_index=index,video=records[index]["video"],counts=counts,
            input_sha=digest,input_nll=nll,text_targets=n,cache_sha=sha(cache_path),pool_sha=sha(pool_path)))
        del inputs,cache,pool


def weighted_quantile(values,weights,q):
    order=np.argsort(values,kind="stable")
    values,weights=values[order],weights[order]
    return float(np.interp(q*np.sum(weights),np.cumsum(weights)-weights/2,values))


def freeze_gates(backend,out,protocol):
    path=out/"gates.json"
    if path.exists():
        backend.controller.thresholds=json.loads(path.read_text())["thresholds"]
        return json.loads(path.read_text())
    chunks=defaultdict(list)
    for i in protocol["calibration_indices"]:
        pools=torch.load(out/"calibration"/f"pool_{i}.pt",weights_only=False)
        for layer,rows in pools.items():
            chunks[layer].extend(rows)
    scope=protocol[backend.name]["scope"]
    p=torch.cat([r["probs"] for l,rs in chunks.items() if scope[0]<=l<scope[1] for r in rs])
    w=torch.cat([torch.full((len(r["probs"]),),r["weight"]) for l,rs in chunks.items() if scope[0]<=l<scope[1] for r in rs]).numpy()
    if backend.name=="qwen":
        signals=p.topk(4,dim=-1).values.sum(-1).numpy()
        thresholds=dict(tau_reduce=weighted_quantile(signals,w,.4))
    else:
        real=p[:,:4].max(-1).values.numpy()
        null=p[:,4].numpy()
        thresholds=dict(tau_skip=weighted_quantile(real,w,.85),tau_hard=weighted_quantile(real,w,.15),
                        tau_null=weighted_quantile(null,w,.92))
    backend.controller.thresholds=thresholds
    means=[]
    for layer,rows in chunks.items():
        num=den=0
        for r in rows:
            with backend.controller.arm(dict(kind="cgcc",scope=scope)):
                counts=backend.controller.decide(r["probs"],layer,r["native_budget"])
            num+=float(counts.sum())*r["weight"]
            den+=len(counts)*r["weight"]
        means.append((layer,num/den))
    means=dict(means)
    targets={"full":sum(means.values())/len(means),"scope":sum(means[l] for l in range(*scope))/(scope[1]-scope[0])}
    aset={}
    scopes={"full":[0,len(backend.blocks)]}
    if backend.name=="qwen":
        scopes["scope"]=scope
    for scope_name,(lo,hi) in scopes.items():
        pp=torch.cat([r["probs"] for l,rs in chunks.items() if lo<=l<hi for r in rs]).cuda()
        nn=torch.cat([r["native_budget"] for l,rs in chunks.items() if lo<=l<hi for r in rs]).cuda()
        ww=torch.cat([torch.full((len(r["probs"]),),r["weight"]) for l,rs in chunks.items() if lo<=l<hi for r in rs]).cuda()
        curve=[]
        for high in np.linspace(.2,1.,65):
            low=max(0.,float(high)-.2)
            mean=float((aset_budget(pp,nn,low,float(high),2).float()*ww).sum()/ww.sum())
            curve.append(dict(theta_min=low,theta_max=float(high),gamma=2.,assigned_mean=mean))
        wanted={"cgmatch":targets[scope_name]}
        if scope_name=="full":
            wanted["neighbor"]=6. if backend.name=="qwen" else 2.
        for tag,target in wanted.items():
            best=min(curve,key=lambda c:abs(c["assigned_mean"]-target))
            aset[f"{scope_name}_{tag}"]=dict(**best,target=target,scope=[lo,hi])
        write_json(out/f"aset_curve_{scope_name}.json",curve)
        del pp,nn,ww
    result=dict(thresholds=thresholds,layer_calibration_assigned_means=means,targets=targets,aset=aset)
    write_json(path,result)
    return result


def sensitivity(backend,out,protocol):
    layers=len(backend.blocks)
    maximum=8 if backend.name=="qwen" else 4
    scopes={"full":[0,layers]}
    if backend.name=="qwen":
        scopes["scope"]=protocol["qwen"]["scope"]
    scopes=protocol.get("baseline_scopes",scopes)
    logpath=out/"sensitivity_raw.jsonl"
    observed={(r["scope"],r["sample_index"],r["layer"],r["k"]):r for r in read_rows(logpath)}
    done=set(observed)
    for index in protocol["calibration_indices"]:
        cache=torch.load(out/"calibration"/f"input_{index}.pt",weights_only=False)
        cache=tree_to(cache,backend.model.device)
        for scope_name,(lo,hi) in scopes.items():
            schedule=[maximum]*layers
            for layer in range(hi-1,lo-1,-1):
                for k in range(maximum,0,-1):
                    schedule[layer]=k
                    key=(scope_name,index,layer,k)
                    if key in done:
                        continue
                    started=time.time()
                    carry=(scope_name,index,layer+1,1)
                    reused=None
                    if k==maximum and carry in observed:
                        # Algorithm 1 carries this exactly identical configuration.
                        old=observed[carry]
                        nll,n=old["nll_sum"],old["text_targets"]
                        reused=list(carry)
                    else:
                        with backend.controller.arm(dict(kind="schedule",scope=[lo,hi],schedule=schedule.copy())):
                            nll,n=backend.input_nll(cache)
                    if not math.isfinite(nll):
                        raise RuntimeError("Nonfinite calibration loss")
                    row=dict(scope=scope_name,sample_index=index,layer=layer,k=k,nll_sum=nll,
                             text_targets=n,seconds=time.time()-started,reused_from=reused)
                    append(logpath,row)
                    observed[key]=row
                    done.add(key)
                    progress(out,"alloc_sensitivity",sample_index=index,scope=scope_name,layer=layer,k=k,completed=len(done))
        del cache
    matrices={}
    rows=read_rows(logpath)
    for name,(lo,hi) in scopes.items():
        matrix=[]
        for layer in range(lo,hi):
            values=[]
            for k in range(1,maximum+1):
                rr=[r for r in rows if r["scope"]==name and r["layer"]==layer and r["k"]==k]
                assert len(rr)==len(protocol["calibration_indices"])
                values.append(math.exp(sum(r["nll_sum"] for r in rr)/sum(r["text_targets"] for r in rr)))
            matrix.append(values)
        matrices[name]=matrix
    write_json(out/"sensitivity.json",matrices)
    return matrices


def freeze_arms(backend,out,protocol,gates,matrices):
    layers=len(backend.blocks)
    scope=protocol[backend.name]["scope"]
    arms={"native":dict(kind="native"),"cgcc":dict(kind="cgcc",scope=scope)}
    for k in ([4,6,8] if backend.name=="qwen" else [1,2,4]):
        arms[f"fixed{k}"]=dict(kind="fixed",scope=scope,budget=k)
    for tag,aset in gates["aset"].items():
        arms[f"aset_{tag}"]=dict(kind="aset",**aset)
        lo,hi=aset["scope"]
        B=round(aset["target"]*(hi-lo))
        B=min((hi-lo)*(8 if backend.name=="qwen" else 4),max(hi-lo,B))
        scope_name=tag.split("_")[0]
        exact=scope_name=="scope"
        schedule=[8 if backend.name=="qwen" else 4]*layers
        schedule[lo:hi]=alloc_layer_dp(matrices[scope_name],B,exact=exact)
        arms[f"alloc_{tag}"]=dict(kind="alloc",scope=[lo,hi],schedule=schedule,
                                     global_assigned_budget=B,used_assigned_budget=sum(schedule[lo:hi]),
                                     exact_budget_control=exact)
    write_json(out/"arms.json",arms)
    return arms


def evaluate(backend,out,records,protocol,arms,limit=None):
    assert json.loads((out/"preflight.json").read_text())["passed"]
    predictions=out/"predictions"
    claims=out/"claims"
    predictions.mkdir(exist_ok=True); claims.mkdir(exist_ok=True)
    done={int(p.stem) for p in predictions.glob("*.json")}
    indices=protocol["evaluation_indices"]
    if limit is not None:
        indices=indices[:limit]
    # Warm-up uses calibration inputs; never contributes measured observations.
    for wi,index in enumerate(protocol["calibration_indices"][:5]):
        inputs,_=backend.prepare(records[index])
        for name,policy in arms.items():
            with backend.controller.arm(policy):
                backend.generate(inputs,4)
        del inputs
        progress(out,"evaluation_warmup",completed=wi+1,expected=5)
    write_json(out/f"warmup_{os.getpid()}.json",dict(passed=True,indices=protocol["calibration_indices"][:5],
               arms_sha=sha(out/"arms.json"),worker_environment_sha=sha(out/f"worker_environment_{os.getpid()}.json")))
    for index in indices:
        if index in done:
            continue
        destination=predictions/f"{index:04d}.json"
        if destination.exists():
            continue
        claim=claims/f"{index:04d}.json"
        try:
            with claim.open("x") as f:
                json.dump(dict(pid=os.getpid(),host=socket.gethostname()),f)
        except FileExistsError:
            try:
                owner=json.loads(claim.read_text())
            except (json.JSONDecodeError,FileNotFoundError):
                # Another worker is creating/removing this short-lived claim.
                continue
            assert owner["host"]==socket.gethostname(),"cross-host claims require explicit recovery"
            try:
                os.kill(owner["pid"],0)
                continue
            except ProcessLookupError:
                claim.unlink()
                with claim.open("x") as f:
                    json.dump(dict(pid=os.getpid(),host=socket.gethostname()),f)
        inputs,counts=backend.prepare(records[index])
        digest=input_digest(inputs)
        order=list(arms)
        np.random.default_rng(SEED+index).shuffle(order)
        results={}
        started=time.time()
        for name in order:
            with backend.controller.arm(arms[name]):
                timed=backend.generate(inputs,4)
            results[name]={k:v for k,v in timed.items() if k in ("answer","output_text","end_to_end_latency_ms")}
        # Full separate accounting passes reconcile actual padded/real execution
        # and verify deterministic outputs; no hooks contaminate timed calls.
        for name in order[::-1]:
            with backend.controller.arm(arms[name]):
                with backend.controller.profile() as ledger,backend.audit_embeddings(inputs) as embedding:
                    profiled=backend.generate(inputs,4)
            if results[name]["output_text"] != profiled["output_text"]:
                raise RuntimeError(f"Timed/profile prediction mismatch: {index} {name}")
            results[name]["work"]={k:dict(v) for k,v in ledger.items()}
            results[name]["embedding_rows"]=embedding
        assert input_digest(inputs)==digest
        write_json(destination,dict(sample_index=index,video=records[index]["video"],input_sha=digest,counts=counts,
                           arm_order=order,arms=results,total_seconds=time.time()-started,
                           protocol_sha=sha(PROTOCOL),arms_sha=sha(out/"arms.json"),worker_pid=os.getpid(),
                           worker_environment_sha=sha(out/f"worker_environment_{os.getpid()}.json")))
        claim.unlink()
        done.add(index)
        progress(out,"evaluation",completed=len(done),expected=len(indices),sample_index=index)
        del inputs
    completed=len(list(predictions.glob("*.json")))
    progress(out,"worker_inference_complete",completed=completed,expected=len(indices),analysis_required=True)
    if completed==len(protocol["evaluation_indices"]):
        subprocess.run([str(ROOT/".venv/bin/python"),str(ROOT/"research_paper/iclr2027/scripts/analyze_todo3_direct_comparison.py"),str(out)],check=True)
        progress(out,"analysis_complete",completed=completed,expected=len(indices))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--backbone",choices=["qwen","unimoe"])
    parser.add_argument("--out",type=Path)
    parser.add_argument("--stage",choices=["freeze","smoke","calibrate","evaluate","all"],default="all")
    parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    protocol=freeze_protocol()
    if args.stage=="freeze":
        print(PROTOCOL)
        return
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(8)
    try:
        self_test()
        records=load_records()
        progress(out,"loading",backbone=args.backbone)
        backend=Backend(args.backbone)
        source_record(out,backend)
        if args.stage in ("smoke","all") and not (out/"preflight.json").exists():
            smoke(backend,out,records,protocol)
        if args.stage=="smoke":
            return
        if args.stage in ("calibrate","all"):
            prepare_calibration(backend,out,records,protocol)
            gates=freeze_gates(backend,out,protocol)
            matrices=sensitivity(backend,out,protocol)
            arms=freeze_arms(backend,out,protocol,gates,matrices)
        else:
            gates=json.loads((out/"gates.json").read_text())
            backend.controller.thresholds=gates["thresholds"]
            arms=json.loads((out/"arms.json").read_text())
        if args.stage in ("evaluate","all"):
            evaluate(backend,out,records,protocol,arms,args.limit)
    except Exception:
        append(out/"errors.jsonl",dict(time=time.time(),traceback=traceback.format_exc()))
        progress(out,"error")
        raise


if __name__=="__main__":
    main()

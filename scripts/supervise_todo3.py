#!/usr/bin/env python3
"""Two-H20 queue for the author-authorized TODO 3 comparison.

No automatic methodological repair or package installation. Failures remain
visible; healthy independent work continues. The spare GPU joins Qwen only
after Uni-MoE's complete analysis has passed.
"""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/"research_paper/iclr2027/evidence/runs"
OUT={b:RUNS/f"todo3_{b}_full1000_20260921_v1" for b in ["qwen","unimoe"]}
STATE=RUNS/"todo3_queue_20260921_v1.json"


def launch(backbone,gpu,stage):
    out=OUT[backbone]
    out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="8",MKL_NUM_THREADS="8",
               TOKENIZERS_PARALLELISM="false",PYTHONUNBUFFERED="1",
               TRITON_CACHE_DIR=f"/tmp/todo3_triton_{backbone}_{gpu}")
    Path(env["TRITON_CACHE_DIR"]).mkdir(parents=True,exist_ok=True)
    env["PATH"]=str(ROOT/".runtime/todo3/bin")+os.pathsep+env.get("PATH","")
    interpreter=ROOT/".venv/bin/python" if backbone=="qwen" else Path("/home/tiger/uni_moe_venv/bin/python")
    command=[str(interpreter),str(ROOT/"scripts/run_todo3_direct_comparison.py"),
             "--backbone",backbone,"--out",str(out),"--stage",stage]
    with (out/f"launch_gpu{gpu}.log").open("a") as f:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,gpu=gpu,backbone=backbone,stage=stage,out=str(out),command=command,start=time.time())
    (out/f"launch_{child.pid}.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record),flush=True)
    return child,record


def main():
    assert not STATE.exists(),"Queue already exists; inspect it before a new launch"
    active=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
    assert not active,f"GPU processes already running: {active}"
    children={}
    manifest=dict(supervisor_pid=os.getpid(),result_status="planned",started=time.time(),workers={})
    for b,gpu in [("qwen",0),("unimoe",1)]:
        children[b],manifest["workers"][b]=launch(b,gpu,"all")
    helper_decided=False
    while True:
        for name,child in list(children.items()):
            manifest["workers"][name]["exit_code"]=child.poll()
        if not helper_decided and children["unimoe"].poll() is not None:
            if children["unimoe"].returncode!=0:
                manifest["helper_note"]="Uni-MoE failed; spare-GPU reassignment paused for diagnosis"
                helper_decided=True
            elif children["qwen"].poll() is not None:
                helper_decided=True
            elif (OUT["qwen"]/"arms.json").exists():
                uni=json.loads((OUT["unimoe"]/"summary.json").read_text())
                assert uni["result_status"]=="verified"
                children["qwen_helper"],manifest["workers"]["qwen_helper"]=launch("qwen",1,"evaluate")
                helper_decided=True
        manifest["updated"]=time.time()
        manifest["queue_finished"]=all(p.poll() is not None for p in children.values())
        tmp=STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest,indent=2)+"\n")
        os.replace(tmp,STATE)
        if manifest["queue_finished"]:
            print(json.dumps(manifest),flush=True)
            return
        time.sleep(20)


if __name__=="__main__":
    main()

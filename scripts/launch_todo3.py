#!/usr/bin/env python3
"""Launch a single explicitly assigned TODO 3 process; no worker lifecycle changes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument("--backbone",choices=["qwen","unimoe"],required=True)
p.add_argument("--gpu",choices=["0","1"],required=True)
p.add_argument("--out",type=Path,required=True)
p.add_argument("--stage",default="all",choices=["smoke","calibrate","evaluate","all"])
p.add_argument("--limit",type=int)
args=p.parse_args()
out=args.out.resolve()
out.mkdir(parents=True,exist_ok=True)
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=args.gpu,OMP_NUM_THREADS="8",MKL_NUM_THREADS="8",TOKENIZERS_PARALLELISM="false",
           PYTHONUNBUFFERED="1",TRITON_CACHE_DIR=f"/tmp/todo3_triton_{args.backbone}")
env["PATH"]=str(ROOT/".runtime/todo3/bin")+os.pathsep+env.get("PATH","")
Path(env["TRITON_CACHE_DIR"]).mkdir(parents=True,exist_ok=True)
interpreter=ROOT/".venv/bin/python" if args.backbone=="qwen" else Path("/home/tiger/uni_moe_venv/bin/python")
command=[str(interpreter),str(ROOT/"scripts/run_todo3_direct_comparison.py"),"--backbone",args.backbone,
         "--stage",args.stage,"--out",str(out)]
if args.limit is not None:
    command += ["--limit",str(args.limit)]
with (out/"launch.log").open("a") as log:
    child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=child.pid,gpu=args.gpu,command=command,time=time.time())
(out/f"launch_{child.pid}.json").write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record))

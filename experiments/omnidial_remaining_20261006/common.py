"""Shared outcome-free dataset normalization and durable run bookkeeping."""
import hashlib,json,os,math,time
from pathlib import Path
ROOT=Path('/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner')
BASE=ROOT/'experiments/omnidial_remaining_20261006'
PAPER=ROOT/'research_paper/acl_arr_oct2027'
# The shared mount does not consistently implement flock. All supervisors and
# backfills run on one worker, so their advisory locks belong on its local /tmp.
LOCK_DIR=Path('/tmp/omnidial_remaining_20261006_locks')
def lock_path(name):
 LOCK_DIR.mkdir(parents=True,exist_ok=True)
 return LOCK_DIR/name
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def optional_read(p,fallback=None,attempts=3,delay=.1):
 """Bounded reads for replace-written administrative state, never frozen inputs."""
 for attempt in range(attempts):
  try:return read(p)
  except (OSError,json.JSONDecodeError):
   if attempt+1<attempts:time.sleep(delay)
 return fallback
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(f'.{os.getpid()}.tmp')
 with tmp.open('w') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def append(p,v):
 line=json.dumps(v,allow_nan=False)+'\n'
 with Path(p).open('a') as f:f.write(line);f.flush();os.fsync(f.fileno())
def dataset(bench,labels=False):
 if bench=='omnivideobench':
  path=ROOT/'eval_data/OmniVideoBench/data.json';directory=path.parent
  raw=read(path);data=[dict(video=r['video'],question=r['question'],options=r['options']) for r in raw]
  gold=[str(r['correct_option']).upper() for r in raw] if labels else None
 elif bench=='avut':
  path=ROOT/'eval_data/AVUTBenchmark/AV_Human_filtered_data.json';directory=path.parent;raw=read(path)
  data=[dict(video=Path(r['video_path']).name,question=r['question'],options=[f'{a}. {r["option_"+a]}' for a in 'ABCDEF' if r.get('option_'+a) is not None]) for r in raw]
  gold=[str(r.get('answer','')).strip().upper() for r in raw] if labels else None
 else:
  path=ROOT/'eval_data/WorldSense/worldsense_qa.json';directory=path.parent/'videos';raw=read(path);data=[];gold=[] if labels else None
  for e in raw.values():
   for k in sorted(k for k in e if k.startswith('task')):
    t=e[k]
    if not t.get('candidates'):continue
    data.append(dict(video=str(e['video_id'])+'.mp4',question=t['question'],options=list(t['candidates'])))
    if labels:gold.append(str(t.get('answer','')).strip().upper())
 data=[dict(sample_index=i,**r) for i,r in enumerate(data)]
 return data,directory,path,gold

def load_rows(out,recover_tail=False):
 p=out/'raw_results.jsonl'
 if not p.exists():return {}
 contents=p.read_bytes()
 if contents and not contents.endswith(b'\n'):
  if not recover_tail:raise ValueError('Uncommitted final JSONL line')
  stop=contents.rfind(b'\n')+1
  (out/f'uncommitted_tail_{os.getpid()}.bin').write_bytes(contents[stop:])
  with p.open('r+b') as f:f.truncate(stop);f.flush();os.fsync(f.fileno())
  contents=contents[:stop]
 rows={}
 for line in contents.splitlines():
  r=json.loads(line);i=r['sample_index'];assert i not in rows and r['complete'];rows[i]=r
 return rows

def verify(out):
 manifest=read(out/'manifest.json')
 for file,digest in manifest['files'].items():assert sha(out/file)==digest,file
 p=read(out/'protocol.json');assert sha(p['dataset'])==p['dataset_sha256']
 return p

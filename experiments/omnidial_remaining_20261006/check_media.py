"""Outcome-blind recovery of the decodable AVUT/WorldSense subsets."""
import concurrent.futures,json,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner')
BASE=ROOT/'experiments/omnidial_remaining_20261006'
FF=ROOT/'.runtime/scope_bin/ffmpeg'
for name in ('avut','worldsense'):
 if name=='avut':
  data=json.loads((ROOT/'eval_data/AVUTBenchmark/AV_Human_filtered_data.json').read_text())
  rows=[Path(r['video_path']).name for r in data]; directory=ROOT/'eval_data/AVUTBenchmark'
 else:
  data=json.loads((ROOT/'eval_data/WorldSense/worldsense_qa.json').read_text());rows=[]
  for e in data.values():
   rows += [str(e['video_id'])+'.mp4' for k in sorted(e) if k.startswith('task') and e[k].get('candidates')]
  directory=ROOT/'eval_data/WorldSense/videos'
 def probe(video):
  try:
   r=subprocess.run([str(FF),'-v','error','-threads','1','-i',str(directory/video),'-map','0:v:0','-map','0:a:0','-t','0.1','-f','null','-'],capture_output=True,text=True,timeout=45)
   return video,dict(ok=r.returncode==0,returncode=r.returncode,error=r.stderr[-3000:])
  except subprocess.TimeoutExpired:return video,dict(ok=False,error='header/first-frame probe timed out')
 results={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
  for v,r in pool.map(probe,dict.fromkeys(rows)):
   results[v]=r
   if len(results)%200==0: print(name,len(results),flush=True)
 excluded=[dict(sample_index=i,video=v,**results[v]) for i,v in enumerate(rows) if not results[v]['ok']]
 result=dict(benchmark=name,total_questions=len(rows),expected_decodable=1733 if name=='avut' else 3167,
  included_indices=[i for i,v in enumerate(rows) if results[v]['ok']],excluded=excluded,
  method='ffmpeg 7.0.2 header + 0.1 s AV decode; no labels used; old per-question inclusion file was deleted',videos=results,time=time.time())
 (BASE/(name+'_media_check.json')).write_text(json.dumps(result,indent=2)+'\n')
 print(name,'included',len(result['included_indices']),'excluded',excluded,flush=True)

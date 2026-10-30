import concurrent.futures,subprocess,sys,json
from common import *
# Each video gets an isolated, bounded decoder; no model inference or labels.
code='''import sys, numpy as np
from decord import VideoReader,cpu
v=VideoReader(sys.argv[1],ctx=cpu(0),num_threads=1)
assert len(v)>0
v.get_batch(np.linspace(0,len(v)-1,min(64,len(v))).round().astype(int)).asnumpy()
'''
for name in ['avut','worldsense']:
 records,directory,_,_=dataset(name);media=read(BASE/(name+'_media_check.json'))
 existing={r['video'] for r in media['excluded']}
 videos=list(dict.fromkeys(r['video'] for r in records if r['video'] not in existing))
 def test(v):
  try:
   r=subprocess.run([sys.executable,'-B','-c',code,str(directory/v)],capture_output=True,text=True,timeout=60)
   return v,dict(ok=r.returncode==0,returncode=r.returncode,error=r.stderr[-2500:])
  except subprocess.TimeoutExpired:return v,dict(ok=False,error='decord sampled-frame timeout 60s')
 results={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
  for v,r in pool.map(test,videos):
   results[v]=r
   if len(results)%200==0:print(name,len(results),flush=True)
 excluded={r['sample_index']:r for r in media['excluded']}
 for r in records:
  if r['video'] in results and not results[r['video']]['ok']:
   excluded[r['sample_index']]=dict(sample_index=r['sample_index'],video=r['video'],**results[r['video']])
 if name=='avut':excluded[1495]=dict(sample_index=1495,video=records[1495]['video'],ok=False,error='Retained documented historical exclusion: scripts/cgcc_sweep.py --skip-indices 1495; initial-frame ffmpeg probe alone does not reproduce old failure.')
 media['frame_decode_check']=results
 media['excluded']=[excluded[i] for i in sorted(excluded)];media['included_indices']=[r['sample_index'] for r in records if r['sample_index'] not in excluded]
 media['method']+='; decord 64 evenly spaced frames, including first/last; retain explicitly documented AVUT exclusion 1495'
 write(BASE/(name+'_media_check.json'),media)
 print(name,'included',len(media['included_indices']),'excluded',media['excluded'],flush=True)

#!/usr/bin/env python3
"""Backfill only full verified results; preserve every historical table cell."""
import argparse,csv,fcntl,io,subprocess,time,sys
from common import *

def atomic_text(p,s):
 p=Path(p);tmp=p.with_suffix(f'.{os.getpid()}.tmp')
 with tmp.open('w') as f:f.write(s);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def cell(value,native,accuracy,bold=True,marker=False):
 displayed=round(value,1 if accuracy else 3);delta=100*(displayed/native-1)
 if abs(round(delta,1))==0:delta=0.;color='mtneutral'
 else:color='mtgood' if (delta>0)==accuracy else 'mtbad'
 number=f'{displayed:.1f}' if accuracy else f'{displayed:.3f}'
 change=f'{delta:+.1f}' if delta else '0.0'
 return '\\mtmetric'+('bold' if bold else '')+'{'+number+'}{'+color+'}{'+change+'}'+(r'$^{\S}$' if marker else '')
def new_rows():
 found=[]
 for out in BASE.iterdir():
  if not out.is_dir() or not (out/'summary.json').exists():continue
  s=read(out/'summary.json');p=verify(out)
  if s.get('result_status')!='verified':continue
  assert s['raw_sha256']==sha(out/'raw_results.jsonl') and s['manifest_sha256']==sha(out/'manifest.json')
  found.append((out,p,s))
 return found

def backfill(out):
 # Recompute scoring independently from the just-finished inference process.
 subprocess.run([sys.executable,'-B',str(out/'source/analyze.py'),'--out',str(out)],check=True)
 with lock_path('paper_backfill.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  found=new_rows();table=PAPER/'results/main_inference_table.tex';text=table.read_text();lines=text.splitlines()
  display=PAPER/'results/main_inference_display.csv';old=list(csv.DictReader(display.open()));fieldnames=list(old[0])
  historical_display=[dict(row) for row in old]
  original_lines=list(lines);editable_cells={}
  provenance=[];coverage=[]
  for directory,p,s in found:
   if p['kind']=='terminal':continue
   backbone='Qwen3-Omni Thinker' if p['model']=='qwen' else 'Uni-MoE-2.0-Omni'
   bench={'avut':'AVUT','worldsense':'WorldSense','omnivideobench':'OmniVideoBench'}[p['benchmark']]
   if p['kind']=='transfer':
    prefix=r'\textbf{OmniDial (all)}$^{\ddagger}$' if p['model']=='qwen' else r'\textbf{OmniDial (partial)}$^{\ddagger}$'
    matches=[i for i,line in enumerate(lines) if line.startswith(prefix+' &')];assert len(matches)==1
    i=matches[0];parts=lines[i].split(' & ');pos=3 if bench=='AVUT' else 5
    editable_cells.setdefault(i,set()).update((pos,pos+1))
    native=next(r for r in old if r['backbone']==backbone and r['benchmark']==bench and r['method'].startswith('Native'))
    a=s['arms']['omnidial'];parts[pos]=cell(a['accuracy_percent'],float(native['accuracy_percent']),True,marker=True)
    ending=' \\\\' if pos+1==6 else ''
    parts[pos+1]=cell(a['generation_seconds'],float(native['generation_seconds']),False)+ending
    lines[i]=' & '.join(parts)
    method='OmniDial (all [0,48))' if p['model']=='qwen' else 'OmniDial (partial [7,19))'
    old=[r for r in old if (r['backbone'],r['benchmark'],r['method'])!=(backbone,bench,method)]
    old.append(dict(backbone=backbone,benchmark=bench,method=method,samples=s['n'],accuracy_percent=f"{a['accuracy_percent']:.1f}",generation_seconds=f"{a['generation_seconds']:.3f}",source=str((directory/'summary.json').relative_to(ROOT))))
    coverage.append(f"{('Qwen all-layer' if p['model']=='qwen' else 'Uni-MoE partial-layer')} OmniDial on {bench} ({s['n']:,} evaluated, {s['timed_n']:,} timed)")
   else:
    for key,label in [('alloc','Alloc-MoE'),('aset','ASET')]:
     matches=[i for i,line in enumerate(lines) if line.startswith(label+' (partial layers) &')];assert len(matches)==1
     i=matches[0];parts=lines[i].split(' & ');a=s['arms'][key]
     editable_cells.setdefault(i,set()).update((1,2))
     if i>0:lines[i-1]=lines[i-1].replace('unimoe=None',f'unimoe={key}_partial_20261006')
     parts[1]=cell(a['accuracy_percent'],38.6,True,bold=False,marker=True);parts[2]=cell(a['generation_seconds'],2.986,False,bold=False);lines[i]=' & '.join(parts)
     method=label+' (partial [7,19))'
     old=[r for r in old if (r['backbone'],r['benchmark'],r['method'])!=(backbone,bench,method)]
     old.append(dict(backbone=backbone,benchmark=bench,method=method,samples=s['n'],accuracy_percent=f"{a['accuracy_percent']:.1f}",generation_seconds=f"{a['generation_seconds']:.3f}",source=str((directory/'summary.json').relative_to(ROOT))))
     coverage.append(f"Uni-MoE partial-layer {label} on OmniVideoBench ({s['n']:,} evaluated, {s['timed_n']:,} timed)")
   provenance.append(dict(run=str(directory.relative_to(ROOT)),summary_sha256=sha(directory/'summary.json'),raw_sha256=s['raw_sha256'],tasks=p['task_ids'],scope=p['scope'],backend=p['backend'],n=s['n'],timed_n=s['timed_n'],selectors={name:{metric:f'arms.{name}.{metric}' for metric in ('accuracy_percent','generation_seconds')} for name in s['arms']}))
  # Only the explicitly targeted new cells may change. In particular the
  # historical Qwen 39.68 / 13.151 operating point stays byte-for-byte intact.
  for i,original in enumerate(original_lines):
   if ' & ' not in original:continue
   before=original.split(' & ');after=lines[i].split(' & ')
   assert len(before)==len(after),(i,'table structure changed')
   for j,value in enumerate(before):
    if j not in editable_cells.get(i,set()):assert after[j]==value,(i,j,'historical table cell changed')
  new_methods={'OmniDial (all [0,48))','OmniDial (partial [7,19))','Alloc-MoE (partial [7,19))','ASET (partial [7,19))'}
  replaceable={(('Qwen3-Omni Thinker' if p['model']=='qwen' else 'Uni-MoE-2.0-Omni'),{'avut':'AVUT','worldsense':'WorldSense','omnivideobench':'OmniVideoBench'}[p['benchmark']]) for _,p,_ in found if p['kind']!='terminal'}
  for row in historical_display:
   if row['method'] in new_methods and (row['backbone'],row['benchmark']) in replaceable:continue
   assert row in old,('historical CSV row changed',row)
  atomic_text(table,'\n'.join(lines)+'\n')
  buff=io.StringIO();writer=csv.DictWriter(buff,fieldnames=fieldnames);writer.writeheader();writer.writerows(old);atomic_text(display,buff.getvalue())
  prose=PAPER/'sections/experiments.tex';s=prose.read_text()
  if any(p['kind']=='baselines' for _,p,_ in found):
   s=s.replace('Partial-layer Alloc-MoE and ASET have not been evaluated on Uni-MoE.','The added Uni-MoE partial-layer baselines use $[7,19)$ in a separate evaluation.')
  if any(p['kind']=='transfer' for _,p,_ in found):
   s=s.replace('These points have not\nbeen evaluated on AVUT or WorldSense.','Completed scope-transfer results are marked $\\S$; unreached cells remain blank.')
  marker='% OCTOBER 6 VERIFIED BACKFILL'
  if marker not in s:s=s.replace('\\maininferencetable',marker+'\n\\input{results/remaining_run_summary}\n\n\\maininferencetable')
  atomic_text(prose,s)
  text=''
  if coverage:
   text='\\paragraph{Additional scope evaluations.}\nCompleted configurations are '+ '; '.join(coverage)+'. '
   text+='Qwen uses the Megatron-Core GroupedMLP backend and Uni-MoE uses its DeepSpeed runtime. Timing excludes the first five evaluated questions. Points marked $\\S$ are separate-session measurements with frozen OmniVideoBench gates. Table percentages use the displayed Native and are descriptive cross-run ratios. Appendix~\\ref{app:remaining-runs} records input-subset recovery and timing details.\n'
  atomic_text(PAPER/'results/remaining_run_summary.tex',text)
  terminal=next(((d,p,s) for d,p,s in found if p['kind']=='terminal'),None)
  if terminal:
   d,p,s=terminal;a=s['arms']['omnidial']
   atomic_text(PAPER/'results/terminal_full_result.tex',f"\\paragraph{{Full-set terminal-layer protection.}}\nUni-MoE scope $[16,27)$ records {a['accuracy_percent']:.1f}\\% on all 1,000 OmniVideoBench questions, reusing 32 pilot predictions after input and output checks. The remaining 968 questions were newly evaluated. Mean generation time is {a['generation_seconds']:.3f}\\,s over {s['timed_n']} newly timed questions; the pilot timings are excluded. This timing window differs from the archived late-scope run.\n")
   # Replace the provisional pilot continuation sentence only after completion.
   s=prose.read_text().replace('The full-set extension remains pending.','The full-set continuation is reported below.')
   if '\\input{results/terminal_full_result}' not in s:s+='\n\\input{results/terminal_full_result}\n'
   atomic_text(prose,s)
  write(PAPER/'results/remaining_run_provenance.json',dict(updated=time.time(),runs=provenance))
  tasks=read(PAPER/'plan/main_table_gap_tasks.json');completed={t for _,p,_ in found for t in p['task_ids']}
  for task in tasks['tasks']:
   if task['task_id'] in completed:task.update(result_status='verified',execution='complete_and_backfilled')
  tasks['blank_numeric_cells']=sum(t['result_status']!='verified' for t in tasks['tasks'])*2
  tasks['table_sha256']=sha(table);tasks['status']='verified' if tasks['blank_numeric_cells']==0 else 'planned; partially completed'
  write(PAPER/'plan/main_table_gap_tasks.json',tasks)
  buff=io.StringIO();fields=list(tasks['tasks'][0]);writer=csv.DictWriter(buff,fieldnames=fields);writer.writeheader();writer.writerows(tasks['tasks']);atomic_text(PAPER/'plan/main_table_gap_tasks.csv',buff.getvalue())
  with (PAPER/'plan/progress.md').open('a') as f:f.write(f"\nAutomatic verified backfill — {time.strftime('%Y-%m-%d %H:%M:%S')}: `{out.name}`. Remaining Main Table numeric cells: {tasks['blank_numeric_cells']}. See results/remaining_run_provenance.json.\n")
  status=ROOT/'research_paper/MY_BUDGETING_RESEARCH_STATUS.md'
  current=status.read_text();heading='# OmniDial research status\n'
  entry=f"\n## Verified automatic backfill — {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n`{out.name}` completed and passed full-result checks. Main Table remaining empty numeric cells: {tasks['blank_numeric_cells']}. Historical controls are preserved. Exact metrics and provenance: `acl_arr_oct2027/results/remaining_run_provenance.json`. Other tasks follow the queue recorded in `../experiments/omnidial_remaining_20261006/queue.json`.\n\n"
  atomic_text(status,heading+entry+current[len(heading):])
  todo=PAPER/'plan/layer_scope_followups.md'
  current=todo.read_text()
  if tasks['blank_numeric_cells']==0:
   current=current.replace('- [ ] **Required follow-up: fill all six current Main Table configuration gaps.**','- [x] **Completed: all six Main Table configuration gaps filled after full-result verification.**')
  current+=f"\nVerified automatic update — {time.strftime('%Y-%m-%d %H:%M:%S')}: `{out.name}` completed and backfilled; {tasks['blank_numeric_cells']} Main Table numeric cells remain.\n"
  atomic_text(todo,current)
  appendix=PAPER/'appendix.tex';current=appendix.read_text()
  if any(p['kind']=='baselines' for _,p,_ in found):
   current=current.replace('The planned Uni-MoE partial Alloc-MoE/ASET run','The Uni-MoE partial Alloc-MoE/ASET run')
  atomic_text(appendix,current)
  with (out/'paper_compile.log').open('a') as log:
   build=subprocess.run(['make','review'],cwd=PAPER,stdout=log,stderr=subprocess.STDOUT)
  write(out/'backfill.json',dict(table_sha256=sha(table),build_exit_code=build.returncode,pdf_sha256=sha(PAPER/'build/main.pdf') if build.returncode==0 else None,summary_sha256=sha(out/'summary.json'),raw_sha256=sha(out/'raw_results.jsonl'),manifest_sha256=sha(out/'manifest.json'),backfill_script_sha256=sha(BASE/'backfill.py'),lock_path=str(lock_path('paper_backfill.lock')),time=time.time()))
  assert build.returncode==0,'Paper updated; compilation failed; inspect paper_compile.log'
if __name__=='__main__':
 import sys
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();backfill(a.out)

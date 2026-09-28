import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[3]
os.chdir(root)
driver=Path(__file__).resolve().parent
plan=json.loads((driver/'launch.json').read_text())
assert subprocess.check_output(['git','rev-parse','--short','HEAD'],text=True).strip()==plan['native_commit']
env=dict(os.environ,PYTHONPATH='src:.',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLCONFIGDIR='/tmp/pfm-mpl-natural')
record=dict(status='running',stage='cache_completion',started_at=time.time(),native_commit=plan['native_commit'])
def save(): (driver/'status.json').write_text(json.dumps(record,indent=2)+'\n')
def run(command,name):
 with (driver/(name+'.log')).open('w') as log:
  subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
base=driver.parent
try:
 save()
 run([sys.executable,'scripts/server/complete_reference_topology_cache.py',
 '--full-segments','/home/tuv43532/PangenomeFM/server_workspace/data/processed/hprc_r2_sv/full_segments.csv.gz',
 '--manifest','/home/tuv43532/PangenomeFM/server_workspace/data/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv',
 '--cache-root','/home/tuv43532/PangenomeFM/results/entex/v1/topology_cache_all_reference',
 '--results-root','/home/tuv43532/PangenomeFM/server_workspace/results/full_multicohort_server_20260806',
 '--target-cache','/home/tuv43532/PangenomeFM/server_workspace/data/processed/hprc_r2_ccre_screen_v4_features.npz',
 '--out-root',str(base/'reference_topology_completion')],'cache_completion')
 record['stage']='smoke';save();run(plan['smoke'],'smoke')
 status=json.loads((base/'sv_type_natural_recovered_smoke/status.json').read_text())
 assert status['status']=='complete' and status['n_events']==174267 and status['n_excluded']==0
 record['stage']='full_matrix';save()
 template=plan['smoke'][:plan['smoke'].index('--folds')]
 commands=[]
 for fold in plan['folds']:
  command=template.copy();command[command.index('--out-root')+1]=str(base/'sv_type_natural_shards'/fold)
  command+=['--folds',fold];commands.append(command)
 (driver/'full_commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
  futures=[pool.submit(run,c,fold) for c,fold in zip(commands,plan['folds'])]
  for f in futures:f.result()
 record['stage']='merge_and_replay';save()
 run([sys.executable,'-m','tasks.transfer.sv_type_shards','--shards',str(base/'sv_type_natural_shards'),'--out-dir',str(base/'sv_type_natural_full')],'merge')
 run([sys.executable,'-m','tasks.transfer.sv_type_report','--root',str(base/'sv_type_natural_full'),'--examples',str(base/'sv_type_preparation/natural_events.parquet'),'--out-dir',str(base/'sv_type_natural_full_analysis')],'report')
 record.update(status='complete',completed_at=time.time())
except Exception as e:
 record.update(status='failed',error=repr(e));raise
finally:save()

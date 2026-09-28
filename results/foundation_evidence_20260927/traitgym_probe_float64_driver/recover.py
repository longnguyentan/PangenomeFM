import json,os,subprocess,time
from pathlib import Path
root=Path('/home/tuv43532/PangenomeFM_evidence_report_20260927')
here=root/'results/foundation_evidence_20260927/traitgym_probe_float64_driver'
plan=json.loads((here/'recovery_launch.json').read_text())
assert subprocess.check_output(['git','rev-parse','--short=7','HEAD'],cwd=root,text=True).strip()==plan['recovery_commit']
env=dict(os.environ,PYTHONPATH='src:.',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/pfm-mpl')
state=dict(status='running',completed_stages=0,native_fitting_commit='ee466cb',recovery_commit=plan['recovery_commit'],started=time.time())
def save(): (here/'recovery_status.json').write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 for i,command in enumerate(plan['commands']):
  state.update(stage=i,command=command); save()
  with (here/f'recovery_{i}.log').open('x') as log:
   subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  state['completed_stages']=i+1; save()
 state['status']='complete'
except Exception as error:
 state.update(status='failed',error=repr(error)); raise
finally:
 state['updated']=time.time();save()

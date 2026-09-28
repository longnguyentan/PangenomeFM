import json,os,subprocess,time
from pathlib import Path
root=Path('/home/tuv43532/PangenomeFM_evidence_report_20260927')
here=root/'results/foundation_evidence_20260927/sv_type_driver'
plan=json.load(open(here/'launch.json'))
assert subprocess.check_output(['git','rev-parse','--short=7','HEAD'],cwd=root,text=True).strip()==plan['native_commit']
env=dict(os.environ,PYTHONPATH='src:.',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLCONFIGDIR='/tmp/pfm-mpl')
state=dict(status='running',completed_stages=0,native_commit=plan['native_commit'],started=time.time())
def save():
    (here/'status.json').write_text(json.dumps(state,indent=2)+'\n')
save()
try:
    for i,cmd in enumerate(plan['commands']):
        state.update(stage=i,command=cmd)
        save()
        with (here/f'stage_{i}.log').open('x') as log:
            subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        state['completed_stages']=i+1
        save()
    state['status']='complete'
except Exception as error:
    state.update(status='failed',error=repr(error))
    raise
finally:
    state['updated']=time.time()
    save()

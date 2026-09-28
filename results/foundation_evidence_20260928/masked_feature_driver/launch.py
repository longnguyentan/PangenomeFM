import json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[3];os.chdir(root)
base=root/'results/foundation_evidence_20260927'
checkpoints=list((base/'frozen_branch_seed_replication/seed_314159/sequence_trained').glob('*/run_001/ckpt_*.pt'))
assert len(checkpoints)==1
command=[sys.executable,'-u','scripts/server/run_masked_feature_campaign.py',
 '--config','configs/masked_nt_objective_20260928.json','--template-checkpoint',str(checkpoints[0]),
 '--main-checkout','/home/tuv43532/PangenomeFM',
 '--nt-cache',str(base/'benchmark_nt_completion/benchmark_nt.npz'),
 '--topology-control-cache','/home/tuv43532/PangenomeFM_review_20260927/results/v2_review_20260927/controls/topology_control.npz',
 '--out-root',str(base/'masked_feature_full'),'--gpus','0','1','2','3']
record=dict(native_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),command=command)
(Path(__file__).parent/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
subprocess.run(command,env=dict(os.environ,PYTHONPATH='src:.'),check=True)

"""Check every fixed-C output against the earlier complete allele-score study."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser()
parser.add_argument('--base',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
new=args.base/'traitgym_probe_float64_full'
receipt=json.loads((new/'status.json').read_text())
assert receipt['status']=='complete' and receipt['scope']=='full_matrix' and receipt['completed_runs']==60
plan=receipt['plan']
comparisons=[]
for job in receipt['jobs']:
    for dataset in plan['datasets']:
        suffix=Path(dataset)/job['fold']/f"seed_{job['seed']}"/job['closure']/'predictions.parquet'
        before=pd.read_parquet(args.base/'traitgym_allele_full'/suffix)
        after=pd.read_parquet(new/suffix)
        for name,spec in plan['selected_probe']['outputs'].items():
            if spec['estimator']!='fixed':
                continue
            a=before.loc[before.feature_set.eq(spec['input'])].sort_values('variant_id')
            b=after.loc[after.feature_set.eq(name)].sort_values('variant_id')
            for col in ['variant_id','label','y_true','p_raw','p_calibrated','threshold','y_pred']:
                np.testing.assert_array_equal(a[col],b[col])
            comparisons.append(dict(dataset=dataset,fold=job['fold'],seed=job['seed'],context=job['closure'],feature=name,n=len(a),bitwise_equal=True))
assert len(comparisons)==360
with args.out.open('x') as f:
    json.dump(dict(status='pass',n_comparisons=len(comparisons),comparisons=comparisons),f,indent=2)
print('All 360 fixed-C comparisons reproduce the original predictions bitwise')

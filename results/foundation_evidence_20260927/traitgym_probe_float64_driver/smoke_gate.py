import json
from pathlib import Path
import numpy as np
import pandas as pd
base=Path('/home/tuv43532/PangenomeFM_evidence_report_20260927/results/foundation_evidence_20260927')
plan=json.load(open(base.parent.parent/'configs/traitgym_probe_study_20260927.json'))
audit=json.load(open(base/'traitgym_probe_float64_smoke_analysis/audit.json'))
assert audit['status']=='complete' and audit['numerical_gate']=='pass'
comparisons=[]
for dataset in plan['datasets']:
    suffix=Path(dataset)/'fold_a/seed_42/strict/predictions.parquet'
    before=pd.read_parquet(base/'traitgym_allele_full'/suffix)
    after=pd.read_parquet(base/'traitgym_probe_float64_smoke'/suffix)
    for name,spec in plan['selected_probe']['outputs'].items():
        if spec['estimator']!='fixed':
            continue
        a=before.loc[before.feature_set.eq(spec['input'])].sort_values('variant_id')
        b=after.loc[after.feature_set.eq(name)].sort_values('variant_id')
        for col in ['variant_id','label','y_true','p_raw','p_calibrated','threshold','y_pred']:
            np.testing.assert_array_equal(a[col],b[col])
        comparisons.append(dict(dataset=dataset,feature=name,n=len(a),bitwise_equal=True))
(base/'traitgym_probe_float64_driver/smoke_gate.json').write_text(json.dumps(dict(status='pass',comparisons=comparisons),indent=2)+'\n')
print('All 12 fixed-C real-data comparisons reproduce original predictions bitwise',flush=True)

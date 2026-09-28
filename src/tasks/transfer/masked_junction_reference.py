"""Compare audited masked-feature validation arms with the frozen junction Q."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tasks.transfer.masked_feature_report import compare as validate_masked_matrix
from tasks.transfer.traitgym import verified_fingerprint, write_json


def compare(candidate: pd.DataFrame, reference: pd.DataFrame, plan: dict) -> pd.DataFrame:
    reference=reference.loc[reference.model.eq(plan['reference_model'])]
    rows=[]
    for (seed,task), group in candidate.groupby(['seed','task']):
        old=reference.loc[reference.seed.eq(seed)&reference.task.eq(task)]
        if len(old)!=4 or old.feature.duplicated().any() or set(old.feature)!={'cs','csh','cst','csht'}:
            raise ValueError('Incomplete unique junction reference')
        all_rows=pd.concat([group,old],ignore_index=True)
        for key in ['fold','context','targets_sha256','n_train','n_val','n_test','n_evaluated','evaluation_partition']:
            if all_rows[key].nunique()!=1:
                raise ValueError('Junction and masked-feature populations differ')
        prevalence = all_rows['positive_prevalence'].to_numpy()
        # Prior report CSV rounds a replayed proportion by one ULP; target hashes
        # and exact evaluated counts above still establish identical labels.
        if not np.allclose(prevalence, prevalence[0], rtol=0, atol=1e-15):
            raise ValueError('Junction and masked-feature prevalence differs')
        for feature in ['cs','csh']:
            if all_rows.loc[all_rows.feature.eq(feature),'scores_sha256'].nunique()!=1:
                raise ValueError('Junction and masked-feature baseline scores differ')
        old=old.set_index('feature')
        for row in group.loc[group.feature.isin(plan['features'])].itertuples():
            for metric in plan['metrics']:
                value=float(getattr(row,metric))
                baseline=float(old.loc[row.feature,metric])
                rows.append(dict(seed=seed,task=task,feature=row.feature,metric=metric,model=row.model,
                    masked_value=value,junction_q_value=baseline,difference=value-baseline))
    return pd.DataFrame(rows)


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--masked-report',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    args=ap.parse_args()
    plan=json.loads(args.config.read_text())
    source=verified_fingerprint(Path(plan['reference_table']),plan['reference_sha256'])
    audit=json.loads((args.masked_report/'audit.json').read_text())
    if not audit['all_metrics_replayed'] or not audit['all_probes_converged']:
        raise ValueError('New model report has not passed replay/convergence')
    candidate=pd.read_csv(args.masked_report/'audited_per_run.csv',float_precision='round_trip')
    seeds=sorted(candidate.seed.unique().tolist())
    if not set(seeds)<=set(plan['seeds']):
        raise ValueError('Unexpected seeds')
    validate_masked_matrix(candidate,dict(arms=plan['candidate_arms'],fold='fold_a'),seeds)
    reference=pd.read_csv(plan['reference_table'],float_precision='round_trip')
    result=compare(candidate,reference,plan)
    args.out_dir.mkdir(parents=True,exist_ok=False)
    result.to_csv(args.out_dir/'paired_per_seed.csv',index=False)
    result.groupby(['task','feature','metric','model']).difference.agg(['mean','min','max']).reset_index().to_csv(args.out_dir/'seed_descriptives.csv',index=False)
    write_json(args.out_dir/'audit.json',dict(status='complete',plan=plan,reference=source,
        all_reference_baselines_bitwise_identical=True,seeds=seeds,full_seed_matrix=set(seeds)==set(plan['seeds']),
        scope='Exploratory development validation; no CI; no gate or training change'))


if __name__=='__main__':
    main()

from itertools import product

import numpy as np
import pandas as pd
import pytest

from tasks.transfer.masked_feature_report import compare
from tasks.transfer.masked_junction_reference import compare as compare_junction


def test_paired_development_matrix_and_baseline_identity():
    arms=['full_trained','full_random','coordinate_trained','coordinate_random']
    rows=[]
    for seed,model,task,feature in product([42,314159,20260806],arms,['sv','ccre'],['cs','csh','cst','csht']):
        rows.append(dict(seed=seed,model=model,task=task,feature=feature,n_test=0,
            evaluation_partition='development_validation',fold='fold_a',context='1hop',
            targets_sha256='same',n_train=100,n_val=20,n_evaluated=20,positive_prevalence=.5,
            scores_sha256=feature,auprc=.7 if model.endswith('trained') else .6,
            auroc=.7,normalized_ap=.4,balanced_accuracy=.6,f1=.6))
    frame=pd.DataFrame(rows)
    plan=dict(arms=arms,fold='fold_a')
    result=compare(frame,plan,[42,314159,20260806])
    assert len(result)==180
    part=result.loc[result.metric.eq('auprc')&result.comparison.eq('full_trained minus full_random')]
    assert part.difference.to_numpy()==pytest.approx(.1)
    with pytest.raises(ValueError,match='matrix'):
        compare(frame.iloc[1:],plan,[42,314159,20260806])
    broken=frame.copy()
    broken.loc[0,'scores_sha256']='changed'
    with pytest.raises(ValueError,match='baseline'):
        compare(broken,plan,[42,314159,20260806])
    broken=frame.copy()
    broken.loc[0,'targets_sha256']='changed'
    with pytest.raises(ValueError,match='examples'):
        compare(broken,plan,[42,314159,20260806])
    old=frame.loc[frame.model.eq('full_trained')].assign(model='Q',auprc=.65)
    reference_plan=dict(reference_model='Q',features=['cst','csht'],metrics=['auprc'])
    paired=compare_junction(frame,old,reference_plan)
    assert paired.loc[paired.model.eq('full_trained'),'difference'].to_numpy()==pytest.approx(.05)
    assert paired.loc[paired.model.eq('full_random'),'difference'].to_numpy()==pytest.approx(-.05)
    broken=old.copy()
    broken.loc[broken.feature.eq('cs'),'scores_sha256']='changed'
    with pytest.raises(ValueError,match='baseline'):
        compare_junction(frame,broken,reference_plan)

    rounded=old.copy()
    rounded['positive_prevalence']=np.nextafter(.5, 0.)
    assert len(compare_junction(frame,rounded,reference_plan))==len(paired)
    rounded['positive_prevalence']=.499
    with pytest.raises(ValueError,match='prevalence'):
        compare_junction(frame,rounded,reference_plan)

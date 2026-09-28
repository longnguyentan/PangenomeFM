"""Replay masked-feature controls on development validation, without fold CIs."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import pandas as pd

from tasks.entex.prepare import fingerprint
from tasks.transfer.hr_control_report import audited_run
from tasks.transfer.report import save_figure
from tasks.transfer.traitgym import write_json


def compare(frame: pd.DataFrame, plan: dict, seeds: list[int]) -> pd.DataFrame:
    keys=['seed','model','task','feature']
    expected=set(product(seeds,plan['arms'],['sv','ccre'],['cs','csh','cst','csht']))
    if frame.duplicated(keys).any() or set(frame[keys].itertuples(index=False,name=None)) != expected:
        raise ValueError('Incomplete declared model/seed/task/feature matrix')
    if (not frame.n_test.eq(0).all() or not frame.evaluation_partition.eq('development_validation').all()
            or not frame.fold.eq(plan['fold']).all() or not frame.context.eq('1hop').all()):
        raise ValueError('Unexpected chromosome or test evaluation')
    rows=[]
    for (seed,task), group in frame.groupby(['seed','task']):
        for key in ['targets_sha256','n_train','n_val','n_evaluated','positive_prevalence']:
            if group[key].nunique()!=1:
                raise ValueError('Paired examples/labels differ')
        for feature in ['cs','csh']:
            if group.loc[group.feature.eq(feature),'scores_sha256'].nunique()!=1:
                raise ValueError('Non-embedding baseline predictions differ')
        for feature,metric in product(['cst','csht'],['auprc','auroc','normalized_ap','balanced_accuracy','f1']):
            values=group.loc[group.feature.eq(feature)].set_index('model')[metric]
            for left,right in [('full_trained','full_random'),('coordinate_trained','coordinate_random'),('full_trained','coordinate_trained')]:
                rows.append(dict(seed=seed,task=task,feature=feature,metric=metric,comparison=f'{left} minus {right}',
                    candidate=float(values[left]),control=float(values[right]),difference=float(values[left]-values[right])))
    return pd.DataFrame(rows)


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--seeds',type=int,nargs='+',help='Explicit completed subset for a partial development report')
    args=ap.parse_args()
    receipt_bytes=(args.root/'status.json').read_bytes()
    receipt=json.loads(receipt_bytes)
    plan=receipt['plan']
    seeds=args.seeds or plan['seeds']
    if len(set(seeds))!=len(seeds) or not set(seeds)<=set(receipt['completed_seeds']):
        raise ValueError('Requested seeds have not finished')
    records,optimization,pretraining=[],[],[]
    for seed in seeds:
        directory=args.root/f'seed_{seed}'
        source=json.loads((directory/'biological/status.json').read_text())
        if (source['status']!='complete' or source['completed_commands']!=len(source['commands'])
                or source['evaluation_partition']!='development_validation' or not source['encoders_frozen']
                or source['probe_max_iter_override']!=4000 or source['extraction_candidate_policy']!='manuscript'):
            raise ValueError('Incomplete or mismatched biological execution')
        initial={}
        for arm in plan['arms']:
            status=json.loads((directory/arm/'status.json').read_text())
            if (status['status']!='complete' or status['biological_labels_used'] or status['test_windows_loaded']
                    or status['plan']!=plan or status['seed']!=seed or status['arm']!=arm):
                raise ValueError('Invalid pretraining provenance')
            initial[arm]=status['initial_encoder_sha256']
            if arm.endswith('random') and status['final_encoder_sha256']!=initial[arm]:
                raise ValueError('Random backbone changed')
            if fingerprint(directory/arm/'checkpoint.pt')['sha256']!=status['checkpoint']['sha256']:
                raise ValueError('Checkpoint changed')
            pretraining.append(dict(seed=seed,arm=arm,best_validation_loss=status['best_validation_loss'],
                epochs_completed=status['epochs_completed'],initial_encoder_sha256=initial[arm],
                final_encoder_sha256=status['final_encoder_sha256']))
            for task in ['sv','ccre']:
                path=directory/'biological/probes'/arm/task/'1hop'
                checked=audited_run(path,arm,task,plan['fold'],seed,'1hop',validation_only=True)
                if not checked.checkpoint_sha256.eq(status['checkpoint']['sha256']).all():
                    raise ValueError('Probe checkpoint differs from selected pretraining checkpoint')
                records.append(checked)
                m=pd.read_csv(path/'metrics.csv')
                if not m.probe_converged.all() or not m.probe_max_iter.eq(4000).all():
                    raise ValueError('Probe convergence incomplete')
                optimization.append(m.assign(model=arm,task=task))
        for kind in ['full','coordinate']:
            if initial[kind+'_trained']!=initial[kind+'_random']:
                raise ValueError('Matched initializations differ')
    frame=pd.concat(records,ignore_index=True)
    # Keep native feature keys for exact paired replay, but identify the changed
    # representation explicitly: these sequence-conditioned vectors are not T.
    frame['embedding']=frame.model.map(lambda name: 'E_random' if name.endswith('random') else 'E')
    frame['embedding_representation']='sequence_conditioned_masked_features'
    frame['feature_display']=frame.feature.map(dict(cs='C+S',csh='C+S+H',cst='C+S+E',csht='C+S+H+E'))
    differences=compare(frame,plan,seeds)
    args.out_dir.mkdir(parents=True,exist_ok=False)
    source_snapshot=args.out_dir/"execution_receipt.json"
    source_snapshot.write_bytes(receipt_bytes)
    frame.to_csv(args.out_dir/'audited_per_run.csv',index=False)
    differences.to_csv(args.out_dir/'paired_differences.csv',index=False)
    pd.concat(optimization,ignore_index=True).to_csv(args.out_dir/'probe_optimization.csv',index=False)
    pd.DataFrame(pretraining).to_csv(args.out_dir/'pretraining.csv',index=False)
    summary=differences.groupby(['task','feature','metric','comparison']).difference.agg(['mean','min','max']).reset_index()
    summary.to_csv(args.out_dir/'seed_descriptives.csv',index=False)
    selected=differences.loc[differences.feature.eq('csht') & differences.metric.eq('auprc') & differences.comparison.eq('full_trained minus full_random')]
    gate=dict(status='eligible_for_chromosome_replication' if set(seeds)==set(plan['seeds']) and selected.difference.gt(0).all() else 'not_promoted',
        scope='Exploratory fold-A validation; seeds are not independent chromosome replicates; no CI',
        all_declared_seeds_complete=set(seeds)==set(plan['seeds']), checks=selected.to_dict('records'),
        source=fingerprint(source_snapshot), live_status_path=str((args.root/'status.json').resolve()),
        all_metrics_replayed=True, all_probes_converged=True,
        graph_contribution_requires_separate_full_vs_coordinate_comparison=True,
        representation='E is sequence-conditioned; native cst/csht keys are retained solely for pipeline compatibility',
        report_implementation=fingerprint(Path(__file__)))
    write_json(args.out_dir/'audit.json',gate)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,task in zip(axes,['sv','ccre']):
        part=differences.loc[differences.task.eq(task)&differences.feature.eq('csht')&differences.metric.eq('auprc')]
        for comparison,g in part.groupby('comparison'):
            ax.plot(g.seed.astype(str),g.difference,'o-',label=comparison)
        ax.axhline(0,color='.5',lw=.8)
        ax.set(title=task,xlabel='Initialization seed',ylabel='Validation Δ AUPRC after C+S+H')
    axes[0].legend(fontsize=7)
    fig.suptitle('Masked-feature objective adaptation: one development fold, no confidence intervals')
    save_figure(fig,args.out_dir,'masked_feature_controls')
    plt.close(fig)


if __name__=='__main__':
    main()

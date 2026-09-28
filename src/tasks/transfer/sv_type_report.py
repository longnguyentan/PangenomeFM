"""Replay frozen SV-type predictions and report all three classes and macro AP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from evaluation.paired_inference import bh_adjust, fold_sign_flip
from scripts.server.run_ccre_frozen_probe_fold import binary_metrics
from tasks.entex.analyze import estimate
from tasks.transfer.report import save_figure
from tasks.transfer.sv_types import CLASSES
from tasks.transfer.traitgym import verified_fingerprint, write_json


def replay(directory: Path, examples: pd.DataFrame, plan: dict, job: dict) -> pd.DataFrame:
    audit = json.loads((directory/'audit.json').read_text())
    if (audit.get('status') != 'complete' or audit.get('n_excluded') != 0
            or audit.get('feature_coverage') != 1 or not audit.get('all_probes_converged')):
        raise ValueError('Incomplete, filtered or unconverged SV-type run')
    verified_fingerprint(directory/'predictions.parquet', audit['predictions']['sha256'])
    p = pd.read_parquet(directory/'predictions.parquet')
    metrics = pd.read_csv(directory/'metrics.csv', float_precision='round_trip')
    keys = ['target_class', 'feature_set']
    expected = {(c, f) for c in CLASSES for f in plan['feature_sets']}
    if (metrics.duplicated(keys).any() or set(metrics[keys].itertuples(index=False, name=None)) != expected
            or set(p[keys].itertuples(index=False, name=None)) != expected):
        raise ValueError('Missing/duplicated class or feature output')
    truth = examples.loc[examples.chrom.isin(job['test'])].sort_values('variant_id')
    identity = ['variant_id', 'locus_id', 'chrom', 'start', 'end', 'event_end', 'svtype', 'svlen', 'label']
    for m in metrics.itertuples():
        part = p.loc[p.target_class.eq(m.target_class) & p.feature_set.eq(m.feature_set)].sort_values('variant_id')
        pd.testing.assert_frame_equal(part[identity].reset_index(drop=True), truth[identity].reset_index(drop=True), check_dtype=False)
        if (part.variant_id.duplicated().any() or not part.y_true.eq(part.label.eq(CLASSES[m.target_class]).astype(int)).all()
                or not part.chromosome.eq(part.chrom).all() or len(part) != m.n_test
                or m.probe_max_iter != plan['probe_max_iter'] or not m.probe_converged
                or not part.y_pred.eq((part.p_calibrated >= m.threshold).astype(int)).all()
                or not part.threshold.eq(m.threshold).all()):
            raise ValueError('SV class, prediction universe or optimizer identity differs')
        for key, value in dict(fold=job['fold'], seed=job['seed'], context=job['closure'], task=plan['task']).items():
            if getattr(m, key) != value or not part[key].eq(value).all():
                raise ValueError('Run identity differs')
        checked = binary_metrics(part.y_true.to_numpy(), part.p_calibrated.to_numpy(), m.threshold)
        checked['balanced_accuracy'] = balanced_accuracy_score(part.y_true, part.y_pred)
        checked['normalized_ap'] = (checked['auprc']-checked['positive_fraction'])/(1-checked['positive_fraction'])
        for key in plan['metrics']:
            if not np.isclose(getattr(m, key), checked[key], rtol=0, atol=1e-10):
                raise ValueError('Stored SV metric differs from prediction replay')
    return metrics


def summarize(metrics: pd.DataFrame, plan: dict):
    keys = ['context', 'fold', 'seed', 'feature_set']
    if metrics.duplicated(['target_class', *keys]).any():
        raise ValueError('Duplicate class/run metrics')
    macro = metrics.groupby(keys)[plan['metrics']].mean().reset_index().assign(target_class='macro')
    frame = pd.concat([metrics, macro], ignore_index=True)
    absolute, pairs = [], []
    for (target, context), group in frame.groupby(['target_class', 'context']):
        for metric in plan['metrics']:
            for feature, part in group.groupby('feature_set'):
                absolute.append(dict(target_class=target, context=context, metric=metric, feature_set=feature,
                    **estimate(part, metric, plan['n_bootstrap'], plan['statistics_seed'])))
            if metric not in ['auprc', 'auroc', 'normalized_ap']:
                continue
            wide = group.pivot(index=['fold', 'seed'], columns='feature_set', values=metric)
            if wide.isna().any().any():
                raise ValueError('Incomplete paired SV results')
            for name, full, base in plan['comparisons']:
                pairs.append((wide[full]-wide[base]).rename('gain').reset_index().assign(
                    target_class=target, context=context, metric=metric, contrast=name))
    paired = pd.concat(pairs, ignore_index=True)
    rows = []
    for key, group in paired.groupby(['target_class', 'context', 'metric', 'contrast']):
        rows.append(dict(zip(['target_class', 'context', 'metric', 'contrast'], key),
            **estimate(group, 'gain', plan['n_bootstrap'], plan['statistics_seed']), sign_flip_p=fold_sign_flip(group)))
    contrasts = pd.DataFrame(rows)
    contrasts['bh_q_within_metric'] = contrasts.groupby('metric').sign_flip_p.transform(bh_adjust)
    return pd.DataFrame(absolute), contrasts, paired


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--examples', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    receipt = json.loads((args.root/'status.json').read_text())
    if receipt['status'] != 'complete' or receipt['completed_runs'] != receipt['planned_runs']:
        raise ValueError('Incomplete declared SV experiment')
    plan = receipt['plan']
    verified_fingerprint(args.examples, plan['examples_sha256'])
    examples = pd.read_parquet(args.examples)
    rows = [replay(args.root/j['fold']/f"seed_{j['seed']}"/j['closure'], examples, plan, j) for j in receipt['jobs']]
    metrics = pd.concat(rows, ignore_index=True)
    absolute, contrasts, paired = summarize(metrics, plan)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, frame in [('per_run', metrics), ('absolute', absolute), ('contrasts', contrasts), ('paired', paired)]:
        frame.to_csv(args.out_dir/(name+'.csv'), index=False)
    write_json(args.out_dir/'audit.json', dict(status='complete', n_runs=len(rows), n_evaluations=len(metrics),
        all_probes_converged=True, all_metrics_replayed=True, n_events=len(examples), n_excluded=0,
        population=plan.get('population', 'Chromosome/length-bin matched known events; not natural prevalence'),
        class_counts=examples.svtype.value_counts().to_dict(),
        no_random_encoder_superiority_claim=True))
    lines = ['# '+plan.get('report_title', 'Frozen SV type: length-matched common-support experiment'), '',
        plan.get('population', 'Chromosome/length-bin matched common-support events.'), '',
        'Class counts: '+', '.join(f'{k}={v:,}' for k, v in examples.svtype.value_counts().items())+'.', '',
        '| Class | Context | Contrast | ΔAP | 95% CI | Fold p | BH q |', '|---|---|---|---:|---|---:|---:|']
    for r in contrasts.loc[contrasts.metric.eq('auprc')].itertuples():
        lines.append(f'| {r.target_class} | {r.context} | {r.contrast} | {r.mean:+.6f} | [{r.ci95_low:+.6f}, {r.ci95_high:+.6f}] | {r.sign_flip_p:.4f} | {r.bh_q_within_metric:.4f} |')
    (args.out_dir/'README.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), layout='constrained')
    for ax, target in zip(axes, [*CLASSES, 'macro']):
        part = contrasts.loc[contrasts.target_class.eq(target) & contrasts.metric.eq('auprc')]
        for i, r in enumerate(part.itertuples()):
            ax.plot(r.mean, i, 'o', color='#245a81' if r.context == 'strict' else '#be6831')
            ax.hlines(i, r.ci95_low, r.ci95_high, color='.4')
        ax.axvline(0, color='.6', lw=.8)
        ax.set(yticks=range(len(part)), yticklabels=[f'{r.context}: {r.contrast}' for r in part.itertuples()], title=target, xlabel='Δ AUPRC (95% CI)')
        ax.invert_yaxis()
    fig.suptitle(plan.get('figure_title', 'Known SV type: 223 events per class, frozen graph/sequence encoders'))
    save_figure(fig, args.out_dir, 'sv_type_topology_gains')
    plt.close(fig)


if __name__ == '__main__':
    main()

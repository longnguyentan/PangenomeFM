"""Validation-only constant fallback for an already completed frozen regression study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tasks.entex.prepare import fingerprint
from tasks.transfer.genotyping_report import replay, summarize
from tasks.transfer.regression_probe import regression_metrics
from tasks.transfer.traitgym import verified_fingerprint, write_json


def choose_ridge(ridge_validation_mae: float, median_validation_mae: float) -> bool:
    """No test outcomes enter this decision; the simpler reference wins exact ties."""
    values = np.asarray([ridge_validation_mae, median_validation_mae], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Validation MAE must be finite and nonnegative")
    return bool(values[0] < values[1])


def derive_run(metrics: pd.DataFrame, predictions: pd.DataFrame, loci: pd.DataFrame,
               job: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Derive fallback arms after the caller has replayed every original prediction."""
    records, frames, selections = [], [], []
    train = ~loci.chrom.isin([*job['validation'], *job['test']])
    val = loci.chrom.isin(job['validation'])
    for target, group in metrics.groupby('target'):
        reference = group.loc[group.feature_set.eq('train_median')]
        if len(reference) != 1:
            raise ValueError("Require exactly one training-median reference")
        reference = reference.iloc[0]
        median = float(np.median(loci.loc[train, target]))
        val_mae = float(np.mean(np.abs(loci.loc[val, target] - median)))
        if not np.isclose(reference.validation_mae, val_mae, atol=1e-12, rtol=0):
            raise ValueError("Training-median validation error does not replay")
        for row in group.loc[~group.feature_set.eq('train_median')].itertuples(index=False):
            use_ridge = choose_ridge(row.validation_mae, val_mae)
            selected = row.feature_set if use_ridge else 'train_median'
            p = predictions.loc[predictions.target.eq(target) & predictions.feature_set.eq(selected)].copy()
            original = predictions.loc[predictions.target.eq(target) & predictions.feature_set.eq(row.feature_set)]
            if (not len(p) or p.locus_id.duplicated().any()
                    or set(p.locus_id) != set(original.locus_id)):
                raise ValueError("Fallback prediction identities differ")
            p = p.assign(feature_set='fallback__' + row.feature_set)
            frames.append(p)
            records.append(row._asdict() | dict(
                feature_set='fallback__' + row.feature_set, alpha=row.alpha if use_ridge else np.nan,
                validation_mae=min(row.validation_mae, val_mae),
                **regression_metrics(p.y_true, p.prediction)))
            selections.append(dict(target=target, fold=job['fold'], seed=job['seed'], context=job['closure'],
                feature_set=row.feature_set, selected_feature=selected, selected_ridge=use_ridge,
                ridge_validation_mae=row.validation_mae, median_validation_mae=val_mae))
    return pd.DataFrame(records), pd.concat(frames, ignore_index=True), pd.DataFrame(selections)


def comparison_plan(original: dict, followup: dict) -> dict:
    comparisons = list(original['comparisons'])
    comparisons.extend(('fallback_' + name, 'fallback__' + full, 'fallback__' + base)
                       for name, full, base in original['comparisons'])
    for feature in original['feature_sets']:
        if feature != 'train_median':
            comparisons.extend([
                ('fallback_vs_original__' + feature, 'fallback__' + feature, feature),
                ('fallback_vs_median__' + feature, 'fallback__' + feature, 'train_median')])
    return dict(followup, comparisons=comparisons)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'root', 'loci', 'out-dir']:
        ap.add_argument('--' + name, type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.config.read_text())
    source_plan = Path('configs/cosigt_quality_20260927.json')
    verified_fingerprint(source_plan, plan['source_plan_sha256'])
    verified_fingerprint(args.root/'status.json', plan['source_status_sha256'])
    verified_fingerprint(args.loci, plan['loci_sha256'])
    receipt = json.loads((args.root/'status.json').read_text())
    original = receipt['plan']
    if original != json.loads(source_plan.read_text()):
        raise ValueError("Original plan differs from the pinned configuration")
    # The status checksum pins the exact original plan and complete fold/seed matrix.
    if (receipt['status'] != 'complete' or receipt['completed_runs'] != plan['n_runs']
            or len(receipt['jobs']) != plan['n_runs']
            or original['targets'] != plan['targets'] or original['feature_sets'] != plan['feature_sets']):
        raise ValueError("Original completed study differs from the declared source")
    loci = pd.read_parquet(args.loci)
    if len(loci) != plan['n_loci']:
        raise ValueError("Locus universe differs")
    rows, frames, selections, sources = [], [], [], [fingerprint(args.config), fingerprint(args.root/'status.json'), fingerprint(args.loci)]
    for job in receipt['jobs']:
        directory = args.root/job['fold']/f"seed_{job['seed']}"/job['closure']
        metrics = replay(directory, loci, job, original)
        p = pd.read_parquet(directory/'predictions.parquet')
        new, predicted, selection = derive_run(metrics, p, loci, job)
        rows.extend([metrics, new])
        frames.extend([p, predicted])
        selections.append(selection)
        sources.extend(fingerprint(directory/f) for f in ['audit.json', 'metrics.csv', 'predictions.parquet', 'validation_selection.csv'])
    metrics = pd.concat(rows, ignore_index=True)
    predictions = pd.concat(frames, ignore_index=True)
    selection = pd.concat(selections, ignore_index=True)
    absolute, contrasts, paired = summarize(metrics, comparison_plan(original, plan))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, frame in [('per_run', metrics), ('absolute', absolute), ('contrasts', contrasts),
                        ('paired', paired), ('selection', selection)]:
        frame.to_csv(args.out_dir/(name + '.csv'), index=False)
    predictions.to_parquet(args.out_dir/'predictions.parquet', index=False)
    write_json(args.out_dir/'audit.json', dict(status='complete', n_runs=len(receipt['jobs']), n_loci=len(loci),
        n_evaluations=len(metrics), n_derived_evaluations=len(selection), new_fits=0,
        n_ridge_selected=int(selection.selected_ridge.sum()), n_median_selected=int((~selection.selected_ridge).sum()),
        all_original_predictions_replayed=True, no_test_selection=True, plan=plan, sources=sources,
        implementation=fingerprint(Path(__file__)), predictions=fingerprint(args.out_dir/'predictions.parquet')))
    primary = contrasts.loc[contrasts.target.eq(plan['primary_target']) & contrasts.metric.eq('mae')
                            & contrasts.contrast.str.startswith('fallback_T_')]
    lines = ['# Validation-only genotyping-quality fallback', '',
             'Exploratory follow-up; no encoders or probes retrained. Positive gains mean lower error.', '',
             '| Context | Contrast | MAE reduction | 95% CI |', '|---|---|---:|---|']
    for row in primary.itertuples():
        lines.append(f'| {row.context} | {row.contrast} | {row.mean:+.6f} | [{row.ci95_low:+.6f}, {row.ci95_high:+.6f}] |')
    (args.out_dir/'README.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()

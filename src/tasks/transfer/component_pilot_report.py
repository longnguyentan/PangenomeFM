"""Audit matched component-context validation probes; no test access or promotion."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tasks.entex.prepare import fingerprint
from tasks.transfer.hr_control_report import audited_run
from tasks.transfer.masked_replication import verify_saved_probes
from tasks.transfer.traitgym import write_json
from training.component_pilot import validate_plan


def compare(frame: pd.DataFrame, plan: dict) -> pd.DataFrame:
    keys = ['seed', 'context', 'model', 'task', 'feature']
    expected = set(product(plan['seeds'], plan['contexts'], plan['arms'], ['sv', 'ccre'], ['cs', 'csh', 'cst', 'csht']))
    if frame.duplicated(keys).any() or set(frame[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError('Incomplete/duplicate context pilot matrix')
    if (not frame.n_test.eq(0).all() or not frame.fold.eq('fold_a').all()
            or not frame.evaluation_partition.eq('development_validation').all()):
        raise ValueError('Pilot contains an unexpected partition')
    rows = []
    metrics = ['auprc', 'auroc', 'normalized_ap', 'balanced_accuracy', 'f1']
    if not np.isfinite(frame[metrics].to_numpy()).all():
        raise ValueError('Nonfinite biological development metrics')
    for (seed, task), group in frame.groupby(['seed', 'task']):
        for field in ['targets_sha256', 'train_targets_sha256', 'validation_targets_sha256',
                      'n_train', 'n_val', 'n_evaluated', 'positive_prevalence']:
            if group[field].nunique() != 1:
                raise ValueError('Matched biological training/validation population differs')
        for feature in ['cs', 'csh']:
            if group.loc[group.feature.eq(feature), 'scores_sha256'].nunique() != 1:
                raise ValueError('Non-embedding baseline predictions differ across contexts/arms')
        for feature, metric in product(['cst', 'csht'], metrics):
            values = group.loc[group.feature.eq(feature)].set_index(['context', 'model'])[metric]
            def add(comparison, left, right):
                rows.append(dict(seed=seed, task=task, feature=feature, metric=metric,
                    comparison=comparison, difference=float(left - right)))
            for arm in plan['arms']:
                add(f'component minus 1hop / {arm}', values['component', arm], values['1hop', arm])
            for context in plan['contexts']:
                for left, right in [('full_trained', 'full_random'), ('coordinate_trained', 'coordinate_random'),
                                    ('full_trained', 'coordinate_trained')]:
                    add(f'{left} minus {right} / {context}', values[context, left], values[context, right])
            add('context difference in full trained-minus-random',
                values['component', 'full_trained'] - values['component', 'full_random'],
                values['1hop', 'full_trained'] - values['1hop', 'full_random'])
    return pd.DataFrame(rows)


def report(root: Path, out: Path):
    source_bytes = (root/'status.json').read_bytes()
    receipt = json.loads(source_bytes)
    plan = receipt['plan']
    validate_plan(plan)
    jobs = [(r['seed'], r['context']) for r in receipt['completed_jobs']]
    if (receipt['stage'] not in ['report', 'complete'] or len(jobs) != 6
            or set(jobs) != set(product(plan['seeds'], plan['contexts']))):
        raise ValueError('Incomplete pilot execution')
    rows, pretraining, artifacts = [], [], []
    for seed in plan['seeds']:
        initial, moments, populations = {}, None, None
        for context, arm in product(plan['contexts'], plan['arms']):
            folder = root/f'seed_{seed}'/context
            training = folder/arm
            status = json.loads((training/'status.json').read_text())
            if (status.get('status') != 'complete' or status['plan'] != plan
                    or status.get('biological_labels_used') or status.get('test_windows_loaded')
                    or status['source_context'] != context or status['seed'] != seed or status['arm'] != arm
                    or fingerprint(training/'checkpoint.pt')['sha256'] != status['checkpoint']['sha256']):
                raise ValueError('Invalid pretraining provenance')
            initial[context, arm] = status['initial_encoder_sha256']
            if arm.endswith('random') and status['final_encoder_sha256'] != initial[context, arm]:
                raise ValueError('Random encoder changed')
            with np.load(training/'target_moments.npz', allow_pickle=False) as archive:
                actual_moments = {k: archive[k].copy() for k in archive.files}
            if moments is None:
                moments = actual_moments
            elif set(moments) != set(actual_moments) or any(not np.array_equal(moments[k], actual_moments[k]) for k in moments):
                raise ValueError('Target identities/moments differ across contexts or controls')
            windows = pd.read_csv(training/'windows.csv').sort_values('window').reset_index(drop=True)
            columns = ['window', 'chrom', 'exclusion', 'n_target_segments', 'target_ids_sha256']
            if populations is None:
                populations = windows[columns]
            else:
                pd.testing.assert_frame_equal(populations, windows[columns], check_dtype=False)
            pretraining.append(dict(seed=seed, context=context, arm=arm,
                best_validation_loss=status['best_validation_loss'], epochs_completed=status['epochs_completed'],
                initial_encoder_sha256=status['initial_encoder_sha256'], final_encoder_sha256=status['final_encoder_sha256']))
            for task in ['sv', 'ccre']:
                path = folder/'probes'/arm/task
                audit = json.loads((path/'audit.json').read_text())
                extracted = audit['canonical_candidate_audit']['extracted_chromosomes']
                if (audit.get('test_extraction_excluded') is not True
                        or set(extracted) & set(audit['test_chromosomes'])):
                    raise ValueError('Test windows unexpectedly extracted')
                checked = audited_run(path, arm, task, 'fold_a', seed, context, validation_only=True)
                if not checked.checkpoint_sha256.eq(status['checkpoint']['sha256']).all():
                    raise ValueError('Probe used another checkpoint')
                metrics = pd.read_csv(path/'metrics.csv')
                if not metrics.probe_converged.all() or not metrics.probe_max_iter.eq(4000).all():
                    raise ValueError('Probe convergence incomplete')
                for column in ['train_targets_sha256', 'validation_targets_sha256']:
                    if metrics[column].nunique() != 1:
                        raise ValueError('Feature-specific population mismatch')
                    checked[column] = metrics[column].iloc[0]
                artifacts.extend(dict(seed=seed, context=context, arm=arm, task=task, **r)
                    for r in verify_saved_probes(path, metrics))
                rows.append(checked)
        for kind in ['full', 'coordinate']:
            if len({initial[c, kind + '_' + state] for c, state in product(plan['contexts'], ['trained', 'random'])}) != 1:
                raise ValueError('Context/control initialization differs')
    frame = pd.concat(rows, ignore_index=True)
    frame['embedding'] = np.where(frame.model.str.endswith('random'), 'E_random', 'E')
    differences = compare(frame, plan)
    out.mkdir(parents=True, exist_ok=False)
    (out/'execution_receipt.json').write_bytes(source_bytes)
    frame.to_csv(out/'audited_per_run.csv', index=False)
    differences.to_csv(out/'paired_differences.csv', index=False)
    differences.groupby(['task', 'feature', 'metric', 'comparison']).difference.agg(['mean', 'min', 'max']).reset_index().to_csv(out/'seed_descriptives.csv', index=False)
    pd.DataFrame(pretraining).to_csv(out/'pretraining.csv', index=False)
    pd.DataFrame(artifacts).to_csv(out/'fitted_probe_artifacts.csv', index=False)
    write_json(out/'audit.json', dict(status='complete', n_pretraining_runs=24, n_probe_runs=48,
        biological_test_predictions=False, test_windows_extracted=False,
        exact_target_moments_and_populations=True, all_metrics_replayed=True,
        all_probes_converged=True, all_fitted_probes_verified=True,
        scope='Fold-A development-validation descriptions; no genomic confidence intervals or automatic promotion',
        context_interpretation='Context changes encoder/decoder neighborhoods and native per-slice SO/LN scaling, SR min/max and degree normalization; this is a context-pipeline comparison, not an isolated adjacency effect.',
        tables={name: fingerprint(out/name) for name in ['audited_per_run.csv', 'paired_differences.csv',
            'seed_descriptives.csv', 'pretraining.csv', 'fitted_probe_artifacts.csv']},
        source=fingerprint(out/'execution_receipt.json'), implementation=fingerprint(Path(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    report(args.root, args.out_dir)


if __name__ == '__main__':
    main()

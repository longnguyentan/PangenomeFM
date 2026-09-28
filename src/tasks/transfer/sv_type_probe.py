"""Frozen one-versus-rest SV-type probes on a prespecified event population."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from evaluation.modality_factorial import build_modality_factorial, concatenate_modalities, factorial_feature_access
from scripts.server.run_ccre_frozen_probe_fold import evaluate_feature_sets
from scripts.server.run_ccre_frozen_probe_matrix import build_jobs
from tasks.entex.prepare import fingerprint
from tasks.transfer.locus_features import FrozenLocusFeatures
from tasks.transfer.sv_types import CLASSES
from tasks.transfer.traitgym import verified_fingerprint, write_json


def validate_population(examples: pd.DataFrame, loci: pd.DataFrame, plan: dict) -> None:
    """Require the declared event denominator and an exact, unique anchor table."""
    if (len(examples) != plan['n_events']
            or examples.svtype.value_counts().to_dict() != plan['class_counts']
            or examples.variant_id.duplicated().any()
            or loci.locus_id.duplicated().any()
            or set(examples.locus_id) != set(loci.locus_id)):
        raise ValueError('Event population or unique anchor universe differs from plan')
    merged = examples.merge(loci, on='locus_id', suffixes=('', '_anchor'), validate='many_to_one')
    if any(not merged[c].eq(merged[c+'_anchor']).all() for c in ['chrom', 'start', 'end']):
        raise ValueError('Event/anchor coordinates differ')


def evaluate_types(examples: pd.DataFrame, matrices: dict, plan: dict, job):
    """Reuse the manuscript binary classifier separately for each supplied class."""
    if (examples.variant_id.duplicated().any() or set(examples.svtype) != set(CLASSES)
            or not examples.svtype.map(CLASSES).eq(examples.label).all()):
        raise ValueError("Malformed multiclass identities or labels")
    frames, predictions = [], []
    for target, label in CLASSES.items():
        y = examples.label.eq(label).astype(int).to_numpy()
        for chroms in [set(job.test), set(job.validation), set(examples.chrom)-set(job.test)-set(job.validation)]:
            mask = examples.chrom.isin(chroms)
            if y[mask].sum() < 10 or (1-y[mask]).sum() < 10:
                raise ValueError("Insufficient class support in a chromosome partition")
        access = {**factorial_feature_access(), **{k: '+'.join(v) for k, v in plan['custom_feature_sets'].items()}}
        metrics, _, p = evaluate_feature_sets(segids=np.arange(len(examples)), chromosomes=examples.chrom.to_numpy(),
            labels=y, features={name: matrices[name] for name in plan['feature_sets']},
            test_chrs=set(job.test), val_chrs=set(job.validation), seed=job.seed, probe_max_iter=plan['probe_max_iter'], feature_access=access)
        metrics['balanced_accuracy'] = [balanced_accuracy_score(g.y_true, g.y_pred) for name in metrics.feature_set
            for g in [p.loc[p.feature_set.eq(name)]]]
        metrics['normalized_ap'] = (metrics.auprc-metrics.positive_fraction)/(1-metrics.positive_fraction)
        metrics['positive_prevalence'], metrics['n_val'] = metrics.positive_fraction, metrics.n_validation
        p = p.rename(columns={'segid': 'example_id'}).merge(examples.assign(example_id=np.arange(len(examples))), on='example_id', validate='many_to_one')
        for frame in [metrics, p]:
            for key, value in dict(task=plan['task'], target_class=target, fold=job.fold, seed=job.seed, context=job.closure).items():
                frame[key] = value
        frames.append(metrics)
        predictions.append(p)
    return pd.concat(frames, ignore_index=True), pd.concat(predictions, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'examples', 'loci', 'mapping-dir', 'manifest', 'feature-cache', 'sequence-cache',
                 'topology-control-cache', 'topology-cache-root', 'results-root', 'out-root']:
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--folds', nargs='+')
    ap.add_argument('--seeds', nargs='+', type=int)
    ap.add_argument('--contexts', nargs='+', choices=['strict', '1hop'])
    args = ap.parse_args()
    plan = json.loads(args.config.read_text())
    config = json.loads(Path(plan['reference_config']).read_text())
    manuscript = json.loads(Path(config['manuscript_config']).read_text())
    source = verified_fingerprint(args.examples, plan['examples_sha256'])
    loci_source = verified_fingerprint(args.loci, plan['loci_sha256'])
    examples, loci = pd.read_parquet(args.examples), pd.read_parquet(args.loci)
    validate_population(examples, loci, plan)
    mapping = json.loads((args.mapping_dir/'mapping_qc.json').read_text())
    if (mapping['source_loci']['sha256'] != loci_source['sha256']
            or mapping['source_graph']['sha256'] != config['full_segments_sha256'] or mapping['fraction_mapped'] != 1):
        raise ValueError('Require complete exact-graph mapping of the declared anchor universe')
    overlaps = pd.read_parquet(args.mapping_dir/'overlaps.parquet')
    required = set(examples.locus_id)
    loci = loci.loc[loci.locus_id.isin(required)].reset_index(drop=True)
    overlaps = overlaps.loc[overlaps.locus_id.isin(required)]
    if overlaps.locus_id.duplicated().any() or not overlaps.overlap_bp.eq(1).all():
        raise ValueError('Every anchor needs exactly one containing segment')
    features = FrozenLocusFeatures(loci, overlaps, config, args.feature_cache, args.sequence_cache,
        args.topology_control_cache, args.manifest, plan['topology_control_sha256'])
    row = examples.locus_id.map(pd.Series(np.arange(len(loci)), index=loci.locus_id)).to_numpy(int)
    static = {name: values[row] for name, values in features.static.items()}
    static['event_log_length'] = np.log1p(examples.svlen.to_numpy())[:, None].astype(np.float32)
    jobs = build_jobs(manuscript, seeds=set(args.seeds) if args.seeds else None, contexts=set(args.contexts) if args.contexts else None)
    jobs = [j for j in jobs if not args.folds or j.fold in args.folds]
    if not jobs:
        raise ValueError('No jobs selected')
    args.out_root.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='running', plan=plan, jobs=[asdict(j) for j in jobs], completed_runs=0, planned_runs=len(jobs),
        sources=[source, loci_source, *features.sources, fingerprint(args.config), fingerprint(args.mapping_dir/'mapping_qc.json'),
                 fingerprint(args.mapping_dir/'overlaps.parquet')], command=sys.argv, implementation=fingerprint(Path(__file__)),
        n_events=len(examples), class_counts=examples.svtype.value_counts().to_dict(), encoder_training=False, n_excluded=0)
    write_json(args.out_root/'status.json', receipt)
    try:
        for job in jobs:
            t, topology = features.topology(job, args.topology_cache_root, args.results_root)
            components = dict(static, frozen_pangenomefm=t[row])
            matrices = build_modality_factorial(components, include_external_sequence=True, include_topology_control=True)
            if set(matrices) & set(plan['custom_feature_sets']):
                raise ValueError('Do not overwrite manuscript feature definitions')
            matrices.update({name: concatenate_modalities(components, keys) for name, keys in plan['custom_feature_sets'].items()})
            metrics, predictions = evaluate_types(examples, matrices, plan, job)
            out = args.out_root/job.fold/f'seed_{job.seed}'/job.closure
            out.mkdir(parents=True, exist_ok=False)
            metrics.to_csv(out/'metrics.csv', index=False)
            predictions.to_parquet(out/'predictions.parquet', index=False)
            write_json(out/'audit.json', dict(status='complete', topology=topology, predictions=fingerprint(out/'predictions.parquet'),
                all_probes_converged=bool(metrics.probe_converged.all()), n_excluded=0, feature_coverage=1., test_chromosomes=list(job.test)))
            if not metrics.probe_converged.all():
                raise ValueError('Unconverged SV-type probe; diagnostic predictions retained')
            receipt['completed_runs'] += 1
            write_json(args.out_root/'status.json', receipt)
            print(f"Completed {receipt['completed_runs']}/{len(jobs)}: {job.name}", flush=True)
        receipt['status'] = 'complete'
    except Exception as error:
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        write_json(args.out_root/'status.json', receipt)


if __name__ == '__main__':
    main()

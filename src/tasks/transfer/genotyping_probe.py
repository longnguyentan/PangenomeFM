"""Chromosome-held-out regression of measured genotyping quality at frozen loci."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from evaluation.modality_factorial import build_modality_factorial, concatenate_modalities
from scripts.server.run_ccre_frozen_probe_matrix import build_jobs
from tasks.entex.prepare import fingerprint
from tasks.entex.probe import check_splits
from tasks.transfer.locus_features import FrozenLocusFeatures
from tasks.transfer.regression_probe import regression_metrics, select_ridge
from tasks.transfer.traitgym import verified_fingerprint, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "preparation-dir", "mapping-dir", "manifest", "feature-cache", "sequence-cache",
                 "topology-control-cache", "topology-cache-root", "results-root", "out-root"]:
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--folds", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--contexts", nargs="+", choices=["strict", "1hop"])
    args = ap.parse_args()
    plan = json.loads(args.config.read_text())
    config = json.loads(Path(plan["reference_config"]).read_text())
    manuscript = json.loads(Path(config["manuscript_config"]).read_text())
    locus_path = args.preparation_dir / "loci.parquet"
    source = verified_fingerprint(locus_path, plan["loci_sha256"])
    mapping = json.loads((args.mapping_dir / "mapping_qc.json").read_text())
    if (mapping["source_loci"]["sha256"] != source["sha256"]
            or mapping["source_graph"]["sha256"] != config["full_segments_sha256"]
            or mapping["fraction_mapped"] != 1):
        raise ValueError("Mapping provenance or full coverage differs")
    loci = pd.read_parquet(locus_path)
    overlaps = pd.read_parquet(args.mapping_dir / "overlaps.parquet")
    check_splits(loci, manuscript["rotating_chromosome_folds"])
    if (len(loci) != plan["n_loci"] or (loci.n_observed_donors < 1).any()
            or not np.isfinite(loci[plan["targets"]].to_numpy()).all()):
        raise ValueError("Outcome universe differs or contains unmeasured loci")
    features = FrozenLocusFeatures(loci, overlaps, config, args.feature_cache, args.sequence_cache,
        args.topology_control_cache, args.manifest, plan["topology_control_sha256"], plan["aggregation"])
    features.static["locus_log_length"] = np.log1p(loci.end-loci.start).to_numpy()[:, None]
    jobs = build_jobs(manuscript, seeds=set(args.seeds) if args.seeds else None,
                      contexts=set(args.contexts) if args.contexts else None)
    jobs = [j for j in jobs if not args.folds or j.fold in args.folds]
    if not jobs:
        raise ValueError("No jobs selected")
    args.out_root.mkdir(parents=True, exist_ok=False)
    receipt = dict(status="running", plan=plan, jobs=[asdict(j) for j in jobs], completed_runs=0,
        command=sys.argv, planned_runs=len(jobs), encoder_training=False, sources=[source, *features.sources,
        fingerprint(args.config), fingerprint(args.preparation_dir / "qc.json"),
        fingerprint(args.mapping_dir / "mapping_qc.json"), fingerprint(args.mapping_dir / "overlaps.parquet")],
        implementation=fingerprint(Path(__file__)), n_excluded=0, no_test_selection=True)
    write_json(args.out_root / "status.json", receipt)
    try:
        for job in jobs:
            topology, audit = features.topology(job, args.topology_cache_root, args.results_root)
            components = dict(features.static, frozen_pangenomefm=topology)
            matrices = build_modality_factorial(components, include_external_sequence=True, include_topology_control=True)
            if set(matrices) & set(plan["custom_feature_sets"]):
                raise ValueError("Do not overwrite existing feature definitions")
            matrices.update({k: concatenate_modalities(components, names) for k, names in plan["custom_feature_sets"].items()})
            test = loci.chrom.isin(job.test).to_numpy()
            val = loci.chrom.isin(job.validation).to_numpy()
            train = ~(test | val)
            if any(mask.sum() < 10 for mask in [train, val, test]):
                raise ValueError("Insufficient chromosome-fold support")
            output, predictions, selection = [], [], []
            for target in plan["targets"]:
                y = loci[target].to_numpy(float)
                if any(np.std(y[mask]) <= 0 for mask in [train, val, test]):
                    raise ValueError("Constant regression target in a partition")
                for name in plan["feature_sets"]:
                    if name == "train_median":
                        predicted = np.full(test.sum(), np.median(y[train]))
                        selected = dict(alpha=np.nan, validation_mae=float(np.mean(np.abs(y[val]-np.median(y[train])))))
                    else:
                        x = matrices[name]
                        model, selected, candidates = select_ridge(x[train], y[train], x[val], y[val], plan["ridge_alphas"])
                        predicted = model.predict(x[test])
                        selection.extend(dict(target=target, feature_set=name, **item) for item in candidates)
                    common = dict(task=plan["task"], target=target, fold=job.fold, seed=job.seed, context=job.closure, feature_set=name)
                    output.append(dict(**common, n_train=int(train.sum()), n_val=int(val.sum()), n_test=int(test.sum()),
                        **selected, **regression_metrics(y[test], predicted)))
                    p = loci.loc[test, ["locus_id", "chrom", "start", "end", "n_observed_donors"]].copy()
                    p = p.assign(y_true=y[test], prediction=predicted, **common)
                    predictions.append(p)
            out = args.out_root / job.fold / f"seed_{job.seed}" / job.closure
            out.mkdir(parents=True, exist_ok=False)
            pd.DataFrame(output).to_csv(out / "metrics.csv", index=False)
            pd.DataFrame(selection).to_csv(out / "validation_selection.csv", index=False)
            pd.concat(predictions, ignore_index=True).to_parquet(out / "predictions.parquet", index=False)
            write_json(out / "audit.json", dict(status="complete", topology=audit,
                predictions=fingerprint(out / "predictions.parquet"), n_excluded=0, feature_coverage=1.,
                test_chromosomes=list(job.test), validation_chromosomes=list(job.validation),
                target_unit="one genomic locus; observed-donor aggregation only; equal locus weights"))
            receipt["completed_runs"] += 1
            write_json(args.out_root / "status.json", receipt)
            print(f"Completed {receipt['completed_runs']}/{len(jobs)}: {job.name}", flush=True)
        receipt["status"] = "complete"
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.out_root / "status.json", receipt)


if __name__ == "__main__":
    main()

"""Run P0 with the unmodified manuscript frozen extraction and logistic probe."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from evaluation.modality_factorial import (
    build_modality_factorial,
    load_frozen_node_embedding_cache,
)
from scripts.server.run_ccre_frozen_probe_fold import (
    evaluate_feature_sets,
    validate_checkpoint_holdout,
)
from scripts.server.run_ccre_frozen_probe_matrix import build_jobs, checkpoint_for
from tasks.ccre.embedding_baseline import _extract_embeddings
from tasks.entex.mapping import aggregate_features
from tasks.entex.prepare import fingerprint
from tasks.entex.cache import cached_topology
from tasks.entex.sensitivity import match_exposure

FEATURES = [
    "coordinate",
    "sequence_kmer",
    "frozen_sequence_fm",
    "frozen_pangenomefm",
    "coordinate_plus_frozen_sequence_fm",
    "coordinate_plus_frozen_pangenomefm",
    "coordinate_plus_frozen_sequence_fm_plus_frozen_pangenomefm",
]


def complete_loci(
    loci: pd.DataFrame, overlaps: pd.DataFrame, available: set[int]
) -> pd.DataFrame:
    bad = set(overlaps.loc[~overlaps.segid.isin(available), "locus_id"])
    return loci.loc[
        loci.locus_id.isin(overlaps.locus_id) & ~loci.locus_id.isin(bad)
    ].copy()


def check_splits(loci: pd.DataFrame, folds: list[dict]) -> None:
    tests = [chrom for fold in folds for chrom in fold["test"]]
    if len(tests) != len(set(tests)):
        raise ValueError("Test chromosomes overlap between folds")
    if not set(loci.chrom).issubset(tests):
        raise ValueError("Loci outside manuscript chromosome folds")
    for fold in folds:
        if set(fold["test"]) & set(fold["validation"]):
            raise ValueError("Test/validation chromosome overlap")
    if loci.locus_id.duplicated().any():
        raise ValueError("P0 requires exactly one row per locus")
    if loci.groupby("locus_id").chrom.nunique().max() != 1:
        raise ValueError("Locus appears on multiple chromosomes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/entex_v1.json"))
    for name in [
        "loci",
        "mapping-dir",
        "full-segments",
        "manifest",
        "feature-cache",
        "sequence-cache",
        "results-root",
        "out-root",
    ]:
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument(
        "--sensitivity",
        choices=["primary", "exposure_matched", "h3k27ac", "ctcf"],
        default="primary",
    )
    ap.add_argument("--task", choices=["p0", "p1", "p2"], default="p0")
    ap.add_argument("--subtask")
    ap.add_argument("--measurements", type=Path)
    ap.add_argument("--cache-all-reference-targets", action="store_true")
    ap.add_argument("--topology-cache-root", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--folds", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--contexts", nargs="+", choices=["strict", "1hop"])
    ap.add_argument("--aggregation", choices=["mean", "length_weighted"])
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    manuscript = json.loads(Path(config["manuscript_config"]).read_text())
    # Check every resource before expensive embedding extraction.
    inputs = {
        name: fingerprint(getattr(args, name))
        for name in [
            "loci",
            "full_segments",
            "manifest",
            "feature_cache",
            "sequence_cache",
        ]
    }
    if inputs["full_segments"]["sha256"] != config["full_segments_sha256"]:
        raise ValueError("Graph differs from exact manuscript resource")
    mapping_audit = json.loads((args.mapping_dir / "mapping_qc.json").read_text())
    if mapping_audit["source_graph"]["sha256"] != config["full_segments_sha256"]:
        raise ValueError("Mapping and graph resource mismatch")
    if mapping_audit["source_loci"]["sha256"] != inputs["loci"]["sha256"]:
        raise ValueError("Mapping was built for different loci")
    if mapping_audit["fraction_mapped"] < config["minimum_mapping"]:
        raise ValueError("Mapping below prespecified gate")
    loci = pd.read_parquet(args.loci)
    if args.task == "p1":
        if args.sensitivity != "primary" or not args.subtask:
            raise ValueError("P1 requires --subtask tissue and primary sensitivity")
        if (
            "task" not in loci
            or "subtask" not in loci
            or not loci.task.eq("p1").all()
            or not loci.subtask.eq(args.subtask).all()
        ):
            raise ValueError("P1 tissue metadata mismatch")
    elif args.task == "p2":
        if (
            args.sensitivity != "primary"
            or args.subtask not in {"ctcf", "h3k27ac"}
            or args.measurements is None
        ):
            raise ValueError("P2 requires assay subtask and measurement cache")
        inputs["measurements"] = fingerprint(args.measurements)
        measurements = pd.read_parquet(args.measurements)
        if (
            not measurements.task.eq("p2").all()
            or not measurements.subtask.eq(args.subtask).all()
        ):
            raise ValueError("P2 assay metadata mismatch")
        if set(measurements.locus_id) != set(loci.locus_id):
            raise ValueError("Measurement/locus universe mismatch")
        coords = measurements[["locus_id", "chrom", "start", "end"]].drop_duplicates()
        if len(coords) != len(loci) or not coords.merge(
            loci, on=["locus_id", "chrom", "start", "end"]
        ).shape[0] == len(loci):
            raise ValueError("Measurement/locus coordinates mismatch")
    elif "task" in loci:
        raise ValueError("P0 cannot consume a different task dataset")
    if args.sensitivity in {"h3k27ac", "ctcf"}:
        if "sensitivity" not in loci or not loci.sensitivity.eq(args.sensitivity).all():
            raise ValueError(
                "Assay sensitivity requires its assay-specific prepared labels"
            )
    elif "sensitivity" in loci:
        raise ValueError("Assay dataset cannot be labeled as primary/exposure matched")
    check_splits(loci, manuscript["rotating_chromosome_folds"])
    loci["example_id"] = np.arange(len(loci))
    overlaps = pd.read_parquet(args.mapping_dir / "overlaps.parquet")
    if args.task == "p2" and (
        overlaps.locus_id.duplicated().any() or not overlaps.overlap_bp.eq(1).all()
    ):
        raise ValueError("P2 requires exactly one containing segment per mapped SNV")
    s_ids, s_values, s_audit = load_frozen_node_embedding_cache(args.sequence_cache)
    # Merged caches contain shard provenance; validate each original shard audit.
    expected_audit = Path(str(args.sequence_cache) + ".audit.json")
    candidates = [s_audit]
    if "source_shards" in s_audit:
        candidates = []
        for shard in s_audit["source_shards"]:
            sidecar = args.sequence_cache.parent / (
                Path(shard["path"]).name + ".audit.json"
            )
            if fingerprint(sidecar)["sha256"] != shard["audit_sha256"]:
                raise ValueError("NT shard audit checksum mismatch")
            candidates.append(json.loads(sidecar.read_text()))
    if not candidates or any(
        a.get("model_name") != config["nt_model"]
        or a.get("resolved_revision") != config["nt_revision"]
        or a.get("full_segments_sha256") != config["full_segments_sha256"]
        or a.get("maximum_token_length") != config["nt_max_length"]
        or a.get("maximum_raw_bases") != config["nt_max_bases"]
        or a.get("model_parameters_frozen") is not True
        or a.get("fine_tuned") is not False
        or a.get("pooling")
        != "mean final hidden state over non-special, non-padding tokens"
        for a in candidates
    ):
        raise ValueError(
            f"NT cache model/preprocessing provenance mismatch: {expected_audit}"
        )
    if s_audit.get("output_sha256") != inputs["sequence_cache"]["sha256"]:
        raise ValueError("NT cache checksum mismatch")
    cache_audit = json.loads(Path(str(args.feature_cache) + ".audit.json").read_text())
    if (
        cache_audit.get("inputs", {}).get("full_segments_sha256")
        != config["full_segments_sha256"]
    ):
        raise ValueError("C/K cache graph provenance mismatch")
    with np.load(args.feature_cache, allow_pickle=False) as cache:
        c_ids = cache["segid"].copy()
        c_values = {key: cache[key].copy() for key in ["coordinate", "sequence_kmer"]}
    jobs = build_jobs(
        manuscript,
        seeds=set(args.seeds) if args.seeds else None,
        contexts=set(args.contexts) if args.contexts else None,
    )
    jobs = [j for j in jobs if not args.folds or j.fold in args.folds]
    if not jobs:
        raise ValueError("No jobs selected")
    for job in jobs:
        out = args.out_root / job.fold / f"seed_{job.seed}" / job.closure
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"Refusing to overwrite run: {out}")
        checkpoint = checkpoint_for(args.results_root, job)
        holdout = validate_checkpoint_holdout(
            checkpoint, test_chrs=set(job.test), closure=job.closure, seed=job.seed
        )

        requested_targets = set(overlaps.segid.astype(int))
        if args.cache_all_reference_targets:
            requested_targets.update(c_ids.astype(int))

        def extract():
            return _extract_embeddings(
                checkpoint=checkpoint,
                manifest=args.manifest,
                full_segments=args.full_segments,
                labeled_segids=requested_targets,
                closure=job.closure,
                device_name=args.device,
                seed=job.seed,
                max_slices=None,
                canonical_conflict_policy="exclude",
                return_canonical_audit=True,
            )

        if args.topology_cache_root is None:
            frozen, _, extraction = extract()
        else:
            identity = dict(
                checkpoint_sha256=fingerprint(checkpoint)["sha256"],
                graph_sha256=inputs["full_segments"]["sha256"],
                manifest_sha256=inputs["manifest"]["sha256"],
                seed=job.seed,
                closure=job.closure,
                canonical_conflict_policy="exclude",
            )
            frozen, extraction = cached_topology(
                args.topology_cache_root
                / job.fold
                / f"seed_{job.seed}"
                / f"{job.closure}.npz",
                requested_targets,
                identity,
                extract,
            )
        coverage = {}
        for name, ids in [("C", c_ids), ("K", c_ids), ("S", s_ids), ("T", frozen)]:
            coverage[name] = len(complete_loci(loci, overlaps, set(ids))) / len(loci)
        selected = complete_loci(loci, overlaps, set(c_ids) & set(s_ids) & set(frozen))
        out.mkdir(parents=True)
        audit = dict(
            task=args.task,
            subtask=args.subtask or args.sensitivity,
            sensitivity=args.sensitivity,
            sensitivity_definitions=config["sensitivities"],
            fold=job.fold,
            seed=job.seed,
            closure=job.closure,
            inputs=inputs,
            checkpoint=fingerprint(checkpoint),
            holdout=holdout,
            extraction=extraction,
            feature_coverage=coverage,
            joint_feature_coverage=len(selected) / len(loci),
            n_excluded=len(loci) - len(selected),
            aggregation=args.aggregation or config["aggregation"],
            encoder_training=False,
            status="coverage_checked",
        )
        (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        loci.loc[~loci.locus_id.isin(selected.locus_id)].to_parquet(
            out / "excluded_loci.parquet", index=False
        )
        if len(selected) / len(loci) < config["minimum_feature_coverage"]:
            raise ValueError(f"Feature coverage below gate; inspect {out}")
        if args.sensitivity == "exposure_matched":
            before_matching = selected.copy()
            selected, balance = match_exposure(
                selected, config["sensitivities"]["matching_seed"]
            )
            balance.to_csv(out / "exposure_balance.csv", index=False)
            before_matching.loc[
                ~before_matching.locus_id.isin(selected.locus_id)
            ].to_parquet(out / "exposure_excluded_loci.parquet", index=False)
            audit["n_exposure_excluded"] = len(before_matching) - len(selected)
            audit["n_exposure_selected"] = len(selected)
        selected = selected.reset_index(drop=True)
        method = audit["aggregation"]
        components = {
            key: aggregate_features(selected, overlaps, c_ids, value, method)
            for key, value in c_values.items()
        }
        components["frozen_sequence_fm"] = aggregate_features(
            selected, overlaps, s_ids, s_values, method
        )
        t_ids = np.array(sorted(frozen), dtype=np.int64)
        components["frozen_pangenomefm"] = aggregate_features(
            selected, overlaps, t_ids, np.stack([frozen[i] for i in t_ids]), method
        )
        factorial = build_modality_factorial(components, include_external_sequence=True)
        if args.task == "p2":
            from tasks.entex.measurement_probe import evaluate_measurements

            retained = measurements.locus_id.isin(selected.locus_id)
            audit["measurement_coverage"] = float(retained.mean())
            audit["measurement_exclusions"] = int((~retained).sum())
            audit["probe_optimization"] = (
                "count-equivalent locus/label fitting; scaler occurrence weights; explicit original-row class weights"
            )
            if retained.mean() < config["minimum_feature_coverage"]:
                raise ValueError("P2 measurement coverage below gate")
            metrics, _, predictions = evaluate_measurements(
                loci=selected,
                measurements=measurements,
                features={k: factorial[k] for k in FEATURES},
                test_chrs=set(job.test),
                val_chrs=set(job.validation),
                seed=job.seed,
            )
        else:
            metrics, _, predictions = evaluate_feature_sets(
                segids=selected.example_id.to_numpy(),
                chromosomes=selected.chrom.to_numpy(),
                labels=selected.label.to_numpy(),
                features={k: factorial[k] for k in FEATURES},
                test_chrs=set(job.test),
                val_chrs=set(job.validation),
                seed=job.seed,
            )
        for frame in [metrics, predictions]:
            for key, value in [
                ("task", args.task),
                ("subtask", args.subtask or args.sensitivity),
                ("fold", job.fold),
                ("seed", job.seed),
                ("closure", job.closure),
                ("context", job.closure),
            ]:
                frame[key] = value
        metrics["balanced_accuracy"] = [
            balanced_accuracy_score(p.y_true, p.y_pred)
            for key in metrics.feature_set
            for p in [predictions.loc[predictions.feature_set == key]]
        ]
        metrics["n_val"] = metrics.n_validation
        metrics["positive_prevalence"] = metrics.positive_fraction
        metrics["normalized_ap"] = (metrics.auprc - metrics.positive_prevalence) / (
            1 - metrics.positive_prevalence
        )
        metrics.loc[metrics.positive_prevalence.isin([0, 1]), "normalized_ap"] = np.nan
        predictions = predictions.rename(columns={"segid": "example_id"}).merge(
            selected[["example_id", "locus_id", "chrom", "start", "end"]],
            on="example_id",
            validate="many_to_one",
        )
        metrics.to_csv(out / "metrics.csv", index=False)
        predictions.to_parquet(out / "predictions.parquet", index=False)
        selected.to_parquet(out / "feature_universe.parquet", index=False)
        audit["status"] = "complete"
        (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()

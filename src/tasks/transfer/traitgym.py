"""Frozen TraitGym locus priors using manuscript mapping, features and probes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from evaluation.modality_factorial import (build_modality_factorial, concatenate_modalities,
    factorial_feature_access, load_frozen_node_embedding_cache)
from scripts.server.audit_traitgym_coverage import normalize
from scripts.server.run_ccre_frozen_probe_fold import evaluate_feature_sets, validate_checkpoint_holdout
from scripts.server.run_ccre_frozen_probe_matrix import build_jobs, checkpoint_for
from tasks.entex.mapping import aggregate_features
from tasks.entex.prepare import fingerprint
from tasks.transfer.selected_probe import evaluate_selected, completion_summary


IDENTITY = ["variant_id", "locus_id", "chrom", "start", "end", "ref", "alt", "label", "match_group"]


def validate_examples(examples: pd.DataFrame, source: pd.DataFrame,
                      overlaps: pd.DataFrame, folds: list[dict]) -> dict:
    """Reject dropped controls, allele collapse, and genomic/group split leakage."""
    expected = normalize(source)
    for frame in [examples, expected]:
        if frame.variant_id.duplicated().any() or frame[IDENTITY].isna().any().any():
            raise ValueError("Missing or duplicated variant identities")
    pd.testing.assert_frame_equal(
        examples[IDENTITY].sort_values("variant_id").reset_index(drop=True),
        expected[IDENTITY].sort_values("variant_id").reset_index(drop=True), check_dtype=False)
    groups = examples.groupby("match_group").agg(n=("label", "size"), positives=("label", "sum"),
                                                chromosomes=("chrom", "nunique"))
    if not (groups.n.eq(10) & groups.positives.eq(1) & groups.chromosomes.eq(1)).all():
        raise ValueError("Require intact single-chromosome groups with one positive and nine controls")
    tests = [c for fold in folds for c in fold["test"]]
    if len(tests) != len(set(tests)) or not set(examples.chrom) <= set(tests):
        raise ValueError("Chromosomes outside non-overlapping manuscript test folds")
    for fold in folds:
        if set(fold["test"]) & set(fold["validation"]):
            raise ValueError("Test and validation chromosomes overlap")
        test, val = examples.chrom.isin(fold["test"]), examples.chrom.isin(fold["validation"])
        for mask in [test, val, ~(test | val)]:
            if examples.loc[mask, "label"].nunique() != 2:
                raise ValueError("Every train/validation/test partition needs both classes")
    if (overlaps.locus_id.duplicated().any() or set(overlaps.locus_id) != set(examples.locus_id)
            or not overlaps.overlap_bp.eq(1).all()):
        raise ValueError("Require exactly one containing segment for every original SNV locus")
    mapped = examples.merge(overlaps, on="locus_id", validate="many_to_one")
    if mapped.groupby("segid").chrom.nunique().max() != 1:
        raise ValueError("A canonical segment occurs on multiple chromosomes")
    collision_ids = set(mapped.groupby("segid").label.nunique().loc[lambda v: v > 1].index)
    return dict(n=len(examples), unique_loci=examples.locus_id.nunique(),
        positives=int(examples.label.sum()), negatives=int((examples.label == 0).sum()),
        positive_prevalence=float(examples.label.mean()), match_groups=len(groups),
        n_segments=mapped.segid.nunique(), mixed_label_segments=len(collision_ids),
        variants_in_mixed_label_segments=int(mapped.segid.isin(collision_ids).sum()),
        groups_with_identical_segment=int(mapped.groupby("match_group").segid.nunique().eq(1).sum()),
        chromosome_counts=examples.chrom.value_counts().to_dict(), n_excluded=0)


def align_component(examples, overlaps, ids, values):
    """Pool unique loci with the existing adapter, then restore variant rows."""
    loci = examples[["locus_id", "chrom", "start", "end"]].drop_duplicates().reset_index(drop=True)
    pooled = aggregate_features(loci, overlaps, ids, values, "mean")
    positions = pd.Series(np.arange(len(loci)), index=loci.locus_id)
    return pooled[examples.locus_id.map(positions).to_numpy(int)]


def weighted_chromosome_ap(per_chromosome: pd.DataFrame) -> float:
    if per_chromosome.auprc.isna().any() or per_chromosome.chromosome.duplicated().any():
        raise ValueError("Require one defined AP per evaluated chromosome")
    return float(np.average(per_chromosome.auprc, weights=per_chromosome.n))


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def verified_fingerprint(path: Path, expected: str) -> dict:
    actual = fingerprint(path)
    if actual["sha256"] != expected:
        raise ValueError(f"Stale or mismatched artifact: {path}")
    return actual


def validate_nt_provenance(path: Path, audit: dict, config: dict, output_sha256: str) -> list[dict]:
    """Follow the manuscript merged cache's per-shard preprocessing receipts."""
    if audit.get("output_sha256") != output_sha256:
        raise ValueError("NT cache checksum mismatch")
    candidates, sources = [audit], []
    if "source_shards" in audit:
        candidates = []
        for shard in audit["source_shards"]:
            sidecar = path.parent / (Path(shard["path"]).name + ".audit.json")
            sources.append(verified_fingerprint(sidecar, shard["audit_sha256"]))
            candidates.append(json.loads(sidecar.read_text()))
    required = dict(model_name=config["nt_model"], resolved_revision=config["nt_revision"],
        full_segments_sha256=config["full_segments_sha256"],
        maximum_token_length=config["nt_max_length"], maximum_raw_bases=config["nt_max_bases"],
        model_parameters_frozen=True, fine_tuned=False,
        pooling="mean final hidden state over non-special, non-padding tokens")
    if not candidates or any(any(a.get(k) != v for k, v in required.items()) for a in candidates):
        raise ValueError("NT model, preprocessing or frozen-state provenance mismatch")
    return sources


def author_variant_scores(path: Path, expected_sha256: str, original: pd.DataFrame,
                          prepared: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Honor the pinned publisher's positional score contract, then join by variant."""
    receipt = verified_fingerprint(path, expected_sha256)
    scores = pd.read_parquet(path)
    if (list(scores) != ["score"] or len(scores) != len(original)
            or not scores.index.equals(pd.RangeIndex(len(original)))
            or not original.index.equals(scores.index) or not np.isfinite(scores.score).all()):
        raise ValueError("Author scores do not follow the original finite row-order contract")
    source = normalize(original)
    indexed = pd.Series(scores.score.to_numpy(), index=source.variant_id)
    ordered = prepared.variant_id.map(indexed).to_numpy()
    if not np.isfinite(ordered).all() or set(prepared.variant_id) != set(source.variant_id):
        raise ValueError("Author score join changed the original variant universe")
    return np.column_stack([ordered, np.abs(ordered)]).astype(np.float32), receipt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/traitgym_locus_prior_20260927.json"))
    for name in ["source-dir", "mapping-dir", "topology-audit", "topology-cache-root",
                 "feature-cache", "sequence-cache", "topology-control-cache", "results-root", "out-root"]:
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--folds", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--contexts", nargs="+", choices=["strict", "1hop"])
    ap.add_argument("--variant-score-dir", type=Path)
    args = ap.parse_args()
    plan = json.loads(args.config.read_text())
    config = json.loads(Path(plan["reference_config"]).read_text())
    manuscript = json.loads(Path(config["manuscript_config"]).read_text())
    mapping = json.loads((args.mapping_dir / "audit.json").read_text())
    topology = json.loads(args.topology_audit.read_text())
    source = json.loads((args.source_dir / "source.json").read_text())
    if (source["revision"] != plan["dataset_revision"] or source["dataset"] != "songlab/TraitGym"
            or mapping["graph"]["sha256"] != config["full_segments_sha256"]
            or mapping.get("reference_verification", {}).get("mismatches") != 0
            or topology.get("status") != "complete" or not topology.get("all_variants_covered")):
        raise ValueError("Require pinned source, exact graph and completed REF/topology coverage audits")
    audited_files = {Path(item["path"]).name: item["sha256"] for item in topology["sources"]}
    audited_caches = {item["cache"]["path"]: item for item in topology["caches"]}
    sources = [fingerprint(args.config), fingerprint(args.topology_audit),
               fingerprint(Path(config["manuscript_config"])), fingerprint(args.source_dir / "source.json")]
    sources.append(verified_fingerprint(args.mapping_dir / "audit.json", audited_files["audit.json"]))
    datasets, qc = {}, {}
    for name in plan["datasets"]:
        example_path = args.mapping_dir / (name + "_examples.parquet")
        overlap_path = args.mapping_dir / (name + "_overlaps.parquet")
        for p in [example_path, overlap_path]:
            sources.append(verified_fingerprint(p, audited_files[p.name]))
        source_path = args.source_dir / (name + "_matched_9.parquet")
        receipt = next(item for item in mapping["datasets"] if item["dataset"] == name)
        sources.append(verified_fingerprint(source_path, receipt["source"]["sha256"]))
        examples, overlaps = pd.read_parquet(example_path), pd.read_parquet(overlap_path)
        qc[name] = validate_examples(examples, pd.read_parquet(source_path), overlaps,
                                     manuscript["rotating_chromosome_folds"])
        examples = examples.reset_index(drop=True).assign(example_id=np.arange(len(examples)))
        datasets[name] = (examples, overlaps)

    # Static cache arrays are loaded once and immediately restricted to task loci.
    ck_audit = json.loads(Path(str(args.feature_cache) + ".audit.json").read_text())
    if (ck_audit.get("status") != "complete"
            or ck_audit["inputs"]["full_segments_sha256"] != config["full_segments_sha256"]):
        raise ValueError("C/K cache provenance mismatch")
    sources.append(verified_fingerprint(args.feature_cache, ck_audit["output_sha256"]))
    sources.append(fingerprint(Path(str(args.feature_cache) + ".audit.json")))
    static = {name: {} for name in datasets}
    if plan.get("variant_scores"):
        if args.variant_score_dir is None:
            raise ValueError("This secondary plan requires its pinned author allele scores")
        for name, (examples, _) in datasets.items():
            spec = plan["variant_scores"][name]
            matrix, score_source = author_variant_scores(args.variant_score_dir / spec["filename"],
                spec["sha256"], pd.read_parquet(args.source_dir / (name + "_matched_9.parquet")), examples)
            static[name]["variant_llr"] = matrix
            sources.append(score_source)
    elif args.variant_score_dir is not None:
        raise ValueError("Allele scores require an explicitly separate analysis plan")
    with np.load(args.feature_cache, allow_pickle=False) as cache:
        for key in ["coordinate", "sequence_kmer"]:
            for name, (examples, overlaps) in datasets.items():
                static[name][key] = align_component(examples, overlaps, cache["segid"], cache[key])
    for key, path in [("frozen_sequence_fm", args.sequence_cache),
                      ("topology_control", args.topology_control_cache)]:
        ids, values, audit = load_frozen_node_embedding_cache(path)
        source_fp = fingerprint(path)
        sources.extend([source_fp, fingerprint(Path(str(path) + ".audit.json"))])
        if key == "frozen_sequence_fm":
            sources.extend(validate_nt_provenance(path, audit, config, source_fp["sha256"]))
        elif (audit.get("kind") != "handcrafted_topology_control" or values.shape[1] != 14
                or audit.get("full_segments_sha256") != config["full_segments_sha256"]):
            raise ValueError("Expected the existing 14-statistic H control")
        for name, (examples, overlaps) in datasets.items():
            static[name][key] = align_component(examples, overlaps, ids, values)
        del ids, values
    jobs = build_jobs(manuscript, seeds=set(args.seeds) if args.seeds else None,
                      contexts=set(args.contexts) if args.contexts else None)
    jobs = [j for j in jobs if not args.folds or j.fold in args.folds]
    if not jobs:
        raise ValueError("No jobs selected")
    args.out_root.mkdir(parents=True, exist_ok=False)
    receipt = dict(status="running", plan=plan, sources=sources, command=sys.argv,
        implementation=fingerprint(Path(__file__)), jobs=[asdict(j) for j in jobs],
        datasets=plan["datasets"], completed_runs=0, planned_runs=len(jobs) * len(datasets),
        encoder_training=False, evaluation_partition="test", feature_coverage="100% C/K/S/H required",
        scope="full_matrix" if len(jobs) == len(build_jobs(manuscript)) else "partial_matrix")
    if plan.get("selected_probe"):
        receipt["probe_implementation"] = fingerprint(Path(__file__).with_name("selected_probe.py"))
        receipt["runtime"] = {name: version(name) for name in ["numpy", "scipy", "scikit-learn", "torch", "pandas"]}
    write_json(args.out_root / "status.json", receipt)
    write_json(args.out_root / "qc.json", qc)
    try:
        for job in jobs:
            path = args.topology_cache_root / job.fold / f"seed_{job.seed}" / (job.closure + ".npz")
            cache_receipt = audited_caches[str(path)]
            cache_fp = verified_fingerprint(path, cache_receipt["cache"]["sha256"])
            verified_fingerprint(Path(str(path) + ".audit.json"), cache_receipt["sidecar"]["sha256"])
            checkpoint = checkpoint_for(args.results_root, job)
            holdout = validate_checkpoint_holdout(checkpoint, test_chrs=set(job.test),
                val_chrs=set(job.validation), closure=job.closure, seed=job.seed)
            ids, values, audit = load_frozen_node_embedding_cache(path)
            identity = dict(checkpoint_sha256=fingerprint(checkpoint)["sha256"],
                graph_sha256=config["full_segments_sha256"], manifest_sha256=topology["manifest"]["sha256"],
                seed=job.seed, closure=job.closure, canonical_conflict_policy="exclude")
            if (audit.get("identity") != identity or audit.get("output_sha256") != cache_fp["sha256"]
                    or audit.get("model_parameters_frozen") is not True):
                raise ValueError("Topology cache and held-out checkpoint provenance mismatch")
            for name, (examples, overlaps) in datasets.items():
                components = dict(static[name], frozen_pangenomefm=align_component(examples, overlaps, ids, values))
                matrices = build_modality_factorial(components, include_external_sequence=True,
                                                     include_topology_control=True)
                custom = plan.get("custom_feature_sets", {})
                if set(custom) & set(matrices):
                    raise ValueError("Custom features must not replace manuscript feature meanings")
                matrices.update({k: concatenate_modalities(components, names) for k, names in custom.items()})
                access = {**factorial_feature_access(), **{k: " + ".join(names) for k, names in custom.items()}}
                arguments = dict(segids=examples.example_id.to_numpy(), chromosomes=examples.chrom.to_numpy(),
                    labels=examples.label.to_numpy(), test_chrs=set(job.test), val_chrs=set(job.validation), seed=job.seed)
                selection = validation_predictions = None
                if plan.get("selected_probe"):
                    metrics, per_chromosome, predictions, selection, validation_predictions = evaluate_selected(
                        **arguments, features=matrices, plan=plan)
                else:
                    metrics, per_chromosome, predictions = evaluate_feature_sets(
                        **arguments, features={k: matrices[k] for k in plan["feature_sets"]},
                        probe_max_iter=plan["probe_max_iter"], feature_access=access)
                metrics["balanced_accuracy"] = [balanced_accuracy_score(p.y_true, p.y_pred)
                    for feature in metrics.feature_set for p in [predictions.loc[predictions.feature_set.eq(feature)]]]
                metrics["chromosome_weighted_auprc"] = [weighted_chromosome_ap(
                    per_chromosome.loc[per_chromosome.feature_set.eq(feature)]) for feature in metrics.feature_set]
                metrics["normalized_ap"] = (metrics.auprc - metrics.positive_fraction) / (1 - metrics.positive_fraction)
                metrics["positive_prevalence"], metrics["n_val"] = metrics.positive_fraction, metrics.n_validation
                predictions = predictions.rename(columns={"segid": "example_id"}).merge(
                    examples[["example_id", *IDENTITY]], on="example_id", validate="many_to_one")
                frames = [metrics, per_chromosome, predictions]
                if validation_predictions is not None:
                    validation_predictions = validation_predictions.rename(columns={"segid": "example_id"}).merge(
                        examples[["example_id", *IDENTITY]], on="example_id", validate="many_to_one")
                    frames.append(validation_predictions)
                for frame in frames:
                    for key, value in dict(task=plan["task"], dataset=name, fold=job.fold,
                                           seed=job.seed, context=job.closure).items():
                        frame[key] = value
                out = args.out_root / name / job.fold / f"seed_{job.seed}" / job.closure
                out.mkdir(parents=True, exist_ok=False)
                metrics.to_csv(out / "metrics.csv", index=False)
                per_chromosome.to_csv(out / "per_chromosome.csv", index=False)
                predictions.to_parquet(out / "predictions.parquet", index=False)
                extra = {}
                if selection is not None:
                    selection.to_csv(out / "validation_selection.csv", index=False)
                    validation_predictions.to_parquet(out / "validation_predictions.parquet", index=False)
                    extra = {key: fingerprint(out / filename) for key, filename in [
                        ("validation_selection", "validation_selection.csv"),
                        ("validation_predictions", "validation_predictions.parquet")]}
                write_json(out / "audit.json", dict(status="complete", holdout=holdout,
                    identity=identity, feature_coverage={k: 1.0 for k in components}, n_excluded=0,
                    **completion_summary(metrics), **extra,
                    predictions=fingerprint(out / "predictions.parquet"), topology_cache=cache_fp,
                    interpretation=plan["interpretation"]))
                receipt["completed_runs"] += 1
                write_json(args.out_root / "status.json", receipt)
                print(f"Completed {receipt['completed_runs']}/{receipt['planned_runs']}: {name} {job.name}", flush=True)
            del ids, values
        receipt["status"] = "complete"
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.out_root / "status.json", receipt)


if __name__ == "__main__":
    main()

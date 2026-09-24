"""Reconstruct original HGSVC probes, verify them, then score HG008 held-out chromosomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from evaluation.calibration import apply_temperature, fit_temperature
from evaluation.modality_factorial import load_frozen_node_embedding_cache
from scripts.server.run_ccre_frozen_probe_fold import (
    binary_metrics,
    validate_checkpoint_holdout,
)
from scripts.server.run_ccre_frozen_probe_matrix import build_jobs, checkpoint_for
from scripts.server.run_sv_frozen_probe_fold import (
    build_pair_matrix,
    complete_feature_mask,
)
from tasks.ccre.aligned_baselines import _fit_model
from tasks.ccre.binary import _choose_threshold
from tasks.entex.cache import cached_topology
from tasks.entex.prepare import fingerprint


def masks(chromosomes: np.ndarray, test: set[str], validation: set[str]):
    if test & validation:
        raise ValueError("Overlapping train/validation/test chromosomes")
    is_test = np.isin(chromosomes, sorted(test))
    is_val = np.isin(chromosomes, sorted(validation))
    return ~(is_test | is_val), is_val, is_test


def fit_original(
    matrix: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    seed: int,
):
    """HG008 labels are deliberately absent from this interface."""
    if np.any(train & val) or any(np.unique(labels[m]).size != 2 for m in [train, val]):
        raise ValueError("Invalid original-training partitions")
    model = _fit_model("logistic", seed)
    model.fit(matrix[train], labels[train])
    raw = model.predict_proba(matrix[val])[:, 1]
    temperature = fit_temperature(labels[val], raw)
    threshold = _choose_threshold(labels[val], apply_temperature(raw, temperature))
    return model, temperature, threshold


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", type=Path, default=Path("configs/downstream_transfer_v1.json")
    )
    ap.add_argument("--external-examples", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--context", choices=["strict", "1hop"], required=True)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    resource = json.loads(Path(config["resource_config"]).read_text())
    task = config["hg008"]
    jobs = build_jobs(
        json.loads(Path(resource["manuscript_config"]).read_text()),
        seeds={args.seed},
        contexts={args.context},
    )
    job = next(j for j in jobs if j.fold == args.fold)
    root = Path("server_workspace")
    graph = root / "data/processed/hprc_r2_sv/full_segments.csv.gz"
    manifest = root / "data/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv"
    feature_path = root / "data/processed/hprc_r2_ccre_screen_v4_features.npz"
    sequence_path = (
        root
        / "results/frozen_sequence_fm_cache_20260815/hprc_target_union_sequence_fm.npz"
    )
    train_path = Path(task["training_examples"])
    expected = {
        graph: resource["full_segments_sha256"],
        feature_path: "c4d646f52bd86178689654138a34f700bea604e2f778d261327260c0c4a61c68",
        sequence_path: "80934eb12b6eb01b0e5de5132a36d35e3cab2f51f7d72c13a254b642c2f791de",
        train_path: task["training_examples_sha256"],
    }
    inputs = {str(p): fingerprint(p) for p in expected}
    if any(inputs[str(p)]["sha256"] != sha for p, sha in expected.items()):
        raise ValueError("Original manuscript resource checksum mismatch")
    mapping_audit = json.loads(
        (args.external_examples.parent / "audit.json").read_text()
    )
    if mapping_audit["full_segments_sha256"] != resource["full_segments_sha256"]:
        raise ValueError("External mapping uses a different graph")
    if mapping_audit["mapped_fraction_of_eligible"] < resource["minimum_mapping"]:
        raise ValueError("HG008 mapping below gate")
    original = pd.read_csv(train_path)
    external = pd.read_csv(args.external_examples)
    if not external.binary_svtype_label.eq(external.svtype.eq("INS").astype(int)).all():
        raise ValueError("Incompatible external labels")
    external = external.loc[external.chrom.isin(job.test)].copy()
    if external.empty:
        raise ValueError("No external test records for fold")
    checkpoint = checkpoint_for(root / "results/full_multicohort_server_20260806", job)
    holdout = validate_checkpoint_holdout(
        checkpoint, test_chrs=set(job.test), closure=job.closure, seed=job.seed
    )
    required = set(
        pd.concat(
            [
                original.start_segid,
                original.end_segid,
                external.start_segid,
                external.end_segid,
            ]
        ).astype(int)
    )
    identity = dict(
        checkpoint_sha256=fingerprint(checkpoint)["sha256"],
        graph_sha256=resource["full_segments_sha256"],
        manifest_sha256=fingerprint(manifest)["sha256"],
        seed=job.seed,
        closure=job.closure,
        canonical_conflict_policy="exclude",
    )

    def no_extraction():
        raise FileNotFoundError(
            "Exact completed EN-TEx all-reference topology cache required"
        )

    frozen, extraction = cached_topology(
        Path("results/entex/v1/topology_cache_all_reference")
        / job.fold
        / f"seed_{job.seed}"
        / f"{job.closure}.npz",
        required,
        identity,
        no_extraction,
    )
    sid, sequence, _ = load_frozen_node_embedding_cache(sequence_path)
    with np.load(feature_path, allow_pickle=False) as stored:
        cid, coordinate = stored["segid"].copy(), stored["coordinate"].copy()
    available = set(cid) & set(sid) & set(frozen)
    keep = complete_feature_mask(
        original, embedded_segids=available, cached_segids=available
    )
    original = original.loc[keep].sort_values("example_id").reset_index(drop=True)
    complete = complete_feature_mask(
        external, embedded_segids=available, cached_segids=available
    )
    out = args.out_root / job.fold / f"seed_{job.seed}" / job.closure
    out.mkdir(parents=True, exist_ok=False)
    external.loc[~complete].to_csv(out / "excluded_external.csv", index=False)
    coverage = float(complete.mean())
    if coverage < resource["minimum_feature_coverage"]:
        raise ValueError(f"External feature coverage {coverage} below gate")
    external = external.loc[complete].reset_index(drop=True)
    tid = np.array(sorted(frozen))
    tvalues = np.stack([frozen[int(s)] for s in tid])
    components = [(cid, coordinate), (sid, sequence), (tid, tvalues)]
    old_pairs, new_pairs = [], []
    for ids, values in components:
        positions = {int(s): i for i, s in enumerate(ids)}
        old_pairs.append(
            build_pair_matrix(original, node_values=values, positions=positions)
        )
        new_pairs.append(
            build_pair_matrix(external, node_values=values, positions=positions)
        )
    train, val, test = masks(
        original.chrom.to_numpy(), set(job.test), set(job.validation)
    )
    labels = original.binary_svtype_label.to_numpy()
    original_dir = (
        Path(task["manuscript_result_root"])
        / job.fold
        / f"seed_{job.seed}"
        / job.closure
    )
    prior = pd.read_csv(original_dir / "test_predictions.csv.gz")
    result, predictions, regressions = [], [], []
    for index, name in enumerate(task["primary_feature_sets"]):
        print(
            f"Fitting original HGSVC: {job.fold} seed={job.seed} {job.closure} {name}",
            flush=True,
        )
        count = 2 if index == 0 else 3
        matrix = np.concatenate(old_pairs[:count], axis=1)
        model, temperature, threshold = fit_original(
            matrix, labels, train, val, job.seed
        )
        old_scores = apply_temperature(
            model.predict_proba(matrix[test])[:, 1], temperature
        )
        old_metric = binary_metrics(labels[test], old_scores, threshold)
        expected_predictions = prior.loc[prior.feature_set.eq(name)].sort_values(
            "example_id"
        )
        if not np.array_equal(
            expected_predictions.example_id, original.loc[test, "example_id"]
        ):
            raise ValueError("Reconstructed original test universe changed")
        if not np.array_equal(expected_predictions.y_true, labels[test]):
            raise ValueError("Reconstructed original labels changed")
        expected_metric = binary_metrics(
            expected_predictions.y_true.to_numpy(),
            expected_predictions.p_calibrated.to_numpy(),
            threshold,
        )
        error = old_metric["auprc"] - expected_metric["auprc"]
        regression = dict(
            feature_set=name,
            reconstructed_auprc=old_metric["auprc"],
            cached_auprc=expected_metric["auprc"],
            auprc_difference=error,
            max_score_error=float(
                np.max(
                    np.abs(old_scores - expected_predictions.p_calibrated.to_numpy())
                )
            ),
        )
        regressions.append(regression)
        pd.DataFrame(regressions).to_csv(
            out / "original_probe_regression.csv", index=False
        )
        if abs(error) > task["regression_auprc_tolerance"]:
            raise ValueError(
                "Reconstructed original probe fails manuscript regression; external evaluation stopped"
            )
        import joblib

        joblib.dump(
            dict(
                model=model,
                temperature=temperature,
                threshold=threshold,
                training_data_sha256=task["training_examples_sha256"],
                checkpoint=identity,
                hg008_label_access="none",
            ),
            out / f"{name}.joblib",
        )
        raw = model.predict_proba(np.concatenate(new_pairs[:count], axis=1))[:, 1]
        scores = apply_temperature(raw, temperature)
        for scope, mask in [
            ("all", np.ones(len(external), dtype=bool)),
            (
                "overlap_only",
                (
                    external.start_mapping_distance.eq(0)
                    & external.end_mapping_distance.eq(0)
                ).to_numpy(),
            ),
        ]:
            if not mask.any():
                continue
            y = external.binary_svtype_label.to_numpy()[mask]
            result.append(
                dict(
                    feature_set=name.removesuffix("_pair"),
                    scope=scope,
                    n_train=int(train.sum()),
                    n_val=int(val.sum()),
                    n_test=int(mask.sum()),
                    positive_prevalence=float(y.mean()),
                    balanced_accuracy=float(
                        balanced_accuracy_score(y, scores[mask] >= threshold)
                    ),
                    **binary_metrics(y, scores[mask], threshold),
                )
            )
        prediction = external.copy()
        prediction["feature_set"] = name.removesuffix("_pair")
        prediction["y_true"], prediction["p_raw"], prediction["p_calibrated"] = (
            external.binary_svtype_label,
            raw,
            scores,
        )
        prediction["threshold"], prediction["y_pred"] = (
            threshold,
            (scores >= threshold).astype(int),
        )
        predictions.append(prediction)
        del matrix, model
    metrics, predicted = pd.DataFrame(result), pd.concat(predictions)
    for frame in [metrics, predicted]:
        for k, v in dict(
            task="hg008",
            subtask="clonal_INS_vs_DEL",
            fold=job.fold,
            seed=job.seed,
            context=job.closure,
            closure=job.closure,
        ).items():
            frame[k] = v
    metrics.to_csv(out / "metrics.csv", index=False)
    predicted.to_parquet(out / "predictions.parquet", index=False)
    audit = dict(
        status="complete",
        inputs=inputs,
        external_examples=fingerprint(args.external_examples),
        checkpoint=identity,
        holdout=holdout,
        extraction=extraction,
        external_feature_coverage=coverage,
        n_external=len(external),
        original_training_source=str(train_path),
        encoder_training=False,
        hg008_training_or_calibration=False,
        probe="reconstructed original HGSVC logistic probe",
        original_probe_regression=regressions,
    )
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Refit a fixed original feature matrix to diagnose numerical probe repeatability.

This diagnostic never reads HG008 labels and never promotes a passing refit to
an external result. All prespecified thread/repeat settings are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_info, threadpool_limits

from evaluation.calibration import apply_temperature
from evaluation.modality_factorial import load_frozen_node_embedding_cache
from scripts.server.run_ccre_frozen_probe_fold import binary_metrics
from scripts.server.run_ccre_frozen_probe_matrix import build_jobs
from scripts.server.run_sv_frozen_probe_fold import (
    build_pair_matrix,
    complete_feature_mask,
)
from tasks.entex.analyze import BASE, FULL
from tasks.entex.prepare import fingerprint
from tasks.transfer.external_sv import fit_original, masks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", default="fold_d")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context", default="strict", choices=["strict", "1hop"])
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    config = json.loads(
        Path("configs/server_full_multicohort_20260806.json").read_text()
    )
    job = next(
        j
        for j in build_jobs(config)
        if (j.fold, j.seed, j.closure) == (args.fold, args.seed, args.context)
    )
    root = Path("server_workspace")
    examples_path = (
        root
        / "data/processed/hgsvc3_sv_breakpoint_examples_20260809/sv_breakpoint_examples.csv.gz"
    )
    sequence_path = (
        root
        / "results/frozen_sequence_fm_cache_20260815/hprc_target_union_sequence_fm.npz"
    )
    coordinate_path = root / "data/processed/hprc_r2_hgsvc3_sv_features_20260809.npz"
    topology_path = (
        Path("results/entex/v1/topology_cache_all_reference")
        / job.fold
        / f"seed_{job.seed}"
        / f"{job.closure}.npz"
    )
    tid, topology, topology_audit = load_frozen_node_embedding_cache(topology_path)
    if topology_audit["output_sha256"] != fingerprint(topology_path)["sha256"]:
        raise ValueError("Topology cache hash changed")
    sid, sequence, _ = load_frozen_node_embedding_cache(sequence_path)
    with np.load(coordinate_path) as data:
        cid, coordinates = data["segid"].copy(), data["coordinate"].copy()
    available = set(tid) & set(sid) & set(cid)
    examples = pd.read_csv(examples_path)
    complete = complete_feature_mask(
        examples, embedded_segids=available, cached_segids=available
    )
    examples = examples.loc[complete].sort_values("example_id").reset_index(drop=True)
    train, val, test = masks(
        examples.chrom.to_numpy(), set(job.test), set(job.validation)
    )
    labels = examples.binary_svtype_label.to_numpy()
    archived_path = (
        root
        / "results/hgsvc3_sv_sequence_fm_factorial_20260815"
        / job.fold
        / f"seed_{job.seed}"
        / job.closure
        / "test_predictions.csv.gz"
    )
    archived = pd.read_csv(archived_path)
    pairs = [
        build_pair_matrix(
            examples,
            node_values=values,
            positions={int(s): i for i, s in enumerate(ids)},
        )
        for ids, values in [(cid, coordinates), (sid, sequence), (tid, topology)]
    ]
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows, sources = (
        [],
        [examples_path, sequence_path, coordinate_path, topology_path, archived_path],
    )
    for feature, count, settings in [
        (BASE, 2, [("control", 1)]),
        (FULL, 3, [("first", 1), ("repeat", 1), ("four_threads", 4)]),
    ]:
        matrix = np.concatenate(pairs[:count], axis=1)
        matrix_hash = hashlib.sha256(memoryview(matrix)).hexdigest()
        prior = archived.loc[archived.feature_set.eq(feature + "_pair")].sort_values(
            "example_id"
        )
        if not np.array_equal(
            prior.example_id, examples.loc[test, "example_id"]
        ) or not np.array_equal(prior.y_true, labels[test]):
            raise ValueError("Original test universe changed")
        prior_ap = binary_metrics(labels[test], prior.p_calibrated.to_numpy(), 0.5)[
            "auprc"
        ]
        first_scores = None
        for name, threads in settings:
            with threadpool_limits(limits=threads):
                model, temperature, threshold = fit_original(
                    matrix, labels, train, val, job.seed
                )
                scores = apply_temperature(
                    model.predict_proba(matrix[test])[:, 1], temperature
                )
            if first_scores is None:
                first_scores = scores.copy()
            metrics = binary_metrics(labels[test], scores, threshold)
            row = dict(
                feature_set=feature,
                setting=name,
                threads=threads,
                matrix_sha256=matrix_hash,
                matrix_dtype=str(matrix.dtype),
                n_train=int(train.sum()),
                n_val=int(val.sum()),
                n_test=int(test.sum()),
                iterations=int(model[-1].n_iter_.max()),
                temperature=temperature,
                archived_auprc=prior_ap,
                max_error_vs_archive=float(
                    np.abs(scores - prior.p_calibrated.to_numpy()).max()
                ),
                max_error_vs_first_fit=float(np.abs(scores - first_scores).max()),
                **metrics,
            )
            row["auprc_difference_from_archive"] = row["auprc"] - prior_ap
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.out_dir / "repeatability.csv", index=False)
            print(row, flush=True)
    (args.out_dir / "audit.json").write_text(
        json.dumps(
            dict(
                status="complete",
                fold=job.fold,
                seed=job.seed,
                context=job.closure,
                sources=[fingerprint(p) for p in sources],
                numpy_version=np.__version__,
                sklearn_version=sklearn.__version__,
                libraries=threadpool_info(),
                purpose="Fixed cache, same seed, same training/validation labels: one-thread duplicate fits and four-thread diagnostic; all outcomes retained",
                external_labels_read=False,
                alters_external_regression_gate=False,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

"""Measure frozen extraction numerical repeatability without biological fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from scripts.server.run_ccre_frozen_probe_matrix import build_jobs, checkpoint_for
from tasks.ccre.embedding_baseline import _extract_embeddings
from tasks.entex.prepare import fingerprint


def compare(reference: np.ndarray, observed: np.ndarray) -> dict:
    if reference.shape != observed.shape or not np.isfinite(observed).all():
        raise ValueError(
            "Embedding comparison has mismatched shape or nonfinite values"
        )
    difference = np.abs(reference.astype(float) - observed)
    scale = StandardScaler().fit(reference)
    standardized = np.abs(scale.transform(reference) - scale.transform(observed))
    return dict(
        n_segments=len(reference),
        n_dimensions=reference.shape[1],
        identical_fraction=float((difference == 0).mean()),
        maximum_absolute_error=float(difference.max()),
        p99_absolute_error=float(np.quantile(difference, 0.99)),
        relative_l2_error=float(
            np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-30)
        ),
        minimum_channel_sd=float(np.sqrt(scale.var_).min()),
        maximum_standardized_error=float(standardized.max()),
        p99_standardized_error=float(np.quantile(standardized, 0.99)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", default="fold_d")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context", default="strict", choices=["strict", "1hop"])
    ap.add_argument("--max-slices", type=int, default=20)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    if args.max_slices < 1:
        raise ValueError("Expected positive diagnostic slice count")
    config = json.loads(
        Path("configs/server_full_multicohort_20260806.json").read_text()
    )
    job = next(
        j
        for j in build_jobs(config)
        if (j.fold, j.seed, j.closure) == (args.fold, args.seed, args.context)
    )
    resource = Path("server_workspace")
    checkpoint = checkpoint_for(
        resource / "results/full_multicohort_server_20260806", job
    )
    graph = resource / "data/processed/hprc_r2_sv/full_segments.csv.gz"
    manifest = resource / "data/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv"
    raw = pd.read_csv(
        resource
        / "data/processed/hgsvc3_sv_breakpoint_examples_20260809/sv_breakpoint_examples.csv.gz",
        usecols=["start_segid", "end_segid"],
    )
    requested = set(raw.start_segid) | set(raw.end_segid)
    rows, audits, reference, ids, counts = [], {}, None, None, None
    for name, device in [("gpu_first", "cuda"), ("gpu_repeat", "cuda"), ("cpu", "cpu")]:
        frozen, n, audit = _extract_embeddings(
            checkpoint=checkpoint,
            manifest=manifest,
            full_segments=graph,
            labeled_segids=requested,
            closure=job.closure,
            device_name=device,
            seed=job.seed,
            max_slices=args.max_slices,
            canonical_conflict_policy="exclude",
            return_canonical_audit=True,
        )
        current_ids = np.array(sorted(frozen))
        current = np.stack([frozen[int(s)] for s in current_ids])
        if reference is None:
            reference, ids, counts = current, current_ids, n
        elif not np.array_equal(ids, current_ids) or n != counts:
            raise ValueError(
                "Repeated extraction changed the segment universe or pooling counts"
            )
        rows.append(
            dict(comparison=f"{name}_versus_gpu_first", **compare(reference, current))
        )
        audits[name] = audit
        print(rows[-1], flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(args.out_dir / "repeatability.csv", index=False)
    (args.out_dir / "audit.json").write_text(
        json.dumps(
            dict(
                status="complete",
                fold=job.fold,
                seed=job.seed,
                context=job.closure,
                max_slices=args.max_slices,
                checkpoint=fingerprint(checkpoint),
                graph=fingerprint(graph),
                manifest=fingerprint(manifest),
                torch_version=torch.__version__,
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),
                extraction_audits=audits,
                scope="First fixed manifest windows, GPU repeated twice and CPU once; no biological labels fitted or external scores inspected",
                limitation="Numerical differences alone do not establish the cause of archived probe AP disagreement. Original fitted probes/embeddings were not retained.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check actual frozen v1 topology coverage for every TraitGym variant and fold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from scripts.server.run_ccre_frozen_probe_fold import validate_checkpoint_holdout
from scripts.server.run_ccre_frozen_probe_matrix import build_jobs, checkpoint_for
from tasks.entex.prepare import fingerprint


def coverage(examples: pd.DataFrame, overlaps: pd.DataFrame, segids: np.ndarray) -> pd.DataFrame:
    if len(np.unique(segids)) != len(segids) or examples.variant_id.duplicated().any():
        raise ValueError("Duplicate cache or variant identities")
    if overlaps.locus_id.duplicated().any() or set(overlaps.locus_id) != set(examples.locus_id):
        raise ValueError("Require exactly one mapped segment per TraitGym SNV locus")
    complete = overlaps.set_index("locus_id").segid.isin(segids)
    frame = examples.copy()
    frame["topology_covered"] = frame.locus_id.map(complete)
    return pd.DataFrame([dict(group=name, n=len(part), n_covered=int(part.topology_covered.sum()),
        coverage=float(part.topology_covered.mean()), positives=int(part.label.sum()))
        for name, part in [("all", frame), *[(f"label={k}", g) for k, g in frame.groupby("label")],
                           *[(f"chrom={k}", g) for k, g in frame.groupby("chrom")]]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mapping-dir", type=Path, required=True)
    ap.add_argument("--topology-cache-root", type=Path, required=True)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    config = json.loads(Path("configs/entex_v1.json").read_text())
    manuscript = json.loads(Path(config["manuscript_config"]).read_text())
    mapping_audit = json.loads((args.mapping_dir / "audit.json").read_text())
    if (mapping_audit["graph"]["sha256"] != config["full_segments_sha256"]
            or mapping_audit.get("reference_verification", {}).get("mismatches") != 0):
        raise ValueError("Require the exact graph and completed reference-allele audit")
    datasets, sources = {}, [fingerprint(args.mapping_dir / "audit.json")]
    for name in ["complex_traits", "mendelian_traits"]:
        paths = [args.mapping_dir / (name + suffix) for suffix in ["_examples.parquet", "_overlaps.parquet"]]
        datasets[name] = tuple(pd.read_parquet(p) for p in paths)
        sources.extend(fingerprint(p) for p in paths)
    manifest = fingerprint(args.manifest)
    rows, receipts = [], []
    for job in build_jobs(manuscript):
        path = args.topology_cache_root / job.fold / f"seed_{job.seed}" / (job.closure + ".npz")
        sidecar = Path(str(path) + ".audit.json")
        audit = json.loads(sidecar.read_text())
        checkpoint = checkpoint_for(args.results_root, job)
        identity = dict(checkpoint_sha256=fingerprint(checkpoint)["sha256"],
            graph_sha256=config["full_segments_sha256"], manifest_sha256=manifest["sha256"],
            seed=job.seed, closure=job.closure, canonical_conflict_policy="exclude")
        cache = fingerprint(path)
        if (audit.get("identity") != identity or audit.get("output_sha256") != cache["sha256"]
                or audit.get("status") != "complete" or audit.get("downstream_label_access") != "none"
                or audit.get("model_parameters_frozen") is not True):
            raise ValueError("Frozen topology cache provenance mismatch")
        holdout = validate_checkpoint_holdout(checkpoint, test_chrs=set(job.test),
            val_chrs=set(job.validation), closure=job.closure, seed=job.seed)
        with np.load(path, allow_pickle=False) as stored:
            ids = stored["segid"]
        receipts.append(dict(cache=cache, sidecar=fingerprint(sidecar), checkpoint=str(checkpoint), holdout=holdout))
        for name, (examples, overlaps) in datasets.items():
            rows.append(coverage(examples, overlaps, ids).assign(dataset=name, fold=job.fold,
                seed=job.seed, context=job.closure))
    frame = pd.concat(rows, ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.out_dir / "coverage.csv", index=False)
    (args.out_dir / "audit.json").write_text(json.dumps(dict(status="complete", sources=sources,
        implementation=fingerprint(Path(__file__)), command=sys.argv,
        manifest=manifest, caches=receipts, n_caches=len(receipts),
        all_variants_covered=bool(frame.coverage.eq(1).all()), original_examples_retained=True,
        interpretation="Actual v1 frozen checkpoint-cache coverage; not v2 coverage or a fitted TraitGym result",
        no_model_fitted=True), indent=2) + "\n")


if __name__ == "__main__":
    main()

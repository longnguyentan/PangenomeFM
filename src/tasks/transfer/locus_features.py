"""Audited, frozen manuscript features pooled with the existing locus mapper."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.modality_factorial import load_frozen_node_embedding_cache
from scripts.server.run_ccre_frozen_probe_fold import validate_checkpoint_holdout
from scripts.server.run_ccre_frozen_probe_matrix import checkpoint_for
from tasks.entex.mapping import aggregate_features
from tasks.entex.prepare import fingerprint
from tasks.transfer.traitgym import validate_nt_provenance, verified_fingerprint


class FrozenLocusFeatures:
    """Load label-free static components once; verify each fold's frozen T cache."""

    def __init__(self, loci: pd.DataFrame, overlaps: pd.DataFrame, config: dict,
                 feature_cache: Path, sequence_cache: Path, topology_control_cache: Path,
                 manifest: Path, topology_control_sha256: str, aggregation: str = "mean"):
        if loci.locus_id.duplicated().any() or set(loci.locus_id) != set(overlaps.locus_id):
            raise ValueError("Require unique and completely mapped loci")
        if (overlaps.duplicated(["locus_id", "segid"]).any() or (overlaps.overlap_bp <= 0).any()
                or overlaps.segid.isna().any()):
            raise ValueError("Invalid or duplicated locus/segment overlap")
        self.loci, self.overlaps, self.config = loci, overlaps, config
        self.aggregation = aggregation
        self.sources = [fingerprint(manifest)]
        self.manifest_sha = self.sources[0]["sha256"]
        self.static = {}
        audit_path = Path(str(feature_cache) + ".audit.json")
        audit = json.loads(audit_path.read_text())
        if (audit.get("status") != "complete"
                or audit.get("inputs", {}).get("full_segments_sha256") != config["full_segments_sha256"]):
            raise ValueError("C/K cache graph provenance mismatch")
        self.sources.extend([fingerprint(audit_path), verified_fingerprint(feature_cache, audit["output_sha256"])])
        with np.load(feature_cache, allow_pickle=False) as cache:
            for name in ["coordinate", "sequence_kmer"]:
                self.static[name] = self.pool(cache["segid"], cache[name])
        for name, path in [("frozen_sequence_fm", sequence_cache), ("topology_control", topology_control_cache)]:
            ids, values, audit = load_frozen_node_embedding_cache(path)
            source = fingerprint(path)
            # Historical H receipts identify graph/processing but do not store
            # the output hash; require its independently pinned plan checksum.
            expected_sha = audit.get("output_sha256") if name == "frozen_sequence_fm" else topology_control_sha256
            if expected_sha != source["sha256"]:
                raise ValueError("Frozen feature checksum mismatch")
            self.sources.extend([source, fingerprint(Path(str(path) + ".audit.json"))])
            if name == "frozen_sequence_fm":
                self.sources.extend(validate_nt_provenance(path, audit, config, source["sha256"]))
            elif (audit.get("kind") != "handcrafted_topology_control" or values.shape[1] != 14
                    or audit.get("status") != "complete" or audit.get("downstream_label_access") != "none"
                    or audit.get("full_segments_sha256") != config["full_segments_sha256"]):
                raise ValueError("Expected original 14-statistic graph control")
            self.static[name] = self.pool(ids, values)

    def pool(self, ids, values):
        return aggregate_features(self.loci, self.overlaps, ids, values, self.aggregation)

    def topology(self, job, cache_root: Path, results_root: Path) -> tuple[np.ndarray, dict]:
        path = cache_root / job.fold / f"seed_{job.seed}" / (job.closure + ".npz")
        ids, values, audit = load_frozen_node_embedding_cache(path)
        checkpoint = checkpoint_for(results_root, job)
        holdout = validate_checkpoint_holdout(checkpoint, test_chrs=set(job.test),
            val_chrs=set(job.validation), closure=job.closure, seed=job.seed)
        expected = dict(checkpoint_sha256=fingerprint(checkpoint)["sha256"],
            graph_sha256=self.config["full_segments_sha256"], manifest_sha256=self.manifest_sha,
            seed=job.seed, closure=job.closure, canonical_conflict_policy="exclude")
        if (audit.get("identity") != expected or audit.get("model_parameters_frozen") is not True
                or audit.get("downstream_label_access") != "none"):
            raise ValueError("Topology cache identity or frozen state differs")
        source = verified_fingerprint(path, audit["output_sha256"])
        return self.pool(ids, values), dict(cache=source, sidecar=fingerprint(Path(str(path) + ".audit.json")),
            identity=expected, holdout=holdout)

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_hprc_chr8_cohort import audit_cohort


def test_audit_cohort_requires_exact_finite_embedding_ids(tmp_path: Path) -> None:
    donor = "D1"
    cohort = tmp_path / "cohort.tsv"
    pd.DataFrame(
        [
            {
                "donor_id": donor,
                "population": "P1",
                "super_population": "S1",
                "cohort_role": "test",
            }
        ]
    ).to_csv(cohort, sep="\t", index=False)
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(
        [
            {
                "donor_id": donor,
                "chromosome": "chr8",
                "haplotype": haplotype,
                "locus": "chr8",
                "n_loci": 1,
                "resolution_method": "reference_chromosome_block",
                "n_path_fragments": 1,
                "first_path": f"{donor}#{haplotype}#chr8#0",
                "last_path": f"{donor}#{haplotype}#chr8#0",
            }
            for haplotype in [1, 2]
        ]
    ).to_csv(manifest, sep="\t", index=False)
    path_batch = tmp_path / "paths"
    path_batch.mkdir()
    (path_batch / f"{donor}.chr8.paths.txt").write_text(
        f"{donor}#1#chr8#0\n{donor}#2#chr8#0\n", encoding="utf-8"
    )
    chunks = tmp_path / f"{donor}.chr8.chunks2m"
    chunks.mkdir()
    (chunks / "chunk_0.gfa.gz").write_bytes(b"x")
    (chunks / ".complete").write_text(
        "chunk_size_bp=2000000\nchunks=1\npaths=2\n", encoding="utf-8"
    )
    donor_out = tmp_path / "results" / "donors" / donor
    donor_out.mkdir(parents=True)
    pairs = pd.DataFrame(
        [
            {
                "donor_id": donor,
                "h1_embedding_id": "left",
                "h2_embedding_id": "right",
                "shared_nodes": 2,
                "shared_node_bp": 500,
            }
        ]
    )
    pairs.to_csv(
        donor_out / "paired_windows.csv.gz", index=False, compression="gzip"
    )
    np.savez_compressed(
        donor_out / "graphgenomefm_embeddings.npz",
        ids=np.asarray(["left", "right"]),
        embeddings=np.ones((2, 48), dtype=np.float32),
    )
    np.savez_compressed(
        donor_out / "nucleotide_transformer_embeddings.npz",
        ids=np.asarray(["left", "right"]),
        embeddings=np.ones((2, 512), dtype=np.float32),
    )
    (donor_out / "dataset_summary.json").write_text(
        json.dumps({"n_labeled_haplotype_windows": 2}), encoding="utf-8"
    )

    audit, summary = audit_cohort(
        cohort_path=cohort,
        path_manifest=manifest,
        path_batch_dir=path_batch,
        chunks_root=tmp_path,
        result_root=tmp_path / "results",
        require_shared_node_anchors=True,
    )
    assert len(audit) == 1
    assert summary["status"] == "passed"
    assert summary["n_pairs"] == 1
    assert summary["n_embeddings_per_model"] == 2
    assert audit.iloc[0]["anchor_method_provenance"].startswith(
        "inferred_from_shared_nodes"
    )

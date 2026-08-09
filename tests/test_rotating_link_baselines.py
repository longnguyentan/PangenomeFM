from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.server.evaluate_rotating_link_baselines import BASELINE_NAMES, evaluate


def test_rotating_baselines_use_train_val_and_heldout_chromosomes(tmp_path: Path) -> None:
    full_segments = tmp_path / "full_segments.csv.gz"
    segments = pd.DataFrame(
        {
            "name": ["s0", "s1", "s2", "s3"],
            "seq": ["ACGT" * 20, "AAAA" * 20, "CCCC" * 20, "AGAG" * 20],
            "SN": ["chr0"] * 4,
            "SO": [0, 100, 200, 300],
        }
    )
    segments.to_csv(full_segments, index=False, compression="gzip")
    links = pd.DataFrame(
        {
            "from_seg": ["s0", "s1", "s2", "s0"],
            "from_orient": ["+"] * 4,
            "to_seg": ["s1", "s2", "s3", "s2"],
            "to_orient": ["+"] * 4,
            "overlap": ["0M"] * 4,
        }
    )
    candidates = pd.DataFrame(
        {
            "u_oid": ([0, 2, 4, 0, 2] * 8),
            "v_oid": ([2, 4, 6, 4, 6] * 8),
            "label": ([0, 1] * 20),
        }
    )
    manifest_rows = []
    for chromosome in ["chr1", "chr2", "chr3"]:
        links_path = tmp_path / f"{chromosome}_links.csv.gz"
        edges_path = tmp_path / f"{chromosome}_edges.csv.gz"
        links.to_csv(links_path, index=False, compression="gzip")
        candidates.to_csv(edges_path, index=False, compression="gzip")
        manifest_rows.append(
            {
                "name": f"slice_{chromosome}_strict",
                "target_sn": f"GRCh38#0#{chromosome}",
                "closure": "strict",
                "links_path": str(links_path),
                "edge_pred_path": str(edges_path),
            }
        )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest, index=False)

    audit = evaluate(
        manifest_path=manifest,
        full_segments_path=full_segments,
        out_dir=tmp_path / "out",
        closure="strict",
        test_chrs={"chr3"},
        val_chrs={"chr2"},
        seed=42,
        split_seed=7,
        shortest_path_cutoff=3,
    )

    assert audit["status"] == "complete"
    assert audit["training_chromosomes"] == ["chr1"]
    assert audit["validation_chromosomes"] == ["chr2"]
    assert audit["heldout_chromosomes"] == ["chr3"]
    metrics = pd.read_csv(tmp_path / "out" / "metrics.csv")
    assert set(metrics.loc[metrics["chromosome"].isna(), "baseline"]) == set(BASELINE_NAMES)
    predictions = pd.read_csv(tmp_path / "out" / "heldout_predictions.csv.gz")
    assert set(predictions["chromosome"]) == {"chr3"}
    assert set(predictions["fold_evaluation_split"]) == {"heldout"}

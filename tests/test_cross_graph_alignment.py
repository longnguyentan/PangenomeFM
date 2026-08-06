from pathlib import Path

import numpy as np
import pandas as pd

from tasks.haplotype.cross_graph_alignment import (
    build_same_locus_anchors,
    train_cross_graph_alignment,
)


def test_build_and_align_cross_graph_anchors(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    rows = []
    for chromosome in ["chr19", "chr21", "chr22", "chrY"]:
        for index in range(4):
            sequence = "".join(rng.choice(list("ACGT"), size=96))
            rows.append(
                {
                    "name": f"{chromosome}_{index}",
                    "seq": sequence,
                    "LN": len(sequence),
                    "SN": f"REF#0#{chromosome}",
                    "SO": index * 100,
                    "SR": 0,
                }
            )
    left_segments = tmp_path / "left.csv.gz"
    right_segments = tmp_path / "right.csv.gz"
    pd.DataFrame(rows).to_csv(left_segments, index=False, compression="gzip")
    pd.DataFrame(
        [{**row, "name": "R_" + row["name"]} for row in rows]
    ).to_csv(right_segments, index=False, compression="gzip")
    left_links = tmp_path / "left_links.csv.gz"
    right_links = tmp_path / "right_links.csv.gz"
    pd.DataFrame(
        [
            {"from_seg": rows[index]["name"], "to_seg": rows[index + 1]["name"]}
            for index in range(len(rows) - 1)
        ]
    ).to_csv(left_links, index=False, compression="gzip")
    pd.DataFrame(
        [
            {
                "from_seg": "R_" + rows[index]["name"],
                "to_seg": "R_" + rows[index + 1]["name"],
            }
            for index in range(len(rows) - 1)
        ]
    ).to_csv(right_links, index=False, compression="gzip")
    anchors = tmp_path / "anchors.csv.gz"
    summary = build_same_locus_anchors(
        hprc_segments=left_segments,
        hprc_links=left_links,
        hgsvc_segments=right_segments,
        hgsvc_links=right_links,
        out_path=anchors,
        stride=32,
    )
    assert summary["n_anchors"] >= 8
    aligned = train_cross_graph_alignment(
        anchors_path=anchors,
        out_dir=tmp_path / "alignment",
        adversarial_weights=(0.0, 0.01),
        epochs=1,
        batch_size=4,
    )
    assert aligned["best_adversarial_weight"] in {0.0, 0.01}
    assert "recall_at_1" in aligned["locked_test"]

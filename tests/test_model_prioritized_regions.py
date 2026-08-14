from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.server.prepare_model_prioritized_regions import (
    FINAL_REQUIRED_COLUMNS,
    assign_priorities,
    clip_regions_to_reference,
    count_variant_anchors,
    distance_to_nearest_gene,
    fasta_covariates,
    load_gene_intervals,
    mappability_covariates,
    validate_regions,
)


def write_gzip(path: Path, text: str) -> None:
    with gzip.open(path, "wt") as handle:
        handle.write(text)


def synthetic_regions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region_id": [f"r{index}" for index in range(6)],
            "chromosome": ["chr1"] * 6,
            "start": np.arange(0, 60, 10),
            "end": np.arange(10, 70, 10),
            "region_length": [10] * 6,
            "fold": ["fold_a"] * 6,
            "graph_complexity": np.linspace(1.0, 1.5, 6),
            "model_score": np.arange(6, 0, -1, dtype=float),
            "is_eligible": [True] * 6,
            "eligibility_reason": ["eligible"] * 6,
        }
    )


def test_streaming_covariates_and_priority_rule(tmp_path: Path) -> None:
    regions = validate_regions(synthetic_regions())
    fasta = tmp_path / "reference.fa.gz"
    write_gzip(fasta, ">chr1\nGCGCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n")
    gc, acgt, observed, chromosome_lengths = fasta_covariates(regions, fasta)
    assert observed.tolist() == [10] * 6
    assert acgt.tolist() == [10] * 6
    assert gc[0] == 4
    assert chromosome_lengths == {"chr1": 60}
    regions["gc_content"] = gc / acgt

    bedgraph = tmp_path / "map.bedgraph.gz"
    # Umap's omitted positions are zero: r0 therefore has mean 0.5.
    write_gzip(bedgraph, "chr1\t0\t5\t1\nchr1\t10\t20\t0.5\n")
    mappability, reported = mappability_covariates(regions, bedgraph)
    assert mappability[0] == 0.5
    assert reported[0] == 5
    assert mappability[2] == 0.0
    regions["mappability"] = mappability

    gtf = tmp_path / "genes.gtf.gz"
    write_gzip(
        gtf,
        'chr1\ttest\tgene\t16\t20\t.\t+\t.\tgene_id "g1";\n',
    )
    genes = load_gene_intervals(gtf)
    distance = distance_to_nearest_gene(regions, genes)
    assert distance[0] == 5
    assert distance[1] == 0
    regions["distance_to_gene"] = distance

    variants = tmp_path / "variants.csv.gz"
    pd.DataFrame(
        {"chrom": ["chr1", "chr1", "1"], "pos": [1, 10, 11]}
    ).to_csv(variants, index=False, compression="gzip")
    counts, rows, assigned = count_variant_anchors(
        regions, variants, chunksize=2
    )
    assert counts.tolist() == [2, 1, 0, 0, 0, 0]
    assert rows == 3 and assigned == 3
    regions["variant_density"] = counts / regions["region_length"] * 1_000_000

    prioritized, region_exclusions, chromosome_exclusions = assign_priorities(
        regions,
        priority_fraction=0.10,
        controls_per_case=5,
    )
    assert region_exclusions.empty
    assert chromosome_exclusions.empty
    assert len(prioritized) == 6
    assert prioritized["is_prioritized"].sum() == 1
    assert prioritized.loc[prioritized["is_prioritized"], "region_id"].iloc[0] == "r0"
    assert set(FINAL_REQUIRED_COLUMNS).issubset(prioritized.columns)


def test_terminal_benchmark_region_is_clipped_to_reference_boundary(
    tmp_path: Path,
) -> None:
    regions = validate_regions(synthetic_regions().iloc[:2].copy())
    fasta = tmp_path / "reference.fa.gz"
    write_gzip(fasta, ">chr1\nAAAAAAAAAAAAAAA\n")

    gc, acgt, observed, chromosome_lengths = fasta_covariates(regions, fasta)
    clipped, clipping = clip_regions_to_reference(regions, chromosome_lengths)

    assert gc.tolist() == [0, 0]
    assert acgt.tolist() == [10, 5]
    assert observed.tolist() == [10, 5]
    assert clipped["end"].tolist() == [10, 15]
    assert clipped["region_length"].tolist() == [10, 5]
    assert clipped["benchmark_end"].tolist() == [10, 20]
    assert clipped["reference_clipped"].tolist() == [False, True]
    assert len(clipping) == 1
    assert clipping.loc[0, "region_id"] == "r1"
    assert clipping.loc[0, "reference_clipped_bp"] == 5


def test_ineligible_scores_never_become_cases() -> None:
    regions = validate_regions(synthetic_regions())
    regions["gc_content"] = 0.5
    regions["mappability"] = 0.5
    regions["variant_density"] = 1.0
    regions["distance_to_gene"] = 0.0
    regions.loc[0, "is_eligible"] = False
    regions.loc[0, "eligibility_reason"] = "incomplete_seed_coverage"
    regions.loc[1, "model_score"] = 5.0
    retained, region_exclusions, _ = assign_priorities(
        regions,
        priority_fraction=0.10,
        controls_per_case=4,
    )
    assert "r0" not in set(retained["region_id"])
    assert "r0" in set(region_exclusions["region_id"])
    assert retained.loc[retained["is_prioritized"], "region_id"].iloc[0] == "r1"

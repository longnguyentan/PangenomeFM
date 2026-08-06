import gzip
import random
from pathlib import Path

import pandas as pd
import pytest

from tasks.haplotype.methylation import (
    _load_selected_gfa,
    _parse_walk,
    _reciprocal_anchor_pairs,
    _walk_matches_active_target,
    _walk_windows,
    benchmark_methylation_pairs,
    build_methylation_pairs,
)


def _write_gzip(path: Path, text: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)


def test_parse_chunked_p_walk_restores_absolute_path_coordinate() -> None:
    walk = _parse_walk(
        [
            "P",
            "HG00438#1#CM089174.1#9507[2000000-4000000]",
            "10+,11-",
            "*",
        ]
    )
    assert walk is not None
    assert walk["source_path_name"] == "HG00438#1#CM089174.1#9507"
    assert walk["contig"] == "CM089174.1"
    assert walk["path_start"] == 2_009_507


def test_walk_windows_trims_vg_chunk_boundary_overlap() -> None:
    walks = [
        {
            "sample": "D",
            "haplotype": "1",
            "contig": "chr8",
            "path_start": 0,
            "path_name": "D#1#chr8#0[0-8]",
            "steps": [("a", "+"), ("b", "+")],
        },
        {
            "sample": "D",
            "haplotype": "1",
            "contig": "chr8",
            "path_start": 4,
            "path_name": "D#1#chr8#0[4-12]",
            "steps": [("b", "+"), ("c", "+")],
        },
    ]
    windows = _walk_windows(
        sequences={"a": "AAAA", "b": "CCCC", "c": "GGGG"},
        degree={"a": 1, "b": 2, "c": 1},
        walks=walks,
        window_bp=20,
    )
    assert len(windows) == 1
    assert windows.loc[0, "path_bp"] == 12


def test_selected_gfa_restores_walk_phase_block_from_chunk_target(
    tmp_path: Path,
) -> None:
    chunk = tmp_path / "chunk_0_D#1#CONTIG#0_0_4.gfa"
    chunk.write_text(
        "\n".join(
            [
                "S\t1\tACGT",
                "S\t2\tTTTT",
                "W\tD\t1\tCONTIG\t0\t4\t>1",
                "W\tD\t2\tOTHER\t0\t4\t>2",
                "",
            ]
        ),
        encoding="utf-8",
    )
    sequences, _, walks = _load_selected_gfa(
        chunk,
        sample_ids={"D"},
        path_names={"D#1#CONTIG#0"},
    )
    assert sequences == {"1": "ACGT"}
    assert len(walks) == 1
    assert walks[0]["path_name"] == "D#1#CONTIG[0-4]"


def test_walk_target_restoration_rejects_nonzero_phase_context() -> None:
    walk = _parse_walk(["W", "D", "1", "CONTIG", "0", "4", ">1"])
    assert walk is not None
    assert _walk_matches_active_target(walk, {"D#1#CONTIG#0"})
    assert not _walk_matches_active_target(walk, {"D#1#CONTIG#100"})


def test_unique_spaced_kmers_rescue_disjoint_graph_node_ids() -> None:
    def sequence(seed: int) -> str:
        rng = random.Random(seed)
        return "".join(rng.choice("ACGT") for _ in range(512))

    rows = []
    for haplotype, nodes in (("1", ["a", "b"]), ("2", ["c", "d"])):
        for index, node in enumerate(nodes):
            rows.append(
                {
                    "sample": "D",
                    "haplotype": haplotype,
                    "contig": "chr8",
                    "track_contig": f"D#{haplotype}#chr8",
                    "window_start": index * 10_000,
                    "node_ids": node,
                    "oriented_node_ids": f"{node}+",
                    "sequence_sample": sequence(index),
                    "methylation": 0.8 - 0.2 * int(haplotype),
                }
            )
    pairs = _reciprocal_anchor_pairs(
        pd.DataFrame(rows),
        node_lengths={node: 1000 for node in "abcd"},
        min_shared_bp=500,
        min_shared_nodes=2,
    )
    assert len(pairs) == 2
    assert set(pairs["anchor_method"]) == {
        "reciprocal_unique_spaced_31mer"
    }
    assert pairs["shared_kmer31"].min() >= 2


def test_build_methylation_pairs_from_shared_graph_anchors(tmp_path: Path) -> None:
    gfa = tmp_path / "pair.gfa"
    gfa.write_text(
        "\n".join(
            [
                "S\ta\tAACCGGTT",
                "S\tshared\tCGCGCGCG",
                "S\tb\tTTGGCCAA",
                "L\ta\t+\tshared\t+\t0M",
                "L\tshared\t+\tb\t+\t0M",
                "W\tD\t1\tchr1\t0\t16\t>a>shared",
                "W\tD\t2\tchr1\t0\t16\t>b>shared",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    h1 = tmp_path / "h1.gz"
    h2 = tmp_path / "h2.gz"
    h1_lines = "\n".join(
        f"D#1#chr1\t{i}\t{i+1}\tCG\t0.8\t+\t20" for i in range(0, 16, 2)
    )
    h2_lines = "\n".join(
        f"D#2#chr1\t{i}\t{i+1}\tCG\t0.2\t+\t20" for i in range(0, 16, 2)
    )
    _write_gzip(h1, h1_lines + "\n")
    _write_gzip(h2, h2_lines + "\n")

    out = tmp_path / "out"
    summary = build_methylation_pairs(
        gfa_path=gfa,
        methylation_paths=[h1, h2],
        out_dir=out,
        window_bp=16,
        min_cpgs=2,
        min_depth=10,
        min_shared_bp=4,
        min_shared_nodes=1,
    )
    pairs = pd.read_csv(out / "paired_windows.csv.gz")
    assert summary["n_reciprocal_h1_h2_pairs"] == 1
    assert pairs.loc[0, "methylation_delta"] == pytest.approx(0.6)
    assert pairs.loc[0, "shared_nodes"] == 1


def test_benchmark_accepts_multiple_donors_and_holds_each_out(
    tmp_path: Path,
) -> None:
    paths = []
    for donor_index, donor in enumerate(["D1", "D2"]):
        rows = []
        for index in range(20):
            signal = (index - 10) / 10
            rows.append(
                {
                    "donor_id": donor,
                    "h1_contig": "chr1",
                    "h1_window_start": index * 10_000,
                    "methylation_delta": signal + donor_index * 0.01,
                    "delta_gc_fraction": signal,
                    "delta_cpg_density": signal / 2,
                    "delta_node_count": signal * 2,
                    "delta_path_bp": signal,
                    "delta_mean_node_length": signal / 3,
                    "delta_mean_degree": signal / 4,
                    "delta_branching_fraction": signal / 5,
                    "delta_reverse_fraction": signal / 6,
                }
            )
        path = tmp_path / f"{donor}.csv.gz"
        pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
        paths.append(path)

    out = tmp_path / "benchmark"
    summary = benchmark_methylation_pairs(
        pairs_path=paths,
        out_dir=out,
        split_mode="donor",
        n_bootstrap=10,
    )
    predictions = pd.read_csv(out / "blocked_predictions.csv.gz")
    assert summary["n_donors"] == 2
    assert summary["n_splits"] == 2
    assert set(predictions["donor_id"]) == {"D1", "D2"}

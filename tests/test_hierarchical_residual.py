from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tasks.haplotype.hierarchical_residual import (
    HierarchicalResidualNetwork,
    _select_residual_shrinkage,
    train_hierarchical_residual,
)


def test_hierarchical_residual_runs_leave_one_donor_out(tmp_path: Path) -> None:
    rows = []
    ids = []
    seq_embeddings = []
    graph_embeddings = []
    rng = np.random.default_rng(1)
    for donor_index, donor in enumerate(["D1", "D2", "D3"]):
        for index in range(8):
            left, right = f"{donor}-h1-{index}", f"{donor}-h2-{index}"
            ids.extend([left, right])
            seq_left, seq_right = rng.normal(size=4), rng.normal(size=4)
            graph_left, graph_right = rng.normal(size=3), rng.normal(size=3)
            seq_embeddings.extend([seq_left, seq_right])
            graph_embeddings.extend([graph_left, graph_right])
            rows.append(
                {
                    "donor_id": donor,
                    "h1_embedding_id": left,
                    "h2_embedding_id": right,
                    "h1_contig": "chr8",
                    "h1_window_start": index * 10_000,
                    "delta_repeat_fraction": float(
                        graph_left[0] - graph_right[0]
                    ),
                    "methylation_delta": float(
                        0.1 * (seq_left - seq_right).sum()
                        + 0.02 * donor_index
                    ),
                }
            )
    pairs = tmp_path / "pairs.csv.gz"
    pd.DataFrame(rows).to_csv(pairs, index=False, compression="gzip")
    seq = tmp_path / "seq.npz"
    graph = tmp_path / "graph.npz"
    np.savez_compressed(seq, ids=np.asarray(ids), embeddings=np.asarray(seq_embeddings))
    np.savez_compressed(
        graph, ids=np.asarray(ids), embeddings=np.asarray(graph_embeddings)
    )
    cohort = tmp_path / "cohort.tsv"
    pd.DataFrame(
        [
            {
                "donor_id": donor,
                "population": f"P{index}",
                "super_population": f"S{index}",
            }
            for index, donor in enumerate(["D1", "D2", "D3"])
        ]
    ).to_csv(cohort, sep="\t", index=False)
    summary = train_hierarchical_residual(
        pairs_path=pairs,
        sequence_embeddings=seq,
        graph_embeddings=graph,
        cohort_path=cohort,
        out_dir=tmp_path / "out",
        epochs=2,
        patience=1,
    )
    assert summary["n_donors"] == 3
    assert {row["model"] for row in summary["metrics"]} == {
        "frozen_sequence_ridge",
        "hierarchical_graph_only_residual",
        "hierarchical_graph_context_residual",
        "hierarchical_annotation_only_residual",
        "zero",
    }
    assert summary["primary_metric_aggregation"] == (
        "macro-average across heldout groups"
    )
    assert {row["model"] for row in summary["macro_metrics"]} == {
        "frozen_sequence_ridge",
        "hierarchical_graph_only_residual",
        "hierarchical_graph_context_residual",
        "hierarchical_annotation_only_residual",
        "zero",
    }
    assert all(row["n_groups"] == 3 for row in summary["macro_metrics"])
    assert summary["residual_feature_sets"] == {
        "hierarchical_graph_only_residual": 3,
        "hierarchical_graph_context_residual": 4,
        "hierarchical_annotation_only_residual": 1,
    }
    assert {
        row["metric"] for row in summary["paired_group_bootstrap_differences"]
    } == {"spearman", "mae", "r2", "direction_accuracy"}
    assert all(
        row["bootstrap_replicates"] == 10_000
        for row in summary["paired_group_bootstrap_differences"]
    )
    assert (tmp_path / "out" / "macro_metrics.csv").is_file()
    assert (
        tmp_path
        / "out"
        / "paired_group_bootstrap_differences.csv"
    ).is_file()


def test_validation_shrinkage_can_select_zero() -> None:
    y = np.asarray([0.1, -0.1, 0.2])
    baseline = y.copy()
    harmful_residual = np.asarray([1.0, 1.0, 1.0])
    alpha, sweep = _select_residual_shrinkage(
        y, baseline, harmful_residual
    )
    assert alpha == 0.0
    assert [row["alpha"] for row in sweep] == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_hierarchical_residual_is_exactly_swap_antisymmetric() -> None:
    model = HierarchicalResidualNetwork(
        3,
        hidden_dim=8,
        output_dim=4,
        swap_signs=np.asarray([-1.0, -1.0, 1.0]),
    )
    values = torch.randn(6, 3)
    node_to_window = torch.arange(6)
    window_to_event = torch.tensor([0, 0, 1, 1, 2, 2])
    event_to_path = torch.tensor([0, 0, 0])
    prediction = model(
        values, node_to_window, window_to_event, event_to_path
    )
    swapped = values * torch.tensor([-1.0, -1.0, 1.0])
    swapped_prediction = model(
        swapped, node_to_window, window_to_event, event_to_path
    )
    assert torch.allclose(prediction, -swapped_prediction, atol=1e-6)

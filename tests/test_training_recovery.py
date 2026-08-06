from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.pretrain import (
    WarmupCosineScheduler,
    leakage_safe_mask_indices,
    mask_positive_query_edges,
    maybe_tensorize_slice,
    split_candidate_indices,
)
from evaluation.splits import normalize_chrom, validate_chromosome_split


def test_warmup_cosine_scheduler_state_roundtrip() -> None:
    source_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    source_optimizer = torch.optim.AdamW([source_parameter], lr=5e-4)
    source = WarmupCosineScheduler(source_optimizer, warmup_epochs=2, total_epochs=10)
    for _ in range(4):
        source.step()

    target_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    target_optimizer = torch.optim.AdamW([target_parameter], lr=5e-4)
    target = WarmupCosineScheduler(target_optimizer, warmup_epochs=2, total_epochs=10)
    target.load_state_dict(source.state_dict())

    assert target._step_count == 4
    assert target.base_lrs == [5e-4]
    source.step()
    target.step()
    assert target_optimizer.param_groups[0]["lr"] == pytest.approx(
        source_optimizer.param_groups[0]["lr"]
    )


def test_warmup_cosine_scheduler_rejects_incompatible_budget() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=5e-4)
    source = WarmupCosineScheduler(optimizer, warmup_epochs=2, total_epochs=10)
    incompatible = WarmupCosineScheduler(optimizer, warmup_epochs=2, total_epochs=11)

    with pytest.raises(ValueError, match="total_epochs"):
        incompatible.load_state_dict(source.state_dict())


def test_candidate_split_is_controlled_only_by_split_seed() -> None:
    first = split_candidate_indices(101, split_seed=20260804)
    second = split_candidate_indices(101, split_seed=20260804)
    different = split_candidate_indices(101, split_seed=42)

    assert all((left == right).all() for left, right in zip(first, second))
    assert any((left != right).any() for left, right in zip(first, different))
    assert [len(values) for values in first] == [71, 10, 20]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("chr8", "chr8"),
        ("8", "chr8"),
        ("GRCh38#0#chr8", "chr8"),
        ("CHM13#0#chr8", "chr8"),
        ("id=CHM13|chr8", "chr8"),
        ("HG002#1#contig42", "HG002#1#contig42"),
    ],
)
def test_normalize_chrom_across_graph_releases(value: str, expected: str) -> None:
    assert normalize_chrom(value) == expected


def test_multicohort_chromosome_aliases_cannot_cross_split() -> None:
    with pytest.raises(ValueError, match="val/test overlap"):
        validate_chromosome_split(
            val_chrs=["GRCh38#0#chr19"],
            test_chrs=["id=CHM13|chr19"],
        )


def test_query_edge_masking_vectorizes_multiple_positive_edges() -> None:
    src = torch.tensor([0, 1, 2, 3, 4])
    dst = torch.tensor([1, 2, 3, 4, 0])
    edge_attr = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    query_u = torch.tensor([1, 3, 0])
    query_v = torch.tensor([2, 4, 4])
    labels = torch.tensor([1.0, 1.0, 0.0])
    idx = torch.tensor([0, 1, 2])

    masked_src, masked_dst, masked_attr = mask_positive_query_edges(
        src, dst, edge_attr, query_u, query_v, labels, idx
    )

    assert list(zip(masked_src.tolist(), masked_dst.tolist())) == [
        (0, 1),
        (2, 3),
        (4, 0),
    ]
    assert masked_attr is not None and masked_attr.shape == (3, 2)


def test_lazy_tensorization_converts_raw_slice_without_mutating_it() -> None:
    import argparse
    import numpy as np

    raw = {
        "node_feats": np.ones((2, 3), dtype=np.float32),
        "so_arr": np.array([10, 20]),
        "src": np.array([0]),
        "dst": np.array([1]),
        "temps": np.ones(2, dtype=np.float32),
        "labels": np.array([1.0, 0.0], dtype=np.float32),
        "query_u": np.array([0, 1]),
        "query_v": np.array([1, 0]),
        "train_idx": np.array([0]),
        "val_idx": np.array([], dtype=np.int64),
        "test_idx": np.array([1]),
        "edge_attr": None,
        "orient_arr": np.array([0, 1]),
        "pop_ids_arr": np.zeros(2, dtype=np.int64),
        "branching_frac": 0.0,
        "name": "tiny",
        "target_sn": "chr1",
        "closure": "strict",
        "nodes": np.array([0, 1]),
        "n_pos": 1,
        "n_neg": 1,
    }
    args = argparse.Namespace(device="cpu", orientation_rope=True, pop_cond=False)

    tensorized, temporary = maybe_tensorize_slice(raw, args)

    assert temporary is True
    assert tensorized["X"].shape == (2, 3)
    assert tensorized["orient"].tolist() == [0, 1]
    assert "X" not in raw


def test_leakage_safe_mask_hides_val_test_and_current_train_queries() -> None:
    split = {
        "train_idx": torch.tensor([0, 1, 2]),
        "val_idx": torch.tensor([3]),
        "test_idx": torch.tensor([4, 5]),
    }
    mask = leakage_safe_mask_indices(split, torch.tensor([1, 2]))
    assert mask.tolist() == [1, 2, 3, 4, 5]

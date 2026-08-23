from __future__ import annotations

import numpy as np
import pytest
import torch

from models.multimodal_pangenome import (
    InvariantPairHead,
    ObjectiveWeights,
    OrderedPathEncoder,
    PangenomeFoundationModelV2,
)
from training.multimodal_pretrain import (
    split_batch,
    synthetic_arrays,
    tensorize,
    validate_arrays,
)


def test_invariant_edge_head_is_symmetric() -> None:
    torch.manual_seed(1)
    head = InvariantPairHead(8, dropout=0.0).eval()
    left = torch.randn(5, 8)
    right = torch.randn(5, 8)
    assert torch.equal(head(left, right), head(right, left))


def test_ordered_path_encoder_uses_path_order() -> None:
    torch.manual_seed(2)
    encoder = OrderedPathEncoder(
        8, n_heads=2, n_layers=1, dropout=0.0, max_path_length=8
    ).eval()
    nodes = torch.randn(5, 8)
    forward = encoder(nodes, torch.tensor([[0, 1, 2, 3, -1]]))
    reverse = encoder(nodes, torch.tensor([[3, 2, 1, 0, -1]]))
    assert not torch.allclose(forward, reverse)


def test_multimodal_model_computes_every_objective() -> None:
    arrays = synthetic_arrays(7)
    validate_arrays(arrays)
    batch = tensorize(arrays, torch.device("cpu"))
    train = split_batch(batch, 0)
    model = PangenomeFoundationModelV2(
        topology_dim=arrays["topology_embeddings"].shape[1],
        hidden_dim=16,
        coordinate_dim=4,
        n_heads=4,
        sequence_layers=1,
        path_layers=1,
        max_sequence_length=64,
        max_path_length=16,
        n_populations=4,
        n_haplotype_states=3,
        dropout=0.0,
        modality_dropout=0.0,
    )
    outputs = model.compute_objectives(train, weights=ObjectiveWeights())
    assert {"edge", "masked_token", "path_order", "branch_choice", "cross_graph", "total"}.issubset(outputs)
    assert torch.isfinite(outputs["total"])
    outputs["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_split_batch_never_leaks_edge_examples() -> None:
    batch = tensorize(synthetic_arrays(9), torch.device("cpu"))
    train = split_batch(batch, 0)
    validation = split_batch(batch, 1)
    test = split_batch(batch, 2)
    assert len(train["edge_u"]) == len(validation["edge_u"]) == len(test["edge_u"]) == 8
    assert set(train["edge_u"].tolist()).isdisjoint(validation["edge_u"].tolist())


def test_validate_requires_exactly_one_sequence_representation() -> None:
    arrays = synthetic_arrays(3)
    arrays["sequence_features"] = np.zeros((len(arrays["topology_embeddings"]), 5), dtype=np.float32)
    with pytest.raises(ValueError, match="exactly one"):
        validate_arrays(arrays)


import torch

from models.hierarchical_path import (
    AntisymmetricHaplotypeHead,
    CrossGraphContrastiveLoss,
    HierarchicalPathEncoder,
)


def test_hierarchical_path_shapes() -> None:
    encoder = HierarchicalPathEncoder(input_dim=4, hidden_dim=6, output_dim=5)
    nodes = torch.randn(8, 4)
    output = encoder(
        nodes,
        node_to_window=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        window_to_event=torch.tensor([0, 0, 1, 1]),
        event_to_path=torch.tensor([0, 1]),
    )
    assert output["window_embeddings"].shape == (4, 6)
    assert output["event_embeddings"].shape == (2, 5)
    assert output["path_embeddings"].shape == (2, 5)


def test_haplotype_head_is_antisymmetric() -> None:
    torch.manual_seed(4)
    head = AntisymmetricHaplotypeHead(embedding_dim=5, hidden_dim=7)
    first = torch.randn(3, 5)
    second = torch.randn(3, 5)
    assert torch.allclose(head(first, second), -head(second, first), atol=1e-6)


def test_contrastive_loss_prefers_matched_pairs() -> None:
    loss = CrossGraphContrastiveLoss(temperature=0.1)
    anchors = torch.eye(4)
    matched = loss(anchors, anchors)
    mismatched = loss(anchors, anchors.roll(1, dims=0))
    assert matched < mismatched

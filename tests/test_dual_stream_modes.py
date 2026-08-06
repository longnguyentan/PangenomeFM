import torch

from models.dual_stream_gat import DualStreamPangenomeGAT


class _Bomb(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("inactive stream was evaluated")


def test_graph_only_checkpoint_skips_linear_stream() -> None:
    model = DualStreamPangenomeGAT(
        in_dim=7,
        hidden_dim=8,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
        edge_mlp_dim=16,
        stream_mode="graph",
    ).eval()
    model.linear_layers[0] = _Bomb()
    with torch.inference_mode():
        encoded = model.encode_nodes(
            torch.zeros(3, 7),
            torch.arange(3),
            torch.tensor([0, 1]),
            torch.tensor([1, 2]),
            torch.ones(3),
        )
    assert encoded.shape == (3, 8)

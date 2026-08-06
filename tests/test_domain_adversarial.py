import pytest

torch = pytest.importorskip("torch")

from training.pretrain import _GradientReversal


def test_gradient_reversal_changes_only_backward_sign() -> None:
    value = torch.tensor([[1.0, -2.0]], requires_grad=True)
    output = _GradientReversal.apply(value, 0.25)
    assert torch.equal(output, value)
    output.sum().backward()
    assert torch.allclose(value.grad, torch.full_like(value, -0.25))

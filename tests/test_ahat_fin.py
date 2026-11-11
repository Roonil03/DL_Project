import pytest
import torch

from src.models.ahat_fin import AHATFinSignalModel


def test_ahat_fin_output_shape_range_and_gradients():
    torch.manual_seed(7)
    model = AHATFinSignalModel(
        lookback=12,
        num_features=5,
        num_assets=3,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )
    x = torch.randn(4, 12, 5, requires_grad=True)

    output = model(x)
    assert output.shape == (4, 3)
    assert torch.isfinite(output).all()
    assert torch.all(output >= -1.0) and torch.all(output <= 1.0)

    output.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert model.temporal_input.weight.grad is not None
    assert model.feature_input.weight.grad is not None
    assert model.gate[0].weight.grad is not None


def test_ahat_fin_rejects_incompatible_inputs_and_dimensions():
    model = AHATFinSignalModel(lookback=8, num_features=4, num_assets=2, d_model=8, n_heads=2)
    with pytest.raises(ValueError, match="expected"):
        model(torch.randn(2, 7, 4))
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(2, 8))
    with pytest.raises(ValueError, match="divisible"):
        AHATFinSignalModel(lookback=8, num_features=4, num_assets=2, d_model=10, n_heads=4)

import numpy as np
import torch
from src.backtest.backtester import Backtester
from src.losses.sharpe_loss import NegativeSharpeLoss


def test_sharpe_loss_matches_gross_backtest_with_sizing_and_cap():
    signals = torch.tensor([[.5, -.8], [.9, .4], [-.2, .6]], requires_grad=True)
    returns = torch.tensor([[.01, -.02], [-.01, .03], [.02, .01]])
    vol = torch.tensor([[.05, .2], [.01, .3], [.1, .05]])
    loss = NegativeSharpeLoss(target_vol=.12, max_leverage=2.0)(signals, returns, vol)
    actual = Backtester(target_vol=.12, max_leverage=2.0, tc_bps=0).run(
        signals.detach().numpy(), vol.numpy(), returns.numpy())['gross_returns'].to_numpy()
    expected = -np.sqrt(252) * np.mean(actual) / np.sqrt(np.var(actual, ddof=1) + 1e-6)
    assert np.isclose(loss.item(), expected, rtol=1e-5)
    loss.backward()
    assert torch.isfinite(signals.grad).all()


def test_bounded_candidates_and_validation_only_selection():
    import copy
    from src.training.tune import bounded_model_candidates, select_validation_candidate
    models = {'dlinear': {'trend_window': 20}, 'lstm': {'hidden_dim': 64},
              'mamba': {'d_model': 64}, 'patchtst': {'d_model': 64, 'heads': 4},
              'ahat_fin': {'d_model': 64, 'heads': 4}}
    before = copy.deepcopy(models)
    for name in models:
        candidates = bounded_model_candidates(name, models, [.001, .0003], 2)
        assert len(candidates) == 2
        assert candidates[0]['models'][name] != candidates[1]['models'][name]
    assert models == before
    trials = [{'negative_validation_sharpe': -.2, 'test_score': 100},
              {'negative_validation_sharpe': -.5, 'test_score': -100}]
    assert select_validation_candidate(trials) is trials[1]

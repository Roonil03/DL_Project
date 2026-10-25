import numpy as np
import pandas as pd
import pytest
import torch

from src.backtest.backtester import Backtester
from src.data.scaling import TimeSeriesScaler
from src.data.splits import chronological_split
from src.evaluation.regime_metrics import calculate_regime_robustness, evaluate_by_regime
from src.evaluation.statistical_tests import paired_sharpe_difference_test
from src.losses.sharpe_loss import NegativeSharpeLoss
from src.regimes.rules import detect_regimes


def test_split_is_chronological_and_test_is_untouched():
    frame = pd.DataFrame({'x': np.arange(100.0)}, index=pd.date_range('2020-01-01', periods=100))
    train, validation, test = chronological_split(frame)
    original_test = test.copy(deep=True)
    assert train.index.max() < validation.index.min() < test.index.min()
    train.iloc[:] = -1
    validation.iloc[:] = -2
    pd.testing.assert_frame_equal(test, original_test)


def test_scaler_fits_train_only_and_rejects_unfitted_use():
    train = np.arange(20.0).reshape(10, 2)
    future = np.full((3, 2), 10_000.0)
    scaler = TimeSeriesScaler()
    with pytest.raises(ValueError):
        scaler.transform(future)
    scaler.fit(train)
    np.testing.assert_allclose(scaler.mean, train.mean(axis=0))
    assert np.abs(scaler.transform(future)).min() > 100


def test_transaction_costs_and_leverage_cap_are_enforced():
    signals = np.full((8, 2), 100.0)
    volatility = np.full((8, 2), 0.01)
    returns = np.tile([0.01, -0.005], (8, 1))
    zero = Backtester(target_vol=0.1, max_leverage=2.0, tc_bps=0).run(signals, volatility, returns)
    costed = Backtester(target_vol=0.1, max_leverage=2.0, tc_bps=10).run(signals, volatility, returns)
    assert np.abs(costed['positions'].to_numpy()).max() <= 2.0
    np.testing.assert_allclose(zero['gross_returns'], costed['gross_returns'])
    assert (costed['net_returns'] <= costed['gross_returns'] + 1e-15).all()


def test_regime_construction_is_causal_and_metrics_report_counts():
    index = pd.date_range('2018-01-01', periods=260)
    prices = pd.DataFrame({'a': 100 * np.exp(np.arange(260) * 0.001)}, index=index)
    base = detect_regimes(prices, vol_window=20, trend_window=40)
    changed = prices.copy()
    changed.iloc[-1] *= 4
    perturbed = detect_regimes(changed, vol_window=20, trend_window=40)
    pd.testing.assert_series_equal(base.iloc[:-1], perturbed.iloc[:-1])
    returns = pd.Series(np.linspace(-0.01, 0.01, len(index)), index=index)
    result = evaluate_by_regime(returns, base, confidence_intervals=True, block_size=5, num_bootstraps=20)
    assert {'observations', 'sharpe_ci_lower', 'sharpe_ci_upper'}.issubset(result.columns)
    robustness = calculate_regime_robustness(returns, base)
    assert np.isfinite(list(robustness.values())).all()


def test_sharpe_loss_is_finite_and_differentiable():
    signals = torch.zeros((16, 3), requires_grad=True)
    targets = torch.linspace(-0.01, 0.01, 48).reshape(16, 3)
    loss = NegativeSharpeLoss()(signals, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert signals.grad is not None and torch.isfinite(signals.grad).all()


def test_paired_block_bootstrap_validates_hypothesis_direction():
    a = np.linspace(-0.005, 0.01, 60)
    b = a - 0.001
    result = paired_sharpe_difference_test(a, b, num_bootstraps=40, block_size=6, alternative='greater')
    assert result['alternative'] == 'greater'
    assert result['observations'] == 60
    assert 0 <= result['p_value'] <= 1
    with pytest.raises(ValueError):
        paired_sharpe_difference_test(a, b, alternative='ambiguous')
